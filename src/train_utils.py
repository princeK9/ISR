"""Training and evaluation loops shared by every model.

Both models are trained through exactly this code, with the same optimizer, loss,
schedule, seeds and batch order. That is the point: any difference in the reported
numbers comes from the architectures rather than from one of them getting a better
training setup.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    """Hyperparameters shared by both models.

    Starting points, not tuned results — adjust these first if training looks wrong:
    `lr` (1e-3 is a sane AdamW default; halve it if the loss spikes or oscillates),
    `batch_size` (32 balances gradient noise against CPU throughput here), and
    `num_epochs`/`patience` (raise if val accuracy is still climbing when it stops).
    `grad_clip` mostly matters for the Transformer; it is applied to both so the
    comparison stays like-for-like.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 32
    num_epochs: int = 30
    patience: int = 5
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42


@dataclass
class History:
    """Per-epoch metrics, for plotting and for the final report."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_acc: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return the history as plain types, ready for JSON."""
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "epoch_seconds": self.epoch_seconds,
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "total_seconds": self.total_seconds,
        }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch so runs are reproducible.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if it is available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    train_set: Dataset,
    val_set: Dataset,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders with a seeded shuffle.

    The generator is seeded from `config.seed`, so every model trained with the same
    config sees batches in the same order — one less confound in the comparison.

    Args:
        train_set: Training dataset.
        val_set: Validation dataset.
        config: Hyperparameters.

    Returns:
        Tuple of (train loader, val loader).
    """
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: Model taking (landmarks, mask) and returning logits.
        loader: Training data loader.
        optimizer: Optimizer to step.
        criterion: Loss function.
        device: Device to train on.
        grad_clip: Max gradient norm; values <= 0 disable clipping.

    Returns:
        Tuple of (mean loss per sample, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for batch in loader:
        landmarks = batch["landmarks"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(landmarks, mask)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_size

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate a model without updating it.

    Args:
        model: Model taking (landmarks, mask) and returning logits.
        loader: Data loader to evaluate over.
        criterion: Loss function.
        device: Device to run on.
        return_predictions: Also return predicted and true labels, for the confusion
            matrix and per-class report.

    Returns:
        (loss, accuracy), plus (predictions, labels) when `return_predictions` is set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        landmarks = batch["landmarks"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(landmarks, mask)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        seen += batch_size

        if return_predictions:
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    loss_value = total_loss / seen
    accuracy = correct / seen
    if return_predictions:
        return loss_value, accuracy, np.concatenate(all_preds), np.concatenate(all_labels)
    return loss_value, accuracy


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: Path,
    verbose: bool = True,
) -> History:
    """Train a model to convergence with early stopping on validation accuracy.

    The best checkpoint by validation accuracy is written to `checkpoint_path`, so the
    reported test score comes from the best epoch rather than the last one.

    Args:
        model: Model to train.
        train_loader: Training batches.
        val_loader: Validation batches.
        config: Hyperparameters.
        device: Device to train on.
        checkpoint_path: Where to save the best state dict.
        verbose: Print a line per epoch.

    Returns:
        The training history.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = History()
    epochs_without_improvement = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    for epoch in range(1, config.num_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - epoch_started

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.epoch_seconds.append(elapsed)

        improved = val_acc > history.best_val_acc
        if improved:
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"  epoch {epoch:>2}/{config.num_epochs}  "
                f"train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}  "
                f"({elapsed:.0f}s){'  *best' if improved else ''}"
            )

        if epochs_without_improvement >= config.patience:
            if verbose:
                print(f"  early stopping: no val improvement for {config.patience} epochs")
            break

    history.total_seconds = time.perf_counter() - run_started
    return history
