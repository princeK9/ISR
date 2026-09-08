"""Train the baseline GRU and the Transformer under identical conditions.

Both models go through the same splits, the same cache, the same TrainConfig and the
same seeded batch order, so the gap between their scores reflects the architectures
rather than the training setup.

Usage (from the project root):
    python -m src.train
    python -m src.train --models gru
    python -m src.train --epochs 40 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to disk without needing a display
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.dataset import (
    CACHE_ROOT,
    DEFAULT_LANDMARK_TYPES,
    DEFAULT_MAX_LEN,
    NUM_TOP_SIGNS,
    ASLLandmarkDataset,
    cache_dir_for,
    make_splits,
    select_top_signs,
)
from src.models import BaselineGRU, TransformerClassifier
from src.train_utils import History, TrainConfig, fit, get_device, make_loaders, set_seed

FIGURE_DIR = Path("reports/figures")


def build_model(name: str, num_features: int, num_classes: int, max_len: int) -> torch.nn.Module:
    """Construct a model by name.

    Args:
        name: Either "gru" or "transformer".
        num_features: Width of the per-frame feature vector.
        num_classes: Number of sign classes.
        max_len: Sequence length.

    Returns:
        The instantiated model.

    Raises:
        ValueError: If `name` is not a known model.
    """
    if name == "gru":
        return BaselineGRU(num_features=num_features, num_classes=num_classes)
    if name == "transformer":
        return TransformerClassifier(
            num_features=num_features, num_classes=num_classes, max_len=max_len
        )
    raise ValueError(f"Unknown model {name!r}; expected 'gru' or 'transformer'")


def plot_history(histories: dict[str, History], out_path: Path) -> None:
    """Plot loss and accuracy curves for every trained model.

    Args:
        histories: Model name to its training history.
        out_path: Destination PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"gru": "#4C72B0", "transformer": "#C44E52"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, history in histories.items():
        epochs = range(1, len(history.train_loss) + 1)
        color = colors.get(name, None)
        axes[0].plot(epochs, history.train_loss, color=color, ls="--", label=f"{name} train")
        axes[0].plot(epochs, history.val_loss, color=color, label=f"{name} val")
        axes[1].plot(epochs, history.train_acc, color=color, ls="--", label=f"{name} train")
        axes[1].plot(epochs, history.val_acc, color=color, label=f"{name} val")
        if history.best_epoch > 0:
            axes[1].scatter([history.best_epoch], [history.best_val_acc], color=color, zorder=5)

    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy (dot = best epoch)")
    axes[1].axhline(1 / NUM_TOP_SIGNS, color="grey", ls=":", label="chance")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    """Train the requested models and write checkpoints, histories and curves."""
    parser = argparse.ArgumentParser(description="Train the ASL sign classifiers.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--models", nargs="+", default=["gru", "transformer"])
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--landmark-types", nargs="+", default=list(DEFAULT_LANDMARK_TYPES))
    parser.add_argument("--epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--tag", default="", help="Suffix for checkpoint/history filenames.")
    args = parser.parse_args()

    config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    device = get_device()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    signs = select_top_signs(train_df, num_signs=NUM_TOP_SIGNS)
    splits = make_splits(train_df)
    cache_dir = cache_dir_for(args.cache_root, args.landmark_types, args.max_len)

    datasets = {
        name: ASLLandmarkDataset(
            split_df,
            data_dir=args.data_dir,
            signs=signs,
            landmark_types=args.landmark_types,
            max_len=args.max_len,
            cache_dir=cache_dir,
        )
        for name, split_df in splits.items()
    }

    print(f"device: {device}  |  cache: {cache_dir}")
    print(f"features/frame: {datasets['train'].num_features}  classes: {datasets['train'].num_classes}")
    print(f"sequences: train {len(datasets['train'])}, val {len(datasets['val'])}, test {len(datasets['test'])}")
    print(f"config: {config}")
    print()

    histories: dict[str, History] = {}
    for name in args.models:
        # Re-seed before each model so both get the same init RNG stream and the same
        # batch order - otherwise the comparison quietly depends on training order.
        set_seed(config.seed)
        train_loader, val_loader = make_loaders(datasets["train"], datasets["val"], config)
        model = build_model(name, datasets["train"].num_features, datasets["train"].num_classes, args.max_len)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"=== {name} ({params:,} parameters) ===")
        checkpoint = args.checkpoint_dir / f"{name}{args.tag}_best.pt"
        history = fit(model, train_loader, val_loader, config, device, checkpoint)
        histories[name] = history

        print(f"  best val acc {history.best_val_acc:.4f} at epoch {history.best_epoch}")
        print(f"  total {history.total_seconds:.0f}s "
              f"({history.total_seconds / max(len(history.train_loss), 1):.0f}s/epoch)")
        print(f"  checkpoint -> {checkpoint}")
        print()

        history_path = args.checkpoint_dir / f"{name}{args.tag}_history.json"
        history_path.write_text(json.dumps(history.to_dict(), indent=2), encoding="utf-8")

    figure_path = args.figure_dir / f"training_curves{args.tag}.png"
    plot_history(histories, figure_path)
    print(f"curves -> {figure_path}")
    print()

    print("=== summary ===")
    print(f"{'model':>14} {'best val acc':>13} {'final train acc':>16} {'epochs':>7} {'time':>8}")
    for name, history in histories.items():
        print(
            f"{name:>14} {history.best_val_acc:>13.4f} {history.train_acc[-1]:>16.4f} "
            f"{len(history.train_loss):>7} {history.total_seconds:>7.0f}s"
        )


if __name__ == "__main__":
    main()
