"""Evaluate trained checkpoints on the held-out (unseen-signer) test split.

Reports test accuracy for every model, a per-class precision/recall table, and a
confusion-matrix heatmap for the best one. Also supports the face-landmark ablation:
pass a cache built with face included and compare the resulting accuracy.

Usage (from the project root):
    python -m src.evaluate
    python -m src.evaluate --models transformer --landmark-types face left_hand pose right_hand --tag _face
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

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
from src.train import build_model
from src.train_utils import evaluate, get_device

FIGURE_DIR = Path("reports/figures")


def plot_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Save a row-normalised confusion matrix heatmap.

    Rows are normalised to recall per true class, so classes with slightly different
    support stay comparable by eye.

    Args:
        labels: True label indices.
        predictions: Predicted label indices.
        class_names: Class names in label order.
        out_path: Destination PNG.
        title: Figure title.
    """
    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(normalised, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(image, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    """Score every requested checkpoint on the test split and write reports."""
    parser = argparse.ArgumentParser(description="Evaluate trained ASL classifiers.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--models", nargs="+", default=["gru", "transformer"])
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--landmark-types", nargs="+", default=list(DEFAULT_LANDMARK_TYPES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tag", default="", help="Checkpoint filename suffix.")
    args = parser.parse_args()

    device = get_device()
    train_df = pd.read_csv(args.data_dir / "train.csv")
    signs = select_top_signs(train_df, num_signs=NUM_TOP_SIGNS)
    splits = make_splits(train_df)
    cache_dir = cache_dir_for(args.cache_root, args.landmark_types, args.max_len)

    def build_loader(split: str) -> tuple[ASLLandmarkDataset, DataLoader]:
        dataset = ASLLandmarkDataset(
            splits[split],
            data_dir=args.data_dir,
            signs=signs,
            landmark_types=args.landmark_types,
            max_len=args.max_len,
            cache_dir=cache_dir,
        )
        return dataset, DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    val_set, val_loader = build_loader("val")
    test_set, loader = build_loader("test")
    criterion = torch.nn.CrossEntropyLoss()

    print(f"test split: {len(test_set)} sequences from "
          f"{splits['test']['participant_id'].nunique()} unseen signers")
    print(f"features/frame: {test_set.num_features}  |  cache: {cache_dir}")
    print()

    results: dict[str, dict[str, object]] = {}
    for name in args.models:
        checkpoint = args.checkpoint_dir / f"{name}{args.tag}_best.pt"
        if not checkpoint.exists():
            print(f"skipping {name}: no checkpoint at {checkpoint}")
            continue

        model = build_model(name, test_set.num_features, test_set.num_classes, args.max_len)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)

        _, val_accuracy = evaluate(model, val_loader, criterion, device)
        loss, accuracy, predictions, labels = evaluate(
            model, loader, criterion, device, return_predictions=True
        )
        results[name] = {
            "loss": loss,
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "predictions": predictions,
            "labels": labels,
        }
        print(f"{name:>14}: val accuracy {val_accuracy:.4f}  |  "
              f"test loss {loss:.4f}  test accuracy {accuracy:.4f}")

    if not results:
        print("No checkpoints found; train first with: python -m src.train")
        return

    # Model selection uses validation accuracy, never test: picking the "best" model by
    # its test score would leak the held-out split into the choice and inflate the
    # headline number. Test is read once, for reporting, after the choice is made.
    best_name = max(results, key=lambda k: results[k]["val_accuracy"])  # type: ignore[arg-type]
    best = results[best_name]
    print()
    print(f"=== per-class report: {best_name} (selected on val accuracy) ===")
    print(
        classification_report(
            best["labels"],
            best["predictions"],
            labels=range(len(test_set.signs)),
            target_names=test_set.signs,
            digits=3,
            zero_division=0,
        )
    )

    figure_path = args.figure_dir / f"confusion_{best_name}{args.tag}.png"
    plot_confusion(
        best["labels"],  # type: ignore[arg-type]
        best["predictions"],  # type: ignore[arg-type]
        test_set.signs,
        figure_path,
        title=f"{best_name} — test confusion matrix (row-normalised)",
    )
    print(f"confusion matrix -> {figure_path}")

    summary_path = args.checkpoint_dir / f"test_results{args.tag}.json"
    summary_path.write_text(
        json.dumps(
            {
                name: {
                    "test_loss": r["loss"],
                    "test_accuracy": r["accuracy"],
                    "val_accuracy": r["val_accuracy"],
                    "selected_on_val": name == best_name,
                }
                for name, r in results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
