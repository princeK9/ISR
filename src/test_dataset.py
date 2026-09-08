"""Smoke test: build the train split, pull one batch, and print its shapes.

Confirms the Dataset, the signer-disjoint splits, and the DataLoader are wired
together correctly before any model exists.

Usage (from the project root):
    python -m src.test_dataset
    python -m src.test_dataset --landmark-types face left_hand pose right_hand
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DEFAULT_LANDMARK_TYPES,
    DEFAULT_MAX_LEN,
    NUM_TOP_SIGNS,
    ASLLandmarkDataset,
    make_splits,
    select_top_signs,
)


def main() -> None:
    """Build the train split, fetch one batch, and print shapes and sanity checks."""
    parser = argparse.ArgumentParser(description="Smoke test the ASL landmark Dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num-signs", type=int, default=NUM_TOP_SIGNS)
    parser.add_argument("--landmark-types", nargs="+", default=list(DEFAULT_LANDMARK_TYPES))
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    top_signs = select_top_signs(train_df, num_signs=args.num_signs)
    splits = make_splits(train_df)

    print("=== signer-disjoint splits ===")
    for name, split_df in splits.items():
        participants = sorted(split_df["participant_id"].unique())
        in_subset = split_df[split_df["sign"].isin(top_signs)]
        print(
            f"{name:>5}: {len(participants)} participants, "
            f"{len(in_subset)} sequences in the {args.num_signs}-sign subset"
        )
    overlap = set(splits["train"]["participant_id"]) & set(splits["test"]["participant_id"])
    print(f"train/test participant overlap: {len(overlap)} (must be 0)")
    print()

    dataset = ASLLandmarkDataset(
        splits["train"],
        data_dir=args.data_dir,
        signs=top_signs,
        landmark_types=args.landmark_types,
        max_len=args.max_len,
    )
    print("=== dataset ===")
    print(f"Landmark types: {dataset.landmark_types}")
    print(f"Sequences: {len(dataset)}, classes: {dataset.num_classes}, "
          f"features/frame: {dataset.num_features}")
    print()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print("=== one batch ===")
    for key, tensor in batch.items():
        print(f"{key:>10}: shape {tuple(tensor.shape)}, dtype {tensor.dtype}")
    print(f"Real frames per sequence: {batch['mask'].sum(dim=1).tolist()}")
    print(f"Labels: {batch['label'].tolist()}")
    print(f"Signs:  {[dataset.signs[i] for i in batch['label'].tolist()]}")
    print()

    assert not torch.isnan(batch["landmarks"]).any(), "NaNs survived preprocessing"
    assert batch["landmarks"].shape == (args.batch_size, args.max_len, dataset.num_features)
    assert batch["mask"].shape == (args.batch_size, args.max_len)
    padding = batch["landmarks"][~batch["mask"]]
    assert padding.numel() == 0 or torch.all(padding == 0), "Padded frames are not zero"

    print("=== missing landmark data (this batch) ===")
    for key, value in dataset.missing_summary().items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
