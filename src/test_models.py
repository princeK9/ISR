"""Smoke test: run one real cached batch through both models before training.

Beyond checking shapes, this asserts the property that is easy to get silently wrong —
that padded positions cannot influence the prediction. A model with an inverted or
missing padding mask still trains and still produces plausible-looking numbers, so the
bug tends to surface only as unexplained accuracy loss much later.

Usage (from the project root):
    python -m src.test_models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
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
from src.models import BaselineGRU, TransformerClassifier


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_padding_is_ignored(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    """Check that corrupting padded positions leaves the logits unchanged.

    Args:
        model: Model to test, in eval mode.
        batch: A batch containing "landmarks" and "mask".

    Returns:
        The largest absolute logit change observed.

    Raises:
        AssertionError: If the batch has no padding to corrupt, or if the outputs move.
    """
    model.eval()
    landmarks, mask = batch["landmarks"], batch["mask"]
    padding = ~mask
    assert padding.any(), "batch has no padded positions, so this test proves nothing"

    with torch.no_grad():
        before = model(landmarks, mask)
        corrupted = landmarks.clone()
        # Large, obviously out-of-distribution values: if padding leaks at all, it shows.
        corrupted[padding] = torch.randn_like(corrupted[padding]) * 100.0
        after = model(corrupted, mask)

    delta = (before - after).abs().max().item()
    assert delta < 1e-4, f"padding changed the logits by {delta:.3e} - mask is not applied correctly"
    return delta


def main() -> None:
    """Instantiate both models, run one real batch, and verify shapes and masking."""
    parser = argparse.ArgumentParser(description="Smoke test both models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--landmark-types", nargs="+", default=list(DEFAULT_LANDMARK_TYPES))
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    signs = select_top_signs(train_df, num_signs=NUM_TOP_SIGNS)
    splits = make_splits(train_df)
    cache_dir = cache_dir_for(args.cache_root, args.landmark_types, args.max_len)

    dataset = ASLLandmarkDataset(
        splits["train"],
        data_dir=args.data_dir,
        signs=signs,
        landmark_types=args.landmark_types,
        max_len=args.max_len,
        cache_dir=cache_dir,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))

    print("=== batch (from cache) ===")
    for key, tensor in batch.items():
        print(f"{key:>10}: {tuple(tensor.shape)}  {tensor.dtype}")
    print(f"real frames per sequence: {batch['mask'].sum(dim=1).tolist()}")
    print()

    models = {
        "BaselineGRU": BaselineGRU(
            num_features=dataset.num_features, num_classes=dataset.num_classes
        ),
        "TransformerClassifier": TransformerClassifier(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            max_len=args.max_len,
        ),
    }

    for name, model in models.items():
        logits = model(batch["landmarks"], batch["mask"])
        expected = (args.batch_size, dataset.num_classes)

        print(f"=== {name} ===")
        print(f"parameters: {count_parameters(model):,}")
        print(f"logits:     {tuple(logits.shape)}  (expected {expected})")

        assert logits.shape == expected, f"{name} returned {tuple(logits.shape)}, expected {expected}"
        assert torch.isfinite(logits).all(), f"{name} produced non-finite logits"

        delta = assert_padding_is_ignored(model, batch)
        print(f"padding-leak check: max logit change {delta:.2e} (must be < 1e-4)  OK")

        # A freshly initialised model should sit near uniform, i.e. ln(num_classes).
        loss = torch.nn.functional.cross_entropy(logits, batch["label"]).item()
        import math

        print(f"initial loss: {loss:.3f} (uniform baseline ln(30) = {math.log(30):.3f})")
        print()

    print("All model smoke tests passed.")


if __name__ == "__main__":
    main()
