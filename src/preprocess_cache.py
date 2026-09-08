"""Precompute the whole 30-sign subset into memory-mappable arrays.

Decoding a sequence means reading a parquet file, reindexing it onto the full landmark
grid, interpolating gaps and padding — a few milliseconds each. Repeated for every item
of every epoch that dominates training time, so it is done once here and the result is
memory-mapped thereafter.

The cache stores sign names rather than integer labels, so it stays valid regardless of
how the label space is ordered later, and carries participant_id so signer-disjoint
splitting still works straight from the cache.

Usage (from the project root):
    python -m src.preprocess_cache
    python -m src.preprocess_cache --landmark-types face left_hand pose right_hand
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import (
    CACHE_ROOT,
    DEFAULT_LANDMARK_TYPES,
    DEFAULT_MAX_LEN,
    NUM_TOP_SIGNS,
    ASLLandmarkDataset,
    cache_dir_for,
    select_top_signs,
)


def build_cache(
    train_df: pd.DataFrame,
    data_dir: Path,
    signs: list[str],
    cache_dir: Path,
    landmark_types: tuple[str, ...],
    max_len: int,
    report_every: int = 1000,
) -> dict[str, object]:
    """Decode every sequence for `signs` and write the cache arrays.

    Args:
        train_df: Contents of train.csv.
        data_dir: Directory the `path` column is relative to.
        signs: Sign vocabulary to cache.
        cache_dir: Destination directory, created if absent.
        landmark_types: Landmark groups to include as features.
        max_len: Fixed sequence length.
        report_every: Progress print interval, in sequences.

    Returns:
        Summary dict with the row count, feature width, elapsed seconds and bytes written.
    """
    # A cache-less dataset over the full subset: this is the slow parquet path, used
    # here deliberately, and it owns the single definition of the preprocessing.
    source = ASLLandmarkDataset(
        train_df,
        data_dir=data_dir,
        signs=signs,
        landmark_types=landmark_types,
        max_len=max_len,
    )
    num_rows = len(source)
    if num_rows == 0:
        raise ValueError("No sequences matched the requested signs; nothing to cache.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    landmarks_path = cache_dir / "landmarks.npy"
    mask_path = cache_dir / "mask.npy"

    # open_memmap writes straight to disk, so peak memory stays at one sequence rather
    # than the whole (num_rows, max_len, num_features) array.
    landmarks_mm = np.lib.format.open_memmap(
        landmarks_path, mode="w+", dtype=np.float32, shape=(num_rows, max_len, source.num_features)
    )
    mask_mm = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=bool, shape=(num_rows, max_len)
    )

    started = time.perf_counter()
    for i, row in enumerate(source.df.itertuples()):
        features, mask = source.process_sequence(data_dir / row.path)
        landmarks_mm[i] = features
        mask_mm[i] = mask
        if (i + 1) % report_every == 0:
            rate = (i + 1) / (time.perf_counter() - started)
            print(f"  {i + 1}/{num_rows} sequences ({rate:.0f}/s)")

    landmarks_mm.flush()
    mask_mm.flush()
    del landmarks_mm, mask_mm

    np.savez(
        cache_dir / "meta.npz",
        sequence_id=source.df["sequence_id"].to_numpy(dtype=np.int64),
        participant_id=source.df["participant_id"].to_numpy(dtype=np.int64),
        sign=source.df["sign"].to_numpy(dtype=np.str_),
        landmark_types=np.array(source.landmark_types, dtype=np.str_),
        hand_types=np.array(source.hand_types, dtype=np.str_),
        max_len=np.int64(max_len),
        num_features=np.int64(source.num_features),
    )

    elapsed = time.perf_counter() - started
    written = sum(p.stat().st_size for p in cache_dir.glob("*") if p.is_file())
    return {
        "rows": num_rows,
        "num_features": source.num_features,
        "elapsed_s": elapsed,
        "bytes": written,
        "missing_summary": source.missing_summary(),
    }


def main() -> None:
    """Parse CLI arguments and build the cache for one configuration."""
    parser = argparse.ArgumentParser(description="Precompute the landmark cache.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num-signs", type=int, default=NUM_TOP_SIGNS)
    parser.add_argument("--landmark-types", nargs="+", default=list(DEFAULT_LANDMARK_TYPES))
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the cache already exists."
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    signs = select_top_signs(train_df, num_signs=args.num_signs)
    cache_dir = cache_dir_for(args.cache_root, args.landmark_types, args.max_len)

    if (cache_dir / "meta.npz").exists() and not args.force:
        print(f"Cache already present at {cache_dir} (use --force to rebuild).")
        return

    print(f"Building cache -> {cache_dir}")
    print(f"  landmark_types: {args.landmark_types}")
    print(f"  max_len: {args.max_len}, signs: {len(signs)}")
    summary = build_cache(
        train_df,
        data_dir=args.data_dir,
        signs=signs,
        cache_dir=cache_dir,
        landmark_types=tuple(args.landmark_types),
        max_len=args.max_len,
    )

    print()
    print(f"Cached {summary['rows']} sequences x {summary['num_features']} features "
          f"in {summary['elapsed_s']:.0f}s")
    print(f"Cache size: {summary['bytes'] / 1e9:.2f} GB")
    print(f"Missing-landmark stats over the full subset: {summary['missing_summary']}")


if __name__ == "__main__":
    main()
