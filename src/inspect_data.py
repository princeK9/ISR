"""Quick sanity-check inspection of the downloaded asl-signs dataset.

Loads train.csv, one sample landmark parquet file, and the sign-to-index
mapping, and prints summary stats to confirm everything looks as expected
before building the real data pipeline.

Usage:
    python src/inspect_data.py
    python src/inspect_data.py --data-dir data/raw
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def inspect_train_csv(data_dir: Path) -> pd.DataFrame:
    """Load train.csv and print shape, unique sign count, and unique participant count.

    Args:
        data_dir: Root directory containing train.csv.

    Returns:
        The loaded train.csv as a DataFrame.
    """
    train_csv_path = data_dir / "train.csv"
    df = pd.read_csv(train_csv_path)

    print("=== train.csv ===")
    print(f"Shape: {df.shape}")
    print(f"Unique signs: {df['sign'].nunique()}")
    print(f"Unique participants: {df['participant_id'].nunique()}")
    print()
    return df


def inspect_sample_parquet(data_dir: Path, relative_path: str) -> pd.DataFrame:
    """Load one landmark parquet file and print its columns, dtypes, and shape.

    Args:
        data_dir: Root directory the parquet path is relative to.
        relative_path: Value of the "path" column for the chosen sample row.

    Returns:
        The loaded parquet file as a DataFrame.
    """
    parquet_path = data_dir / relative_path
    df = pd.read_parquet(parquet_path)

    print("=== sample parquet file ===")
    print(f"Path: {parquet_path}")
    print(f"Shape: {df.shape}")
    print("Columns and dtypes:")
    print(df.dtypes)
    print()
    return df


def inspect_frame_counts(data_dir: Path, train_df: pd.DataFrame, sample_size: int) -> None:
    """Print min/max/mean frame count per sequence_id across a sample of sequences.

    Args:
        data_dir: Root directory the parquet paths are relative to.
        train_df: The loaded train.csv DataFrame.
        sample_size: Number of sequences to sample for the frame-count stats
            (reading every parquet file would be slow for a quick sanity check).
    """
    sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)

    frame_counts = []
    for _, row in sample.iterrows():
        df = pd.read_parquet(data_dir / row["path"], columns=["frame"])
        frame_counts.append(df["frame"].nunique())

    counts = pd.Series(frame_counts)
    print(f"=== frame counts (sample of {len(counts)} sequences) ===")
    print(f"Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.1f}")
    print()


def inspect_sign_mapping(data_dir: Path, num_examples: int) -> None:
    """Load sign_to_prediction_index_map.json and print a few example mappings.

    Args:
        data_dir: Root directory containing sign_to_prediction_index_map.json.
        num_examples: Number of example mappings to print.
    """
    mapping_path = data_dir / "sign_to_prediction_index_map.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping: dict[str, int] = json.load(f)

    print("=== sign_to_prediction_index_map.json ===")
    print(f"Total signs: {len(mapping)}")
    for sign, index in list(mapping.items())[:num_examples]:
        print(f"  {sign!r} -> {index}")
    print()


def main() -> None:
    """Parse CLI arguments and run all inspection steps."""
    parser = argparse.ArgumentParser(description="Inspect the asl-signs dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing train.csv, train_landmark_files/, etc. (default: data/raw)",
    )
    parser.add_argument(
        "--frame-count-sample-size",
        type=int,
        default=200,
        help="Number of sequences to sample when computing frame count stats (default: 200)",
    )
    args = parser.parse_args()

    train_df = inspect_train_csv(args.data_dir)
    inspect_sample_parquet(args.data_dir, train_df.iloc[0]["path"])
    inspect_frame_counts(args.data_dir, train_df, args.frame_count_sample_size)
    inspect_sign_mapping(args.data_dir, num_examples=5)


if __name__ == "__main__":
    main()
