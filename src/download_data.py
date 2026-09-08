"""Download and unzip the Kaggle "asl-signs" competition dataset.

Kaggle API setup (one-time, manual):
    1. Accept the competition rules first, or every download 403s:
         https://www.kaggle.com/competitions/asl-signs/rules
    2. Authenticate, using either option:

       (a) OAuth — simplest, and what the current CLI recommends:
             kaggle auth login
           Credentials are cached locally; there is no token file to manage.

       (b) API token file — go to https://www.kaggle.com/settings/api and click
           "Create New Token" to download `kaggle.json`, then place it at:
             Windows:      C:\\Users\\<you>\\.kaggle\\kaggle.json
             Linux/macOS:  ~/.kaggle/kaggle.json  (then: chmod 600 ~/.kaggle/kaggle.json)
           Verified against kaggle 2.2.4, which still reads this file, though it now
           prefers the OAuth flow above.

    Either way the SDK is already declared in requirements.txt (pip install kaggle).

Usage:
    python src/download_data.py
    python src/download_data.py --data-dir data/raw
"""

import argparse
import zipfile
from pathlib import Path

COMPETITION = "asl-signs"


def download_and_extract(data_dir: Path) -> None:
    """Download the competition dataset via the Kaggle API and unzip it.

    Args:
        data_dir: Destination directory for the raw dataset files.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading competition '{COMPETITION}' to {data_dir} ...")
    api.competition_download_files(COMPETITION, path=str(data_dir), quiet=False)

    zip_path = data_dir / f"{COMPETITION}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected downloaded archive at {zip_path}, but it was not found."
        )

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink()
    print(f"Done. Dataset available at {data_dir}")


def main() -> None:
    """Parse CLI arguments and run the download."""
    parser = argparse.ArgumentParser(description="Download the Kaggle asl-signs dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to download and extract the dataset into (default: data/raw)",
    )
    args = parser.parse_args()
    download_and_extract(args.data_dir)


if __name__ == "__main__":
    main()
