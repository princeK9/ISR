"""PyTorch Dataset and signer-disjoint splitting for the Kaggle asl-signs dataset.

Each training example is one sequence of MediaPipe Holistic landmarks stored as a
parquet file (columns: frame, row_id, type, landmark_index, x, y, z). This module
turns those files into fixed-length (max_len, num_features) float tensors plus an
attention mask, restricted to a subset of the most frequent signs.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Landmark counts per MediaPipe Holistic group. Features are always emitted in this
# canonical order regardless of the order `landmark_types` is passed in, so the
# feature layout is reproducible across runs and ablations.
LANDMARK_COUNTS: dict[str, int] = {
    "face": 468,
    "left_hand": 21,
    "pose": 33,
    "right_hand": 21,
}
CANONICAL_TYPE_ORDER: tuple[str, ...] = ("face", "left_hand", "pose", "right_hand")
HAND_TYPES: tuple[str, ...] = ("left_hand", "right_hand")

# Face is excluded by default: 468 of the 543 landmarks are face points, and they
# contribute little to the meaning of most isolated signs relative to their cost.
DEFAULT_LANDMARK_TYPES: tuple[str, ...] = ("left_hand", "pose", "right_hand")

# MediaPipe Pose indices used to build the normalization reference point.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# Chosen from the sequence-length distribution measured in notebooks/eda.ipynb over the
# 30-sign subset: median 24, mean 39, p95 131, max 453. 64 frames cover 84.4% of sequences
# outright while keeping 47% of each padded batch as real data (128 would leave 71% padding
# for only 10 points more coverage), and it makes the Transformer's quadratic attention 4x
# cheaper. The 16% that run longer are uniformly subsampled rather than truncated, so a long
# sign keeps its ending at coarser temporal resolution instead of losing it.
DEFAULT_MAX_LEN = 64
NUM_TOP_SIGNS = 30

# Where preprocessed caches live; one subdirectory per feature configuration.
CACHE_ROOT = Path("data/processed/cache")


def select_top_signs(train_df: pd.DataFrame, num_signs: int = NUM_TOP_SIGNS) -> list[str]:
    """Select the most frequent signs in the dataset.

    Ties on sample count are broken by sign name, and the result is returned
    alphabetically sorted, so the selection (and therefore the label mapping
    derived from it) is identical across runs and across notebook/script.

    Args:
        train_df: Contents of train.csv.
        num_signs: How many signs to keep.

    Returns:
        Alphabetically sorted list of the selected sign names.
    """
    counts = train_df["sign"].value_counts()
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return sorted(sign for sign, _ in ranked[:num_signs])


def make_splits(
    train_df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split rows into train/val/test by participant, so no signer spans two splits.

    Participants are partitioned first and their sequences follow, which is what
    makes the val/test scores a measure of generalization to unseen signers rather
    than to unseen recordings of signers the model already memorized.

    Args:
        train_df: Contents of train.csv.
        ratios: Fractions of *participants* (not sequences) per split.
        seed: Seed for the participant shuffle.

    Returns:
        Mapping of "train"/"val"/"test" to the row subset for that split.

    Raises:
        ValueError: If the ratios do not sum to 1, or if there are too few
            participants to give every split at least one.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}")

    participants = np.sort(train_df["participant_id"].unique())
    shuffled = np.random.default_rng(seed).permutation(participants)

    n_total = len(shuffled)
    n_train = int(round(ratios[0] * n_total))
    n_val = int(round(ratios[1] * n_total))
    bounds = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    empty = [name for name, ids in bounds.items() if len(ids) == 0]
    if empty:
        raise ValueError(
            f"Splits {empty} got zero participants from {n_total} total with ratios {ratios}."
        )

    return {
        name: train_df[train_df["participant_id"].isin(ids)].reset_index(drop=True)
        for name, ids in bounds.items()
    }


def _read_landmark_array(path: Path, types: tuple[str, ...]) -> np.ndarray:
    """Read one sequence parquet file into a dense (num_frames, num_landmarks, 3) array.

    Rows are reindexed onto the full (frame x landmark) grid rather than trusting
    the file's row order or completeness, so a frame that omits an undetected hand
    entirely produces NaNs in the same positions as one that stores NaN coordinates.

    Args:
        path: Path to the sequence's parquet file.
        types: Landmark types to read, in canonical order.

    Returns:
        Array of shape (num_frames, sum(LANDMARK_COUNTS[t] for t in types), 3),
        with NaN wherever a landmark was not detected.
    """
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = df[df["type"].isin(types)]

    frames = np.sort(df["frame"].unique())
    type_labels = np.array([t for t in types for _ in range(LANDMARK_COUNTS[t])], dtype=object)
    type_indices = np.array([i for t in types for i in range(LANDMARK_COUNTS[t])])
    num_landmarks = len(type_labels)

    expected = pd.MultiIndex.from_arrays(
        [
            np.repeat(frames, num_landmarks),
            np.tile(type_labels, len(frames)),
            np.tile(type_indices, len(frames)),
        ],
        names=["frame", "type", "landmark_index"],
    )
    values = (
        df.set_index(["frame", "type", "landmark_index"])[["x", "y", "z"]]
        .reindex(expected)
        .to_numpy(dtype=np.float32)
    )
    return values.reshape(len(frames), num_landmarks, 3)


def _reference_points(pose_xyz: np.ndarray) -> tuple[np.ndarray, bool]:
    """Compute the per-frame normalization reference: the shoulder midpoint.

    The midpoint of the left and right shoulder is used rather than the nose or a
    hand landmark because it is near-stationary while the hands move, it sits at the
    center of the signing space so hand coordinates become directly interpretable as
    offsets from the body, and the shoulders are the pose points least likely to
    drop out (the nose disappears whenever the head turns, and hands leave frame
    constantly). It is recomputed per frame so the sequence stays normalized even if
    the signer shifts position mid-recording.

    Args:
        pose_xyz: Pose landmarks of shape (num_frames, 33, 3), possibly containing NaN.

    Returns:
        Tuple of (reference of shape (num_frames, 3), whether any usable shoulder
        was found in the sequence at all).
    """
    shoulders = pose_xyz[:, [POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER], :]
    with warnings.catch_warnings():
        # Frames with neither shoulder detected are expected, and handled just below;
        # numpy's "mean of empty slice" warning for them would otherwise flood training logs.
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = np.nanmean(shoulders, axis=1)

    if np.isnan(reference).all():
        # No shoulders anywhere in this sequence: fall back to no translation, and
        # let the caller record it rather than silently shifting by an arbitrary point.
        return np.zeros_like(reference), False

    # Frames whose shoulders were both undetected borrow the reference from
    # neighbouring frames, which is far closer than skipping normalization for them.
    filled = pd.DataFrame(reference).interpolate(
        method="linear", axis=0, limit_direction="both"
    )
    return filled.to_numpy(dtype=np.float32), True


def _fill_missing(landmarks: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Fill NaN landmark coordinates, reporting how much had to be invented.

    Gaps are interpolated along the time axis first, which is accurate for the common
    case of a hand blinking out for a few frames. Coordinates still missing afterwards
    belong to landmarks absent for the whole sequence (e.g. a one-handed sign never
    shows the other hand) and are set to 0.0 — which, post-normalization, means "at
    the shoulder midpoint": a neutral in-distribution value rather than an outlier.

    Args:
        landmarks: Array of shape (num_frames, num_landmarks, 3), possibly with NaN.

    Returns:
        Tuple of (filled array, count of originally-missing coordinates, count of
        coordinates that fell through to the zero fill).
    """
    num_frames, num_landmarks, _ = landmarks.shape
    flat = landmarks.reshape(num_frames, num_landmarks * 3)
    num_missing = int(np.isnan(flat).sum())

    interpolated = pd.DataFrame(flat).interpolate(
        method="linear", axis=0, limit_direction="both"
    )
    num_zero_filled = int(interpolated.isna().to_numpy().sum())
    filled = interpolated.fillna(0.0).to_numpy(dtype=np.float32)

    return filled.reshape(num_frames, num_landmarks, 3), num_missing, num_zero_filled


def _pad_or_subsample(features: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Force a sequence to `max_len` frames and build its attention mask.

    Over-long sequences are uniformly subsampled rather than truncated, because the
    end of a sign carries as much meaning as its start and cutting it off would
    discard the handshape the sign resolves to.

    Args:
        features: Array of shape (num_frames, num_features).
        max_len: Target sequence length.

    Returns:
        Tuple of (padded features of shape (max_len, num_features), boolean mask of
        shape (max_len,) that is True for real frames and False for padding).
    """
    num_frames, num_features = features.shape

    if num_frames >= max_len:
        keep = np.linspace(0, num_frames - 1, max_len).round().astype(int)
        return features[keep], np.ones(max_len, dtype=bool)

    padded = np.zeros((max_len, num_features), dtype=np.float32)
    padded[:num_frames] = features
    mask = np.zeros(max_len, dtype=bool)
    mask[:num_frames] = True
    return padded, mask


def _hand_presence(raw_landmarks: np.ndarray, hand_slices: list[tuple[int, int]]) -> np.ndarray:
    """Flag, per frame, whether each hand was actually detected.

    Must be called on the *raw* array, before `_fill_missing` runs: afterwards an absent
    hand is indistinguishable from a hand resting exactly at the shoulder midpoint, since
    both are zeros. Handing the model this bit explicitly means it never has to infer
    "hand absent" from an all-zero block — and absence is itself informative, because
    one-handed signs are a large fraction of the vocabulary.

    Args:
        raw_landmarks: Array of shape (num_frames, num_landmarks, 3) containing NaN
            wherever a landmark was not detected.
        hand_slices: (start, stop) row ranges of the hand groups within `raw_landmarks`.

    Returns:
        Float array of shape (num_frames, len(hand_slices)), 1.0 where that hand had at
        least one detected landmark in that frame and 0.0 otherwise.
    """
    columns = [
        (~np.isnan(raw_landmarks[:, start:stop, :]).all(axis=(1, 2))).astype(np.float32)
        for start, stop in hand_slices
    ]
    if not columns:
        return np.zeros((len(raw_landmarks), 0), dtype=np.float32)
    return np.stack(columns, axis=1)


def cache_dir_for(cache_root: Path, landmark_types: Iterable[str], max_len: int) -> Path:
    """Return the cache directory for one preprocessing configuration.

    The configuration is encoded in the directory name so that caches for different
    landmark sets or sequence lengths (e.g. the face ablation) coexist rather than
    silently overwriting each other.

    Args:
        cache_root: Directory holding all caches.
        landmark_types: Landmark groups included in the features.
        max_len: Fixed sequence length the cache was built with.

    Returns:
        Path to this configuration's cache directory.
    """
    ordered = [t for t in CANONICAL_TYPE_ORDER if t in set(landmark_types)]
    return Path(cache_root) / f"{'+'.join(ordered)}_len{max_len}"


class ASLLandmarkDataset(Dataset):
    """Fixed-length landmark sequences for a subset of signs, with attention masks.

    Attributes:
        signs: The sign vocabulary, alphabetically sorted.
        label_map: Sign name to integer label in [0, len(signs)).
        num_features: Width of the per-frame feature vector.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        signs: Iterable[str],
        landmark_types: Iterable[str] = DEFAULT_LANDMARK_TYPES,
        max_len: int = DEFAULT_MAX_LEN,
        cache_dir: Path | None = None,
    ) -> None:
        """Build a dataset over the rows of `df` whose sign is in `signs`.

        Args:
            df: Rows of train.csv for one split (see `make_splits`).
            data_dir: Directory that the `path` column is relative to.
            signs: Sign vocabulary to keep; all other rows are dropped.
            landmark_types: Which MediaPipe groups to emit as features. Pose is read
                regardless, since the normalization reference is derived from it.
            max_len: Fixed output sequence length.
            cache_dir: Preprocessed cache to read from (the fast path, built by
                src/preprocess_cache.py). When None, every item is decoded from its
                parquet file instead — same output, roughly two orders of magnitude
                slower, kept for debugging and for building the cache itself.

        Raises:
            ValueError: If `landmark_types` contains an unknown group or is empty.
            FileNotFoundError: If `cache_dir` is given but incomplete.
        """
        requested = set(landmark_types)
        unknown = requested - set(LANDMARK_COUNTS)
        if unknown:
            raise ValueError(
                f"Unknown landmark types {sorted(unknown)}; expected any of {sorted(LANDMARK_COUNTS)}"
            )
        if not requested:
            raise ValueError("landmark_types must not be empty")

        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.landmark_types = tuple(t for t in CANONICAL_TYPE_ORDER if t in requested)
        self._read_types = tuple(
            t for t in CANONICAL_TYPE_ORDER if t in requested | {"pose"}
        )

        # Slice of the read array holding the requested types, in canonical order.
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for landmark_type in self._read_types:
            offsets[landmark_type] = (cursor, cursor + LANDMARK_COUNTS[landmark_type])
            cursor += LANDMARK_COUNTS[landmark_type]
        self._pose_slice = offsets["pose"]
        self._feature_rows = np.concatenate(
            [np.arange(*offsets[t]) for t in self.landmark_types]
        )

        # One presence flag per hand group actually included in the features. A config
        # without hands (pose-only, say) simply gets none, keeping num_features honest.
        self.hand_types = tuple(t for t in self.landmark_types if t in HAND_TYPES)
        self._hand_slices = [offsets[t] for t in self.hand_types]

        self.signs = sorted(signs)
        self.label_map = {sign: index for index, sign in enumerate(self.signs)}
        self.df = df[df["sign"].isin(self.label_map)].reset_index(drop=True)

        self.num_coord_features = 3 * sum(LANDMARK_COUNTS[t] for t in self.landmark_types)
        self.num_features = self.num_coord_features + len(self.hand_types)

        self._missing: Counter[str] = Counter()

        # Cache is memory-mapped lazily, on first access inside each DataLoader worker:
        # np.memmap handles do not survive the spawn used for workers on Windows.
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache_rows: np.ndarray | None = None
        self._cache_landmarks: np.ndarray | None = None
        self._cache_mask: np.ndarray | None = None
        if self.cache_dir is not None:
            self._bind_cache()

    def __len__(self) -> int:
        """Number of sequences in this split."""
        return len(self.df)

    @property
    def num_classes(self) -> int:
        """Size of the label space."""
        return len(self.signs)

    def _bind_cache(self) -> None:
        """Map this split's rows onto rows of the preprocessed cache.

        Validates that the cache was built with the same feature configuration, so a
        stale cache fails loudly here instead of silently feeding the model the wrong
        feature layout.

        Raises:
            FileNotFoundError: If the cache directory is missing files.
            ValueError: If the cache config disagrees with this dataset, or if it does
                not contain every sequence in this split.
        """
        assert self.cache_dir is not None
        meta_path = self.cache_dir / "meta.npz"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No cache at {self.cache_dir}. Build it with: python -m src.preprocess_cache"
            )
        meta = np.load(meta_path, allow_pickle=False)

        cached_types = tuple(str(t) for t in meta["landmark_types"])
        if cached_types != self.landmark_types:
            raise ValueError(
                f"Cache at {self.cache_dir} holds landmark_types={cached_types}, "
                f"but this dataset expects {self.landmark_types}"
            )
        for name, expected in (("max_len", self.max_len), ("num_features", self.num_features)):
            if int(meta[name]) != expected:
                raise ValueError(
                    f"Cache at {self.cache_dir} has {name}={int(meta[name])}, expected {expected}"
                )

        row_of = {int(sid): i for i, sid in enumerate(meta["sequence_id"])}
        missing = [int(s) for s in self.df["sequence_id"] if int(s) not in row_of]
        if missing:
            raise ValueError(
                f"{len(missing)} sequences of this split are absent from the cache "
                f"(first few: {missing[:5]}). Rebuild it with src/preprocess_cache.py."
            )
        self._cache_rows = np.array(
            [row_of[int(sid)] for sid in self.df["sequence_id"]], dtype=np.int64
        )

    def _open_cache_arrays(self) -> None:
        """Memory-map the cached arrays, once per process."""
        assert self.cache_dir is not None
        self._cache_landmarks = np.load(self.cache_dir / "landmarks.npy", mmap_mode="r")
        self._cache_mask = np.load(self.cache_dir / "mask.npy", mmap_mode="r")

    def process_sequence(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Decode one parquet file into padded features and its attention mask.

        This is the single definition of the preprocessing pipeline: the cache builder
        and the parquet fallback both call it, so the two paths cannot drift apart.

        Args:
            path: Path to the sequence's parquet file.

        Returns:
            Tuple of (features of shape (max_len, num_features), boolean mask of shape
            (max_len,) that is True for real frames).
        """
        landmarks = _read_landmark_array(path, self._read_types)

        # Presence is read off the raw array, before any filling erases the distinction
        # between "absent" and "at the origin".
        presence = _hand_presence(landmarks, self._hand_slices)

        # Order matters: the reference is subtracted while gaps are still NaN, so that
        # filled-in values are neutral offsets rather than raw image coordinates.
        pose_xyz = landmarks[:, self._pose_slice[0] : self._pose_slice[1], :]
        reference, found_reference = _reference_points(pose_xyz)
        landmarks = landmarks - reference[:, None, :]

        landmarks = landmarks[:, self._feature_rows, :]
        landmarks, num_missing, num_zero_filled = _fill_missing(landmarks)

        self._missing["sequences"] += 1
        self._missing["coordinates"] += landmarks.size
        self._missing["missing"] += num_missing
        self._missing["zero_filled"] += num_zero_filled
        self._missing["sequences_without_reference"] += int(not found_reference)

        coords = landmarks.reshape(len(landmarks), self.num_coord_features)
        features = np.concatenate([coords, presence], axis=1, dtype=np.float32)
        return _pad_or_subsample(features, self.max_len)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one sequence, from the cache when available and parquet otherwise.

        Args:
            index: Row index into this split.

        Returns:
            Dict with "landmarks" (max_len, num_features) float32, "mask" (max_len,)
            bool where True marks a real frame, and "label" scalar int64.
        """
        row = self.df.iloc[index]

        if self.cache_dir is not None:
            if self._cache_landmarks is None:
                self._open_cache_arrays()
            assert self._cache_landmarks is not None and self._cache_mask is not None
            cache_row = int(self._cache_rows[index])  # type: ignore[index]
            # np.array() copies out of the memmap; torch must not alias the mapping.
            padded = np.array(self._cache_landmarks[cache_row], dtype=np.float32)
            mask = np.array(self._cache_mask[cache_row], dtype=bool)
        else:
            padded, mask = self.process_sequence(self.data_dir / row["path"])

        return {
            "landmarks": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(self.label_map[row["sign"]], dtype=torch.long),
        }

    def missing_summary(self) -> dict[str, float]:
        """Report how much landmark data was absent in the sequences loaded so far.

        Counts accumulate lazily as items are read, so this reflects the sequences
        touched up to now rather than the whole split.

        Returns:
            Dict of sequences seen, percentage of coordinates that were missing, the
            percentage that fell through interpolation to the zero fill, and the
            number of sequences with no usable shoulder reference.
        """
        seen = self._missing["coordinates"]
        return {
            "sequences_loaded": self._missing["sequences"],
            "missing_pct": 100.0 * self._missing["missing"] / seen if seen else 0.0,
            "zero_filled_pct": 100.0 * self._missing["zero_filled"] / seen if seen else 0.0,
            "sequences_without_reference": self._missing["sequences_without_reference"],
        }
