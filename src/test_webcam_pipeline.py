"""Verify the live webcam preprocessing matches the training pipeline exactly.

The webcam demo cannot reuse `ASLLandmarkDataset.process_sequence` directly, because that
starts from a parquet file and live frames have none. It instead rebuilds the same dense
landmark array from MediaPipe output and calls the same helpers. That leaves room for the
two paths to diverge, which would be silent — the demo would run and mislabel everything.

So this test feeds identical synthetic landmarks through both paths and asserts the model
inputs are bit-identical, then exercises the edge cases that a live camera actually produces:
a partly-filled window, no hands in frame, and no pose detected at all.

Needs no camera, no MediaPipe, and no dataset.

Usage (from the project root):
    python -m src.test_webcam_pipeline
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.dataset import (
    DEFAULT_LANDMARK_TYPES,
    DEFAULT_MAX_LEN,
    HAND_TYPES,
    LANDMARK_COUNTS,
    ASLLandmarkDataset,
)
from src.webcam_demo import (
    FALLBACK_SIGNS,
    landmark_offsets,
    result_to_frame_array,
    window_to_features,
)


def make_synthetic_sequence(
    num_frames: int,
    total_landmarks: int,
    offsets: dict[str, tuple[int, int]],
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic landmark sequence with realistic missing-data patterns.

    Args:
        num_frames: Frames to generate.
        total_landmarks: Landmarks per frame.
        offsets: Group row ranges.
        seed: RNG seed.

    Returns:
        Array of shape (num_frames, total_landmarks, 3) containing NaN where a landmark is
        "undetected": the left hand for the whole sequence (a one-handed sign), and the
        right hand for a few frames in the middle (a detection dropout).
    """
    rng = np.random.default_rng(seed)
    # Coordinates roughly in the [0, 1] normalized range MediaPipe emits.
    sequence = rng.uniform(0.2, 0.8, size=(num_frames, total_landmarks, 3)).astype(np.float32)

    left_start, left_stop = offsets["left_hand"]
    sequence[:, left_start:left_stop, :] = np.nan

    right_start, right_stop = offsets["right_hand"]
    dropout = slice(num_frames // 3, num_frames // 3 + 2)
    sequence[dropout, right_start:right_stop, :] = np.nan

    return sequence


def write_parquet(sequence: np.ndarray, offsets, path: Path) -> None:
    """Write a synthetic sequence in the dataset's parquet schema.

    Args:
        sequence: Array of shape (num_frames, total_landmarks, 3).
        offsets: Group row ranges, used to label each landmark's type.
        path: Destination parquet file.
    """
    types: list[str] = []
    indices: list[int] = []
    for name, (start, stop) in sorted(offsets.items(), key=lambda item: item[1][0]):
        types.extend([name] * (stop - start))
        indices.extend(range(stop - start))

    num_frames, total_landmarks, _ = sequence.shape
    frame_column = np.repeat(np.arange(num_frames), total_landmarks)
    flat = sequence.reshape(-1, 3)

    pd.DataFrame(
        {
            "frame": frame_column.astype(np.int16),
            "row_id": [f"{f}-{t}-{i}" for f, t, i in
                       zip(frame_column, types * num_frames, indices * num_frames)],
            "type": types * num_frames,
            "landmark_index": np.array(indices * num_frames, dtype=np.int16),
            "x": flat[:, 0].astype(np.float64),
            "y": flat[:, 1].astype(np.float64),
            "z": flat[:, 2].astype(np.float64),
        }
    ).to_parquet(path, index=False)


def fake_result(frame: np.ndarray, offsets) -> SimpleNamespace:
    """Build a stand-in for a HolisticLandmarkerResult from one frame array.

    Mirrors MediaPipe's behaviour: a group with no detection yields an empty list rather
    than a list of NaNs, which is exactly the case `result_to_frame_array` must map back
    onto NaN.

    Args:
        frame: Array of shape (total_landmarks, 3).
        offsets: Group row ranges.

    Returns:
        Object exposing pose_landmarks, left_hand_landmarks and right_hand_landmarks.
    """
    fields = {}
    for name, (start, stop) in offsets.items():
        block = frame[start:stop]
        if np.isnan(block).all():
            fields[f"{name}_landmarks"] = []
        else:
            fields[f"{name}_landmarks"] = [
                SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in block
            ]
    fields["pose_landmarks"] = fields.pop("pose_landmarks", [])
    return SimpleNamespace(**fields)


def check_matches_training(offsets, total_landmarks, pose_slice, hand_slices) -> None:
    """Assert the live path and the parquet path produce identical model input."""
    print("=== live path vs training path (identical synthetic landmarks) ===")

    for num_frames, description in ((40, "shorter than max_len"), (100, "longer than max_len")):
        sequence = make_synthetic_sequence(num_frames, total_landmarks, offsets)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_parquet(sequence, offsets, tmp_path / "seq.parquet")

            df = pd.DataFrame(
                [{"path": "seq.parquet", "participant_id": 1, "sequence_id": 1, "sign": "cow"}]
            )
            dataset = ASLLandmarkDataset(
                df, data_dir=tmp_path, signs=["cow"],
                landmark_types=DEFAULT_LANDMARK_TYPES, max_len=DEFAULT_MAX_LEN,
            )
            train_features, train_mask = dataset.process_sequence(tmp_path / "seq.parquet")

        # The live path never sees the parquet; it rebuilds frames from MediaPipe results.
        live_frames = [
            result_to_frame_array(fake_result(frame, offsets), offsets, total_landmarks)
            for frame in sequence
        ]
        live_features, live_mask = window_to_features(
            live_frames, pose_slice, hand_slices, DEFAULT_MAX_LEN
        )

        assert live_features.shape == train_features.shape, (
            f"shape mismatch: {live_features.shape} vs {train_features.shape}"
        )
        assert np.array_equal(live_mask, train_mask), "mask mismatch"
        assert np.array_equal(live_features, train_features, equal_nan=True), (
            f"features differ, max abs diff "
            f"{np.nanmax(np.abs(live_features - train_features)):.3e}"
        )
        print(f"  {num_frames:>3} frames ({description:<21}): identical  "
              f"{live_features.shape}, {int(live_mask.sum())} real frames")


def check_edge_cases(offsets, total_landmarks, pose_slice, hand_slices, num_features) -> None:
    """Exercise the conditions a live camera produces that a dataset file never does."""
    print()
    print("=== live-only edge cases ===")

    # A window with nothing detected at all: the camera is on but nobody is in frame.
    empty = [np.full((total_landmarks, 3), np.nan, dtype=np.float32) for _ in range(DEFAULT_MAX_LEN)]
    features, mask = window_to_features(empty, pose_slice, hand_slices, DEFAULT_MAX_LEN)
    assert features.shape == (DEFAULT_MAX_LEN, num_features)
    assert np.isfinite(features).all(), "all-missing window produced non-finite features"
    assert features[:, -2:].sum() == 0.0, "presence flags should be 0 when no hands detected"
    print(f"  nothing detected      : finite, presence flags all 0, shape {features.shape}")

    # Hands present but no pose, so there is no shoulder reference to normalize against.
    no_pose = make_synthetic_sequence(DEFAULT_MAX_LEN, total_landmarks, offsets, seed=1)
    no_pose[:, pose_slice[0] : pose_slice[1], :] = np.nan
    features, _ = window_to_features(no_pose, pose_slice, hand_slices, DEFAULT_MAX_LEN)
    assert np.isfinite(features).all(), "missing pose produced non-finite features"
    print("  no pose / no shoulders: finite, falls back to zero reference")

    # Partly-filled buffer: the demo refuses to predict here, but the function must still
    # behave, because --window can be set below max_len.
    short = make_synthetic_sequence(10, total_landmarks, offsets, seed=2)
    features, mask = window_to_features(list(short), pose_slice, hand_slices, DEFAULT_MAX_LEN)
    assert int(mask.sum()) == 10, f"expected 10 real frames, got {int(mask.sum())}"
    assert not features[mask.sum():].any(), "padded rows should be zero"
    print(f"  partial window (10)   : padded to {features.shape}, mask has 10 real frames")

    # A single frame, the minimum the buffer can hold.
    one = make_synthetic_sequence(1, total_landmarks, offsets, seed=3)
    features, mask = window_to_features(list(one), pose_slice, hand_slices, DEFAULT_MAX_LEN)
    assert int(mask.sum()) == 1 and np.isfinite(features).all()
    print("  single frame          : finite, 1 real frame")


def check_result_conversion(offsets, total_landmarks) -> None:
    """Verify MediaPipe-shaped output maps onto the training layout correctly."""
    print()
    print("=== MediaPipe result -> frame array ===")

    sequence = make_synthetic_sequence(1, total_landmarks, offsets, seed=4)
    frame = sequence[0]
    rebuilt = result_to_frame_array(fake_result(frame, offsets), offsets, total_landmarks)

    assert np.array_equal(rebuilt, frame, equal_nan=True), "round-trip through result failed"
    left_start, left_stop = offsets["left_hand"]
    assert np.isnan(rebuilt[left_start:left_stop]).all(), "undetected hand should stay NaN"
    print("  round-trip exact; undetected groups remain NaN")

    # An object with no landmark attributes at all, which is what a failed detection or a
    # different MediaPipe version could yield.
    blank = result_to_frame_array(SimpleNamespace(), offsets, total_landmarks)
    assert np.isnan(blank).all(), "empty result should be all NaN"
    print("  empty result          : all NaN, no exception")


def check_model_accepts_output(num_features) -> None:
    """Confirm a freshly built model consumes the live feature layout."""
    print()
    print("=== model accepts live features ===")
    from src.models import BaselineGRU

    model = BaselineGRU(num_features=num_features, num_classes=len(FALLBACK_SIGNS)).eval()
    features = np.zeros((DEFAULT_MAX_LEN, num_features), dtype=np.float32)
    mask = np.ones(DEFAULT_MAX_LEN, dtype=bool)

    with torch.no_grad():
        logits = model(
            torch.from_numpy(features).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
        )
    probabilities = torch.softmax(logits, dim=1)[0]

    assert logits.shape == (1, len(FALLBACK_SIGNS)), f"unexpected logits {tuple(logits.shape)}"
    assert torch.isfinite(logits).all()
    assert abs(float(probabilities.sum()) - 1.0) < 1e-5
    print(f"  logits {tuple(logits.shape)} finite, probabilities sum to 1")


def main() -> None:
    """Run every check."""
    landmark_types = DEFAULT_LANDMARK_TYPES
    offsets, total_landmarks = landmark_offsets(landmark_types)
    pose_slice = offsets["pose"]
    hand_types = tuple(t for t in landmark_types if t in HAND_TYPES)
    hand_slices = [offsets[t] for t in hand_types]
    num_features = 3 * total_landmarks + len(hand_slices)

    expected = 3 * sum(LANDMARK_COUNTS[t] for t in landmark_types) + len(hand_types)
    assert num_features == expected == 227, f"feature width drifted: {num_features}"
    print(f"layout: {landmark_types}, {total_landmarks} landmarks, {num_features} features/frame")
    print(f"offsets: {offsets}")
    print()

    check_matches_training(offsets, total_landmarks, pose_slice, hand_slices)
    check_edge_cases(offsets, total_landmarks, pose_slice, hand_slices, num_features)
    check_result_conversion(offsets, total_landmarks)
    check_model_accepts_output(num_features)

    print()
    print("All webcam pipeline checks passed.")


if __name__ == "__main__":
    main()
