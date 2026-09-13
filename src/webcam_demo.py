"""Live webcam (or video file) demo for the trained ASL sign classifier.

The point of this module is that the live preprocessing is **not** a reimplementation of
the training pipeline — it imports and calls the exact same functions from `src.dataset`
(`_hand_presence`, `_reference_points`, `_fill_missing`, `_pad_or_subsample`) in the exact
same order as `ASLLandmarkDataset.process_sequence`. Only one step differs, and it has to:
the training path decodes a parquet file into a dense `(frames, 75, 3)` array with NaN for
undetected landmarks, whereas here that array is assembled from live MediaPipe output. From
that point on the two paths are identical, which `src.test_webcam_pipeline` verifies by
feeding the same synthetic landmarks through both and asserting bit-identical output.

MediaPipe note: `mediapipe.solutions.holistic` — the legacy API used by most tutorials — was
removed in MediaPipe 1.x. This module uses the current Tasks API
(`mediapipe.tasks.vision.HolisticLandmarker`), which needs a downloadable `.task` model
bundle; it is fetched automatically on first run unless `--no-download` is passed.

Expect live accuracy to be well below the 0.5285 test accuracy. The model was trained on
landmarks from a specific recording setup, and a different camera, distance, framing and
lighting is a genuine distribution shift. See docs/MANUAL_TESTING.md.

Usage (from the project root):
    python -m src.webcam_demo
    python -m src.webcam_demo --source clip.mp4 --save annotated.mp4
    python -m src.webcam_demo --predict-every 10 --window 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.dataset import (
    CANONICAL_TYPE_ORDER,
    DEFAULT_LANDMARK_TYPES,
    DEFAULT_MAX_LEN,
    HAND_TYPES,
    LANDMARK_COUNTS,
    NUM_TOP_SIGNS,
    _fill_missing,
    _hand_presence,
    _pad_or_subsample,
    _reference_points,
    select_top_signs,
)

# Verified reachable (HTTP 200, 13.7 MB) at the time of writing. If it moves, download the
# holistic bundle manually from the MediaPipe "Holistic landmarks detection" task docs and
# pass --model-path.
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/holistic_landmarker.task")

# Used only when neither data/processed/top30_signs.json nor data/raw/train.csv is present,
# so the demo still runs from a fresh clone without the 53 GB dataset. This is the recorded
# output of select_top_signs(train_df, 30); the count is checked against the checkpoint's
# classifier width at load time, so a mismatch fails loudly rather than mislabelling.
FALLBACK_SIGNS: tuple[str, ...] = (
    "awake", "bird", "brown", "bye", "cat", "cow", "doll", "donkey", "drink", "duck",
    "fireman", "first", "hear", "icecream", "lips", "listen", "look", "make", "mouse",
    "napkin", "nuts", "pen", "pretend", "shhh", "sleepy", "think", "toothbrush", "uncle",
    "wake", "who",
)

# Which state-dict entry holds the final classifier weight, per model name in src.train.
CLASSIFIER_WEIGHT_KEYS = {"gru": "classifier.weight", "transformer": "head.2.weight"}

# Drawing colours (BGR, because OpenCV).
COLOR_POSE = (176, 114, 76)
COLOR_LEFT = (104, 168, 85)
COLOR_RIGHT = (82, 78, 196)
COLOR_TEXT = (255, 255, 255)
COLOR_GOOD = (80, 175, 76)
COLOR_FAIR = (0, 170, 255)
COLOR_POOR = (60, 60, 220)


def landmark_offsets(
    landmark_types: tuple[str, ...],
) -> tuple[dict[str, tuple[int, int]], int]:
    """Compute where each landmark group sits in the flat per-frame array.

    Derived from LANDMARK_COUNTS rather than hardcoded, so this cannot drift from the
    layout `src.dataset` uses.

    Args:
        landmark_types: Groups to include, in canonical order.

    Returns:
        Tuple of (mapping from group name to (start, stop) row range, total landmark count).
    """
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for landmark_type in CANONICAL_TYPE_ORDER:
        if landmark_type in landmark_types:
            offsets[landmark_type] = (cursor, cursor + LANDMARK_COUNTS[landmark_type])
            cursor += LANDMARK_COUNTS[landmark_type]
    return offsets, cursor


def result_to_frame_array(
    result: object,
    offsets: dict[str, tuple[int, int]],
    total_landmarks: int,
) -> np.ndarray:
    """Convert one HolisticLandmarker result into the training pipeline's frame layout.

    This is the only step that differs from training, because training reads the same
    array out of a parquet file. Undetected groups stay NaN, exactly as an undetected
    landmark appears in the dataset — which is what lets the downstream steps behave
    identically.

    Args:
        result: A `HolisticLandmarkerResult`, or any object exposing `pose_landmarks`,
            `left_hand_landmarks` and `right_hand_landmarks` as sequences of objects with
            `.x`, `.y`, `.z` (empty or None when not detected).
        offsets: Group row ranges from `landmark_offsets`.
        total_landmarks: Total rows in the frame array.

    Returns:
        Array of shape (total_landmarks, 3), NaN wherever a landmark was not detected.
    """
    frame = np.full((total_landmarks, 3), np.nan, dtype=np.float32)

    for name, landmarks in (
        ("pose", getattr(result, "pose_landmarks", None)),
        ("left_hand", getattr(result, "left_hand_landmarks", None)),
        ("right_hand", getattr(result, "right_hand_landmarks", None)),
    ):
        if name not in offsets or not landmarks:
            continue
        start, stop = offsets[name]
        points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        # Guard against a model returning more points than the layout expects.
        usable = min(len(points), stop - start)
        frame[start : start + usable] = points[:usable]

    return frame


def window_to_features(
    window: list[np.ndarray] | np.ndarray,
    pose_slice: tuple[int, int],
    hand_slices: list[tuple[int, int]],
    max_len: int = DEFAULT_MAX_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a buffer of raw frames into model input, exactly as training does.

    Mirrors `ASLLandmarkDataset.process_sequence` step for step, calling the same helpers.
    The ordering is load-bearing and must not be rearranged: hand presence is read from the
    raw NaN pattern *before* filling (afterwards, an absent hand and a hand resting at the
    shoulder midpoint are both zeros), and the reference is subtracted while gaps are still
    NaN (so filled values are neutral offsets rather than raw image coordinates).

    Args:
        window: Sequence of (num_landmarks, 3) frame arrays, oldest first.
        pose_slice: (start, stop) row range of the pose group.
        hand_slices: Row ranges of the hand groups, in canonical order.
        max_len: Fixed output length.

    Returns:
        Tuple of (features of shape (max_len, num_features) float32, boolean mask of shape
        (max_len,) that is True for real frames).
    """
    raw = np.asarray(window, dtype=np.float32)

    presence = _hand_presence(raw, hand_slices)

    pose_xyz = raw[:, pose_slice[0] : pose_slice[1], :]
    reference, _found_reference = _reference_points(pose_xyz)
    centred = raw - reference[:, None, :]

    filled, _num_missing, _num_zero_filled = _fill_missing(centred)

    coords = filled.reshape(len(filled), -1)
    features = np.concatenate([coords, presence], axis=1, dtype=np.float32)
    return _pad_or_subsample(features, max_len)


def resolve_signs(
    signs_json: Path,
    data_dir: Path,
    num_signs: int = NUM_TOP_SIGNS,
) -> tuple[list[str], str]:
    """Load the label vocabulary, preferring artifacts over the built-in fallback.

    Args:
        signs_json: Cached vocabulary written by the EDA notebook.
        data_dir: Directory containing train.csv, used to re-derive if needed.
        num_signs: Vocabulary size.

    Returns:
        Tuple of (alphabetically sorted sign names, description of where they came from).
    """
    if signs_json.exists():
        signs = json.loads(signs_json.read_text(encoding="utf-8"))
        return sorted(signs), str(signs_json)

    train_csv = data_dir / "train.csv"
    if train_csv.exists():
        import pandas as pd

        return select_top_signs(pd.read_csv(train_csv), num_signs=num_signs), str(train_csv)

    return sorted(FALLBACK_SIGNS), "built-in fallback list"


def load_classifier(
    checkpoint: Path,
    model_name: str,
    num_features: int,
    signs: list[str],
    max_len: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained checkpoint and verify it agrees with the label vocabulary.

    Args:
        checkpoint: Path to a state dict saved by src/train.py.
        model_name: "gru" or "transformer".
        num_features: Per-frame feature width the demo will produce.
        signs: Label vocabulary.
        max_len: Sequence length.
        device: Device to load onto.

    Returns:
        The model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        ValueError: If the checkpoint's class count disagrees with the vocabulary.
    """
    from src.train import build_model

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Train one with: python -m src.train"
        )

    state = torch.load(checkpoint, map_location=device)

    # A vocabulary of the wrong size would silently mislabel every prediction, so check it
    # against the checkpoint's actual classifier width rather than trusting the default.
    weight_key = CLASSIFIER_WEIGHT_KEYS.get(model_name)
    if weight_key in state:
        checkpoint_classes = int(state[weight_key].shape[0])
        if checkpoint_classes != len(signs):
            raise ValueError(
                f"{checkpoint} was trained with {checkpoint_classes} classes but the label "
                f"vocabulary has {len(signs)}. Point --signs-json at the vocabulary this "
                f"checkpoint was trained with."
            )

    model = build_model(model_name, num_features, len(signs), max_len)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_window(
    model: torch.nn.Module,
    features: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run the model on one window and return class probabilities.

    Args:
        model: Loaded classifier in eval mode.
        features: Array of shape (max_len, num_features).
        mask: Boolean array of shape (max_len,).
        device: Device to run on.

    Returns:
        Probability vector of shape (num_classes,).
    """
    landmarks = torch.from_numpy(features).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    logits = model(landmarks, mask_tensor)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def ensure_model_bundle(path: Path, url: str, allow_download: bool) -> Path:
    """Make sure the MediaPipe .task bundle is present, downloading it if permitted.

    Args:
        path: Where the bundle should live.
        url: Source to download from if absent.
        allow_download: Whether downloading is permitted.

    Returns:
        The bundle path.

    Raises:
        FileNotFoundError: If the bundle is absent and downloading is not permitted.
    """
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"MediaPipe model bundle not found at {path}. Download holistic_landmarker.task "
            f"and pass --model-path, or drop --no-download to fetch it automatically."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe holistic bundle -> {path}")
    # Download to a temporary name first so an interrupted transfer cannot leave a
    # truncated file that later looks valid.
    partial = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    print(f"  done ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def draw_landmarks(frame: np.ndarray, frame_array: np.ndarray, offsets, width, height) -> None:
    """Draw the captured landmarks onto the video frame.

    Doubles as a sanity check: if the dots do not track the signer, capture is wrong
    regardless of what the classifier reports.

    Args:
        frame: BGR image to draw on, modified in place.
        frame_array: Landmarks of shape (num_landmarks, 3), possibly containing NaN.
        offsets: Group row ranges from `landmark_offsets`.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """
    import cv2

    for name, color, radius in (
        ("pose", COLOR_POSE, 2),
        ("left_hand", COLOR_LEFT, 3),
        ("right_hand", COLOR_RIGHT, 3),
    ):
        if name not in offsets:
            continue
        start, stop = offsets[name]
        for x, y, _z in frame_array[start:stop]:
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(frame, (int(x * width), int(y * height)), radius, color, -1)


def draw_overlay(
    frame: np.ndarray,
    label: str | None,
    confidence: float,
    filled: int,
    window: int,
    hands: tuple[bool, bool],
    fps: float,
    no_hands_in_window: bool = False,
) -> None:
    """Draw the prediction and status text onto the video frame.

    Args:
        frame: BGR image to draw on, modified in place.
        label: Predicted sign, or None if no prediction has been made yet.
        confidence: Confidence of that prediction.
        filled: Frames currently buffered.
        window: Buffer capacity.
        hands: (left detected, right detected) for the current frame.
        fps: Measured capture rate.
        no_hands_in_window: Whether prediction was withheld for lack of any detected hand.
    """
    import cv2

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 74), (32, 32, 32), -1)

    if label is None and no_hands_in_window:
        text = "no hands detected"
        color = COLOR_FAIR
    elif label is None:
        text = f"buffering  {filled}/{window}"
        color = COLOR_TEXT
    else:
        text = f"{label}  {confidence:.0%}"
        color = COLOR_GOOD if confidence >= 0.6 else COLOR_FAIR if confidence >= 0.35 else COLOR_POOR

    cv2.putText(frame, text, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 2, cv2.LINE_AA)

    left, right = hands
    status = (
        f"L:{'Y' if left else '-'}  R:{'Y' if right else '-'}   "
        f"buf {filled}/{window}   {fps:.0f} fps   q=quit"
    )
    cv2.putText(
        frame, status, (14, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        COLOR_TEXT, 1, cv2.LINE_AA,
    )


def parse_source(source: str) -> int | str:
    """Interpret --source as either a camera index or a file path.

    Args:
        source: Raw CLI value.

    Returns:
        An int camera index, or the path string unchanged.
    """
    try:
        return int(source)
    except ValueError:
        return source


def main() -> int:
    """Run the live demo. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Live ASL sign recognition demo.")
    parser.add_argument("--source", default="0", help="Camera index (default 0) or video file path.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/gru_best.pt"))
    parser.add_argument("--model", default="gru", choices=["gru", "transformer"])
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="MediaPipe holistic .task bundle.")
    parser.add_argument("--no-download", action="store_true",
                        help="Fail instead of downloading the MediaPipe bundle.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--signs-json", type=Path,
                        default=Path("data/processed/top30_signs.json"))
    parser.add_argument("--window", type=int, default=DEFAULT_MAX_LEN,
                        help="Sliding window length in frames (default matches training max_len).")
    parser.add_argument("--predict-every", type=int, default=5,
                        help="Run the classifier every N frames (default 5).")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Average probabilities over the last N predictions; 1 disables.")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Hide predictions below this confidence.")
    parser.add_argument("--allow-no-hands", action="store_true",
                        help="Predict even when no hand was detected anywhere in the window. "
                             "Off by default: the classifier always returns some class, so an "
                             "empty frame otherwise yields a confident, meaningless label.")
    parser.add_argument("--save", type=Path, default=None, help="Write the annotated video here.")
    parser.add_argument("--no-draw", action="store_true", help="Do not draw landmark dots.")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a window; useful with --source FILE --save OUT.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after this many frames (0 = unlimited).")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Do not mirror the camera image (mirroring is display-only).")
    args = parser.parse_args()

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HolisticLandmarker,
            HolisticLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
    except ImportError as error:
        print(f"Missing dependency: {error}", file=sys.stderr)
        print("Install with: pip install opencv-python mediapipe", file=sys.stderr)
        return 2

    landmark_types = DEFAULT_LANDMARK_TYPES
    offsets, total_landmarks = landmark_offsets(landmark_types)
    pose_slice = offsets["pose"]
    hand_types = tuple(t for t in landmark_types if t in HAND_TYPES)
    hand_slices = [offsets[t] for t in hand_types]
    num_features = 3 * total_landmarks + len(hand_slices)

    signs, signs_source = resolve_signs(args.signs_json, args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_classifier(
            args.checkpoint, args.model, num_features, signs, args.window, device
        )
        bundle = ensure_model_bundle(args.model_path, HOLISTIC_MODEL_URL, not args.no_download)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"model      : {args.model} ({args.checkpoint}) on {device}")
    print(f"vocabulary : {len(signs)} signs from {signs_source}")
    print(f"features   : {num_features}/frame, window {args.window}, predict every {args.predict_every}")
    print(f"mediapipe  : {bundle}")
    print("press q to quit")

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        print(
            f"Error: could not open video source {args.source!r}. If this is a webcam, check "
            f"that no other application is using it and that camera access is enabled "
            f"(Windows: Settings > Privacy & security > Camera).",
            file=sys.stderr,
        )
        return 1

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(bundle)),
        running_mode=VisionTaskRunningMode.VIDEO,
    )

    buffer: deque[np.ndarray] = deque(maxlen=args.window)
    recent: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    writer = None
    label: str | None = None
    confidence = 0.0
    no_hands_in_window = False
    frame_index = 0
    fps = 0.0
    last_time = time.perf_counter()
    exit_code = 0

    try:
        with HolisticLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    # End of file for a video source; for a camera this means it dropped.
                    break

                if not args.no_mirror:
                    # Mirroring is display-only and applied before landmark extraction, so
                    # the coordinates stay consistent with what the user sees. It swaps
                    # apparent handedness, which MediaPipe resolves from body pose anyway.
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_index * 1000 / max(capture.get(cv2.CAP_PROP_FPS) or 30, 1))

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_array = result_to_frame_array(result, offsets, total_landmarks)
                buffer.append(frame_array)

                hands = tuple(
                    bool(not np.isnan(frame_array[start:stop]).all())
                    for start, stop in hand_slices
                ) if hand_slices else (False, False)

                # Only predict once the window is genuinely full: a partly-filled buffer
                # would be padded, which is a different input distribution from training.
                if len(buffer) == args.window and frame_index % max(args.predict_every, 1) == 0:
                    features, mask = window_to_features(
                        list(buffer), pose_slice, hand_slices, args.window
                    )
                    # The classifier is a closed-set 30-way softmax with no "none of these"
                    # option, so it labels an empty room as confidently as a real sign. If
                    # no hand appeared anywhere in the window, say so instead of guessing.
                    saw_hand = bool(features[:, -len(hand_slices):].any()) if hand_slices else True
                    no_hands_in_window = not saw_hand
                    if saw_hand or args.allow_no_hands:
                        recent.append(predict_window(model, features, mask, device))
                        probabilities = np.mean(recent, axis=0)
                        index = int(np.argmax(probabilities))
                        value = float(probabilities[index])
                        if value >= args.min_confidence:
                            label, confidence = signs[index], value
                    else:
                        recent.clear()
                        label, confidence = None, 0.0

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6) if fps else 1 / max(now - last_time, 1e-6)
                last_time = now

                if not args.no_draw:
                    draw_landmarks(frame, frame_array, offsets, width, height)
                draw_overlay(
                    frame, label, confidence, len(buffer), args.window,
                    (hands + (False, False))[:2], fps, no_hands_in_window,
                )

                if args.save is not None:
                    if writer is None:
                        args.save.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                            capture.get(cv2.CAP_PROP_FPS) or 20.0, (width, height),
                        )
                    writer.write(frame)

                if not args.headless:
                    cv2.imshow("ASL sign recognition - q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
                if args.max_frames and frame_index >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    except Exception as error:  # noqa: BLE001 - surface any runtime failure cleanly
        print(f"Error during capture: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.save is not None and writer is not None:
        print(f"saved -> {args.save}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
