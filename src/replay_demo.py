"""Replay test sequences through the trained model and watch the prediction form.

For each sampled sequence the model is shown the first 25%, 50%, 75% and 100% of its
frames, which shows how much of a sign has to happen before the model commits — and,
for the failures, whether it was ever right and then changed its mind.

An honest caveat about the word "streaming": the baseline GRU is *bidirectional*, so it
reads each prefix forwards and backwards before predicting. That is prefix inference, not
causal online inference — the model is re-run from scratch on a growing window rather than
updating a running state. A truly streaming deployment would need a unidirectional model.
The numbers here are still meaningful (each prediction uses only frames up to that point),
but they are not a latency claim about real-time decoding.

Frames are the cached, normalised frames the model actually consumes: sequences longer
than max_len were uniformly subsampled when the cache was built, so "50% of frames" means
half of that representation, not half of the original video.

Usage (from the project root):
    python -m src.replay_demo
    python -m src.replay_demo --num-sequences 15 --animate 3
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
from matplotlib import animation

from src.dataset import (
    CACHE_ROOT,
    DEFAULT_LANDMARK_TYPES,
    DEFAULT_MAX_LEN,
    LANDMARK_COUNTS,
    NUM_TOP_SIGNS,
    ASLLandmarkDataset,
    cache_dir_for,
    make_splits,
    select_top_signs,
)
from src.train import build_model
from src.train_utils import get_device

FIGURE_DIR = Path("reports/figures")
FRACTIONS = (0.25, 0.50, 0.75, 1.00)

# Skeleton topology, matching notebooks/eda.ipynb.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
# Only the upper-body joints are drawn. Legs and feet are in the landmark set but sit far
# below the signing space, and including them stretches the view until the hands - the
# part that actually carries the sign - are a few unreadable pixels.
POSE_KEEP = sorted({index for edge in POSE_EDGES for index in edge})
BLUE, GREEN, RED = "#4C72B0", "#55A868", "#C44E52"


def _drawable_parts(frame_xyz: np.ndarray) -> list[tuple[np.ndarray, list[tuple[int, int]], str]]:
    """Split one frame into the point groups that are actually rendered.

    Args:
        frame_xyz: Landmarks for one frame, shape (num_landmarks, 3).

    Returns:
        List of (xy points, edges indexing those points, colour). Absent hands, which
        were zero-filled at the origin, are omitted rather than drawn at the shoulders.
    """
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for landmark_type in DEFAULT_LANDMARK_TYPES:
        offsets[landmark_type] = (cursor, cursor + LANDMARK_COUNTS[landmark_type])
        cursor += LANDMARK_COUNTS[landmark_type]

    parts = []
    pose_start = offsets["pose"][0]
    pose_xy = frame_xyz[[pose_start + i for i in POSE_KEEP], :2]
    remap = {original: new for new, original in enumerate(POSE_KEEP)}
    parts.append((pose_xy, [(remap[a], remap[b]) for a, b in POSE_EDGES], BLUE))

    for landmark_type, color in (("left_hand", GREEN), ("right_hand", RED)):
        start, stop = offsets[landmark_type]
        xy = frame_xyz[start:stop, :2]
        if np.allclose(xy, 0.0):
            continue
        parts.append((xy, HAND_EDGES, color))
    return parts


def sample_sequences(dataset: ASLLandmarkDataset, count: int, seed: int) -> list[int]:
    """Pick random test sequences spanning distinct signs.

    Signs are sampled first, then one sequence at random within each, so the sample is
    random but cannot land on the same sign repeatedly and misrepresent coverage.

    Args:
        dataset: Test dataset.
        count: How many sequences to draw.
        seed: Seed for the draw.

    Returns:
        Dataset indices, in the order sampled.
    """
    rng = np.random.default_rng(seed)
    signs = dataset.df["sign"].to_numpy()
    available = np.unique(signs)
    chosen_signs = rng.choice(available, size=min(count, len(available)), replace=False)
    return [int(rng.choice(np.flatnonzero(signs == sign))) for sign in chosen_signs]


@torch.no_grad()
def predict_prefix(
    model: torch.nn.Module,
    landmarks: torch.Tensor,
    num_frames: int,
    device: torch.device,
) -> tuple[int, float]:
    """Predict from the first `num_frames` frames of one sequence.

    Args:
        model: Trained model in eval mode.
        landmarks: Padded sequence of shape (max_len, num_features).
        num_frames: How many leading frames the model may see.
        device: Device to run on.

    Returns:
        Tuple of (predicted class index, softmax confidence of that class).
    """
    prefix = torch.zeros_like(landmarks)
    prefix[:num_frames] = landmarks[:num_frames]
    mask = torch.zeros(landmarks.size(0), dtype=torch.bool)
    mask[:num_frames] = True

    logits = model(prefix.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=1)[0]
    index = int(probabilities.argmax())
    return index, float(probabilities[index])


def draw_skeleton(ax, frame_xyz: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    """Draw one frame's pose and hand skeleton.

    Args:
        ax: Axes to draw on.
        frame_xyz: Landmarks for one frame, shape (num_landmarks, 3).
        bounds: (x_lo, x_hi, y_lo, y_hi) held fixed across frames.
    """
    for xy, edges, color in _drawable_parts(frame_xyz):
        ax.scatter(xy[:, 0], xy[:, 1], s=8, color=color, zorder=3)
        for a, b in edges:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=color, lw=1.3)

    x_lo, x_hi, y_lo, y_hi = bounds
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)  # y grows downward in landmark space
    ax.set_aspect("equal")
    ax.axis("off")


def animate_sequence(
    model: torch.nn.Module,
    landmarks: torch.Tensor,
    num_real: int,
    true_sign: str,
    class_names: list[str],
    device: torch.device,
    out_path: Path,
) -> None:
    """Save a GIF of the skeleton with the running prediction overlaid.

    Args:
        model: Trained model.
        landmarks: Padded sequence of shape (max_len, num_features).
        num_real: Number of real (non-padded) frames.
        true_sign: Ground-truth sign name.
        class_names: Class names in label order.
        device: Device to run on.
        out_path: Destination .gif.
    """
    coords = landmarks[:num_real, : 3 * sum(LANDMARK_COUNTS[t] for t in DEFAULT_LANDMARK_TYPES)]
    coords = coords.numpy().reshape(num_real, -1, 3)

    # Bounds come from exactly the points that get drawn, held fixed for every frame so
    # the skeleton moves against a stable background instead of the axes rescaling.
    drawn = np.concatenate(
        [xy for frame in coords for xy, _, _ in _drawable_parts(frame)], axis=0
    )
    margin = 0.05 * max(np.ptp(drawn[:, 0]), np.ptp(drawn[:, 1]), 1e-3)
    bounds = (
        float(drawn[:, 0].min() - margin), float(drawn[:, 0].max() + margin),
        float(drawn[:, 1].min() - margin), float(drawn[:, 1].max() + margin),
    )

    # One prediction per frame, from that frame's prefix.
    predictions = [predict_prefix(model, landmarks, k + 1, device) for k in range(num_real)]

    # Match the figure's shape to the data box, otherwise `set_aspect("equal")` pads the
    # difference with dead whitespace.
    x_lo, x_hi, y_lo, y_hi = bounds
    data_ratio = (x_hi - x_lo) / max(y_hi - y_lo, 1e-6)
    height = 4.4
    fig, ax = plt.subplots(figsize=(max(height * data_ratio, 2.6), height + 0.9))

    def update(frame_index: int):
        ax.clear()
        draw_skeleton(ax, coords[frame_index], bounds)
        index, confidence = predictions[frame_index]
        predicted = class_names[index]
        correct = predicted == true_sign
        ax.set_title(
            f'true: "{true_sign}"\n'
            f'predicted: "{predicted}" ({confidence:.0%})\n'
            f"frame {frame_index + 1}/{num_real}",
            fontsize=10,
            color="#2E7D32" if correct else "#C62828",
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=num_real, interval=120)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def main() -> None:
    """Replay sampled test sequences and report how predictions evolve."""
    parser = argparse.ArgumentParser(description="Replay test sequences through a model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/gru_best.pt"))
    parser.add_argument("--model", default="gru")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num-sequences", type=int, default=12)
    parser.add_argument("--animate", type=int, default=3, help="How many GIFs to render.")
    parser.add_argument(
        "--animate-signs",
        nargs="*",
        default=None,
        help="Render GIFs for these signs specifically instead of the first sampled ones.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    train_df = pd.read_csv(args.data_dir / "train.csv")
    signs = select_top_signs(train_df, num_signs=NUM_TOP_SIGNS)
    splits = make_splits(train_df)
    cache_dir = cache_dir_for(args.cache_root, DEFAULT_LANDMARK_TYPES, args.max_len)

    test_set = ASLLandmarkDataset(
        splits["test"],
        data_dir=args.data_dir,
        signs=signs,
        max_len=args.max_len,
        cache_dir=cache_dir,
    )
    model = build_model(args.model, test_set.num_features, test_set.num_classes, args.max_len)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    indices = sample_sequences(test_set, args.num_sequences, args.seed)
    print(f"model: {args.model} ({args.checkpoint})")
    print(f"replaying {len(indices)} random test sequences, one per distinct sign, seed={args.seed}")
    print("NOTE: the GRU is bidirectional, so each row is prefix inference, not causal streaming.")
    print()

    header = f"{'true sign':>12} {'signer':>7} {'frames':>7} " + " ".join(
        f"{int(f * 100):>3}%{'':>13}" for f in FRACTIONS
    ) + f" {'result':>9}"
    print(header)
    print("-" * len(header))

    records = []
    correct_count = 0
    for index in indices:
        item = test_set[index]
        row = test_set.df.iloc[index]
        num_real = int(item["mask"].sum())
        true_sign = row["sign"]

        cells = []
        stages = []
        for fraction in FRACTIONS:
            frames = max(1, int(round(fraction * num_real)))
            predicted_index, confidence = predict_prefix(
                model, item["landmarks"], frames, device
            )
            predicted = test_set.signs[predicted_index]
            cells.append(f"{predicted[:11]:>11} {confidence:>4.0%}")
            stages.append(
                {"fraction": fraction, "frames": frames, "predicted": predicted,
                 "confidence": confidence}
            )

        final = stages[-1]
        is_correct = final["predicted"] == true_sign
        correct_count += is_correct
        print(
            f"{true_sign:>12} {row['participant_id']:>7} {num_real:>7} "
            + " ".join(f"{c:>17}" for c in cells)
            + f" {'CORRECT' if is_correct else 'wrong':>9}"
        )
        records.append(
            {"sequence_id": int(row["sequence_id"]), "participant_id": int(row["participant_id"]),
             "true_sign": true_sign, "num_frames": num_real, "correct": bool(is_correct),
             "stages": stages}
        )

    print()
    print(f"{correct_count}/{len(records)} correct on this sample "
          f"({correct_count / len(records):.0%}); full test accuracy is the reliable number.")

    # How often is the final answer already settled at the halfway point?
    settled = sum(
        1 for r in records if r["stages"][1]["predicted"] == r["stages"][-1]["predicted"]
    )
    print(f"final prediction already reached by 50% of frames: {settled}/{len(records)}")
    print(f"mean confidence: "
          + ", ".join(
              f"{int(f * 100)}% -> {np.mean([r['stages'][i]['confidence'] for r in records]):.2f}"
              for i, f in enumerate(FRACTIONS)
          ))

    results_path = args.figure_dir.parent / "replay_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nresults -> {results_path}")

    pairs = list(zip(records, indices))
    if args.animate_signs:
        wanted = list(args.animate_signs)
        pairs = [p for p in pairs if p[0]["true_sign"] in wanted]
    for record, index in pairs[: args.animate]:
        item = test_set[index]
        out_path = args.figure_dir / f"replay_{record['true_sign']}_{record['sequence_id']}.gif"
        animate_sequence(
            model,
            item["landmarks"],
            record["num_frames"],
            record["true_sign"],
            test_set.signs,
            device,
            out_path,
        )
        print(f"animation -> {out_path}")


if __name__ == "__main__":
    main()
