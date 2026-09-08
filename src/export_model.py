"""Export a trained model to TorchScript and benchmark inference latency.

TorchScript is used rather than ONNX because it needs no dependency beyond torch itself,
and it keeps the exported artifact runnable from Python or C++ without a separate runtime.
Freezing and `optimize_for_inference` are applied on top, which inline the parameters as
constants and fold what can be folded. (Both are deprecated in favour of `torch.compile`
in recent torch versions; TorchScript is kept here because it produces a self-contained,
portable artifact, which `torch.compile` does not.)

The measured conclusion is that the export format barely matters for this model, while the
thread count matters a great deal — so the benchmark sweeps both rather than reporting a
single flattering number.

Usage (from the project root):
    python -m src.export_model
    python -m src.export_model --runs 500
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

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
from src.train_utils import get_device


def benchmark(
    model: torch.nn.Module,
    landmarks: torch.Tensor,
    mask: torch.Tensor,
    runs: int,
    warmup: int,
) -> dict[str, float]:
    """Time repeated single-sequence forward passes.

    Warmup runs are discarded: the first calls pay for lazy initialisation and, for the
    scripted module, for its profiling-guided optimisation passes, which would otherwise
    dominate the average.

    Args:
        model: Model or scripted module to time.
        landmarks: Input of shape (1, seq_len, num_features).
        mask: Mask of shape (1, seq_len).
        runs: Timed iterations.
        warmup: Untimed iterations first.

    Returns:
        Dict of mean, median, p95 and min latency in milliseconds.
    """
    with torch.no_grad():
        for _ in range(warmup):
            model(landmarks, mask)

        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            model(landmarks, mask)
            timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    return {
        "mean_ms": statistics.fmean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": timings[int(0.95 * len(timings))],
        "min_ms": timings[0],
    }


def main() -> None:
    """Export the model, verify it, and report before/after latency."""
    parser = argparse.ArgumentParser(description="Export and benchmark a trained model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/gru_best.pt"))
    parser.add_argument("--model", default="gru")
    parser.add_argument("--output", type=Path, default=Path("models/gru_scripted.pt"))
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    train_df = pd.read_csv(args.data_dir / "train.csv")
    signs = select_top_signs(train_df, num_signs=NUM_TOP_SIGNS)
    splits = make_splits(train_df)
    cache_dir = cache_dir_for(args.cache_root, DEFAULT_LANDMARK_TYPES, args.max_len)

    test_set = ASLLandmarkDataset(
        splits["test"], data_dir=args.data_dir, signs=signs,
        max_len=args.max_len, cache_dir=cache_dir,
    )
    model = build_model(args.model, test_set.num_features, test_set.num_classes, args.max_len)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    item = test_set[0]
    landmarks = item["landmarks"].unsqueeze(0).to(device)
    mask = item["mask"].unsqueeze(0).to(device)

    print(f"model: {args.model}  |  device: {device}  |  threads: {torch.get_num_threads()}")
    print(f"input: {tuple(landmarks.shape)}  ({int(mask.sum())} real frames)")
    print()

    scripted = torch.jit.script(model)
    scripted = torch.jit.freeze(scripted.eval())
    scripted = torch.jit.optimize_for_inference(scripted)

    # An export that changed the predictions would be worse than no export at all.
    with torch.no_grad():
        reference = model(landmarks, mask)
        exported = scripted(landmarks, mask)
    max_difference = (reference - exported).abs().max().item()
    agree = bool((reference.argmax(1) == exported.argmax(1)).all())
    print(f"parity: max logit difference {max_difference:.2e}, same prediction: {agree}")
    if max_difference > 1e-4 or not agree:
        raise SystemExit("Exported model does not match the original; refusing to save.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))

    eager_size = args.checkpoint.stat().st_size / 1e6
    scripted_size = args.output.stat().st_size / 1e6
    print(f"saved -> {args.output} ({scripted_size:.2f} MB; state dict was {eager_size:.2f} MB)")
    print()

    # Thread count is swept as well as the export format. For a batch of one, the model is
    # far too small to fill 10 cores, and the cost of splitting and rejoining the work each
    # layer swamps the arithmetic - so the default thread pool is actively harmful here.
    # It turns out to matter several times more than the export format does.
    default_threads = torch.get_num_threads()
    thread_counts = sorted({1, 2, 4, default_threads})

    print(f"benchmarking {args.runs} runs after {args.warmup} warmup runs, batch size 1")
    print()
    print(f"{'threads':>8} {'variant':>13} {'median':>10} {'p95':>10} {'min':>10}")
    measurements: dict[str, dict[str, float]] = {}
    for threads in thread_counts:
        torch.set_num_threads(threads)
        for name, module in (("PyTorch", model), ("TorchScript", scripted)):
            result = benchmark(module, landmarks, mask, args.runs, args.warmup)
            measurements[f"{name}@{threads}"] = {**result, "threads": threads, "variant": name}
            print(f"{threads:>8} {name:>13} {result['median_ms']:>9.3f}ms "
                  f"{result['p95_ms']:>9.3f}ms {result['min_ms']:>9.3f}ms")
    torch.set_num_threads(default_threads)

    baseline_key = f"PyTorch@{default_threads}"
    baseline = measurements[baseline_key]
    best_key = min(measurements, key=lambda k: measurements[k]["median_ms"])
    best = measurements[best_key]

    print()
    print(f"baseline (PyTorch, default {default_threads} threads): {baseline['median_ms']:.3f}ms median")
    print(f"best     ({best['variant']}, {int(best['threads'])} threads): {best['median_ms']:.3f}ms median")
    print(f"speedup: {baseline['median_ms'] / best['median_ms']:.2f}x  "
          f"({1000 / baseline['median_ms']:.0f} -> {1000 / best['median_ms']:.0f} sequences/s)")

    same_threads = measurements[f"TorchScript@{default_threads}"]
    print(f"export format alone (same {default_threads} threads): "
          f"{baseline['median_ms'] / same_threads['median_ms']:.2f}x")

    results_path = Path("reports/export_benchmark.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {"device": str(device), "default_threads": default_threads, "runs": args.runs,
             "measurements": measurements, "baseline": baseline_key, "best": best_key,
             "speedup_vs_baseline": baseline["median_ms"] / best["median_ms"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"results -> {results_path}")


if __name__ == "__main__":
    main()
