"""
Benchmark the Tier 1 ScreeningPipeline to verify the README "<5s CPU" claim.

Measures per-image latency in three configurations:
  1. Baseline mode (segmenter only, vCDR > 0.5)
  2. DINOv2 + classifier (no segmenter)
  3. Full pipeline (segmenter + DINOv2 + classifier + Grad-CAM)

For each configuration the script runs one warm-up pass, then N timed
passes on random synthetic images, and reports:

    p50, p90, p95, p99 (milliseconds), mean, min, max

Why this matters for the original goal:
The whole point of the two-tier design is that Tier 1 must fit on a rural
clinic device with a $500 fundus camera and no GPU. If the full pipeline
can't hit <5s per image on a commodity CPU, the deployment story is broken
regardless of how good the model is. This script gives us a reproducible
number to put in the paper and to guard against latency regressions.

Usage:
    python scripts/benchmark_screening.py --device cpu --iterations 20
    python scripts/benchmark_screening.py --device cuda --iterations 100 --image-size 512
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn as nn

from src.data.transforms import get_fundus_transforms
from src.models.segmentation.boundary_former import BoundaryFormer
from src.pipeline.screening import ScreeningPipeline


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    return float(np.percentile(np.array(data), p))


def _summarize(name: str, timings_ms: list[float]) -> dict[str, float]:
    return {
        "config": name,
        "n": len(timings_ms),
        "mean_ms": mean(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "p50_ms": _percentile(timings_ms, 50),
        "p90_ms": _percentile(timings_ms, 90),
        "p95_ms": _percentile(timings_ms, 95),
        "p99_ms": _percentile(timings_ms, 99),
    }


class _StubEncoder(nn.Module):
    """Lightweight stand-in so benchmarks don't require torch.hub DINOv2.

    The real DINOv2-Base is ~86M params; this stub gives us shape-compatible
    features fast. Use --real-encoder to force the actual DINOv2 load.
    """

    embed_dim = 768

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.conv(x)).flatten(1))

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)
        feats = nn.functional.adaptive_avg_pool2d(feats, 14)
        return feats.flatten(2).transpose(1, 2).repeat(1, 1, self.embed_dim // 64)


def _build_encoder(use_real: bool, variant: str):
    if use_real:
        from src.models.features.dinov2_encoder import DINOv2Encoder

        encoder = DINOv2Encoder(variant=variant, freeze_encoder=True)
    else:
        encoder = _StubEncoder()
    encoder.eval()
    return encoder


def _classifier_head(input_dim: int) -> nn.Module:
    head = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Dropout(0.0),
        nn.Linear(input_dim, 1),
    )
    head.eval()
    return head


def _time_pipeline(
    pipeline: ScreeningPipeline,
    images: torch.Tensor,
    iterations: int,
) -> list[float]:
    """Run `iterations` forward passes and return per-call latencies in ms."""
    timings: list[float] = []
    # Warm-up
    with torch.no_grad():
        pipeline.run(images[0:1])

    for i in range(iterations):
        idx = i % images.shape[0]
        t0 = time.perf_counter()
        with torch.no_grad():
            pipeline.run(images[idx:idx + 1])
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 1 screening latency benchmark")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=4,
                        help="How many distinct synthetic images to rotate through")
    parser.add_argument("--real-encoder", action="store_true",
                        help="Load the actual DINOv2 encoder instead of the stub")
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    parser.add_argument("--output", type=str, default="outputs/benchmark_screening.json")
    parser.add_argument("--target-ms", type=float, default=5000.0,
                        help="Latency budget in milliseconds (README claim)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-generate a pool of synthetic fundus images so we rotate through
    # distinct tensors (avoids a degenerate cache-hit-only benchmark).
    transform = get_fundus_transforms("val", image_size=args.image_size)
    samples = []
    rng = np.random.default_rng(0)
    for _ in range(args.num_samples):
        raw = rng.integers(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)
        samples.append(transform(raw)["image"])
    images = torch.stack(samples).to(device)

    # Build the three configurations
    segmenter = BoundaryFormer(num_classes=3, base_channels=32, growth_rate=16).to(device)
    segmenter.eval()
    encoder = _build_encoder(args.real_encoder, args.dinov2_variant).to(device)
    classifier = _classifier_head(encoder.embed_dim).to(device)

    configs = {
        "baseline_vcdr_only": ScreeningPipeline(
            segmenter=segmenter,
            device=args.device,
            baseline_mode=True,
        ),
        "encoder_plus_classifier": ScreeningPipeline(
            feature_encoder=encoder,
            classifier_head=classifier,
            device=args.device,
            baseline_mode=False,
        ),
        "full_pipeline": ScreeningPipeline(
            segmenter=segmenter,
            feature_encoder=encoder,
            classifier_head=classifier,
            device=args.device,
            baseline_mode=False,
        ),
    }

    print(f"Benchmarking Tier 1 screening on {args.device}")
    print(f"  image size:    {args.image_size}x{args.image_size}")
    print(f"  iterations:    {args.iterations}")
    print(f"  encoder:       {'DINOv2 real' if args.real_encoder else 'stub'}")
    print(f"  target budget: {args.target_ms:.0f} ms ({args.target_ms / 1000:.1f} s)")
    print()

    results = {}
    for name, pipeline in configs.items():
        timings = _time_pipeline(pipeline, images, args.iterations)
        summary = _summarize(name, timings)
        summary["meets_budget"] = summary["p95_ms"] <= args.target_ms
        results[name] = summary
        flag = "PASS" if summary["meets_budget"] else "FAIL"
        print(
            f"[{flag}] {name:<28} "
            f"p50={summary['p50_ms']:7.1f}ms  "
            f"p95={summary['p95_ms']:7.1f}ms  "
            f"max={summary['max_ms']:7.1f}ms"
        )

    with open(output_path, "w") as f:
        json.dump(
            {
                "device": args.device,
                "image_size": args.image_size,
                "iterations": args.iterations,
                "real_encoder": args.real_encoder,
                "target_ms": args.target_ms,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nWrote {output_path}")

    # Non-zero exit so CI / the caller can gate on the budget
    if not all(r["meets_budget"] for r in results.values()):
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
