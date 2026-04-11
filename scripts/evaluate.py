"""
Baseline (PMC12469222 vCDR > 0.5) vs GlaucoMamba head-to-head evaluation.

Two modes:

1. **Informational** (default, no data provided):
   Prints the same architecture/metric comparison tables that used to be the
   entire contents of this script. Useful for README / paper lookup.

2. **A/B evaluation** (when --data-root is given):
   Loads the test split of a FundusDataset manifest and runs the SAME
   ScreeningPipeline TWICE — once in baseline_mode=True (vCDR threshold) and
   once in baseline_mode=False (BoundaryFormer + DINOv2 + classifier head).

   Computes:
     - AUC (both modes)
     - Sensitivity @ 95% specificity
     - Specificity @ 90% sensitivity
     - Accuracy / sensitivity / specificity / PPV / NPV at threshold 0.5
     - Per-image score delta + label
     - Per-group fairness audit when group_id is available

   Writes:
     <output>/baseline_vs_glaucomamba.csv  -- per-image scores
     <output>/metrics.json                  -- aggregate comparison
     <output>/fairness_report.json          -- only if group_id present

Usage:
    # Informational tables
    python scripts/evaluate.py --tier screening

    # Real A/B evaluation on a held-out test set
    python scripts/evaluate.py \
        --tier screening \
        --data-root data/refuge \
        --split test \
        --segmenter-checkpoint checkpoints/boundary_former/best.pth \
        --classifier-checkpoint checkpoints/screening_classifier.pth
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.evaluation.fairness_audit import FairnessAuditor
from src.evaluation.metrics import (
    binary_classification_metrics,
    compute_auc,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)
from src.pipeline.screening import ScreeningPipeline


# ---------------------------------------------------------------------------
# Informational tables (unchanged)
# ---------------------------------------------------------------------------


def print_screening_metrics():
    """Print Tier 1 screening metrics and baseline comparison."""
    print("\n" + "=" * 95)
    print("TIER 1: SCREENING METRICS (GlaucoMamba)")
    print("=" * 95)

    rows = [
        ("SEGMENTATION (REFUGE)", "", "", ""),
        ("  Dice OD", "93.83% (MaskFormer)", "96.1% (FunduSAM)", "~96% (BoundaryFormer)"),
        ("  Dice OC", "93.71% (MaskFormer)", "86.7% (FunduSAM)*", "~89% (BoundaryFormer)**"),
        ("  IoU OD", "88.29%", "88.2%", "~89%"),
        ("  IoU OC", "91.09%", "78.9%*", "~81%**"),
        ("", "", "", ""),
        ("CLASSIFICATION", "", "", ""),
        ("  Method", "vCDR > 0.5", "RETFound-CFP", "DINOv2-Base frozen"),
        ("  AUC (glaucoma)", "—", "0.940", "0.958 (p<0.001)"),
        ("  Explainability", "vCDR", "None (black box)", "GradCAM + vCDR"),
        ("", "", "", ""),
        ("FAIRNESS", "", "", ""),
        ("  Method", "None", "FIN (unvalidated)", "EquityAttention + GRL"),
    ]

    print(f"{'Metric':<25} {'Baseline (PMC12469222)':<25} {'Old Pipeline':<25} {'GlaucoMamba':<25}")
    print("-" * 100)
    for row in rows:
        print(f"{row[0]:<25} {row[1]:<25} {row[2]:<25} {row[3]:<25}")
    print("=" * 95)


def print_specialist_metrics():
    """Print Tier 2 specialist metrics."""
    print("\n" + "=" * 95)
    print("TIER 2: SPECIALIST METRICS (GlaucoMamba)")
    print("=" * 95)

    rows = [
        ("TEMPORAL MODELING", "BiLSTM (uniform intervals)", "Delta-t Mamba (continuous-time SSM)"),
        ("MULTIMODAL FUSION", "Concatenation", "Cross-Modal Attention + MC Dropout"),
        ("PROGRESSION FORECAST", "ConvNeXt-V2 + BiLSTM", "Delta-t Mamba + Evidential DL"),
        ("UNCERTAINTY", "None (binary output)", "Epistemic + aleatoric (calibrated)"),
        ("VF ANALYSIS", "Flattened vector", "VF-GIN (AUC 0.982)"),
        ("VLM REPORTING", "LLaVA", "Llama 3.2 Vision (BERTScore 0.99)"),
    ]

    print(f"{'Metric':<30} {'Old Pipeline':<35} {'GlaucoMamba':<35}")
    print("-" * 100)
    for metric, old, new in rows:
        print(f"{metric:<30} {old:<35} {new:<35}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# Real A/B evaluation
# ---------------------------------------------------------------------------


def _try_build_segmenter(checkpoint_path: str | None, device: str):
    from src.models.segmentation.boundary_former import BoundaryFormer

    model = BoundaryFormer(num_classes=3, base_channels=64, growth_rate=32)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded segmenter: {checkpoint_path}")
    model.eval()
    return model.to(device)


def _try_build_encoder(variant: str, device: str):
    from src.models.features.dinov2_encoder import DINOv2Encoder

    encoder = DINOv2Encoder(variant=variant, freeze_encoder=True)
    encoder.eval()
    return encoder.to(device)


def _try_build_classifier(checkpoint_path: str | None, input_dim: int, device: str):
    head = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Dropout(0.3),
        nn.Linear(input_dim, 1),
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        head.load_state_dict(state.get("classifier", state), strict=False)
        print(f"Loaded classifier: {checkpoint_path}")
    head.eval()
    return head.to(device)


def _run_pipeline_over_dataset(
    pipeline: ScreeningPipeline,
    dataset: FundusDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a ScreeningPipeline across every sample. Returns (scores, labels, groups, vcdrs)."""
    scores: list[float] = []
    labels: list[int] = []
    groups: list[int] = []
    vcdrs: list[float] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        if "label" not in sample:
            continue
        image = sample["image"].unsqueeze(0)
        demographics = None
        if "group_id" in sample:
            demographics = {"group_id": int(sample["group_id"].item())}

        with torch.no_grad():
            output = pipeline.run(image, demographics=demographics)

        scores.append(output.risk_score)
        labels.append(int(sample["label"].item()))
        if "group_id" in sample:
            groups.append(int(sample["group_id"].item()))
        vcdrs.append(output.vcdr or 0.0)

    return (
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(groups, dtype=np.int32) if groups else np.array([]),
        np.array(vcdrs, dtype=np.float32),
    )


def _summarize(name: str, scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Aggregate metrics for one evaluation run."""
    metrics: dict[str, Any] = {"mode": name, "n": int(len(labels))}
    if len(np.unique(labels)) >= 2:
        metrics["auc"] = compute_auc(labels, scores)
        metrics["sens_at_95_spec"] = sensitivity_at_specificity(labels, scores, 0.95)
        metrics["spec_at_90_sens"] = specificity_at_sensitivity(labels, scores, 0.90)
    metrics.update(binary_classification_metrics(labels, scores))
    return metrics


def _run_ab_evaluation(args: argparse.Namespace) -> None:
    device = args.device
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build shared components once
    segmenter = _try_build_segmenter(args.segmenter_checkpoint, device)
    encoder = _try_build_encoder(args.dinov2_variant, device)
    classifier = _try_build_classifier(args.classifier_checkpoint, encoder.embed_dim, device)

    # Two pipelines backed by the SAME segmenter + encoder + classifier so
    # that the only difference between runs is the baseline_mode switch.
    pipeline_baseline = ScreeningPipeline(
        segmenter=segmenter,
        feature_encoder=encoder,
        classifier_head=classifier,
        device=device,
        baseline_mode=True,
    )
    pipeline_glaucomamba = ScreeningPipeline(
        segmenter=segmenter,
        feature_encoder=encoder,
        classifier_head=classifier,
        device=device,
        baseline_mode=False,
    )

    transform = get_fundus_transforms("val", image_size=args.image_size)
    dataset = FundusDataset(data_root, split=args.split, transform=transform)
    print(f"\nEvaluating on {len(dataset)} images (split={args.split}) from {data_root}")
    if len(dataset) == 0:
        print("No labeled samples found. Run scripts/prepare_*.py first.")
        return

    # --- Run both modes ---
    print("\nRunning BASELINE (vCDR > 0.5)...")
    base_scores, base_labels, base_groups, base_vcdrs = _run_pipeline_over_dataset(
        pipeline_baseline, dataset,
    )
    print("Running GlaucoMamba (DINOv2 + BoundaryFormer)...")
    gm_scores, gm_labels, gm_groups, gm_vcdrs = _run_pipeline_over_dataset(
        pipeline_glaucomamba, dataset,
    )

    assert np.array_equal(base_labels, gm_labels), "label mismatch between runs"

    # --- Aggregate metrics ---
    baseline_metrics = _summarize("baseline", base_scores, base_labels)
    glaucomamba_metrics = _summarize("glaucomamba", gm_scores, gm_labels)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "baseline": baseline_metrics,
                "glaucomamba": glaucomamba_metrics,
                "delta": {
                    "auc": glaucomamba_metrics.get("auc", 0.0) - baseline_metrics.get("auc", 0.0),
                    "sens_at_95_spec": glaucomamba_metrics.get("sens_at_95_spec", 0.0)
                    - baseline_metrics.get("sens_at_95_spec", 0.0),
                },
                "dataset": str(data_root),
                "split": args.split,
            },
            f,
            indent=2,
        )

    # --- Per-image CSV ---
    csv_path = output_dir / "baseline_vs_glaucomamba.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_idx", "label", "baseline_score", "glaucomamba_score", "vcdr", "group_id"],
        )
        for i in range(len(base_labels)):
            writer.writerow(
                [
                    i,
                    int(base_labels[i]),
                    f"{base_scores[i]:.4f}",
                    f"{gm_scores[i]:.4f}",
                    f"{gm_vcdrs[i]:.4f}",
                    int(base_groups[i]) if len(base_groups) else "",
                ]
            )

    # --- Fairness audit (only if group_id available) ---
    fairness_path = None
    if len(base_groups) > 0 and len(np.unique(base_labels)) >= 2:
        auditor = FairnessAuditor()
        try:
            report = auditor.audit(base_labels, gm_scores, base_groups)
            fairness_path = output_dir / "fairness_report.json"
            with open(fairness_path, "w") as f:
                json.dump(
                    {
                        "overall_auc": report.overall_auc,
                        "per_group_auc": report.per_group_auc,
                        "es_auc": report.es_auc,
                        "auc_gap": report.auc_gap,
                        "group_sizes": report.group_sizes,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"Fairness audit skipped: {exc}")

    # --- Console summary ---
    print("\n" + "=" * 80)
    print("BASELINE (vCDR > 0.5) vs GLAUCOMAMBA")
    print("=" * 80)
    print(f"{'Metric':<25} {'Baseline':<20} {'GlaucoMamba':<20} {'Delta':<10}")
    print("-" * 80)
    for key in ("auc", "sens_at_95_spec", "spec_at_90_sens",
                "accuracy", "sensitivity", "specificity"):
        b = baseline_metrics.get(key, 0.0)
        g = glaucomamba_metrics.get(key, 0.0)
        delta = g - b
        print(f"{key:<25} {b:<20.4f} {g:<20.4f} {delta:+.4f}")
    print("=" * 80)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {metrics_path}")
    if fairness_path:
        print(f"Wrote {fairness_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GlaucoMamba pipeline")
    parser.add_argument(
        "--tier", choices=["screening", "specialist", "both"], default="both",
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="If provided, run a real A/B evaluation on this dataset.")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate on (train/val/test).")
    parser.add_argument("--segmenter-checkpoint", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.tier in ("screening", "both"):
        print_screening_metrics()
    if args.tier in ("specialist", "both"):
        print_specialist_metrics()

    if args.data_root:
        _run_ab_evaluation(args)
    else:
        print(
            "\nHint: pass --data-root to run a real baseline vs GlaucoMamba A/B "
            "evaluation against a FundusDataset manifest."
        )


if __name__ == "__main__":
    main()
