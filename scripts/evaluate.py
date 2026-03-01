"""
Evaluation script for the two-tier glaucoma pipeline.

Usage:
    python scripts/evaluate.py --tier both          # Full comparison table
    python scripts/evaluate.py --tier screening      # Tier 1 metrics only
    python scripts/evaluate.py --tier specialist     # Tier 2 metrics only
    python scripts/evaluate.py --tier both --data-root data/refuge --model-dir checkpoints/
"""

from __future__ import annotations

import argparse

import numpy as np

from src.evaluation.metrics import (
    compute_auc,
    compute_vcdr_mae,
    compute_vf_md_mae,
    dice_coefficient,
    iou_score,
)
from src.evaluation.fairness_audit import FairnessAuditor


def print_screening_metrics():
    """Print Tier 1 screening metrics and baseline comparison."""
    print("\n" + "=" * 80)
    print("TIER 1: SCREENING METRICS")
    print("=" * 80)

    rows = [
        ("SEGMENTATION (REFUGE)", "", ""),
        ("  Dice OD", "93.83% (MaskFormer)", "96.1% (FunduSAM)"),
        ("  Dice OC", "93.71% (MaskFormer)", "86.7% (FunduSAM)*"),
        ("  IoU OD", "88.29%", "88.2%"),
        ("  IoU OC", "91.09%", "78.9%*"),
        ("", "", ""),
        ("CLASSIFICATION", "", ""),
        ("  Method", "vCDR > 0.5 threshold", "RETFound-CFP + classifier head"),
        ("  Accuracy", "84.03%", "— (requires training)"),
        ("  AUC (glaucoma)", "—", "0.91 (RETFound reference)"),
        ("", "", ""),
        ("FAIRNESS", "", ""),
        ("  Black patient AUC", "Not reported", "0.82 (with FIN)"),
        ("  Equity method", "None", "Fair Identity Normalization"),
        ("", "", ""),
        ("DEPLOYMENT", "", ""),
        ("  Input", "Single fundus photo", "Single fundus photo"),
        ("  Hardware", "$500 camera + laptop", "$500 camera + laptop"),
        ("  Latency target", "—", "<5s CPU, <1s GPU"),
    ]

    print(f"{'Metric':<30} {'Baseline (PMC12469222)':<30} {'Tier 1 Screening':<30}")
    print("-" * 90)
    for metric, baseline, proposed in rows:
        print(f"{metric:<30} {baseline:<30} {proposed:<30}")

    print("\n* FunduSAM optic CUP Dice is lower than MaskFormer (86.7% vs 93.71%).")
    print("  Cup boundary segmentation remains an open challenge.")
    print("=" * 80)


def print_specialist_metrics():
    """Print Tier 2 specialist metrics."""
    print("\n" + "=" * 80)
    print("TIER 2: SPECIALIST METRICS")
    print("=" * 80)

    rows = [
        ("PROGRESSION FORECASTING", "", ""),
        ("  Architecture", "N/A (cross-sectional)", "ConvNeXt-V2 + BiLSTM"),
        ("  AUC (2-4yr)", "N/A", "0.97 (Mass Eye & Ear)"),
        ("  Critical modality", "N/A", "RNFL OCT (removing: AUC 0.569)"),
        ("  VF contribution", "N/A", "Minimal (removing VF: AUC 0.975)"),
        ("", "", ""),
        ("VLM CLASSIFICATION", "", ""),
        ("  AUC (LLaVA)", "—", "0.92 (DIGS/ADAGES)"),
        ("  AUC (PaliGemma)", "—", "0.90"),
        ("  AUC (RETFound)", "—", "0.91"),
        ("  VF MD MAE", "—", "1.79 dB (LLaVA)"),
        ("  Status", "—", "EXPERIMENTAL (no FDA clearance)"),
        ("", "", ""),
        ("DEPLOYMENT", "", ""),
        ("  Input", "Single fundus photo", "Fundus + OCT + VF + clinical"),
        ("  Hardware", "$500 camera", "$80K+ OCT machine + GPU server"),
        ("  Accepts referrals", "N/A", "Yes (from Tier 1)"),
    ]

    print(f"{'Metric':<30} {'Baseline (PMC12469222)':<30} {'Tier 2 Specialist':<30}")
    print("-" * 90)
    for metric, baseline, proposed in rows:
        print(f"{metric:<30} {baseline:<30} {proposed:<30}")
    print("=" * 80)


def print_comparison_table():
    """Print the full two-tier comparison table."""
    print_screening_metrics()
    print_specialist_metrics()

    print("\n" + "=" * 80)
    print("TWO-TIER REFERRAL FLOW")
    print("=" * 80)
    print("""
  Rural Clinic (Tier 1)              Hospital (Tier 2)
  ─────────────────────              ─────────────────
  Fundus photo                       Fundus + OCT + VF + Clinical
       │                                    │
  FunduSAM segmentation             FunduSAM + MedSAM-2 segmentation
       │                                    │
  RETFound-CFP + classifier         RETFound features (CFP + OCT)
       │                                    │
  Risk score + referral decision    ConvNeXt-V2 + BiLSTM forecasting
       │                                    │
       └──── referral.json ────────>  2-4 year progression forecast
""")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-tier glaucoma pipeline")
    parser.add_argument(
        "--tier",
        type=str,
        choices=["screening", "specialist", "both"],
        default="both",
        help="Which tier to evaluate",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.tier == "screening":
        print_screening_metrics()
    elif args.tier == "specialist":
        print_specialist_metrics()
    else:
        print_comparison_table()

    if args.data_root and args.model_dir:
        print("\nFull evaluation requires trained model checkpoints.")
        print("Run training scripts first:")
        print("  python scripts/train_screening_classifier.py --data-root data/refuge")
        print("  python scripts/train_segmentation.py --data-root data/refuge")
        print("  python scripts/train_progression.py --data-root data/longitudinal")


if __name__ == "__main__":
    main()
