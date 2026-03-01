"""
Tier 1 screening entry point.

Usage:
    python scripts/run_screening.py --image path/to/fundus.png
    python scripts/run_screening.py --image-dir path/to/images/ --output-dir results/
    python scripts/run_screening.py --image path/to/fundus.png --baseline  # vCDR-only mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.pipeline.screening import ScreeningPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 1: Glaucoma Screening")
    parser.add_argument("--image", type=str, help="Path to a single fundus image")
    parser.add_argument("--image-dir", type=str, help="Directory of fundus images for batch screening")
    parser.add_argument("--config", type=str, default="configs/tier/screening.yaml")
    parser.add_argument("--baseline", action="store_true", help="Use vCDR > 0.5 baseline mode (PMC12469222)")
    parser.add_argument("--device", type=str, default=None, help="Override config device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config["pipeline"]["device"]
    baseline_mode = args.baseline or config["pipeline"].get("baseline_mode", False)
    output_dir = Path(args.output_dir or config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    # TODO: Load actual model weights from config checkpoint paths
    pipeline = ScreeningPipeline(
        device=device,
        baseline_mode=baseline_mode,
        referral_threshold=config["pipeline"]["referral_threshold"],
        vcdr_alert_threshold=config["pipeline"]["vcdr_alert_threshold"],
    )

    mode = "BASELINE (vCDR > 0.5)" if baseline_mode else "DINOv2 + BoundaryFormer"
    print(f"Tier 1 Screening Pipeline — Mode: {mode}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    if args.image:
        print(f"\nProcessing: {args.image}")
        # TODO: Load image, run pipeline.run(), save outputs
        # result = pipeline.run(fundus_image)
        # if result.refer_to_specialist:
        #     referral = pipeline.create_referral(result, patient_id="...", ...)
        #     with open(output_dir / "referral.json", "w") as f:
        #         json.dump(referral.to_dict(), f, indent=2)
        print("Model weights required. See README for setup instructions.")

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        print(f"\nBatch screening: {len(images)} images from {image_dir}")
        print("Model weights required. See README for setup instructions.")

    else:
        print("\nProvide --image or --image-dir. Use --help for usage.")


if __name__ == "__main__":
    main()
