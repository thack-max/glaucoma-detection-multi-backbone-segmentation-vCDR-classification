"""
Tier 2 specialist assessment entry point.

Usage:
    python scripts/run_specialist.py --patient-dir path/to/patient_data/
    python scripts/run_specialist.py --referral referral.json --oct scan.npy
    python scripts/run_specialist.py --patient-dir data/ --enable-vlm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.pipeline.common import ReferralPacket
from src.pipeline.specialist import SpecialistPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 2: Specialist Assessment")
    parser.add_argument("--patient-dir", type=str, help="Directory with patient multimodal data")
    parser.add_argument("--referral", type=str, help="Path to Tier 1 referral JSON")
    parser.add_argument("--oct", type=str, help="Path to OCT volume (.npy)")
    parser.add_argument("--config", type=str, default="configs/tier/specialist.yaml")
    parser.add_argument("--enable-vlm", action="store_true", help="Enable experimental VLM reports")
    parser.add_argument("--device", type=str, default=None, help="Override config device")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config["pipeline"]["device"]
    enable_vlm = args.enable_vlm or config["pipeline"].get("enable_vlm", False)
    output_dir = Path(args.output_dir or config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load referral if provided
    referral = None
    if args.referral:
        with open(args.referral) as f:
            referral = ReferralPacket.from_dict(json.load(f))
        print(f"Loaded Tier 1 referral for patient: {referral.patient_id}")
        print(f"  Risk: {referral.risk_level} (score: {referral.risk_score:.2f}, vCDR: {referral.vcdr:.3f})")
        print(f"  Urgency: {referral.urgency}")

    # Initialize pipeline
    # TODO: Load actual model weights from config checkpoint paths
    pipeline = SpecialistPipeline(
        device=device,
        enable_vlm=enable_vlm,
    )

    print(f"\nTier 2 Specialist Pipeline")
    print(f"Device: {device}")
    print(f"VLM reports: {'ENABLED (experimental)' if enable_vlm else 'disabled'}")
    print(f"Output: {output_dir}")

    if referral:
        intake = pipeline.ingest_referral(referral)
        print(f"\nReferral intake summary:")
        print(json.dumps(intake, indent=2))

    if args.patient_dir or args.oct:
        print("\nModel weights required. See README for setup instructions.")
    elif not referral:
        print("\nProvide --patient-dir, --oct, or --referral. Use --help for usage.")


if __name__ == "__main__":
    main()
