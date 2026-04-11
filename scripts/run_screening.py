"""
Tier 1 screening entry point.

Usage:
    python scripts/run_screening.py --image path/to/fundus.png
    python scripts/run_screening.py --image-dir path/to/images/ --output-dir results/
    python scripts/run_screening.py --image path/to/fundus.png --baseline  # vCDR-only mode

Loads the screening pipeline (BoundaryFormer + DINOv2 + classifier head) from
the Tier 1 config, runs it on one or more fundus images, and writes per-image
JSON results plus an optional referral.json ready for Tier 2 intake.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.transforms import get_fundus_transforms
from src.data.fundus_dataset import _read_image
from src.models.explainability.gradcam import (
    AttentionRolloutCAM,
    DINOv2GradCAM,
    overlay_heatmap_on_image,
)
from src.pipeline.screening import ScreeningPipeline


def _try_load_segmenter(cfg: dict, device: str):
    """Build BoundaryFormer and optionally load a checkpoint."""
    try:
        from src.models.segmentation.boundary_former import BoundaryFormer
    except ImportError:
        return None

    model = BoundaryFormer(
        num_classes=cfg.get("num_classes", 3),
        base_channels=cfg.get("base_channels", 64),
        growth_rate=cfg.get("growth_rate", 32),
    )
    ckpt = cfg.get("checkpoint")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded segmenter weights: {ckpt}")
    model.eval()
    return model.to(device)


def _try_load_encoder(cfg: dict, device: str):
    try:
        from src.models.features.dinov2_encoder import DINOv2Encoder
    except ImportError:
        return None

    model = DINOv2Encoder(
        variant=cfg.get("variant", "dinov2_vitb14"),
        freeze_encoder=cfg.get("freeze_encoder", True),
        use_registers=cfg.get("use_registers", False),
    )
    model.eval()
    return model.to(device)


def _try_load_classifier(cfg: dict, device: str):
    import torch.nn as nn

    head = nn.Sequential(
        nn.LayerNorm(cfg.get("input_dim", 768)),
        nn.Dropout(0.3),
        nn.Linear(cfg.get("input_dim", 768), 1),
    )
    ckpt = cfg.get("checkpoint")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        head.load_state_dict(state.get("classifier", state), strict=False)
        print(f"Loaded classifier head: {ckpt}")
    head.eval()
    return head.to(device)


def _save_gradcam_overlay(
    image_np: np.ndarray,
    image_tensor: torch.Tensor,
    pipeline: ScreeningPipeline,
    output_path: Path,
) -> bool:
    """Compute a DINOv2-based explainability heatmap and save an overlay.

    Prefers classifier-aware Grad-CAM when a classifier head is available,
    otherwise falls back to attention rollout (classifier-free). Returns
    True if a heatmap was successfully written.
    """
    encoder = pipeline.feature_encoder
    if encoder is None:
        return False

    try:
        if pipeline.classifier_head is not None:
            explainer = DINOv2GradCAM(encoder, pipeline.classifier_head)
            heatmap = explainer(image_tensor)
        else:
            explainer = AttentionRolloutCAM(encoder)
            heatmap = explainer(image_tensor)
    except Exception as exc:  # noqa: BLE001
        print(f"    gradcam failed: {exc}")
        return False

    heatmap_np = heatmap[0].detach().cpu().numpy()
    overlay = overlay_heatmap_on_image(image_np, heatmap_np, alpha=0.45)

    try:
        from PIL import Image

        Image.fromarray(overlay).save(output_path)
        return True
    except ImportError:
        return False


def _screen_one(
    pipeline: ScreeningPipeline,
    image_path: Path,
    transform,
    output_dir: Path,
    save_referral: bool = True,
    save_gradcam: bool = True,
) -> dict:
    image_np = _read_image(image_path)
    tensor = transform(image_np)["image"].unsqueeze(0)

    result = pipeline.run(tensor)

    record = {
        "image": str(image_path),
        "risk_score": round(result.risk_score, 4),
        "risk_level": result.risk_level.value,
        "vcdr": round(result.vcdr or 0.0, 4),
        "refer_to_specialist": result.refer_to_specialist,
        "referral_reason": result.referral_reason,
        "urgency": result.urgency,
        "baseline_vcdr_positive": result.baseline_vcdr_positive,
        "latency_ms": round(result.latency_ms, 2),
    }

    out_json = output_dir / f"{image_path.stem}.json"
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)

    if save_gradcam and not pipeline.baseline_mode:
        gradcam_path = output_dir / f"{image_path.stem}_gradcam.png"
        if _save_gradcam_overlay(image_np, tensor, pipeline, gradcam_path):
            record["gradcam_path"] = str(gradcam_path.relative_to(output_dir))

    if save_referral and result.refer_to_specialist:
        referral = pipeline.create_referral(
            result,
            patient_id=image_path.stem,
            screening_site="unknown",
        )
        referral.fundus_image_path = str(image_path)
        with open(output_dir / f"{image_path.stem}_referral.json", "w") as f:
            json.dump(referral.to_dict(), f, indent=2)

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 1: Glaucoma Screening")
    parser.add_argument("--image", type=str, help="Path to a single fundus image")
    parser.add_argument("--image-dir", type=str, help="Directory of fundus images")
    parser.add_argument("--config", type=str, default="configs/tier/screening.yaml")
    parser.add_argument("--baseline", action="store_true", help="Use vCDR > 0.5 baseline mode")
    parser.add_argument("--device", type=str, default=None)
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

    # Build model components — all optional so the pipeline still runs in
    # pure baseline mode even without downloaded weights.
    segmenter = _try_load_segmenter(config.get("segmentation", {}), device)
    feature_encoder = None
    classifier_head = None
    if not baseline_mode:
        feature_encoder = _try_load_encoder(config.get("feature_extraction", {}), device)
        if feature_encoder is not None:
            cls_cfg = dict(config.get("classifier", {}))
            cls_cfg.setdefault("input_dim", feature_encoder.embed_dim)
            classifier_head = _try_load_classifier(cls_cfg, device)

    pipeline = ScreeningPipeline(
        segmenter=segmenter,
        feature_encoder=feature_encoder,
        classifier_head=classifier_head,
        device=device,
        baseline_mode=baseline_mode,
        referral_threshold=config["pipeline"]["referral_threshold"],
        vcdr_alert_threshold=config["pipeline"]["vcdr_alert_threshold"],
    )

    mode = "BASELINE (vCDR > 0.5)" if baseline_mode else "DINOv2 + BoundaryFormer"
    print(f"Tier 1 Screening Pipeline — Mode: {mode}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    image_size = config.get("feature_extraction", {}).get("image_size", 224)
    transform = get_fundus_transforms("val", image_size=image_size)

    images: list[Path] = []
    if args.image:
        images = [Path(args.image)]
    elif args.image_dir:
        d = Path(args.image_dir)
        images = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg"))

    if not images:
        print("\nProvide --image or --image-dir. Use --help for usage.")
        return

    print(f"\nProcessing {len(images)} image(s)...")
    results = []
    for img in images:
        try:
            rec = _screen_one(pipeline, img, transform, output_dir)
            results.append(rec)
            print(
                f"  {img.name}: risk={rec['risk_score']:.2f} "
                f"({rec['risk_level']}) vCDR={rec['vcdr']:.3f} "
                f"{'REFER' if rec['refer_to_specialist'] else 'ok'}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {img.name}: FAILED — {exc}")

    # Write aggregate summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {"total": len(images), "succeeded": len(results), "results": results},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
