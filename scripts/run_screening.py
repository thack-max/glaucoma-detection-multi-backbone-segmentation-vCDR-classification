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
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.transforms import get_fundus_transforms
from src.data.fundus_dataset import _read_image
from src.models.classifier_head import load_screening_head
from src.models.explainability.gradcam import (
    AttentionRolloutCAM,
    DINOv2GradCAM,
    overlay_heatmap_on_image,
)
from src.pipeline.screening import ScreeningPipeline
from src.utils.checkpoints import load_strict, untrained_warning

# The copy shipped with the package (configs/ ships beside scripts/), so the
# installed `glaucoma-screen` command works from any directory.
PACKAGED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tier" / "screening.yaml"
# The original cwd-relative default: a project directory's own edited copy.
LOCAL_CONFIG = Path("configs") / "tier" / "screening.yaml"


def default_config() -> Path:
    """Default --config: ./configs/tier/screening.yaml if present, else the packaged copy."""
    return LOCAL_CONFIG if LOCAL_CONFIG.is_file() else PACKAGED_CONFIG


def _try_load_segmenter(cfg: dict, device: str):
    """Build BoundaryFormer and, if configured, strictly load its checkpoint.

    Raises FileNotFoundError / RuntimeError when a configured checkpoint is
    missing or does not fit (never falls back to random weights silently).
    """
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
    if ckpt:
        # "model" is the key scripts/train_segmentation.py writes.
        load_strict(model, ckpt, ["model"], "BoundaryFormer segmenter")
        print(f"Loaded segmenter weights: {ckpt}")
    else:
        print(
            untrained_warning("BoundaryFormer segmenter", "segmentation.checkpoint is not set")
            + " vCDR and the vCDR > 0.5 baseline depend on it.",
            file=sys.stderr,
        )
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
    if not getattr(model, "pretrained_loaded", True):
        print(
            untrained_warning(
                "DINOv2 encoder",
                "pretrained weights could not be downloaded via torch.hub: network, "
                "proxy or I/O error; the RuntimeWarning above names it",
            ) + " Risk scores are not meaningful.",
            file=sys.stderr,
        )
    model.eval()
    return model.to(device)


def _try_load_classifier(cfg: dict, device: str, fairness: dict | None = None):
    """Build the classifier head and, for a fairness checkpoint, its EquityModule.

    The head comes from classifier.checkpoint, or from fairness.checkpoint when
    fairness is enabled (a train_fairness.py best.pth holds both the head and
    the EquityModule it was trained behind). Returns (head, equity_or_None).
    """
    fairness = fairness or {}
    # Same default as src/utils/config.py's FairnessSection.
    fair_enabled = bool(fairness.get("enabled", True))
    ckpt = cfg.get("checkpoint")
    fair_ckpt = fairness.get("checkpoint")
    if fair_ckpt and not fair_enabled:
        print(f"fairness.enabled is false: ignoring fairness.checkpoint {fair_ckpt}")
    if fair_ckpt and fair_enabled:
        if ckpt and Path(ckpt).resolve() != Path(fair_ckpt).resolve():
            raise ValueError(
                f"classifier.checkpoint ({ckpt}) and fairness.checkpoint ({fair_ckpt}) "
                f"differ: a train_fairness.py head only fits the EquityModule it was "
                f"trained with. Set one of them (or both to the same file)."
            )
        ckpt = fair_ckpt
    # Same head class train_screening_classifier.py / train_fairness.py save;
    # loaded strictly.
    head, equity = load_screening_head(
        ckpt,
        cfg.get("input_dim", 768),
        with_equity=fair_enabled,
        num_groups=fairness.get("num_groups"),
    )
    if fair_ckpt and fair_enabled and equity is None:
        raise RuntimeError(
            f"fairness.checkpoint {fair_ckpt} holds no EquityModule ('equity') "
            f"weights; expected a scripts/train_fairness.py checkpoint."
        )
    if ckpt:
        print(f"Loaded classifier head: {ckpt}")
        if equity is not None:
            print(f"Loaded EquityModule ({equity.equity_attn.num_groups} groups): {ckpt}")
    else:
        print(
            untrained_warning("Classifier head", "classifier.checkpoint is not set")
            + " Risk scores are not meaningful.",
            file=sys.stderr,
        )
    head.eval()
    if equity is not None:
        equity = equity.eval().to(device)
    return head.to(device), equity


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
            head = pipeline.classifier_head
            if pipeline.equity_module is not None:
                # The head scores EquityModule outputs (group bias unknown here).
                # A fresh wrapper starts in train mode: give it the head's mode.
                head = nn.Sequential(pipeline.equity_module, head)
                head.training = pipeline.classifier_head.training
            explainer = DINOv2GradCAM(encoder, head)
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
    parser.add_argument("--config", type=str, default=str(default_config()))
    parser.add_argument("--baseline", action="store_true", help="Use vCDR > 0.5 baseline mode")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Always say which file was used: the default depends on the cwd.
    print(f"Config: {Path(args.config).resolve()}")
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
    equity_module = None
    if not baseline_mode:
        feature_encoder = _try_load_encoder(config.get("feature_extraction", {}), device)
        if feature_encoder is not None:
            cls_cfg = dict(config.get("classifier", {}))
            cls_cfg.setdefault("input_dim", feature_encoder.embed_dim)
            classifier_head, equity_module = _try_load_classifier(
                cls_cfg, device, fairness=config.get("fairness", {}),
            )

    pipeline = ScreeningPipeline(
        segmenter=segmenter,
        feature_encoder=feature_encoder,
        classifier_head=classifier_head,
        equity_module=equity_module,
        device=device,
        baseline_mode=baseline_mode,
        referral_threshold=config["pipeline"]["referral_threshold"],
        vcdr_alert_threshold=config["pipeline"]["vcdr_alert_threshold"],
    )

    mode = "BASELINE (vCDR > 0.5)" if baseline_mode else "DINOv2 + BoundaryFormer"
    if equity_module is not None:
        mode += " + EquityAttention"
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
