"""
Interactive Gradio demo for Tier 1 GlaucoMamba screening.

Drop a fundus image into the UI; the demo runs:

    BoundaryFormer segmentation
        -> compute vCDR + ISNT + rim ratios
        -> DINOv2-Base feature embedding
        -> classifier head (linear or XGBoost+SHAP if loaded)
        -> Grad-CAM heatmap

and shows the original image alongside the segmentation overlay, the
Grad-CAM heatmap, the computed vCDR, the risk score / risk level,
the referral decision, and (when XGBoost+SHAP is loaded) the top
contributing features.

The demo loads checkpoints from the Tier 1 config when present, but
also runs with random-init models so the UI is exercisable without
any trained weights — useful for verifying the visualization layout.

Usage:
    pip install -e ".[demo]"   # installs gradio
    python scripts/demo_gradio.py

    # With trained checkpoints
    python scripts/demo_gradio.py \
        --segmenter-checkpoint checkpoints/boundary_former/best.pth \
        --classifier-checkpoint checkpoints/screening_classifier.pth

    # With XGBoost+SHAP classifier
    python scripts/demo_gradio.py \
        --xgboost-checkpoint checkpoints/xgboost_screening.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data.transforms import get_fundus_transforms
from src.models.classification.feature_extraction import (
    extract_handcrafted_features,
)
from src.models.explainability.gradcam import (
    AttentionRolloutCAM,
    DINOv2GradCAM,
    overlay_heatmap_on_image,
)
from src.models.features.dinov2_encoder import DINOv2Encoder
from src.models.segmentation.boundary_former import BoundaryFormer
from src.pipeline.common import classify_risk, compute_vcdr
from src.utils.visualization import overlay_segmentation


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_segmenter(checkpoint_path: str | None, device: str) -> BoundaryFormer:
    model = BoundaryFormer(num_classes=3, base_channels=64, growth_rate=32)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded segmenter from {checkpoint_path}")
    else:
        print("No segmenter checkpoint provided — using random init")
    model.eval()
    return model.to(device)


def _load_encoder(variant: str, device: str) -> DINOv2Encoder:
    encoder = DINOv2Encoder(variant=variant, freeze_encoder=True)
    encoder.eval()
    return encoder.to(device)


def _load_classifier(checkpoint_path: str | None, embed_dim: int, device: str) -> nn.Module:
    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(0.0),
        nn.Linear(embed_dim, 1),
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        head.load_state_dict(state.get("classifier", state), strict=False)
        print(f"Loaded classifier head from {checkpoint_path}")
    else:
        print("No classifier checkpoint provided — using random init")
    head.eval()
    return head.to(device)


def _load_xgboost(checkpoint_path: str | None):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    try:
        from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier

        clf = XGBoostScreeningClassifier()
        clf.load(checkpoint_path)
        print(f"Loaded XGBoost classifier from {checkpoint_path}")
        return clf
    except Exception as exc:  # noqa: BLE001
        print(f"XGBoost load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class DemoState:
    """Holds loaded models so each request reuses them."""

    def __init__(
        self,
        segmenter: BoundaryFormer,
        encoder: DINOv2Encoder,
        classifier: nn.Module,
        xgboost: Any | None,
        device: str,
        image_size: int,
    ) -> None:
        self.segmenter = segmenter
        self.encoder = encoder
        self.classifier = classifier
        self.xgboost = xgboost
        self.device = device
        self.image_size = image_size
        self.transform = get_fundus_transforms("val", image_size=image_size)

    @torch.no_grad()
    def _segment(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.segmenter(image_tensor)
        pred = out["logits"].argmax(dim=1)
        disc = (pred >= 1).float()[0]
        cup = (pred == 2).float()[0]
        return disc, cup

    @torch.no_grad()
    def _embed(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(image_tensor)

    def _gradcam(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Best-effort Grad-CAM, falling back to attention rollout."""
        try:
            cam = DINOv2GradCAM(self.encoder, self.classifier)
            heat = cam(image_tensor)
        except Exception:
            cam = AttentionRolloutCAM(self.encoder)
            heat = cam(image_tensor)
        return heat[0].detach().cpu().numpy()

    def run(self, image_np: np.ndarray) -> dict[str, Any]:
        """Process one fundus image. Returns a dict for the Gradio outputs."""
        if image_np is None:
            return {"error": "Please upload a fundus image first."}
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        if image_np.shape[-1] == 4:  # RGBA -> RGB
            image_np = image_np[..., :3]

        tensor = self.transform(image_np)["image"].unsqueeze(0).to(self.device)

        # Segmentation
        disc_mask, cup_mask = self._segment(tensor)
        disc_np = disc_mask.cpu().numpy().astype(np.uint8)
        cup_np = cup_mask.cpu().numpy().astype(np.uint8)

        # Resize masks back to the input image's size for overlaying
        from PIL import Image

        target_size = (image_np.shape[1], image_np.shape[0])
        disc_resized = np.array(
            Image.fromarray((disc_np * 255).astype(np.uint8)).resize(target_size)
        )
        cup_resized = np.array(
            Image.fromarray((cup_np * 255).astype(np.uint8)).resize(target_size)
        )
        seg_overlay = overlay_segmentation(
            image_np, (disc_resized > 127).astype(np.uint8),
            (cup_resized > 127).astype(np.uint8),
            alpha=0.35,
        )

        # Handcrafted features
        feats = extract_handcrafted_features(disc_mask, cup_mask)
        vcdr = compute_vcdr(disc_mask, cup_mask)

        # Risk score
        embedding = self._embed(tensor)
        if self.xgboost is not None:
            combined = np.concatenate(
                [embedding[0].cpu().numpy(), feats.to_array()],
            )[None, :]
            xgb_expl = self.xgboost.explain(combined)
            risk_score = xgb_expl.risk_score
            top_features = xgb_expl.top_contributions(k=5)
        else:
            with torch.no_grad():
                logit = self.classifier(embedding)
            risk_score = float(torch.sigmoid(logit).item())
            top_features = None

        risk_level = classify_risk(risk_score, vcdr)
        refer = risk_level.value in {"moderate", "high"}

        # Grad-CAM heatmap overlay
        heatmap = self._gradcam(tensor)
        gradcam_overlay = overlay_heatmap_on_image(image_np, heatmap, alpha=0.45)

        # Build a clinical summary string
        summary_lines = [
            f"## Risk: **{risk_level.value.upper()}** (score {risk_score:.3f})",
            f"- vCDR: **{vcdr:.3f}** (baseline threshold 0.5)",
            f"- horizontal CDR: {feats.horizontal_cdr:.3f}",
            f"- rim/disc area ratio: {feats.rim_disc_area_ratio:.3f}",
            f"- ISNT rule: **{'violated' if feats.isnt_violation > 0.5 else 'satisfied'}**",
            "",
            f"### Referral: {'**REFER TO SPECIALIST**' if refer else 'no referral needed'}",
        ]
        if top_features:
            summary_lines.extend(["", "### Top SHAP contributions:"])
            for name, val in top_features:
                arrow = "+" if val >= 0 else ""
                summary_lines.append(f"- `{name}`: {arrow}{val:.3f}")

        return {
            "original": image_np,
            "segmentation": seg_overlay,
            "gradcam": gradcam_overlay,
            "summary": "\n".join(summary_lines),
            "error": None,
        }


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app(state: DemoState):
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Gradio is required for the demo. Install with: pip install -e \".[demo]\""
        ) from exc

    def _process(image):
        result = state.run(image)
        if result.get("error"):
            return None, None, None, result["error"]
        return result["original"], result["segmentation"], result["gradcam"], result["summary"]

    with gr.Blocks(title="GlaucoMamba — Tier 1 Screening Demo") as app:
        gr.Markdown(
            "# GlaucoMamba — Tier 1 Glaucoma Screening\n"
            "Upload a fundus photograph. The demo runs BoundaryFormer "
            "segmentation, DINOv2 feature extraction, classifier risk "
            "scoring, and Grad-CAM explainability — all on CPU.\n\n"
            "**This is a research prototype, not a medical device. "
            "Clinical decisions must involve a qualified clinician.**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Fundus image", type="numpy", height=400,
                )
                run_btn = gr.Button("Run screening", variant="primary")
            with gr.Column(scale=2):
                summary_md = gr.Markdown(
                    label="Clinical summary",
                    value="Upload an image and click **Run screening**.",
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original")
                orig_out = gr.Image(label="Original")
            with gr.Column():
                gr.Markdown("### Disc + Cup segmentation")
                seg_out = gr.Image(label="Segmentation overlay")
            with gr.Column():
                gr.Markdown("### Grad-CAM attention")
                cam_out = gr.Image(label="Grad-CAM heatmap")

        run_btn.click(
            _process,
            inputs=[input_image],
            outputs=[orig_out, seg_out, cam_out, summary_md],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 1 Gradio demo")
    parser.add_argument("--config", type=str, default="configs/tier/screening.yaml")
    parser.add_argument("--segmenter-checkpoint", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--xgboost-checkpoint", type=str, default=None)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading models...")
    segmenter = _load_segmenter(args.segmenter_checkpoint, args.device)
    encoder = _load_encoder(args.dinov2_variant, args.device)
    classifier = _load_classifier(
        args.classifier_checkpoint, encoder.embed_dim, args.device,
    )
    xgboost = _load_xgboost(args.xgboost_checkpoint)

    state = DemoState(
        segmenter=segmenter,
        encoder=encoder,
        classifier=classifier,
        xgboost=xgboost,
        device=args.device,
        image_size=args.image_size,
    )

    app = build_app(state)
    print(f"\nLaunching demo at http://localhost:{args.server_port}")
    app.launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
