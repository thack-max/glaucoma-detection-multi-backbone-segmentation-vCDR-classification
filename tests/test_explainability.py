"""Tests for the explainability module (Grad-CAM + attention rollout)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.explainability.gradcam import (
    DINOv2GradCAM,
    AttentionRolloutCAM,
    colorize_heatmap,
    overlay_heatmap_on_image,
)


class _StubEncoder(nn.Module):
    """Minimal ViT-like encoder used to unit-test explainability helpers.

    It mirrors the DINOv2Encoder contract that the explainability module
    relies on: a CLS-token forward pass and an extract_patch_features method
    that returns [B, N, D] patch tokens.
    """

    def __init__(self, embed_dim: int = 16, num_patches: int = 49) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.conv = nn.Conv2d(3, embed_dim, 3, padding=1)
        self.cls_proj = nn.Linear(embed_dim, embed_dim)

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        # Produce [B, N, D] by downsampling the feature map
        feats = self.conv(x)  # [B, D, H, W]
        feats = nn.functional.adaptive_avg_pool2d(
            feats,
            (int(self.num_patches ** 0.5), int(self.num_patches ** 0.5)),
        )  # [B, D, s, s]
        return feats.flatten(2).transpose(1, 2)  # [B, N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patch_features(x)
        return self.cls_proj(patches.mean(dim=1))

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        # Build a fake attention matrix [B, heads, tokens, tokens]
        B = x.shape[0]
        tokens = self.num_patches + 1
        attn = torch.softmax(torch.randn(B, 4, tokens, tokens), dim=-1)
        return attn


class _BinaryHead(nn.Module):
    def __init__(self, input_dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class TestDINOv2GradCAM:
    def test_output_shape(self):
        torch.manual_seed(0)
        encoder = _StubEncoder(embed_dim=16, num_patches=49)
        head = _BinaryHead(16)
        gradcam = DINOv2GradCAM(encoder, head)

        image = torch.randn(2, 3, 64, 64)
        heatmap = gradcam(image)
        assert heatmap.shape == (2, 64, 64)
        assert torch.all(heatmap >= 0)
        assert torch.all(heatmap <= 1)

    def test_nonzero_on_meaningful_input(self):
        torch.manual_seed(0)
        encoder = _StubEncoder()
        head = _BinaryHead()
        gradcam = DINOv2GradCAM(encoder, head)

        image = torch.randn(1, 3, 32, 32)
        heatmap = gradcam(image)
        # Not every pixel should be zero
        assert heatmap.sum() > 0

    def test_multi_class_head(self):
        torch.manual_seed(0)
        encoder = _StubEncoder()
        classifier = nn.Linear(16, 3)
        gradcam = DINOv2GradCAM(encoder, classifier)

        image = torch.randn(1, 3, 32, 32)
        heatmap = gradcam(image, target_class=2)
        assert heatmap.shape == (1, 32, 32)

    def test_frozen_encoder(self):
        """The shipped screening config freezes the encoder (freeze_encoder: true)."""
        torch.manual_seed(0)
        encoder = _StubEncoder()
        encoder.requires_grad_(False)
        head = _BinaryHead()
        gradcam = DINOv2GradCAM(encoder, head)

        heatmap = gradcam(torch.randn(1, 3, 32, 32))
        assert heatmap.shape == (1, 32, 32)
        assert heatmap.sum() > 0
        # Explaining must not unfreeze or touch the encoder weights.
        for p in encoder.parameters():
            assert not p.requires_grad
            assert p.grad is None

    def test_frozen_dinov2_encoder(self, monkeypatch):
        from src.models.features.dinov2_encoder import DINOv2Encoder

        def _offline(*args, **kwargs):
            raise RuntimeError("offline test: use the random-init fallback")

        monkeypatch.setattr(torch.hub, "load", _offline)
        torch.manual_seed(0)
        encoder = DINOv2Encoder(variant="dinov2_vits14", freeze_encoder=True)
        gradcam = DINOv2GradCAM(encoder, _BinaryHead(encoder.embed_dim))

        heatmap = gradcam(torch.randn(1, 3, 56, 56))
        assert heatmap.shape == (1, 56, 56)
        assert torch.isfinite(heatmap).all()

    def test_screening_cli_writes_overlay_with_frozen_encoder(self, tmp_path):
        from scripts.run_screening import _save_gradcam_overlay
        from src.pipeline.screening import ScreeningPipeline

        torch.manual_seed(0)
        encoder = _StubEncoder()
        encoder.requires_grad_(False)
        pipeline = ScreeningPipeline(
            feature_encoder=encoder, classifier_head=_BinaryHead(), device="cpu",
        )
        image_np = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = tmp_path / "x_gradcam.png"

        assert _save_gradcam_overlay(image_np, torch.randn(1, 3, 32, 32), pipeline, out)
        assert out.exists()


class TestAttentionRollout:
    def test_output_shape(self):
        encoder = _StubEncoder(embed_dim=16, num_patches=49)
        rollout = AttentionRolloutCAM(encoder)
        image = torch.randn(2, 3, 64, 64)
        heatmap = rollout(image)
        assert heatmap.shape == (2, 64, 64)
        assert torch.all(heatmap >= 0)
        assert torch.all(heatmap <= 1)

    def test_missing_attention_returns_blank(self):
        encoder = nn.Linear(3, 8)  # has no get_attention_maps
        rollout = AttentionRolloutCAM(encoder)
        image = torch.randn(1, 3, 16, 16)
        heatmap = rollout(image)
        assert heatmap.shape == (1, 16, 16)
        assert (heatmap == 0).all()


class TestVisualizationHelpers:
    def test_colorize_heatmap_shape(self):
        heatmap = np.random.rand(32, 32).astype(np.float32)
        colored = colorize_heatmap(heatmap)
        assert colored.shape == (32, 32, 3)
        assert colored.dtype == np.uint8

    def test_colorize_heatmap_accepts_tensor(self):
        heatmap = torch.rand(16, 16)
        colored = colorize_heatmap(heatmap)
        assert colored.shape == (16, 16, 3)

    def test_overlay_preserves_shape(self):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        heatmap = np.random.rand(64, 64).astype(np.float32)
        out = overlay_heatmap_on_image(image, heatmap, alpha=0.3)
        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_overlay_resizes_heatmap(self):
        """Heatmaps at a lower resolution are auto-resized to the image size."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        heatmap = np.random.rand(16, 16).astype(np.float32)
        out = overlay_heatmap_on_image(image, heatmap)
        assert out.shape == (64, 64, 3)

    def test_overlay_alpha_zero_returns_original(self):
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        heatmap = np.ones((32, 32), dtype=np.float32)
        out = overlay_heatmap_on_image(image, heatmap, alpha=0.0)
        # alpha=0 means fully opaque base image
        assert np.allclose(out, image)
