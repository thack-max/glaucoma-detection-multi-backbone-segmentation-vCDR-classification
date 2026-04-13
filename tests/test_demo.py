"""Tests for the Gradio demo's underlying inference logic (no UI launch)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.demo_gradio import DemoState
from src.models.features.dinov2_encoder import DINOv2Encoder
from src.models.segmentation.boundary_former import BoundaryFormer


def _build_state(xgboost=None, image_size: int = 64):
    """Construct a DemoState with small random-init models (no checkpoints)."""
    segmenter = BoundaryFormer(num_classes=3, base_channels=16, growth_rate=8)
    segmenter.eval()
    encoder = DINOv2Encoder(variant="dinov2_vitb14", freeze_encoder=True)
    encoder.eval()
    classifier = nn.Sequential(
        nn.LayerNorm(encoder.embed_dim),
        nn.Dropout(0.0),
        nn.Linear(encoder.embed_dim, 1),
    )
    classifier.eval()
    return DemoState(
        segmenter=segmenter,
        encoder=encoder,
        classifier=classifier,
        xgboost=xgboost,
        device="cpu",
        image_size=image_size,
    )


class TestDemoState:
    def test_run_produces_all_outputs(self):
        state = _build_state(image_size=32)
        image = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
        result = state.run(image)
        assert result["error"] is None
        assert result["original"].shape == image.shape
        assert result["segmentation"].shape == image.shape
        assert result["gradcam"].shape == image.shape
        assert "Risk:" in result["summary"]
        assert "vCDR" in result["summary"]
        assert "ISNT" in result["summary"]

    def test_run_handles_grayscale(self):
        state = _build_state(image_size=32)
        gray = (np.random.rand(64, 64) * 255).astype(np.uint8)
        result = state.run(gray)
        assert result["error"] is None
        # Output preserves the H, W and adds 3-channel
        assert result["original"].shape[-1] == 3

    def test_run_handles_rgba(self):
        state = _build_state(image_size=32)
        rgba = (np.random.rand(64, 64, 4) * 255).astype(np.uint8)
        result = state.run(rgba)
        assert result["error"] is None
        assert result["original"].shape[-1] == 3

    def test_run_with_no_image_returns_error(self):
        state = _build_state(image_size=32)
        result = state.run(None)
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# XGBoost path through the demo (skipped if xgboost not available)
# ---------------------------------------------------------------------------


_XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="xgboost not installed")
def test_demo_with_xgboost_classifier_includes_shap_summary():
    from src.models.classification.feature_extraction import HANDCRAFTED_FEATURE_NAMES
    from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier

    embed_dim = 768  # match DINOv2-Base
    n_features = embed_dim + len(HANDCRAFTED_FEATURE_NAMES)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n_features)).astype(np.float32)
    y = (X[:, embed_dim] > 0).astype(np.int32)
    clf = XGBoostScreeningClassifier(embed_dim=embed_dim, n_estimators=10)
    clf.fit(X, y)

    state = _build_state(xgboost=clf, image_size=32)
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    result = state.run(image)
    assert result["error"] is None
    # SHAP summary should be embedded in the markdown output if shap is installed
    if importlib.util.find_spec("shap") is not None:
        assert "SHAP" in result["summary"] or "shap" in result["summary"].lower()
