"""
Targeted tests closing coverage gaps identified by the roadmap audit.

Covers:
  - SwinUNETROCT.forward (was construction-only)
  - FairnessAuditor (had zero tests)
  - visualization.overlay_segmentation (had zero tests)
  - LlamaVisionReporter construction path (smoke test only — full
    generation requires multi-GB Llama weights)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# SwinUNETROCT forward pass + RNFL extraction
# ---------------------------------------------------------------------------


class TestSwinUNETRForward:
    def test_forward_matches_input_shape(self):
        from src.models.segmentation.swin_unetr_oct import SwinUNETROCT

        model = SwinUNETROCT(
            in_channels=1,
            num_classes=4,
            embed_dim=16,
            depths=(1, 1, 1, 1),
            num_heads=(2, 2, 2, 2),
            patch_size=(2, 4, 4),
        )
        model.eval()

        volume = torch.randn(1, 1, 8, 32, 32)
        with torch.no_grad():
            logits = model(volume)

        # Output should match input spatial shape with num_classes channels
        assert logits.shape == (1, 4, 8, 32, 32)
        assert torch.isfinite(logits).all()

    def test_segment_rnfl_returns_binary_mask(self):
        from src.models.segmentation.swin_unetr_oct import SwinUNETROCT

        model = SwinUNETROCT(
            in_channels=1,
            num_classes=4,
            embed_dim=16,
            depths=(1, 1, 1, 1),
            num_heads=(2, 2, 2, 2),
            patch_size=(2, 4, 4),
        )
        model.eval()

        volume = torch.randn(1, 1, 8, 32, 32)
        with torch.no_grad():
            mask = model.segment_rnfl(volume)

        # RNFL mask should be binary 0/1 and same spatial size as the volume
        assert mask.shape == (1, 8, 32, 32)
        unique = set(mask.unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_accepts_unbatched_volume(self):
        from src.models.segmentation.swin_unetr_oct import SwinUNETROCT

        model = SwinUNETROCT(
            in_channels=1,
            num_classes=2,
            embed_dim=16,
            depths=(1, 1, 1, 1),
            num_heads=(2, 2, 2, 2),
            patch_size=(2, 4, 4),
        )
        model.eval()

        volume = torch.randn(1, 8, 32, 32)  # [B, D, H, W] without C axis
        with torch.no_grad():
            mask = model.segment_rnfl(volume)
        assert mask.shape == (1, 8, 32, 32)


# ---------------------------------------------------------------------------
# FairnessAuditor
# ---------------------------------------------------------------------------


class TestFairnessAuditor:
    def test_overall_auc(self):
        from src.evaluation.fairness_audit import FairnessAuditor

        auditor = FairnessAuditor(group_names=["Asian", "Black", "White"])
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1] * 3)
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.7, 0.85, 0.3, 0.2, 0.6, 0.9] * 3)
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] * 3)

        report = auditor.audit(labels, scores, groups)
        assert 0.9 <= report.overall_auc <= 1.0
        assert set(report.per_group_auc.keys()) == {"Asian", "Black", "White"}
        for auc in report.per_group_auc.values():
            assert 0.0 <= auc <= 1.0
        assert report.es_auc <= report.overall_auc

    def test_reports_group_sizes(self):
        from src.evaluation.fairness_audit import FairnessAuditor

        auditor = FairnessAuditor(group_names=["A", "B"])
        labels = np.array([0] * 30 + [1] * 30)
        scores = np.random.default_rng(0).random(60)
        groups = np.array([0] * 30 + [1] * 30)

        report = auditor.audit(labels, scores, groups)
        assert report.group_sizes["A"] == 30
        assert report.group_sizes["B"] == 30

    def test_handles_small_group(self):
        """A group with fewer than 10 samples or only one label should
        appear as NaN per-group AUC and not crash the audit."""
        from src.evaluation.fairness_audit import FairnessAuditor

        auditor = FairnessAuditor(group_names=["A", "B", "C"])
        labels = np.array([0, 0, 1, 1] * 5 + [0] * 3 + [0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9] * 5 + [0.5] * 3 + [0.3, 0.7])
        groups = np.array([0] * 20 + [1] * 3 + [2] * 2)

        report = auditor.audit(labels, scores, groups)
        assert np.isnan(report.per_group_auc["B"])  # 3 samples < 10
        assert np.isnan(report.per_group_auc["C"])  # 2 samples < 10
        assert not np.isnan(report.per_group_auc["A"])

    def test_auc_gap_computation(self):
        from src.evaluation.fairness_audit import FairnessAuditor

        auditor = FairnessAuditor(group_names=["A", "B"], max_acceptable_gap=0.2)
        # Group A: perfect classifier. Group B: near-random.
        labels = np.array([0, 0, 1, 1] * 5 + [0, 0, 1, 1] * 5)
        scores_a = np.tile([0.1, 0.2, 0.8, 0.9], 5)
        scores_b = np.tile([0.6, 0.4, 0.3, 0.55], 5)
        scores = np.concatenate([scores_a, scores_b])
        groups = np.array([0] * 20 + [1] * 20)

        report = auditor.audit(labels, scores, groups)
        assert report.auc_gap > 0.3  # A is near-perfect, B is near-random


# ---------------------------------------------------------------------------
# visualization.overlay_segmentation
# ---------------------------------------------------------------------------


class TestVisualizationOverlay:
    def test_overlay_segmentation_preserves_shape(self):
        from src.utils.visualization import overlay_segmentation

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        disc = np.zeros((64, 64), dtype=np.uint8)
        disc[20:44, 20:44] = 1
        cup = np.zeros((64, 64), dtype=np.uint8)
        cup[28:36, 28:36] = 1

        out = overlay_segmentation(image, disc, cup, alpha=0.4)
        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_overlay_segmentation_colors_applied(self):
        from src.utils.visualization import overlay_segmentation

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        disc = np.ones((32, 32), dtype=np.uint8)
        cup = np.ones((32, 32), dtype=np.uint8)

        # With both masks everywhere, alpha > 0, output must differ from the
        # all-zero input.
        out = overlay_segmentation(
            image, disc, cup,
            disc_color=(0, 255, 0),
            cup_color=(255, 0, 0),
            alpha=0.5,
        )
        assert out.sum() > 0

    def test_overlay_segmentation_alpha_zero_identity(self):
        """Alpha=0 should leave the image untouched."""
        from src.utils.visualization import overlay_segmentation

        image = np.full((16, 16, 3), 120, dtype=np.uint8)
        disc = np.ones((16, 16), dtype=np.uint8)
        cup = np.ones((16, 16), dtype=np.uint8)
        out = overlay_segmentation(image, disc, cup, alpha=0.0)
        assert np.array_equal(out, image)

    def test_overlay_segmentation_handles_none_masks(self):
        from src.utils.visualization import overlay_segmentation

        image = np.full((16, 16, 3), 50, dtype=np.uint8)
        out = overlay_segmentation(image, None, None)
        assert out.shape == image.shape


# ---------------------------------------------------------------------------
# LlamaVisionReporter — smoke test only
# ---------------------------------------------------------------------------


class TestLlamaVisionReporterSmoke:
    def test_construction_does_not_load_model(self):
        """LlamaVisionReporter should defer model loading until first use."""
        from src.models.vlm.llama_vision_reporter import LlamaVisionReporter

        reporter = LlamaVisionReporter(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            lora_rank=8,
            max_new_tokens=128,
            load_in_4bit=True,
        )
        # No weights loaded yet
        assert reporter._model is None
        assert reporter._processor is None
        assert reporter.max_new_tokens == 128

    def test_clinical_report_dataclass_roundtrip(self):
        from src.models.vlm.llama_vision_reporter import ClinicalReport

        report = ClinicalReport(
            diagnosis="early glaucoma",
            confidence=0.83,
            structural_findings="RNFL thinning superiorly",
            raw_text="Full report here",
        )
        assert report.requires_human_review is True
        assert report.diagnosis == "early glaucoma"
