"""Tests for handcrafted feature extraction + XGBoost classifier."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from src.models.classification.feature_extraction import (
    HANDCRAFTED_FEATURE_NAMES,
    HandcraftedFeatures,
    extract_handcrafted_features,
)


# ---------------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------------


def test_feature_names_match_dataclass_fields():
    f = HandcraftedFeatures()
    arr = f.to_array()
    assert arr.shape == (len(HANDCRAFTED_FEATURE_NAMES),)
    # All zero by default
    assert np.allclose(arr, 0.0)


def test_extract_features_empty_masks_returns_zeros():
    disc = np.zeros((64, 64), dtype=np.uint8)
    cup = np.zeros((64, 64), dtype=np.uint8)
    feats = extract_handcrafted_features(disc, cup)
    assert feats.vcdr == 0.0
    assert feats.rim_disc_area_ratio == 0.0


def test_extract_features_concentric_disc_cup():
    """Disc bigger than cup centred -> cup/disc area ratio sensible, ISNT satisfied."""
    H = W = 100
    disc = np.zeros((H, W), dtype=np.uint8)
    cup = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]
    disc_r = ((yy - 50) ** 2 + (xx - 50) ** 2) < 25 ** 2
    cup_r = ((yy - 50) ** 2 + (xx - 50) ** 2) < 12 ** 2
    disc[disc_r] = 1
    cup[cup_r] = 1

    feats = extract_handcrafted_features(disc, cup)

    # Vertical CDR ~ cup_diameter / disc_diameter ~ 24/50 = 0.48
    assert 0.4 < feats.vcdr < 0.55
    # Cup area smaller than disc area
    assert 0.0 < feats.cup_disc_area_ratio < 0.5
    # Rim/disc area = 1 - cup/disc area
    assert feats.rim_disc_area_ratio == pytest.approx(
        1.0 - feats.cup_disc_area_ratio, abs=1e-3,
    )
    # Concentric disc -> ISNT should hold (all rim quadrants equal => I>=S>=N>=T)
    assert feats.isnt_violation == 0.0


def test_extract_features_isnt_violation_detected():
    """Cup pushed superiorly -> inferior rim thicker than superior rim, ISNT may fail."""
    H = W = 100
    disc = np.zeros((H, W), dtype=np.uint8)
    cup = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]
    disc[(yy - 50) ** 2 + (xx - 50) ** 2 < 30 ** 2] = 1
    # Cup shifted superiorly (smaller y values)
    cup[(yy - 30) ** 2 + (xx - 50) ** 2 < 18 ** 2] = 1
    cup &= disc  # ensure cup ⊆ disc

    feats = extract_handcrafted_features(disc, cup)
    # Inferior rim should be thicker than superior rim (cup ate the superior rim)
    assert feats.rim_thickness_inferior > feats.rim_thickness_superior


def test_extract_features_accepts_torch_tensor():
    disc = torch.zeros(64, 64)
    disc[20:50, 20:50] = 1.0
    cup = torch.zeros(64, 64)
    cup[28:42, 28:42] = 1.0

    feats = extract_handcrafted_features(disc, cup)
    assert feats.vcdr > 0.0
    assert feats.cup_disc_area_ratio > 0.0


def test_extract_features_squeezes_leading_dims():
    """[1, 1, H, W] tensors should be reduced to [H, W] internally."""
    disc = torch.zeros(1, 1, 32, 32)
    disc[0, 0, 8:24, 8:24] = 1.0
    cup = torch.zeros(1, 1, 32, 32)
    cup[0, 0, 12:20, 12:20] = 1.0

    feats = extract_handcrafted_features(disc, cup)
    assert feats.vcdr > 0.0


# ---------------------------------------------------------------------------
# XGBoost classifier (skipped if xgboost not installed)
# ---------------------------------------------------------------------------


_XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
_SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="xgboost not installed")
class TestXGBoostClassifier:
    def test_fit_and_predict(self):
        from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier

        rng = np.random.default_rng(0)
        n, embed_dim = 50, 8
        n_features = embed_dim + len(HANDCRAFTED_FEATURE_NAMES)

        # Synthetic features where vcdr (penultimate column block) carries signal
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        # Force feature index `embed_dim + 0` (vcdr) to correlate with label
        y = (X[:, embed_dim + 0] > 0).astype(np.int32)

        clf = XGBoostScreeningClassifier(
            embed_dim=embed_dim, n_estimators=30, max_depth=3,
        )
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (n,)
        assert proba.min() >= 0.0 and proba.max() <= 1.0
        # Should fit the training data well (not perfect, but well above chance)
        assert ((proba > 0.5).astype(np.int32) == y).mean() > 0.7

    def test_wrong_feature_dim_rejected(self):
        from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier

        clf = XGBoostScreeningClassifier(embed_dim=16)
        # Wrong feature count
        X = np.zeros((4, 5), dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="Expected"):
            clf.fit(X, y)

    def test_predict_before_fit_raises(self):
        from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier

        clf = XGBoostScreeningClassifier()
        X = np.zeros((1, clf.n_features), dtype=np.float32)
        with pytest.raises(RuntimeError, match="not been fit"):
            clf.predict_proba(X)

    def test_save_load_roundtrip(self, tmp_path):
        from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier

        rng = np.random.default_rng(1)
        embed_dim = 8
        n_features = embed_dim + len(HANDCRAFTED_FEATURE_NAMES)
        X = rng.normal(size=(40, n_features)).astype(np.float32)
        y = (X[:, embed_dim] > 0).astype(np.int32)

        clf = XGBoostScreeningClassifier(embed_dim=embed_dim, n_estimators=10)
        clf.fit(X, y)
        proba_before = clf.predict_proba(X)

        path = tmp_path / "xgb.joblib"
        clf.save(path)

        clf2 = XGBoostScreeningClassifier(embed_dim=embed_dim).load(path)
        proba_after = clf2.predict_proba(X)
        assert np.allclose(proba_before, proba_after)


@pytest.mark.skipif(
    not (_XGBOOST_AVAILABLE and _SHAP_AVAILABLE),
    reason="xgboost+shap not installed",
)
class TestXGBoostExplainability:
    def test_explain_returns_per_feature_attributions(self):
        from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier

        rng = np.random.default_rng(2)
        embed_dim = 4
        n_features = embed_dim + len(HANDCRAFTED_FEATURE_NAMES)
        X = rng.normal(size=(40, n_features)).astype(np.float32)
        y = (X[:, embed_dim] > 0).astype(np.int32)

        clf = XGBoostScreeningClassifier(embed_dim=embed_dim, n_estimators=10)
        clf.fit(X, y)

        expl = clf.explain(X[:1])
        assert len(expl.shap_values) == n_features
        assert expl.feature_names == clf.feature_names

        top = expl.top_contributions(k=3)
        assert len(top) == 3
        # Each entry is (name, value)
        assert all(len(t) == 2 for t in top)
