"""
XGBoost screening classifier with SHAP explainability.

Trains a gradient-boosted decision tree on the concatenation of
DINOv2 CLS embeddings and handcrafted morphometric features (vCDR, rim
ratios, ISNT violation, etc). Provides SHAP-based attribution so each
prediction is explained as a vector of "+0.18 from vCDR, -0.04 from
rim_disc_area_ratio, ..." instead of an opaque embedding.

Why this lives alongside the linear DINOv2 head:

  - On small datasets (REFUGE: 1.2k images, ORIGA: 650 images), trees
    typically beat linear classifiers on rich feature vectors.
  - SHAP attribution is human-readable and matches the way clinicians
    already think about glaucoma — vCDR, rim thickness, ISNT.
  - Provides a controlled comparison against the DINOv2-only path.

xgboost and shap are both lazy-imported so this module loads even when
neither library is installed (Tier 1 deployments don't need them).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.models.classification.feature_extraction import (
    HANDCRAFTED_FEATURE_NAMES,
)


@dataclass
class XGBoostExplanation:
    """Per-prediction SHAP explanation."""

    risk_score: float
    base_value: float  # SHAP base value (the model's mean output)
    feature_names: list[str]
    shap_values: list[float]  # one per feature

    def top_contributions(self, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k features by absolute SHAP value."""
        order = np.argsort(np.abs(self.shap_values))[::-1][:k]
        return [(self.feature_names[i], float(self.shap_values[i])) for i in order]


class XGBoostScreeningClassifier:
    """XGBoost classifier on DINOv2 + handcrafted features.

    Usage:
        clf = XGBoostScreeningClassifier(embed_dim=768)
        clf.fit(features, labels)              # one-shot training
        proba = clf.predict_proba(features)    # [B] in [0, 1]
        expl = clf.explain(features[0])        # SHAP attribution

    The classifier accepts the concatenation of DINOv2 embedding
    [B, embed_dim] and handcrafted features [B, n_handcrafted] as a single
    [B, embed_dim + n_handcrafted] feature matrix.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self._model: Any | None = None
        self._explainer: Any | None = None

        # Feature names = DINOv2 dim_0..dim_N + handcrafted names
        self.feature_names: list[str] = (
            [f"dinov2_dim_{i}" for i in range(embed_dim)] + list(HANDCRAFTED_FEATURE_NAMES)
        )
        self.n_features = len(self.feature_names)

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def _ensure_xgboost(self):
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "XGBoostScreeningClassifier requires xgboost. "
                "Install with: pip install xgboost"
            ) from exc
        return xgb

    def _build_model(self):
        xgb = self._ensure_xgboost()
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "XGBoostScreeningClassifier":
        """Fit the classifier.

        Args:
            features: [B, embed_dim + n_handcrafted] feature matrix.
            labels: [B] binary labels (0 = normal, 1 = glaucoma).
            eval_set: Optional (val_features, val_labels) tuple for
                early-stopping-style monitoring during training.
        """
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features ({self.embed_dim} DINOv2 + "
                f"{len(HANDCRAFTED_FEATURE_NAMES)} handcrafted), got {features.shape[1]}"
            )

        self._model = self._build_model()
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = False
        self._model.fit(features, labels, **fit_kwargs)

        # Reset cached explainer
        self._explainer = None
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(glaucoma) per sample, shape [B]."""
        if self._model is None:
            raise RuntimeError("Classifier has not been fit yet.")
        proba = self._model.predict_proba(features)
        # Binary classifier => proba shape [B, 2]; return positive-class column
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int32)

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------

    def _ensure_explainer(self):
        if self._explainer is not None:
            return self._explainer
        if self._model is None:
            raise RuntimeError("Classifier has not been fit yet.")
        try:
            import shap
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SHAP explanations require shap. Install with: pip install shap"
            ) from exc
        self._explainer = shap.TreeExplainer(self._model)
        return self._explainer

    def explain(self, features: np.ndarray) -> XGBoostExplanation:
        """Produce a SHAP attribution for a single feature vector.

        Args:
            features: 1-D array of length n_features, OR a 2D array with
                a single row [1, n_features].

        Returns:
            XGBoostExplanation with risk_score + per-feature shap_values.
        """
        if features.ndim == 1:
            features = features[None, :]
        if features.shape != (1, self.n_features):
            raise ValueError(
                f"explain() expects shape [1, {self.n_features}], got {features.shape}"
            )

        explainer = self._ensure_explainer()
        # shap.TreeExplainer.shap_values returns [B, n_features] for binary
        shap_vals = explainer.shap_values(features)
        if isinstance(shap_vals, list):
            # Older shap returns [class0_arr, class1_arr]; pick positive class
            shap_arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            shap_arr = shap_vals
        shap_arr = np.asarray(shap_arr).reshape(-1)

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]
        base_value = float(base_value)

        return XGBoostExplanation(
            risk_score=float(self.predict_proba(features)[0]),
            base_value=base_value,
            feature_names=self.feature_names,
            shap_values=shap_arr.tolist(),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the trained model + metadata to disk."""
        import joblib

        if self._model is None:
            raise RuntimeError("Cannot save an untrained classifier.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "embed_dim": self.embed_dim,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "feature_names": self.feature_names,
            },
            path,
        )

    def load(self, path: str | Path) -> "XGBoostScreeningClassifier":
        """Restore a trained model from disk."""
        import joblib

        state = joblib.load(path)
        self._model = state["model"]
        self.embed_dim = state["embed_dim"]
        self.feature_names = state["feature_names"]
        self.n_features = len(self.feature_names)
        self._explainer = None
        return self
