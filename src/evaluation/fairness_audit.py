"""
Fairness auditing for demographic parity in glaucoma detection.

Reference: Harvard Glaucoma Fairness (IEEE TMI, 2024)
  - 3,300 subjects balanced across Asian/Black/White
  - Measures AUC per group + Equalized-Subgroup AUC (ES-AUC)
  - FIN improved Black patient AUC: 0.77 → 0.82

Also: retinIA study (Frontiers in Ophthalmology, 2025) showed
AI accuracy 88.6% with best results from human+AI synergy (80.4% sensitivity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class FairnessReport:
    """Fairness audit results across demographic groups."""

    overall_auc: float
    per_group_auc: dict[str, float]
    es_auc: float  # Equalized-subgroup AUC (minimum across groups)
    auc_gap: float  # Max AUC - Min AUC across groups
    group_sizes: dict[str, int]


class FairnessAuditor:
    """Audit model predictions for demographic disparities.

    Following Harvard-GF methodology: evaluate AUC per racial/ethnic group
    and flag significant performance gaps.
    """

    def __init__(
        self,
        group_names: list[str] | None = None,
        max_acceptable_gap: float = 0.05,
    ) -> None:
        """
        Args:
            group_names: Names for demographic groups (e.g., ["Asian", "Black", "White"]).
            max_acceptable_gap: Maximum acceptable AUC gap between groups.
        """
        self.group_names = group_names or ["Asian", "Black", "White"]
        self.max_acceptable_gap = max_acceptable_gap

    def audit(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        group_ids: np.ndarray,
    ) -> FairnessReport:
        """Run fairness audit on model predictions.

        Args:
            labels: Ground truth binary labels [N].
            scores: Model prediction scores [N].
            group_ids: Integer group IDs [N].

        Returns:
            FairnessReport with per-group AUC analysis.
        """
        overall_auc = float(roc_auc_score(labels, scores))

        per_group_auc = {}
        group_sizes = {}

        for gid, name in enumerate(self.group_names):
            mask = group_ids == gid
            group_sizes[name] = int(mask.sum())

            if group_sizes[name] < 10 or len(np.unique(labels[mask])) < 2:
                per_group_auc[name] = float("nan")
                continue

            per_group_auc[name] = float(roc_auc_score(labels[mask], scores[mask]))

        valid_aucs = [v for v in per_group_auc.values() if not np.isnan(v)]
        es_auc = min(valid_aucs) if valid_aucs else 0.0
        auc_gap = max(valid_aucs) - min(valid_aucs) if len(valid_aucs) >= 2 else 0.0

        report = FairnessReport(
            overall_auc=overall_auc,
            per_group_auc=per_group_auc,
            es_auc=es_auc,
            auc_gap=auc_gap,
            group_sizes=group_sizes,
        )

        if auc_gap > self.max_acceptable_gap:
            print(
                f"WARNING: AUC gap of {auc_gap:.3f} exceeds threshold "
                f"({self.max_acceptable_gap}). Consider applying FIN or "
                f"rebalancing training data."
            )

        return report
