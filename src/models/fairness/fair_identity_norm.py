"""
Fair Identity Normalization (FIN) for equitable glaucoma detection.

Reference: Harvard Glaucoma Fairness (Harvard-GF)
  - Luo et al., IEEE Transactions on Medical Imaging, 2024
  - PMC11251413 / GitHub: Harvard-Ophthalmology-AI-Lab/Harvard-GF
  - Dataset: 3,300 subjects, balanced across Asian/Black/White groups
  - FIN improved AUC for Black patients: 0.77 → 0.82

Also see: FairDist (npj Digital Medicine, 2025) for equity-enhanced
progression prediction using knowledge distillation.

NOTE: The document referenced a model called "Fair-Eye Net" with claims
of 73.4% racial disparity reduction — this model does not exist in the
literature. The actual verified approach is FIN from Harvard-GF.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FairIdentityNormalization(nn.Module):
    """Fair Identity Normalization module.

    Equalizes feature importance across demographic identity groups to
    reduce performance disparities in glaucoma detection.

    Applied as a normalization layer within the feature extraction pipeline.
    During training, features are normalized per-group to prevent the model
    from learning group-specific shortcuts that harm minority populations.
    """

    def __init__(
        self,
        feature_dim: int,
        num_groups: int = 3,  # e.g., Asian, Black, White
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum

        # Per-group running statistics
        self.register_buffer("running_mean", torch.zeros(num_groups, feature_dim))
        self.register_buffer("running_var", torch.ones(num_groups, feature_dim))

        # Learnable affine parameters (shared across groups)
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply fair identity normalization.

        Args:
            features: Input features [B, feature_dim].
            group_ids: Integer group labels [B] (0, 1, ..., num_groups-1).

        Returns:
            Normalized features [B, feature_dim].
        """
        output = torch.zeros_like(features)

        for g in range(self.num_groups):
            mask = group_ids == g
            if not mask.any():
                continue

            group_features = features[mask]

            if self.training:
                mean = group_features.mean(dim=0)
                var = group_features.var(dim=0, unbiased=False)

                # Update running stats
                self.running_mean[g] = (
                    (1 - self.momentum) * self.running_mean[g] + self.momentum * mean
                )
                self.running_var[g] = (
                    (1 - self.momentum) * self.running_var[g] + self.momentum * var
                )
            else:
                mean = self.running_mean[g]
                var = self.running_var[g]

            normalized = (group_features - mean) / (var + self.eps).sqrt()
            output[mask] = normalized * self.weight + self.bias

        return output
