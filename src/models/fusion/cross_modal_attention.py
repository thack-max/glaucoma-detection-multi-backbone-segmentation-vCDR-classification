"""
Cross-Modal Attention fusion with MC Dropout uncertainty estimation.

Reference: "Multimodal Fusion Learning with Dual Attention for Medical Imaging."
(DRIFA-Net) WACV 2025. arXiv:2412.01248.
  - Multi-branch fusion attention for enhanced per-modality representations
  - Multimodal information fusion attention for cross-modal shared features
  - Ensemble Monte Carlo dropout for uncertainty quantification
  - Backbone-agnostic — can wrap any feature extractor

Replaces: Simple concatenation of multimodal features in the original pipeline.

In the original architecture, fundus features, OCT features, and clinical data
were concatenated and fed to the BiLSTM. This loses inter-modal relationships
and provides no uncertainty estimate.

Cross-modal attention learns:
  1. Which modalities are most informative for each patient
  2. How to align and combine features across modalities
  3. Uncertainty estimates via MC dropout at inference time
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttention(nn.Module):
    """Per-modality self-attention for feature enhancement.

    Enhances each modality's representation by attending to its own features,
    allowing the model to focus on the most discriminative aspects of each
    data type before cross-modal fusion.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention on a single modality.

        Args:
            x: [B, 1, dim] (single modality feature, unsqueezed for attention).

        Returns:
            Enhanced features [B, 1, dim].
        """
        attended, _ = self.attn(x, x, x)
        x = self.norm(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossAttention(nn.Module):
    """Cross-modal attention for learning inter-modality relationships.

    Each modality attends to all other modalities, learning which cross-modal
    features are most relevant for the current prediction.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention: query attends to context modalities.

        Args:
            query: [B, 1, dim] — modality to enhance.
            context: [B, M, dim] — all modalities to attend to.

        Returns:
            Cross-attended features [B, 1, dim].
        """
        attended, _ = self.cross_attn(query, context, context)
        query = self.norm(query + attended)
        query = self.norm2(query + self.ffn(query))
        return query


class ModalityGating(nn.Module):
    """Dynamic modality gating for uncertainty-aware weighting.

    Learns to automatically down-weight unreliable modalities.
    For example, if a patient's OCT scan is low quality, the gate
    will reduce OCT contribution and rely more on fundus features.
    """

    def __init__(self, dim: int, num_modalities: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """Compute gating weights.

        Args:
            modality_features: List of [B, dim] tensors, one per modality.

        Returns:
            Gating weights [B, num_modalities] summing to 1.
        """
        concat = torch.cat(modality_features, dim=-1)
        return self.gate(concat)


class CrossModalFusion(nn.Module):
    """Cross-Modal Attention Fusion with MC Dropout uncertainty.

    Fuses multiple modalities (fundus, OCT, VF, clinical) through:
    1. Per-modality self-attention (enhance individual representations)
    2. Cross-modal attention (learn inter-modality relationships)
    3. Dynamic gating (weight modalities by reliability)
    4. MC Dropout (epistemic uncertainty at inference)

    The output is a single fused representation ready for the forecaster.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        fused_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.2,
        mc_samples: int = 10,
    ) -> None:
        """
        Args:
            modality_dims: Dict mapping modality name to feature dim.
                           e.g. {"fundus": 768, "oct": 768, "vf": 256, "clinical": 32}
            fused_dim: Output dimension of fused representation.
            num_heads: Number of attention heads.
            dropout: Dropout rate (also used for MC dropout at inference).
            mc_samples: Number of MC dropout forward passes for uncertainty.
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(modality_dims)
        self.fused_dim = fused_dim
        self.mc_samples = mc_samples

        # Project each modality to common dimension
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
            )
            for name, dim in modality_dims.items()
        })

        # Per-modality self-attention
        self.self_attns = nn.ModuleDict({
            name: ModalityAttention(fused_dim, num_heads, dropout)
            for name in modality_dims
        })

        # Cross-modal attention
        self.cross_attns = nn.ModuleDict({
            name: CrossAttention(fused_dim, num_heads, dropout)
            for name in modality_dims
        })

        # Dynamic gating
        self.gating = ModalityGating(fused_dim, self.num_modalities)

        # MC Dropout (kept active during inference for uncertainty)
        self.mc_dropout = nn.Dropout(dropout)

        # Final fusion
        self.output = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _fuse_once(
        self,
        modality_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Single forward pass fusion.

        Args:
            modality_features: Dict of {name: [B, dim]} feature tensors.
                              Missing modalities are padded with zero vectors
                              so that the gating network always sees the same
                              number of inputs it was initialized with.

        Returns:
            Fused representation [B, fused_dim].
        """
        if not modality_features:
            raise ValueError("At least one modality must be provided")

        # Determine batch size from any present modality
        batch_size = next(iter(modality_features.values())).shape[0]
        device = next(iter(modality_features.values())).device

        # Project every configured modality. Missing ones become zero vectors
        # in the fused space so that downstream layers keep a fixed shape.
        projected: dict[str, torch.Tensor] = {}
        present_mask: list[bool] = []
        for name in self.modality_names:
            if name in modality_features:
                projected[name] = self.projections[name](modality_features[name])
                present_mask.append(True)
            else:
                projected[name] = torch.zeros(
                    batch_size, self.fused_dim, device=device,
                )
                present_mask.append(False)

        # Self-attention per modality (applied uniformly, including zero slots
        # so that the computation graph matches self.modality_names exactly).
        enhanced: dict[str, torch.Tensor] = {}
        for name in self.modality_names:
            x = projected[name].unsqueeze(1)  # [B, 1, dim]
            enhanced[name] = self.self_attns[name](x).squeeze(1)

        # Cross-modal attention against the full context
        context = torch.stack(
            [enhanced[n] for n in self.modality_names], dim=1,
        )  # [B, M, dim]

        cross_attended: list[torch.Tensor] = []
        for name in self.modality_names:
            query = enhanced[name].unsqueeze(1)
            attended = self.cross_attns[name](query, context).squeeze(1)
            cross_attended.append(attended)

        # Dynamic gating — zero-masked modalities get their gate weight zeroed
        # out after the softmax, then we renormalize so the remaining weights
        # still sum to 1.
        gates = self.gating(cross_attended)  # [B, M]
        mask = torch.tensor(present_mask, device=gates.device, dtype=gates.dtype)
        mask = mask.unsqueeze(0).expand_as(gates)  # [B, M]
        masked_gates = gates * mask
        norm = masked_gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        gates = masked_gates / norm

        stacked = torch.stack(cross_attended, dim=1)  # [B, M, dim]
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # [B, dim]

        # MC Dropout + output
        fused = self.mc_dropout(fused)
        return self.output(fused)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        return_uncertainty: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Fuse multimodal features with optional uncertainty estimation.

        Args:
            modality_features: Dict of {modality_name: [B, dim]} tensors.
            return_uncertainty: If True, run MC dropout and return mean + std.

        Returns:
            If return_uncertainty=False: Fused features [B, fused_dim].
            If return_uncertainty=True: Dict with 'mean', 'std', 'samples'.
        """
        if not return_uncertainty:
            return self._fuse_once(modality_features)

        # MC Dropout inference: run multiple forward passes with dropout active
        self.mc_dropout.train()  # Keep dropout active
        samples = []
        for _ in range(self.mc_samples):
            sample = self._fuse_once(modality_features)
            samples.append(sample)

        samples = torch.stack(samples)  # [mc_samples, B, fused_dim]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        return {
            "mean": mean,
            "std": std,
            "samples": samples,
        }
