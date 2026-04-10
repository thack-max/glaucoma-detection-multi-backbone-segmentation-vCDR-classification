"""
Equity Attention mechanism for fair glaucoma detection across demographics.

Reference: "Equity-enhanced glaucoma progression prediction from OCT with
knowledge distillation." (FairDist) npj Digital Medicine, 2025.
  - EfficientNet + fair attention mechanism (EqEffNet)
  - Generalized to progression prediction via knowledge distillation
  - MD Fast Progression AUC: 0.738-0.778, ES-AUC: 0.677-0.693
  - Best among all equity-mitigation methods

Also draws from:
  - Harvard Glaucoma Fairness (IEEE TMI, 2024) — FIN baseline
  - Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation."
    ICML 2015 — Gradient Reversal Layer for adversarial learning
  - Group DRO (Sagawa et al., ICLR 2020) — worst-group optimization

Replaces: Fair Identity Normalization (FIN).

Why Equity Attention over FIN:
  - FIN only normalizes per-group statistics (corrective, not generative)
  - Equity Attention learns to attend to features that are invariant across
    demographic groups, actively producing fairer representations
  - Adversarial training via Gradient Reversal forces the feature extractor
    to remove group-discriminative signal
  - Knowledge distillation transfers equity properties to smaller models
  - Validated specifically for glaucoma progression (FairDist paper)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversalFunction(torch.autograd.Function):
    """Identity on the forward pass, negates gradients on the backward pass.

    Introduced by Ganin & Lempitsky (2015). When placed between a feature
    extractor and a classifier, training that classifier to succeed trains
    the upstream features to FAIL at that task — i.e. to become invariant
    to the signal the classifier is trying to predict.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply a gradient reversal with strength `lambda_` to `x`."""
    return _GradientReversalFunction.apply(x, lambda_)


class EquityAttentionBlock(nn.Module):
    """Attention block that learns demographic-invariant feature weighting.

    The key insight: instead of normalizing features per group (FIN),
    learn which feature dimensions are most predictive ACROSS all groups.
    Features that are only predictive for one demographic group are
    automatically down-weighted, promoting equity.
    """

    def __init__(
        self,
        feature_dim: int,
        num_groups: int = 3,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups

        # Shared attention: learns group-invariant feature importance
        self.shared_attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.GELU(),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid(),
        )

        # Group-specific bias correction (small, learnable per-group offsets)
        self.group_bias = nn.Embedding(num_groups, feature_dim)
        nn.init.zeros_(self.group_bias.weight)

        # Adversarial regularizer: penalizes features that predict group membership
        self.group_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, num_groups),
        )

    def forward(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equity attention.

        Args:
            features: Input features [B, feature_dim].
            group_ids: Demographic group indices [B]. Optional at inference.

        Returns:
            Equity-weighted features [B, feature_dim].
        """
        # Shared attention weights (demographic-invariant)
        attn = self.shared_attn(features)
        weighted = features * attn

        # Apply group-specific bias correction if groups are known
        if group_ids is not None:
            bias = self.group_bias(group_ids)
            weighted = weighted + bias

        return weighted

    def get_group_prediction(
        self,
        features: torch.Tensor,
        reverse_gradient: bool = True,
        lambda_: float = 1.0,
    ) -> torch.Tensor:
        """Predict group membership from features (for adversarial training).

        When `reverse_gradient=True` (the default), gradients flowing from
        the group predictor back into `features` are negated via a Gradient
        Reversal Layer. Training the group predictor to succeed therefore
        pushes the feature extractor to produce demographic-invariant
        representations — exactly the FairDist objective.

        When `reverse_gradient=False`, the features are detached and only
        the group predictor itself is updated. This is useful if you want
        to alternate discriminator and generator steps explicitly.

        Args:
            features: [B, feature_dim].
            reverse_gradient: If True (default), insert a GRL so a single
                backward pass trains both the predictor and the feature
                extractor adversarially.
            lambda_: Strength of the reversed gradient.

        Returns:
            Group logits [B, num_groups].
        """
        if reverse_gradient:
            features = gradient_reversal(features, lambda_)
        else:
            features = features.detach()
        return self.group_predictor(features)


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for transferring equity properties.

    A large teacher model (trained with equity attention on diverse data)
    can distill its fair representations into a smaller student model
    for deployment in resource-constrained settings.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5) -> None:
        """
        Args:
            temperature: Softmax temperature for soft targets.
            alpha: Weight between hard label loss and distillation loss.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KD loss.

        Args:
            student_logits: Student model output [B, C].
            teacher_logits: Teacher model output [B, C].
            targets: Hard class labels [B].
        """
        T = self.temperature
        hard_loss = F.cross_entropy(student_logits, targets)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class EquityModule(nn.Module):
    """Complete equity module for fair glaucoma detection.

    Combines:
      1. Equity attention for demographic-invariant feature weighting
      2. Adversarial training to remove demographic information from features
      3. Optional knowledge distillation for model compression

    Drops in as a replacement for FairIdentityNormalization.
    """

    def __init__(
        self,
        feature_dim: int,
        num_groups: int = 3,
        adversarial_weight: float = 0.1,
    ) -> None:
        """
        Args:
            feature_dim: Feature dimension.
            num_groups: Number of demographic groups.
            adversarial_weight: Weight of adversarial (group prediction) loss.
                               Higher = stronger demographic invariance.
        """
        super().__init__()
        self.equity_attn = EquityAttentionBlock(feature_dim, num_groups)
        self.adversarial_weight = adversarial_weight

    def forward(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equity attention to features.

        Args:
            features: [B, feature_dim].
            group_ids: [B] demographic group indices (optional at inference).

        Returns:
            Fair features [B, feature_dim].
        """
        return self.equity_attn(features, group_ids)

    def adversarial_loss(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor,
        reverse_gradient: bool = True,
    ) -> torch.Tensor:
        """Compute adversarial loss for fairness regularization.

        By default this wires the features through a Gradient Reversal Layer
        so that adding `adversarial_loss(...)` to the main classification
        loss and calling `.backward()` once trains:

          - the group predictor to MINIMIZE cross-entropy on group labels
          - the feature extractor to MAXIMIZE it (because gradients are
            reversed before reaching the features)

        Setting `reverse_gradient=False` recovers the old behavior of
        training the predictor in isolation (useful for a warm-up phase).

        Args:
            features: [B, feature_dim] (after equity attention).
            group_ids: [B] true group labels.
            reverse_gradient: Whether to insert the GRL (default True).

        Returns:
            Weighted adversarial cross-entropy loss.
        """
        group_logits = self.equity_attn.get_group_prediction(
            features, reverse_gradient=reverse_gradient, lambda_=1.0,
        )
        return self.adversarial_weight * F.cross_entropy(group_logits, group_ids)

    def equity_gap(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> dict[str, float]:
        """Compute equity metrics across demographic groups.

        Args:
            predictions: Model predictions [B].
            targets: True labels [B].
            group_ids: Demographic groups [B].

        Returns:
            Dict with per-group AUC and equity gap.
        """
        results = {}
        groups = group_ids.unique().tolist()

        for g in groups:
            mask = group_ids == g
            if mask.sum() < 2:
                continue
            g_pred = predictions[mask]
            g_targ = targets[mask]

            # Simple accuracy as proxy (full AUC requires sklearn)
            correct = ((g_pred > 0.5).float() == g_targ.float()).float().mean()
            results[f"group_{g}_acc"] = correct.item()

        accs = [v for k, v in results.items() if "acc" in k]
        if len(accs) >= 2:
            results["equity_gap"] = max(accs) - min(accs)
            results["min_group_acc"] = min(accs)

        return results
