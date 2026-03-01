"""
Evidential Deep Learning forecaster for glaucoma progression with uncertainty.

Reference: "Glaucoma progression prediction using multi-modal data and trusted
multi-view learning." Expert Systems with Applications, 2025.
  - Uses Evidential Deep Learning (EDL) theory to model prediction uncertainty
  - Estimates belief masses and uncertainty per modality
  - Dynamic fusion adjusts modal contributions based on data quality

Also draws from:
  - Amini et al., "Deep Evidential Regression." NeurIPS 2020 — foundational
    EDL framework for regression with uncertainty.
  - Sensoy et al., "Evidential Deep Learning to Quantify Classification
    Uncertainty." NeurIPS 2018 — EDL for classification.

Why evidential learning for glaucoma progression:
  - Clinical decisions require knowing BOTH the prediction AND the confidence.
    "70% chance of progression" is useless without knowing if the model is
    confident (trained on similar patients) or uncertain (novel presentation).
  - EDL provides principled uncertainty decomposition:
    * Aleatoric uncertainty: inherent noise in the data (e.g., VF test variability)
    * Epistemic uncertainty: model ignorance (e.g., rare demographics, edge cases)
  - Separating these lets clinicians know when to trust vs. override the model.

Replaces: ProgressionForecaster (BiLSTM with binary output, no uncertainty).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialClassificationHead(nn.Module):
    """Evidential classification head using Dirichlet distribution.

    Instead of outputting class probabilities, outputs Dirichlet concentration
    parameters (evidence) from which we derive:
      - Class probabilities (expected value of Dirichlet)
      - Epistemic uncertainty (inverse of total evidence)
      - Per-class uncertainty (variance of Dirichlet)
    """

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute evidential prediction.

        Args:
            x: Input features [B, input_dim].

        Returns:
            Dict containing:
              - 'logits': Raw evidence (before softplus) [B, num_classes]
              - 'evidence': Non-negative evidence [B, num_classes]
              - 'alpha': Dirichlet concentration params [B, num_classes]
              - 'prob': Expected class probabilities [B, num_classes]
              - 'uncertainty': Epistemic uncertainty [B] (lower = more confident)
        """
        logits = self.evidence_layer(x)
        evidence = F.softplus(logits)  # Ensure non-negative
        alpha = evidence + 1.0  # Dirichlet params (alpha_k = e_k + 1)
        S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength

        prob = alpha / S  # Expected class probability
        uncertainty = self.num_classes / S.squeeze(-1)  # Epistemic uncertainty

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "uncertainty": uncertainty,
        }


class EvidentialRegressionHead(nn.Module):
    """Evidential regression head using Normal-Inverse-Gamma distribution.

    For predicting continuous values (e.g., future RNFL thickness) with
    separate aleatoric and epistemic uncertainty estimates.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Outputs 4 parameters: mu (mean), v (evidence), alpha, beta
        self.head = nn.Linear(input_dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute evidential regression prediction.

        Args:
            x: Input features [B, input_dim].

        Returns:
            Dict containing:
              - 'mu': Predicted mean [B]
              - 'aleatoric': Aleatoric uncertainty [B] (data noise)
              - 'epistemic': Epistemic uncertainty [B] (model ignorance)
        """
        out = self.head(x)
        mu = out[:, 0]
        v = F.softplus(out[:, 1]) + 1e-6  # Evidence (positive)
        alpha = F.softplus(out[:, 2]) + 1.0  # Shape (> 1)
        beta = F.softplus(out[:, 3]) + 1e-6  # Rate (positive)

        # Aleatoric uncertainty: E[sigma^2] = beta / (alpha - 1)
        aleatoric = beta / (alpha - 1.0 + 1e-6)
        # Epistemic uncertainty: Var[mu] = beta / (v * (alpha - 1))
        epistemic = beta / (v * (alpha - 1.0 + 1e-6))

        return {
            "mu": mu,
            "v": v,
            "alpha": alpha,
            "beta": beta,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
        }


class EvidentialForecaster(nn.Module):
    """Glaucoma progression forecaster with evidential uncertainty estimation.

    Combines temporal features (from Delta-t Mamba) with optional clinical
    and VF graph features to predict:
      1. Binary progression classification with calibrated confidence
      2. Continuous RNFL thickness prediction with uncertainty bounds
    """

    def __init__(
        self,
        temporal_dim: int = 512,
        clinical_dim: int = 0,
        vf_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        """
        Args:
            temporal_dim: Dimension of temporal embeddings (from DeltaTMamba).
            clinical_dim: Dimension of clinical features (IOP, age, medications).
                          Set to 0 if not using clinical data.
            vf_dim: Dimension of VF graph embeddings (from VFGraphNet).
                    Set to 0 if not using VF data.
            hidden_dim: Internal hidden dimension.
            dropout: Dropout rate.
            num_classes: Number of progression classes.
        """
        super().__init__()

        total_input = temporal_dim + clinical_dim + vf_dim

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head (progression/stable)
        self.classification_head = EvidentialClassificationHead(hidden_dim, num_classes)

        # Regression head (future RNFL thickness)
        self.regression_head = EvidentialRegressionHead(hidden_dim)

    def forward(
        self,
        temporal_features: torch.Tensor,
        clinical_features: torch.Tensor | None = None,
        vf_features: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Predict progression with uncertainty.

        Args:
            temporal_features: [B, temporal_dim] from DeltaTMamba.
            clinical_features: [B, clinical_dim] optional clinical data.
            vf_features: [B, vf_dim] optional VF graph embeddings.

        Returns:
            Dict with:
              - 'classification': evidential classification output
              - 'regression': evidential regression output (RNFL prediction)
        """
        parts = [temporal_features]
        if clinical_features is not None:
            parts.append(clinical_features)
        if vf_features is not None:
            parts.append(vf_features)

        x = torch.cat(parts, dim=-1)
        h = self.shared(x)

        return {
            "classification": self.classification_head(h),
            "regression": self.regression_head(h),
        }


class EvidentialLoss(nn.Module):
    """Combined evidential loss for classification + regression.

    Classification: Type II maximum likelihood loss for Dirichlet distribution
    with KL divergence regularization to penalize excessive evidence on
    incorrect classes.

    Regression: Normal-Inverse-Gamma NLL loss from Deep Evidential Regression.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 0.5,
        kl_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.kl_weight = kl_weight

    def classification_loss(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Dirichlet-based classification loss.

        Args:
            alpha: Dirichlet concentration params [B, K].
            targets: Class indices [B].
        """
        K = alpha.shape[-1]
        S = alpha.sum(dim=-1, keepdim=True)
        targets_onehot = F.one_hot(targets, K).float()

        # Type II ML loss
        loss = (targets_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)

        # KL divergence regularization: penalize evidence on wrong classes
        alpha_tilde = targets_onehot + (1 - targets_onehot) * (alpha - 1) + 1
        S_tilde = alpha_tilde.sum(-1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde.squeeze(-1)) - torch.lgamma(torch.tensor(K, dtype=torch.float32, device=alpha.device))
            - torch.lgamma(alpha_tilde).sum(-1)
            + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(-1)
        )

        return (loss + self.kl_weight * kl).mean()

    def regression_loss(
        self,
        mu: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Normal-Inverse-Gamma NLL loss.

        Args:
            mu, v, alpha, beta: NIG parameters.
            targets: Ground truth values [B].
        """
        omega = 2 * beta * (1 + v)

        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=mu.device) / v)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        return nll.mean()

    def forward(
        self,
        model_output: dict[str, dict[str, torch.Tensor]],
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined evidential loss.

        Args:
            model_output: Output from EvidentialForecaster.forward().
            cls_targets: Classification targets [B].
            reg_targets: Regression targets [B] (optional).
        """
        cls_out = model_output["classification"]
        loss = self.cls_weight * self.classification_loss(cls_out["alpha"], cls_targets)

        if reg_targets is not None:
            reg_out = model_output["regression"]
            loss += self.reg_weight * self.regression_loss(
                reg_out["mu"], reg_out["v"], reg_out["alpha"], reg_out["beta"],
                reg_targets,
            )

        return loss
