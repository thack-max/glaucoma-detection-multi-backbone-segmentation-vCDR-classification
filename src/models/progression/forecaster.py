"""
Glaucoma progression forecaster using BiLSTM over temporal visit features.

Reference: Mass Eye & Ear longitudinal study (medRxiv, Nov 2025)
  - 2-year observation window → 2-4 year progression forecast
  - ConvNeXt-V2 + BiLSTM achieved AUC 0.97
  - Saliency maps localized to clinically relevant arcuate bundles

The forecaster consumes a sequence of per-visit feature vectors (from
TemporalEncoder) and optional clinical covariates, outputting:
  1. Binary progression probability
  2. Predicted future RNFL thickness trajectory (optional regression head)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProgressionForecaster(nn.Module):
    """BiLSTM-based glaucoma progression forecaster."""

    def __init__(
        self,
        input_dim: int = 1024,
        clinical_dim: int = 0,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        """
        Args:
            input_dim: Dimension of per-visit feature vectors (from TemporalEncoder).
            clinical_dim: Dimension of clinical covariates (IOP, age, etc.).
                          Set to 0 to use imaging features only.
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
            num_classes: 2 for binary progression/stable classification.
        """
        super().__init__()

        total_input_dim = input_dim + clinical_dim

        self.lstm = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM doubles the hidden dim
        lstm_output_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        visit_features: torch.Tensor,
        clinical_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict progression from longitudinal visit features.

        Args:
            visit_features: [B, T, input_dim] — sequential visit embeddings.
            clinical_features: [B, T, clinical_dim] — optional clinical data per visit.

        Returns:
            Progression logits [B, num_classes].
        """
        if clinical_features is not None:
            x = torch.cat([visit_features, clinical_features], dim=-1)
        else:
            x = visit_features

        lstm_out, (h_n, _) = self.lstm(x)

        # Use final hidden states from both directions
        # h_n shape: [num_layers * 2, B, hidden_dim]
        forward_final = h_n[-2]   # Last layer, forward
        backward_final = h_n[-1]  # Last layer, backward
        combined = torch.cat([forward_final, backward_final], dim=-1)

        return self.classifier(combined)
