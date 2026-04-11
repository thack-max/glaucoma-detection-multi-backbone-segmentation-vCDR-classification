"""
Delta-t Mamba: Time-aware state-space model for longitudinal progression modeling.

Reference: "Delta-t Mamba3D: A Time-Aware Spatio-Temporal State-Space Model
for Breast Cancer Risk Prediction." arXiv:2510.19003 (Oct 2025).

Adapted for glaucoma progression forecasting:
  - The original Delta-t Mamba3D was designed for sequential screening
    mammograms with irregular inter-exam intervals.
  - We adapt the continuous-time selective scan mechanism for longitudinal
    ophthalmic imaging (OCT RNFL scans over 2+ years).
  - Key insight: real clinical visits are NOT uniformly spaced. Patients
    may have visits weeks apart (post-surgery) or years apart (stable).
    BiLSTM treats all timesteps equally — Delta-t Mamba encodes the actual
    time difference into state transitions.

Architecture:
  - Selective State Space Model (S6) with learned state matrices A, B, C
  - Delta-t modulation: inter-visit time difference directly modulates the
    discretization step size, affecting how much state decays between visits
  - Linear O(n) complexity vs O(n^2) for transformers
  - Multi-scale feature fusion for capturing both local and global patterns

Replaces: ConvNeXt-V2 temporal encoder + BiLSTM forecaster
Advantages:
  - Native irregular interval handling (no padding/interpolation needed)
  - Linear complexity (processes longer patient histories efficiently)
  - 2-5% C-index improvement over RNN/transformer baselines (original paper)
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaTDiscretization(nn.Module):
    """Time-aware discretization for the state-space model.

    The core innovation: instead of a fixed discretization step,
    the step size is modulated by the actual time delta between visits.
    Longer gaps → more state decay → model "forgets" more between visits.
    Shorter gaps → less decay → visits are treated as near-continuous.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Learnable base discretization parameters
        self.log_dt_proj = nn.Linear(d_model, d_model)
        # Time delta embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute time-modulated discretization step.

        Args:
            x: Input features [B, T, D].
            delta_t: Time differences between consecutive visits [B, T, 1]
                     in days/months/years (normalized).

        Returns:
            Discretization steps [B, T, D] modulated by time gaps.
        """
        base_dt = F.softplus(self.log_dt_proj(x))
        time_mod = torch.sigmoid(self.time_embed(delta_t))
        return base_dt * time_mod


class SelectiveSSMBlock(nn.Module):
    """Selective State Space Model block with delta-t modulation.

    Implements the S6 (selective structured state space) mechanism where
    the state transition matrices are input-dependent, allowing the model
    to selectively remember or forget information based on content AND timing.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # SSM parameters (input-dependent)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # State transition matrix A (learned, kept in log space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.log_A = nn.Parameter(torch.log(A))

        # Time-aware discretization
        self.dt_module = DeltaTDiscretization(d_inner)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with selective scan.

        Args:
            x: Input sequence [B, T, d_model].
            delta_t: Time deltas between visits [B, T, 1].
                     If None, assumes uniform spacing (delta_t=1).

        Returns:
            Output sequence [B, T, d_model].
        """
        B, T, D = x.shape
        residual = x

        # Input projection + gate
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        z = F.silu(z)

        # Compute input-dependent SSM parameters
        A = -torch.exp(self.log_A)  # [d_inner, d_state]
        B_t = self.B_proj(x_inner)  # [B, T, d_state]
        C_t = self.C_proj(x_inner)  # [B, T, d_state]

        # Time-aware discretization
        if delta_t is None:
            delta_t = torch.ones(B, T, 1, device=x.device)
        dt = self.dt_module(x_inner, delta_t)  # [B, T, d_inner]

        # Discretize: convert continuous A to discrete A_bar, B_bar
        # A_bar = exp(A * dt)
        # B_bar = (A_bar - I) * A^{-1} * B ≈ dt * B (for small dt)
        d_inner = x_inner.shape[-1]
        d_state = self.d_state

        # Selective scan (sequential for correctness; parallel scan possible)
        y = torch.zeros_like(x_inner)
        h = torch.zeros(B, d_inner, d_state, device=x.device)

        for t in range(T):
            dt_t = dt[:, t].unsqueeze(-1)  # [B, d_inner, 1]
            B_t_t = B_t[:, t].unsqueeze(1)  # [B, 1, d_state]
            C_t_t = C_t[:, t].unsqueeze(1)  # [B, 1, d_state]
            x_t = x_inner[:, t].unsqueeze(-1)  # [B, d_inner, 1]

            # State update with time-modulated decay
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)  # [B, d_inner, d_state]
            B_bar = dt_t * B_t_t  # [B, d_inner, d_state]

            h = A_bar * h + B_bar * x_t  # [B, d_inner, d_state]
            y_t = (h * C_t_t).sum(-1)  # [B, d_inner]
            y[:, t] = y_t

        # Gate and project
        y = y * z
        y = self.out_proj(y)
        y = self.dropout(y)

        return self.norm(y + residual)


class DeltaTMamba(nn.Module):
    """Delta-t Mamba temporal encoder for longitudinal glaucoma imaging.

    Processes a sequence of per-visit feature vectors with their associated
    timestamps, producing a context-aware representation that accounts for
    irregular visit spacing.

    Replaces ConvNeXt-V2 TemporalEncoder + BiLSTM ProgressionForecaster
    with a single unified temporal model.
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        d_state: int = 16,
        num_layers: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            input_dim: Dimension of per-visit feature vectors (from DINOv2).
            d_model: Internal model dimension.
            d_state: SSM state dimension (controls memory capacity).
            num_layers: Number of stacked Mamba blocks.
            expand: Inner dimension expansion factor.
            dropout: Dropout rate.
        """
        super().__init__()

        # Project input features to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Stack of selective SSM blocks
        self.layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state, expand, dropout)
            for _ in range(num_layers)
        ])

        # Bidirectional: run forward and backward, concatenate
        self.reverse_layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state, expand, dropout)
            for _ in range(num_layers)
        ])

        self.output_dim = d_model * 2  # bidirectional

        # Final projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, d_model),
            nn.GELU(),
        )

    @staticmethod
    def _compute_delta_t(timestamps: torch.Tensor) -> torch.Tensor:
        """Compute per-visit delta_t normalized to [0, 1] for an SSM pass.

        Defines the "time since the previous visit" signal consumed by the
        selective scan. By convention the first position has no prior visit,
        so ``delta_t[:, 0] = 0``.

        Args:
            timestamps: [B, T] visit timestamps in consistent units
                (days/months). Ordering does not have to be ascending — we
                take ``abs`` so reverse-ordered timestamps (used for the
                backward pass) produce the same positive gaps.

        Returns:
            delta_t: [B, T, 1] with ``delta_t[:, 0] == 0`` and values
            normalized by the per-sequence max gap.
        """
        dt = torch.zeros_like(timestamps)
        dt[:, 1:] = (timestamps[:, 1:] - timestamps[:, :-1]).abs()
        dt_max = dt.max(dim=1, keepdim=True).values.clamp(min=1.0)
        return (dt / dt_max).unsqueeze(-1)

    def forward(
        self,
        visit_features: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a longitudinal visit sequence.

        Args:
            visit_features: [B, T, input_dim] — per-visit embeddings.
            timestamps: [B, T] — visit timestamps (e.g., days since baseline).
                        Used to compute inter-visit delta_t.
                        If None, assumes uniform spacing.

        Returns:
            Sequence representation [B, d_model] — pooled temporal embedding.
        """
        h_bidir = self.forward_sequence(visit_features, timestamps)
        d_model = h_bidir.shape[-1] // 2

        # Pool from the MOST INFORMED position in each direction:
        #   - Forward: the last index has consumed every prior visit.
        #   - Backward: after the final flip-back, the FIRST index is the
        #     backward SSM state *after* traversing every visit in reverse.
        # Concatenating h_bidir[:, -1] (the legacy behaviour) mixed the fully
        # informed forward state with the LEAST informed backward state
        # (which had only processed the last visit), effectively throwing
        # away the backward pass for progression forecasting.
        h_fwd_last = h_bidir[:, -1, :d_model]
        h_bwd_first = h_bidir[:, 0, d_model:]
        pooled_raw = torch.cat([h_fwd_last, h_bwd_first], dim=-1)
        return self.output_proj(pooled_raw)

    def forward_sequence(
        self,
        visit_features: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return full sequence of temporal embeddings (not pooled).

        Useful for per-visit predictions or attention visualization.

        Args:
            visit_features: [B, T, input_dim].
            timestamps: [B, T].

        Returns:
            Temporal embeddings [B, T, d_model*2].
        """
        # Forward delta_t directly from the original timestamps.
        delta_t_fwd = None
        delta_t_bwd = None
        if timestamps is not None:
            delta_t_fwd = self._compute_delta_t(timestamps)
            # Backward delta_t: compute from the REVERSED timestamps so the
            # first position of the reversed sequence has delta_t = 0.
            # Flipping ``delta_t_fwd`` would incorrectly place
            # ``delta_t_fwd[T-1]`` at position 0 and drop the gap between
            # visit 0 and visit 1 entirely.
            delta_t_bwd = self._compute_delta_t(timestamps.flip(1))

        x = self.input_proj(visit_features)

        # Forward pass over the original sequence order.
        h_fwd = x
        for layer in self.layers:
            h_fwd = layer(h_fwd, delta_t_fwd)

        # Backward pass over the reversed sequence.
        x_rev = x.flip(1)
        h_bwd = x_rev
        for layer in self.reverse_layers:
            h_bwd = layer(h_bwd, delta_t_bwd)
        h_bwd = h_bwd.flip(1)  # Realign with original visit order.

        return torch.cat([h_fwd, h_bwd], dim=-1)
