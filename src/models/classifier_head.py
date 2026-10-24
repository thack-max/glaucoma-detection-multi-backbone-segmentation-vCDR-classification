"""
Tier 1 screening classifier head (DINOv2 embedding -> glaucoma logit).

Training (scripts/train_screening_classifier.py, scripts/train_fairness.py)
and inference (scripts/run_screening.py, scripts/evaluate.py) must share ONE
definition: a state_dict saved from one module layout does not load into
another (e.g. ``net.0.weight`` vs ``0.weight``), and a non-strict load hides
that by silently keeping random weights.

train_fairness.py trains the head on EquityModule outputs, not on raw DINOv2
features, and saves both (``"equity"`` + ``"classifier"``). Such a checkpoint
must be loaded with `load_screening_head`, which rebuilds the EquityModule so
it can sit between the encoder and the head; `load_classifier_head` refuses it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.fairness.equity_attention import EquityModule
from src.utils.checkpoints import load_checkpoint, load_into, random_encoder_warning

EQUITY_KEY = "equity"


class ClassifierHead(nn.Module):
    """LayerNorm + dropout + linear on a frozen DINOv2 embedding -> one logit."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_head(
    head: ClassifierHead, ckpt: dict[str, Any], checkpoint: str | Path,
) -> None:
    if not load_into(head, ckpt, key_candidates=["classifier"], strict=True):
        raise RuntimeError(
            f"No classifier weights in {checkpoint} "
            f"(state dicts found: {sorted(ckpt['state_dicts'])})"
        )
    message = random_encoder_warning(ckpt, checkpoint)
    if message:
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _build_equity(
    state_dict: dict[str, torch.Tensor],
    input_dim: int,
    num_groups: int | None,
    checkpoint: str | Path,
) -> EquityModule:
    if num_groups is None:
        # One learned bias row per demographic group.
        bias = state_dict.get("equity_attn.group_bias.weight")
        if bias is None:
            raise RuntimeError(
                f"EquityModule: {checkpoint} has no equity_attn.group_bias.weight "
                f"(keys: {sorted(state_dict)[:5]}...)"
            )
        num_groups = int(bias.shape[0])
    equity = EquityModule(feature_dim=input_dim, num_groups=num_groups)
    try:
        equity.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"EquityModule: cannot load {checkpoint}: {exc}") from exc
    return equity


def load_classifier_head(
    checkpoint: str | Path | None,
    input_dim: int = 768,
    dropout: float = 0.3,
) -> ClassifierHead:
    """Build a ClassifierHead and, if `checkpoint` is set, load it strictly.

    Raises:
        FileNotFoundError: `checkpoint` is set but does not exist.
        RuntimeError: the checkpoint's weights do not fit ClassifierHead, or
            it also holds EquityModule weights (use `load_screening_head`).
    """
    head = ClassifierHead(input_dim=input_dim, dropout=dropout)
    if checkpoint:
        ckpt = load_checkpoint(checkpoint)
        if EQUITY_KEY in ckpt["state_dicts"]:
            raise RuntimeError(
                f"{checkpoint} also holds EquityModule weights (train_fairness.py): "
                f"its head was trained on EquityModule outputs and cannot score raw "
                f"features on its own. Load it with load_screening_head()."
            )
        _load_head(head, ckpt, checkpoint)
    return head


def load_screening_head(
    checkpoint: str | Path | None,
    input_dim: int = 768,
    dropout: float = 0.3,
    with_equity: bool = True,
    num_groups: int | None = None,
) -> tuple[ClassifierHead, EquityModule | None]:
    """Build the screening head plus, for a train_fairness.py checkpoint, its EquityModule.

    Both load strictly from the same file. The EquityModule is returned only
    when the checkpoint holds ``"equity"`` weights; pass it to
    ScreeningPipeline(equity_module=...) so the head sees the features it was
    trained on.

    Args:
        checkpoint: .pth from train_screening_classifier.py or train_fairness.py.
        input_dim: Embedding width (DINOv2 embed_dim).
        dropout: ClassifierHead dropout (inactive in eval mode).
        with_equity: False when the caller has fairness disabled; a checkpoint
            that needs its EquityModule then raises instead of being misused.
        num_groups: Expected EquityModule group count (None: read it from the
            checkpoint).

    Raises:
        FileNotFoundError: `checkpoint` is set but does not exist.
        RuntimeError: weights do not fit, or an equity checkpoint is loaded
            with `with_equity=False`.
    """
    head = ClassifierHead(input_dim=input_dim, dropout=dropout)
    if not checkpoint:
        return head, None
    ckpt = load_checkpoint(checkpoint)
    _load_head(head, ckpt, checkpoint)
    equity_sd = ckpt["state_dicts"].get(EQUITY_KEY)
    if equity_sd is None:
        return head, None
    if not with_equity:
        raise RuntimeError(
            f"{checkpoint} is a fairness checkpoint (train_fairness.py): its head "
            f"was trained on EquityModule outputs, but fairness is disabled. "
            f"Enable fairness (fairness.enabled: true) or use a "
            f"train_screening_classifier.py checkpoint."
        )
    return head, _build_equity(equity_sd, input_dim, num_groups, checkpoint)
