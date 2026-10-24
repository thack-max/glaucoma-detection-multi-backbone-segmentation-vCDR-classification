"""
Tier 1 screening classifier head (DINOv2 embedding -> glaucoma logit).

Training (scripts/train_screening_classifier.py, scripts/train_fairness.py)
and inference (scripts/run_screening.py, scripts/evaluate.py) must share ONE
definition: a state_dict saved from one module layout does not load into
another (e.g. ``net.0.weight`` vs ``0.weight``), and a non-strict load hides
that by silently keeping random weights.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.utils.checkpoints import load_checkpoint, load_into


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


def load_classifier_head(
    checkpoint: str | Path | None,
    input_dim: int = 768,
    dropout: float = 0.3,
) -> ClassifierHead:
    """Build a ClassifierHead and, if `checkpoint` is set, load it strictly.

    Raises:
        FileNotFoundError: `checkpoint` is set but does not exist.
        RuntimeError: the checkpoint's weights do not fit ClassifierHead.
    """
    head = ClassifierHead(input_dim=input_dim, dropout=dropout)
    if checkpoint:
        ckpt = load_checkpoint(checkpoint)
        if not load_into(head, ckpt, key_candidates=["classifier"], strict=True):
            raise RuntimeError(
                f"No classifier weights in {checkpoint} "
                f"(state dicts found: {sorted(ckpt['state_dicts'])})"
            )
    return head
