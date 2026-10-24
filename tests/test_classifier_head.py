"""Screening-classifier checkpoints must load into the inference heads.

train_screening_classifier.py saves ``ClassifierHead.state_dict()``; the
Tier 1 CLI (run_screening.py) and the A/B evaluator (evaluate.py) must load
exactly those weights, and must fail loudly when they cannot.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest
import torch

from scripts.evaluate import _try_build_classifier
from scripts.run_screening import _try_load_classifier
from scripts.train_screening_classifier import ClassifierHead


INPUT_DIM = 16

LOADERS = {
    "run_screening": lambda path: _try_load_classifier(
        {"checkpoint": str(path), "input_dim": INPUT_DIM}, "cpu",
    ),
    "evaluate": lambda path: _try_build_classifier(str(path), INPUT_DIM, "cpu"),
}


def _save_trained_checkpoint(tmp_path: Path) -> tuple[ClassifierHead, Path]:
    torch.manual_seed(0)
    head = ClassifierHead(input_dim=INPUT_DIM)
    with torch.no_grad():
        # Move every weight (LayerNorm's ones/zeros included) off its
        # fresh-init value so a silently skipped load cannot pass.
        for p in head.parameters():
            p.add_(torch.randn_like(p))
    path = tmp_path / "screening_classifier.pth"
    # Same layout train_screening_classifier.main() writes.
    torch.save({"classifier": head.state_dict(), "epoch": 3, "val_acc": 0.9}, path)
    return head.eval(), path


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_trained_checkpoint_round_trips(loader, tmp_path):
    trained, path = _save_trained_checkpoint(tmp_path)
    head = LOADERS[loader](path)

    x = torch.randn(4, INPUT_DIM)
    with torch.no_grad():
        torch.testing.assert_close(head(x).reshape(-1), trained(x).reshape(-1))


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_incompatible_checkpoint_fails_loudly(loader, tmp_path):
    path = tmp_path / "bare_sequential.pth"
    # Keys of a bare nn.Sequential, i.e. not the ClassifierHead layout.
    torch.save({"classifier": {"0.weight": torch.ones(INPUT_DIM)}}, path)
    with pytest.raises(RuntimeError):
        LOADERS[loader](path)


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_missing_checkpoint_fails_loudly(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        LOADERS[loader](tmp_path / "does_not_exist.pth")


class _PickledPayload:
    """Unpickling this runs os.mkdir(marker): stands in for malicious code."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (str(self.marker),))


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_checkpoint_pickle_payload_is_not_executed(loader, tmp_path):
    marker = tmp_path / "payload_ran"
    path = tmp_path / "evil.pth"
    head = ClassifierHead(input_dim=INPUT_DIM)
    torch.save({"classifier": head.state_dict(), "note": _PickledPayload(marker)}, path)

    with pytest.raises(pickle.UnpicklingError):
        LOADERS[loader](path)
    assert not marker.exists()
