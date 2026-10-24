"""Screening-classifier checkpoints must load into the inference heads.

train_screening_classifier.py saves ``ClassifierHead.state_dict()``; the
Tier 1 CLI (run_screening.py) and the A/B evaluator (evaluate.py) must load
exactly those weights, and must fail loudly when they cannot.

train_fairness.py saves the head together with the EquityModule it was
trained behind; that EquityModule must come along (never be dropped).
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
from src.models.classifier_head import load_classifier_head
from src.models.fairness.equity_attention import EquityModule


INPUT_DIM = 16

# Each returns (head, equity_module_or_None).
LOADERS_WITH_EQUITY = {
    "run_screening": lambda path: _try_load_classifier(
        {"checkpoint": str(path), "input_dim": INPUT_DIM}, "cpu",
    ),
    "evaluate": lambda path: _try_build_classifier(str(path), INPUT_DIM, "cpu"),
}
LOADERS = {name: (lambda path, f=f: f(path)[0]) for name, f in LOADERS_WITH_EQUITY.items()}


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


# ---------------------------------------------------------------------------
# train_fairness.py checkpoints: head + the EquityModule it was trained behind
# ---------------------------------------------------------------------------


def _save_fairness_checkpoint(tmp_path: Path) -> tuple[EquityModule, ClassifierHead, Path]:
    torch.manual_seed(1)
    equity = EquityModule(feature_dim=INPUT_DIM, num_groups=3)
    head = ClassifierHead(input_dim=INPUT_DIM)
    with torch.no_grad():
        for p in list(equity.parameters()) + list(head.parameters()):
            p.add_(torch.randn_like(p))
    path = tmp_path / "fairness_best.pth"
    # Same layout train_fairness.main() writes.
    torch.save(
        {
            "equity": equity.state_dict(),
            "classifier": head.state_dict(),
            "val_metrics": {"cls_loss": 0.5, "accuracy": 0.75, "es_auc": 0.7},
            "args": {"num_groups": 3, "dinov2_variant": "dinov2_vitb14"},
        },
        path,
    )
    return equity.eval(), head.eval(), path


@pytest.mark.parametrize("loader", sorted(LOADERS_WITH_EQUITY))
def test_fairness_checkpoint_brings_its_equity_module(loader, tmp_path):
    equity, trained, path = _save_fairness_checkpoint(tmp_path)
    head, loaded_equity = LOADERS_WITH_EQUITY[loader](path)

    assert loaded_equity is not None
    x = torch.randn(4, INPUT_DIM)
    groups = torch.tensor([0, 1, 2, 1])
    with torch.no_grad():
        for g in (None, groups):
            torch.testing.assert_close(
                head(loaded_equity(x, g)), trained(equity(x, g)),
            )


def test_plain_head_checkpoint_has_no_equity_module(tmp_path):
    _, path = _save_trained_checkpoint(tmp_path)
    for loader in LOADERS_WITH_EQUITY.values():
        assert loader(path)[1] is None


def test_load_classifier_head_refuses_fairness_checkpoint(tmp_path):
    # Its head was trained on EquityModule outputs: loading it alone would
    # score raw DINOv2 features with it.
    _, _, path = _save_fairness_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="EquityModule"):
        load_classifier_head(path, INPUT_DIM)


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_head_trained_on_random_encoder_warns(loader, tmp_path, recwarn):
    # A trainer run with --allow-random-encoder (torch.hub unreachable) records
    # encoder_pretrained=False; the head then scores meaningless features.
    head = ClassifierHead(input_dim=INPUT_DIM)
    path = tmp_path / "smoke.pth"
    torch.save({"classifier": head.state_dict(), "encoder_pretrained": False}, path)
    with pytest.warns(RuntimeWarning, match="RANDOMLY INITIALISED DINOv2"):
        LOADERS[loader](path)

    torch.save({"classifier": head.state_dict(), "encoder_pretrained": True}, path)
    recwarn.clear()
    LOADERS[loader](path)
    assert not [w for w in recwarn if "RANDOMLY" in str(w.message)]
