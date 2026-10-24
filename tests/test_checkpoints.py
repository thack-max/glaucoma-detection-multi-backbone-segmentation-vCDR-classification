"""Tests for the unified checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.utils.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    load_into,
    save_checkpoint,
)


class _Dummy(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Canonical save + load
# ---------------------------------------------------------------------------


def test_save_checkpoint_writes_canonical_format(tmp_path: Path):
    model = _Dummy()
    path = tmp_path / "dummy.pth"
    save_checkpoint(
        path,
        component="dummy_component",
        modules={"model": model},
        metadata={"epoch": 3, "val_loss": 0.42},
    )

    ckpt = load_checkpoint(path)
    assert ckpt["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert ckpt["component"] == "dummy_component"
    assert "model" in ckpt["state_dicts"]
    assert ckpt["metadata"]["epoch"] == 3
    assert ckpt["metadata"]["val_loss"] == 0.42


def test_roundtrip_multi_module(tmp_path: Path):
    a = _Dummy(4, 2)
    b = _Dummy(8, 4)
    save_checkpoint(
        tmp_path / "multi.pth",
        component="specialist_joint",
        modules={"temporal": a, "forecaster": b},
    )

    fresh_a = _Dummy(4, 2)
    fresh_b = _Dummy(8, 4)
    ckpt = load_checkpoint(tmp_path / "multi.pth")

    assert load_into(fresh_a, ckpt, key_candidates=["temporal"])
    assert load_into(fresh_b, ckpt, key_candidates=["forecaster"])

    # Weights actually copied over
    assert torch.allclose(a.linear.weight, fresh_a.linear.weight)
    assert torch.allclose(b.linear.weight, fresh_b.linear.weight)


# ---------------------------------------------------------------------------
# Backwards-compatible loading (legacy formats)
# ---------------------------------------------------------------------------


def test_load_legacy_dict_of_state_dicts(tmp_path: Path):
    """Old training scripts saved {'model': state_dict, 'epoch': N} -style."""
    model = _Dummy()
    path = tmp_path / "legacy.pth"
    torch.save(
        {"model": model.state_dict(), "epoch": 5, "val_acc": 0.88},
        path,
    )

    ckpt = load_checkpoint(path)
    assert "model" in ckpt["state_dicts"]
    assert ckpt["metadata"]["epoch"] == 5
    assert ckpt["metadata"]["val_acc"] == 0.88
    assert ckpt["component"] is None  # legacy has no component tag


def test_load_raw_state_dict(tmp_path: Path):
    """Some checkpoints are just a state_dict with no wrapper."""
    model = _Dummy()
    path = tmp_path / "raw.pth"
    torch.save(model.state_dict(), path)

    ckpt = load_checkpoint(path)
    assert "model" in ckpt["state_dicts"]

    fresh = _Dummy()
    assert load_into(fresh, ckpt)
    assert torch.allclose(model.linear.weight, fresh.linear.weight)


def test_load_into_tries_candidate_keys(tmp_path: Path):
    """load_into picks the first matching key in the candidate list."""
    model = _Dummy()
    path = tmp_path / "ambiguous.pth"
    save_checkpoint(
        path,
        component="screening_classifier",
        modules={"classifier": model},
    )

    ckpt = load_checkpoint(path)
    fresh = _Dummy()
    # Ask for 'model' first (which doesn't exist), then 'classifier' (which does)
    assert load_into(fresh, ckpt, key_candidates=["model", "classifier"])
    assert torch.allclose(model.linear.weight, fresh.linear.weight)


def test_load_into_single_state_dict_fallback(tmp_path: Path):
    """If only one state_dict exists and no candidate matches, use it."""
    model = _Dummy()
    torch.save({"classifier": model.state_dict()}, tmp_path / "x.pth")

    ckpt = load_checkpoint(tmp_path / "x.pth")
    fresh = _Dummy()
    # Candidates that don't match should still fall back to the only sd
    assert load_into(fresh, ckpt, key_candidates=["nonexistent"])
    assert torch.allclose(model.linear.weight, fresh.linear.weight)


def test_load_missing_file_raises(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pth")


def test_load_into_returns_false_when_no_match(tmp_path: Path):
    """No keys match and there are multiple state_dicts: can't guess."""
    a = _Dummy()
    b = _Dummy()
    save_checkpoint(
        tmp_path / "multi.pth",
        component="test",
        modules={"a": a, "b": b},
    )
    ckpt = load_checkpoint(tmp_path / "multi.pth")
    fresh = _Dummy()
    assert load_into(fresh, ckpt, key_candidates=["nonexistent"]) is False


class _PickledPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        import os

        return (os.mkdir, (str(self.marker),))


def test_load_checkpoint_refuses_arbitrary_pickles_by_default(tmp_path: Path):
    """weights_only=True by default: unpickling a checkpoint must not run code."""
    import pickle

    import pytest

    marker = tmp_path / "payload_ran"
    path = tmp_path / "evil.pth"
    torch.save({"model": _Dummy().state_dict(), "note": _PickledPayload(marker)}, path)

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    assert not marker.exists()

    # Explicit opt-in for trusted files still works.
    ckpt = load_checkpoint(path, weights_only=False)
    assert "model" in ckpt["state_dicts"]
    assert marker.exists()
