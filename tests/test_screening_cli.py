"""Checkpoint and fairness wiring of the inference CLIs.

Segmenter checkpoints (run_screening.py, evaluate.py, run_specialist.py fundus
+ OCT) must load strictly and safely; an unconfigured segmenter/head, or a
DINOv2 encoder without its pretrained weights, must be announced; and
glaucoma-screen / evaluate.py must run a train_fairness.py head behind its
EquityModule. DINOv2 is replaced by a small deterministic stub (offline).
"""

from __future__ import annotations

import copy
import csv
import json
import os
import pickle
import sys
import urllib.error
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

from scripts import evaluate, run_screening, run_specialist
from src.data.fundus_dataset import FundusDataset, _read_image
from src.data.transforms import get_fundus_transforms
from src.models.classifier_head import ClassifierHead
from src.models.fairness.equity_attention import EquityModule
from src.models.segmentation.boundary_former import BoundaryFormer
from src.models.segmentation.swin_unetr_oct import SwinUNETROCT
from tests.test_evaluate_script import _build_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SCREENING = REPO_ROOT / "configs" / "tier" / "screening.yaml"
EMBED_DIM = 16
IMAGE_SIZE = 64

SMALL_BF = {"num_classes": 3, "base_channels": 16, "growth_rate": 8}
SMALL_OCT = {"embed_dim": 24, "depths": [1, 1, 1, 1], "num_heads": [3, 3, 3, 3]}


class _StubEncoder(nn.Module):
    """Deterministic stand-in for DINOv2Encoder (CLS forward + patch tokens)."""

    pretrained_loaded = True

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3, embed_dim)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(embed_dim, 3, generator=g))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-2, -1)))

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        patches = nn.functional.adaptive_avg_pool2d(x, 4).flatten(2).transpose(1, 2)
        return self.proj(patches)  # [B, 16, D]


def _perturb(module: nn.Module, seed: int = 123) -> nn.Module:
    """Move every parameter off its fresh-init value (stand-in for training)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    return module.eval()


def _assert_same_state(actual: nn.Module, expected: nn.Module) -> None:
    exp, act = expected.state_dict(), actual.state_dict()
    assert act.keys() == exp.keys()
    for key in exp:
        torch.testing.assert_close(act[key], exp[key], msg=f"{key} differs")


# ---------------------------------------------------------------------------
# Segmenter checkpoints: every inference entry point
# ---------------------------------------------------------------------------

# name -> (build the reference architecture, load it from `path` the CLI way)
SEGMENTER_LOADERS = {
    "run_screening": (
        lambda: BoundaryFormer(**SMALL_BF),
        lambda path: run_screening._try_load_segmenter({**SMALL_BF, "checkpoint": path}, "cpu"),
    ),
    "evaluate": (
        # evaluate.py builds the shipped default BoundaryFormer size.
        lambda: BoundaryFormer(num_classes=3, base_channels=64, growth_rate=32),
        lambda path: evaluate._try_build_segmenter(path, "cpu"),
    ),
    "run_specialist-fundus": (
        lambda: BoundaryFormer(**SMALL_BF),
        lambda path: run_specialist._safe_build(
            run_specialist._load_segmenter, {**SMALL_BF, "checkpoint": path},
        ),
    ),
    "run_specialist-oct": (
        lambda: SwinUNETROCT(
            embed_dim=24, depths=(1, 1, 1, 1), num_heads=(3, 3, 3, 3),
        ),
        lambda path: run_specialist._safe_build(
            run_specialist._load_oct_segmenter, {**SMALL_OCT, "checkpoint": path},
        ),
    ),
}


@pytest.mark.parametrize("loader", sorted(SEGMENTER_LOADERS))
def test_segmenter_checkpoint_round_trips(loader, tmp_path):
    build, load = SEGMENTER_LOADERS[loader]
    trained = _perturb(build())
    path = tmp_path / "best.pth"
    # Same layout scripts/train_segmentation.py writes.
    torch.save(
        {"model": trained.state_dict(), "epoch": 7, "metrics": {"dice_od": 0.9, "dice_oc": 0.8}},
        path,
    )
    _assert_same_state(load(str(path)), trained)


@pytest.mark.parametrize("loader", sorted(SEGMENTER_LOADERS))
def test_missing_segmenter_checkpoint_fails_loudly(loader, tmp_path):
    # A configured path that does not exist must not mean "random weights".
    _, load = SEGMENTER_LOADERS[loader]
    with pytest.raises((FileNotFoundError, run_specialist.CheckpointLoadError), match="nope.pth"):
        load(str(tmp_path / "nope.pth"))


@pytest.mark.parametrize("layout", ["dataparallel-prefix", "other-module"])
@pytest.mark.parametrize("loader", sorted(SEGMENTER_LOADERS))
def test_wrong_layout_segmenter_checkpoint_fails_loudly(loader, layout, tmp_path):
    build, load = SEGMENTER_LOADERS[loader]
    if layout == "dataparallel-prefix":
        state = {f"module.{k}": v for k, v in build().state_dict().items()}
    else:
        state = ClassifierHead(input_dim=8).state_dict()
    path = tmp_path / "wrong.pth"
    torch.save({"model": state}, path)
    # Loaded non-strictly, nothing matched yet "Loaded segmenter weights" printed.
    with pytest.raises(RuntimeError, match="wrong.pth"):
        load(str(path))


class _PickledPayload:
    """Unpickling this runs os.mkdir(marker): stands in for malicious code."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (str(self.marker),))


@pytest.mark.parametrize("loader", sorted(SEGMENTER_LOADERS))
def test_segmenter_checkpoint_pickle_payload_is_not_executed(loader, tmp_path, monkeypatch):
    # torch 2.1-2.5 (allowed by pyproject) default torch.load to
    # weights_only=False; this variable restores that default for any call
    # site that does not pass weights_only explicitly.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    build, load = SEGMENTER_LOADERS[loader]
    marker = tmp_path / "payload_ran"
    path = tmp_path / "evil.pth"
    torch.save({"model": build().state_dict(), "note": _PickledPayload(marker)}, path)

    with pytest.raises((pickle.UnpicklingError, run_specialist.CheckpointLoadError)):
        load(str(path))
    assert not marker.exists()


@pytest.mark.parametrize("loader", sorted(SEGMENTER_LOADERS))
def test_unconfigured_segmenter_warns_untrained(loader, capsys):
    _, load = SEGMENTER_LOADERS[loader]
    model = load(None)
    assert model is not None  # still runs (e.g. smoke tests), but says so
    assert "UNTRAINED" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# DINOv2 without pretrained weights must be announced by every CLI
# ---------------------------------------------------------------------------


ENCODER_LOADERS = {
    "run_screening": lambda: run_screening._try_load_encoder({"variant": "dinov2_vits14"}, "cpu"),
    "evaluate": lambda: evaluate._try_build_encoder("dinov2_vits14", "cpu"),
    "run_specialist": lambda: run_specialist._load_encoder({"variant": "dinov2_vits14"}),
}


@pytest.mark.parametrize("loader", sorted(ENCODER_LOADERS))
def test_cli_warns_when_encoder_is_not_pretrained(loader, monkeypatch, capsys):
    def _offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(torch.hub, "load", _offline)
    with pytest.warns(RuntimeWarning, match="RANDOMLY INITIALISED"):
        encoder = ENCODER_LOADERS[loader]()
    assert encoder.pretrained_loaded is False
    err = capsys.readouterr().err
    assert "DINOv2 encoder is UNTRAINED" in err


SHIPPED_ENCODER_LOADERS = {
    "screening": lambda cfg: run_screening._try_load_encoder(cfg, "cpu"),
    "specialist": lambda cfg: run_specialist._load_encoder(cfg),
}


@pytest.mark.parametrize("tier", sorted(SHIPPED_ENCODER_LOADERS))
def test_shipped_config_encoder_matches_trainer_features(tier, monkeypatch):
    # The CLIs must request a callable torch.hub's DINOv2 repo exports, and the
    # same backbone the trainers fit the heads on: every scripts/train_*.py
    # builds DINOv2Encoder(variant, freeze_encoder=True), i.e. the plain
    # (register-free) `variant`.
    requested: list[str] = []

    def _hub(repo, model, **kwargs):
        requested.append(model)
        return nn.Linear(3, 768)

    monkeypatch.setattr(torch.hub, "load", _hub)
    with open(REPO_ROOT / "configs" / "tier" / f"{tier}.yaml") as f:
        cfg = yaml.safe_load(f)["feature_extraction"]
    encoder = SHIPPED_ENCODER_LOADERS[tier](cfg)

    assert encoder.pretrained_loaded is True
    assert requested == [cfg["variant"]]


# ---------------------------------------------------------------------------
# glaucoma-screen: fairness section + train_fairness.py checkpoints
# ---------------------------------------------------------------------------


def _save_fairness_checkpoint(path: Path) -> tuple[EquityModule, ClassifierHead]:
    equity = _perturb(EquityModule(feature_dim=EMBED_DIM, num_groups=3), seed=5)
    head = _perturb(ClassifierHead(input_dim=EMBED_DIM), seed=6)
    # Same layout train_fairness.main() writes.
    torch.save(
        {
            "equity": equity.state_dict(),
            "classifier": head.state_dict(),
            "val_metrics": {"cls_loss": 0.5, "accuracy": 0.75, "es_auc": 0.7},
            "args": {"num_groups": 3},
        },
        path,
    )
    return equity, head


def _screening_config(tmp_path: Path, *, classifier_ckpt=None, fairness=None) -> Path:
    with open(SHIPPED_SCREENING) as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config["segmentation"].update(base_channels=16, growth_rate=8)
    config["feature_extraction"]["image_size"] = IMAGE_SIZE
    config["classifier"].update(input_dim=EMBED_DIM, checkpoint=classifier_ckpt)
    if fairness is not None:
        config["fairness"].update(fairness)
    path = tmp_path / "screening.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path


def _run_screening(monkeypatch, tmp_path, config_path) -> dict:
    image = tmp_path / "fundus.png"
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (150, 80, 60)).save(image)
    out = tmp_path / "out"
    monkeypatch.setattr(run_screening, "_try_load_encoder", lambda cfg, device: _StubEncoder())
    monkeypatch.setattr(
        sys, "argv",
        ["run_screening.py", "--config", str(config_path), "--image", str(image),
         "--output-dir", str(out)],
    )
    run_screening.main()
    with open(out / "fundus.json") as f:
        return json.load(f)


def _expected_risk(tmp_path, equity, head, image: Path | None = None) -> float:
    transform = get_fundus_transforms("val", image_size=IMAGE_SIZE)
    x = transform(_read_image(image or tmp_path / "fundus.png"))["image"].unsqueeze(0)
    with torch.no_grad():
        emb = _StubEncoder()(x)
        if equity is not None:
            emb = equity(emb, None)  # no demographics on the CLI
        return float(torch.sigmoid(head(emb)).item())


@pytest.mark.parametrize("where", ["classifier.checkpoint", "fairness.checkpoint"])
def test_screening_runs_fairness_head_behind_its_equity_module(where, monkeypatch, tmp_path):
    ckpt = tmp_path / "fairness_best.pth"
    equity, head = _save_fairness_checkpoint(ckpt)
    if where == "classifier.checkpoint":
        config = _screening_config(tmp_path, classifier_ckpt=str(ckpt))
    else:
        config = _screening_config(tmp_path, fairness={"enabled": True, "checkpoint": str(ckpt)})

    record = _run_screening(monkeypatch, tmp_path, config)

    expected = _expected_risk(tmp_path, equity, head)
    assert record["risk_score"] == pytest.approx(expected, abs=1e-4)
    # The head applied to raw features (equity dropped) gives a different score.
    assert abs(_expected_risk(tmp_path, None, head) - expected) > 1e-3


def test_screening_missing_fairness_checkpoint_fails_loudly(monkeypatch, tmp_path):
    config = _screening_config(
        tmp_path, fairness={"enabled": True, "checkpoint": str(tmp_path / "nope.pth")},
    )
    with pytest.raises(FileNotFoundError, match="nope.pth"):
        _run_screening(monkeypatch, tmp_path, config)


def test_screening_fairness_checkpoint_without_equity_fails(monkeypatch, tmp_path):
    ckpt = tmp_path / "plain_head.pth"
    torch.save({"classifier": ClassifierHead(input_dim=EMBED_DIM).state_dict()}, ckpt)
    config = _screening_config(tmp_path, fairness={"enabled": True, "checkpoint": str(ckpt)})
    with pytest.raises(RuntimeError, match="no EquityModule"):
        _run_screening(monkeypatch, tmp_path, config)


def test_screening_fairness_head_with_fairness_disabled_fails(monkeypatch, tmp_path):
    ckpt = tmp_path / "fairness_best.pth"
    _save_fairness_checkpoint(ckpt)
    config = _screening_config(
        tmp_path, classifier_ckpt=str(ckpt), fairness={"enabled": False},
    )
    with pytest.raises(RuntimeError, match="fairness is disabled"):
        _run_screening(monkeypatch, tmp_path, config)


def test_screening_conflicting_head_and_fairness_checkpoints_fail(monkeypatch, tmp_path):
    fair = tmp_path / "fairness_best.pth"
    _save_fairness_checkpoint(fair)
    other = tmp_path / "other_head.pth"
    torch.save({"classifier": ClassifierHead(input_dim=EMBED_DIM).state_dict()}, other)
    config = _screening_config(
        tmp_path, classifier_ckpt=str(other),
        fairness={"enabled": True, "checkpoint": str(fair)},
    )
    with pytest.raises(ValueError, match="differ"):
        _run_screening(monkeypatch, tmp_path, config)


def test_screening_warns_untrained_head_and_segmenter(monkeypatch, tmp_path, capsys):
    # The shipped config has no checkpoints: still runs, but must say so.
    _run_screening(monkeypatch, tmp_path, _screening_config(tmp_path))
    err = capsys.readouterr().err
    assert "BoundaryFormer segmenter is UNTRAINED" in err
    assert "Classifier head is UNTRAINED" in err


def test_screening_gradcam_explains_equity_module_and_head(monkeypatch, tmp_path):
    # Grad-CAM must explain the model that produced the risk score, i.e.
    # head(EquityModule(features)), not the head applied to raw features.
    ckpt = tmp_path / "fairness_best.pth"
    equity, head = _save_fairness_checkpoint(ckpt)
    config = _screening_config(tmp_path, classifier_ckpt=str(ckpt))

    explained: list[nn.Module] = []

    class _RecordingGradCAM(run_screening.DINOv2GradCAM):
        def __init__(self, encoder, classifier):
            super().__init__(encoder, classifier)
            explained.append(classifier)

    monkeypatch.setattr(run_screening, "DINOv2GradCAM", _RecordingGradCAM)
    _run_screening(monkeypatch, tmp_path, config)

    assert (tmp_path / "out" / "fundus_gradcam.png").is_file()
    assert len(explained) == 1
    # Explaining must leave every module in eval mode (dropout off).
    assert not [name for name, m in explained[0].named_modules() if m.training]
    emb = torch.randn(3, EMBED_DIM, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = head(equity(emb, None))  # what the risk score is computed with
        # Guard: dropping the EquityModule would explain a different function.
        assert not torch.allclose(head(emb), expected, atol=1e-3)
        torch.testing.assert_close(explained[0](emb), expected)


@pytest.mark.parametrize("kind", ["plain", "fairness"])
def test_screening_scores_stay_deterministic_after_gradcam(kind, monkeypatch, tmp_path):
    # Grad-CAM runs after each image; it must not leave the head (or the
    # EquityModule in front of it) in training mode, or every later image is
    # scored with dropout active.
    ckpt = tmp_path / "head.pth"
    if kind == "fairness":
        equity, head = _save_fairness_checkpoint(ckpt)
    else:
        equity, head = None, _perturb(ClassifierHead(input_dim=EMBED_DIM), seed=6)
        torch.save({"classifier": head.state_dict()}, ckpt)
    config = _screening_config(tmp_path, classifier_ckpt=str(ckpt))
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("a", "b", "c"):
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (150, 80, 60)).save(image_dir / f"{name}.png")
    out = tmp_path / "out"
    monkeypatch.setattr(run_screening, "_try_load_encoder", lambda cfg, device: _StubEncoder())
    monkeypatch.setattr(
        sys, "argv",
        ["run_screening.py", "--config", str(config), "--image-dir", str(image_dir),
         "--output-dir", str(out)],
    )

    run_screening.main()

    expected = _expected_risk(tmp_path, equity, head, image=image_dir / "a.png")
    scores = []
    for name in ("a", "b", "c"):
        assert (out / f"{name}_gradcam.png").is_file()
        with open(out / f"{name}.json") as f:
            scores.append(json.load(f)["risk_score"])
    assert scores == pytest.approx([expected] * 3, abs=1e-4)


# ---------------------------------------------------------------------------
# evaluate.py: same for --classifier-checkpoint
# ---------------------------------------------------------------------------


def test_evaluate_runs_fairness_head_behind_its_equity_module(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    _build_manifest(data_root, n=4)
    ckpt = tmp_path / "fairness_best.pth"
    equity, head = _save_fairness_checkpoint(ckpt)
    out = tmp_path / "eval"

    monkeypatch.setattr(evaluate, "_try_build_encoder", lambda variant, device: _StubEncoder())
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--tier", "screening", "--data-root", str(data_root),
         "--classifier-checkpoint", str(ckpt), "--image-size", str(IMAGE_SIZE),
         "--output-dir", str(out)],
    )
    evaluate.main()

    with open(out / "baseline_vs_glaucomamba.csv") as f:
        rows = list(csv.DictReader(f))
    dataset = FundusDataset(
        data_root, split="test", transform=get_fundus_transforms("val", image_size=IMAGE_SIZE),
    )
    stub = _StubEncoder()
    assert len(rows) == len(dataset) == 4
    with torch.no_grad():
        for row, sample in zip(rows, (dataset[i] for i in range(len(dataset)))):
            group = torch.tensor([int(sample["group_id"].item())])
            expected = torch.sigmoid(head(equity(stub(sample["image"].unsqueeze(0)), group)))
            assert float(row["glaucomamba_score"]) == pytest.approx(expected.item(), abs=1e-4)
