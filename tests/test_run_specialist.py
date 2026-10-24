"""Tier 2 CLI (scripts/run_specialist.py): config wiring, checkpoints, VLM.

DINOv2 is replaced by a small deterministic stub so these tests stay offline.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

from scripts import run_specialist
from src.pipeline.specialist import SpecialistPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "tier" / "specialist.yaml"

TINY_CONFIG = {
    "tier": "specialist",
    "pipeline": {"device": "cpu", "enable_vlm": False},
    "segmentation": {
        "fundus": {"num_classes": 3, "base_channels": 16, "growth_rate": 8},
        "oct": {"embed_dim": 24, "depths": [1, 1, 1, 1], "num_heads": [3, 3, 3, 3]},
    },
    "feature_extraction": {"variant": "dinov2_vits14"},
    "visual_field": {"hidden_dim": 8, "num_gin_layers": 2, "output_dim": 8},
    "progression": {
        "temporal_encoder": {
            "input_dim": 16, "d_model": 8, "d_state": 4, "num_layers": 1, "expand": 1,
        },
        "forecaster": {"temporal_dim": 8, "clinical_dim": 3, "vf_dim": 0, "hidden_dim": 8},
    },
    "fusion": {
        "modality_dims": {"fundus": 16, "oct": 16, "vf": 16, "clinical": 3},
        "fused_dim": 8,
        "num_heads": 2,
        "mc_samples": 2,
    },
    "vlm": {
        "model": "test/tiny-vlm", "lora_rank": 4, "max_new_tokens": 8, "load_in_4bit": False,
    },
    "output": {"output_dir": "unused"},
}


class _StubEncoder(nn.Module):
    """Deterministic stand-in for DINOv2Encoder: [B, 3, H, W] -> [B, embed_dim]."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3, embed_dim)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(embed_dim, 3, generator=g))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-2, -1)))


def _perturb(module: nn.Module, seed: int = 123) -> nn.Module:
    """Move every parameter off its fresh-init value (stand-in for training)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    return module.eval()


def _assert_same_state(actual: nn.Module, expected: nn.Module) -> None:
    exp = expected.state_dict()
    act = actual.state_dict()
    assert act.keys() == exp.keys()
    for key in exp:
        torch.testing.assert_close(act[key], exp[key], msg=f"{key} differs")


def _make_patient(root: Path, *, clinical: bool = False, vf: bool = False) -> Path:
    patient = root / "P1"
    (patient / "visits").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), (120 + 30 * i, 80, 60)).save(
            patient / "visits" / f"v{i}.png",
        )
    if clinical:
        with open(patient / "clinical.json", "w") as f:
            json.dump({"iop": 21.0, "age": 64, "sex": 1}, f)
    if vf:
        rng = np.random.RandomState(0)
        np.save(patient / "vf.npy", rng.randn(54, 3).astype(np.float32))
    return patient


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "specialist.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path


def _run_cli(monkeypatch, tmp_path, config_path, patient_dir, *extra, embed_dim):
    monkeypatch.setattr(run_specialist, "_load_encoder", lambda cfg: _StubEncoder(embed_dim))
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv",
        [
            "run_specialist.py",
            "--config", str(config_path),
            "--patient-dir", str(patient_dir),
            "--output-dir", str(output_dir),
            "--device", "cpu",
            *extra,
        ],
    )
    run_specialist.main()
    with open(output_dir / f"{patient_dir.name}_specialist.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shipped config: forecaster/fusion dims must match what the pipeline feeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("clinical", "vf"), [(False, False), (True, False), (True, True)],
    ids=["visits-only", "visits+clinical", "visits+clinical+vf"],
)
def test_shipped_config_forecasts_from_visit_history(monkeypatch, tmp_path, clinical, vf):
    patient = _make_patient(tmp_path, clinical=clinical, vf=vf)
    record = _run_cli(monkeypatch, tmp_path, SHIPPED_CONFIG, patient, embed_dim=768)

    assert record["progression_probability"] is not None
    assert 0.0 <= record["progression_probability"] <= 1.0
    assert record["progression_class"] in {"stable", "slow", "rapid"}


@pytest.mark.parametrize(
    ("clinical_dim", "vf_dim"), [(3, 0), (0, 0)], ids=["no-vf-slot", "no-aux-slots"],
)
def test_forecaster_without_aux_slots_ignores_patient_aux_data(
    monkeypatch, tmp_path, clinical_dim, vf_dim,
):
    # e.g. a train_progression.py forecaster (vf_dim=0) on a patient with vf.npy:
    # data for a slot the forecaster was not built with must be left out.
    config = copy.deepcopy(TINY_CONFIG)
    config["progression"]["forecaster"].update(clinical_dim=clinical_dim, vf_dim=vf_dim)
    patient = _make_patient(tmp_path, clinical=True, vf=True)

    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, config), patient, embed_dim=16,
    )
    assert record["progression_probability"] is not None


def test_shipped_config_fusion_accepts_pipeline_features():
    with open(SHIPPED_CONFIG) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(0)
    pipeline = SpecialistPipeline(
        vf_graph_net=run_specialist._load_vf_graph_net(config["visual_field"]),
        fusion_module=run_specialist._load_fusion(config["fusion"]),
        device="cpu",
    )
    with torch.no_grad():
        output = pipeline.run(
            vf_data=torch.randn(1, 54, 3),
            clinical_data={"iop": 21.0, "age": 64, "sex": 1},
        )
    assert output.vf_classification is not None
    assert output.fusion_uncertainty is not None


# ---------------------------------------------------------------------------
# Trained checkpoints must actually be loaded
# ---------------------------------------------------------------------------


def _builders() -> dict:
    prog = TINY_CONFIG["progression"]
    return {
        "vf_net": (run_specialist._load_vf_graph_net, TINY_CONFIG["visual_field"]),
        "temporal": (run_specialist._load_temporal_encoder, prog["temporal_encoder"]),
        "fusion": (run_specialist._load_fusion, TINY_CONFIG["fusion"]),
        "forecaster": (run_specialist._load_forecaster, prog["forecaster"]),
    }


def test_loaders_restore_train_specialist_checkpoint(tmp_path):
    builders = _builders()
    # train_specialist.py's forecaster consumes [temporal | fused] (8 + 8).
    builders["forecaster"] = (run_specialist._load_forecaster, {"temporal_dim": 16, "hidden_dim": 8})
    trained = {key: _perturb(build(cfg)) for key, (build, cfg) in builders.items()}
    ckpt = tmp_path / "best.pth"
    # Same layout train_specialist.py writes.
    torch.save(
        {
            **{key: m.state_dict() for key, m in trained.items()},
            "args": {"d_model": 8, "fused_dim": 8},
            "epoch": 1,
            "loss": 0.5,
        },
        ckpt,
    )

    for key, (build, cfg) in builders.items():
        _assert_same_state(build({**cfg, "checkpoint": str(ckpt)}), trained[key])


def test_loaders_restore_train_progression_checkpoint(tmp_path):
    builders = _builders()
    temporal = _perturb(run_specialist._load_temporal_encoder(TINY_CONFIG["progression"]["temporal_encoder"]))
    forecaster = _perturb(run_specialist._load_forecaster(TINY_CONFIG["progression"]["forecaster"]))
    ckpt = tmp_path / "best.pth"
    # Same layout train_progression.py writes.
    torch.save(
        {"mamba": temporal.state_dict(), "forecaster": forecaster.state_dict(), "args": {"d_model": 8}},
        ckpt,
    )

    build, cfg = builders["temporal"]
    _assert_same_state(build({**cfg, "checkpoint": str(ckpt)}), temporal)
    build, cfg = builders["forecaster"]
    _assert_same_state(build({**cfg, "checkpoint": str(ckpt)}), forecaster)


@pytest.mark.parametrize("component", ["vf_net", "temporal", "fusion", "forecaster"])
def test_missing_checkpoint_fails_loudly(component, tmp_path):
    build, cfg = _builders()[component]
    missing = tmp_path / "nope.pth"
    with pytest.raises(run_specialist.CheckpointLoadError, match="nope.pth"):
        # _safe_build must not downgrade this to a silent "[skip]".
        run_specialist._safe_build(build, {**cfg, "checkpoint": str(missing)})


def test_incompatible_checkpoint_fails_loudly(tmp_path):
    # A forecaster saved with a different input width than the config describes.
    other = run_specialist._load_forecaster({"temporal_dim": 16, "hidden_dim": 8})
    ckpt = tmp_path / "best.pth"
    torch.save({"forecaster": other.state_dict()}, ckpt)

    build, cfg = _builders()["forecaster"]
    with pytest.raises(run_specialist.CheckpointLoadError, match="forecaster"):
        run_specialist._safe_build(build, {**cfg, "checkpoint": str(ckpt)})


@pytest.mark.parametrize("layout", ["foreign-keys", "dataparallel-prefix"])
@pytest.mark.parametrize("component", ["vf_net", "temporal", "fusion", "forecaster"])
def test_wrong_keys_checkpoint_fails_loudly(component, layout, tmp_path):
    # Key (not size) mismatches only raise under strict loading; a non-strict
    # load would silently keep the random init.
    build, cfg = _builders()[component]
    if layout == "foreign-keys":
        state = {"some.other.weight": torch.ones(3)}
    else:
        state = {f"module.{k}": v for k, v in build(cfg).state_dict().items()}
    ckpt = tmp_path / "wrong.pth"
    torch.save({component: state}, ckpt)

    with pytest.raises(run_specialist.CheckpointLoadError, match="wrong.pth"):
        run_specialist._safe_build(build, {**cfg, "checkpoint": str(ckpt)})


class _PickledPayload:
    """Unpickling this runs os.mkdir(marker): stands in for malicious code."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (str(self.marker),))


def test_checkpoint_pickle_payload_is_not_executed(tmp_path):
    build, cfg = _builders()["vf_net"]
    marker = tmp_path / "payload_ran"
    ckpt = tmp_path / "evil.pth"
    torch.save({"vf_net": build(cfg).state_dict(), "note": _PickledPayload(marker)}, ckpt)

    with pytest.raises(run_specialist.CheckpointLoadError) as excinfo:
        run_specialist._safe_build(build, {**cfg, "checkpoint": str(ckpt)})
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)
    assert not marker.exists()


def test_cli_progression_uses_trained_weights(monkeypatch, tmp_path):
    prog = TINY_CONFIG["progression"]
    temporal = _perturb(run_specialist._load_temporal_encoder(prog["temporal_encoder"]), seed=1)
    forecaster = _perturb(run_specialist._load_forecaster(prog["forecaster"]), seed=2)
    ckpt = tmp_path / "progression.pth"
    torch.save({"mamba": temporal.state_dict(), "forecaster": forecaster.state_dict()}, ckpt)

    config = copy.deepcopy(TINY_CONFIG)
    config["progression"]["temporal_encoder"]["checkpoint"] = str(ckpt)
    config["progression"]["forecaster"]["checkpoint"] = str(ckpt)
    patient = _make_patient(tmp_path, clinical=True)

    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, config), patient, embed_dim=16,
    )

    data = run_specialist._load_patient_dir(patient)
    reference = SpecialistPipeline(
        feature_encoder=_StubEncoder(16).eval(),
        temporal_encoder=temporal,
        forecaster=forecaster,
        device="cpu",
    )
    with torch.no_grad():
        expected = reference.run(
            visit_history=data["visit_history"], clinical_data=data["clinical"],
        )
    assert record["progression_probability"] == pytest.approx(
        expected.progression_probability, abs=1e-6,
    )


# ---------------------------------------------------------------------------
# --enable-vlm must actually build a report generator
# ---------------------------------------------------------------------------


def test_enable_vlm_generates_report(monkeypatch, tmp_path):
    import src.models.vlm.llama_vision_reporter as vlm_module

    created: dict = {}

    class _FakeReporter:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def generate_report(self, **kwargs):
            return SimpleNamespace(raw_text="FAKE REPORT")

    monkeypatch.setattr(vlm_module, "LlamaVisionReporter", _FakeReporter)
    patient = _make_patient(tmp_path, clinical=True)

    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, TINY_CONFIG), patient,
        "--enable-vlm", embed_dim=16,
    )

    assert created["model_name"] == "test/tiny-vlm"
    assert created["max_new_tokens"] == 8
    assert record["clinical_report"] == "FAKE REPORT"
    assert record["vlm_report_is_experimental"] is True


def test_vlm_reporter_not_built_unless_enabled(monkeypatch, tmp_path):
    import src.models.vlm.llama_vision_reporter as vlm_module

    def _fail(**kwargs):
        raise AssertionError("VLM reporter must not be built without --enable-vlm")

    monkeypatch.setattr(vlm_module, "LlamaVisionReporter", _fail)
    patient = _make_patient(tmp_path)
    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, TINY_CONFIG), patient, embed_dim=16,
    )
    assert record["clinical_report"] is None


def _fail_mllama_load_on_missing_backend(monkeypatch):
    """Make the real LlamaVisionReporter fail the way transformers does without bitsandbytes."""
    # Patch the class itself: transformers swaps the `transformers` module
    # object in sys.modules on lazy import, so patching that attribute is flaky.
    from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration

    def _from_pretrained(cls, *args, **kwargs):
        raise ImportError(
            "Using `bitsandbytes` 4-bit quantization requires bitsandbytes: "
            "`pip install -U bitsandbytes>=0.46.1`"
        )

    monkeypatch.setattr(
        MllamaForConditionalGeneration, "from_pretrained", classmethod(_from_pretrained),
    )


def test_reporter_load_error_names_missing_backend(monkeypatch):
    from src.models.vlm.llama_vision_reporter import LlamaVisionReporter

    _fail_mllama_load_on_missing_backend(monkeypatch)
    reporter = LlamaVisionReporter(model_name="test/tiny-vlm", load_in_4bit=True)
    with pytest.raises(ImportError, match="bitsandbytes"):
        reporter.generate_report()


def test_enable_vlm_load_failure_keeps_assessment(monkeypatch, tmp_path, capsys):
    # The real reporter (not a fake) loads lazily inside generate_report(); a
    # missing optional backend must not abort the run or drop its output.
    _fail_mllama_load_on_missing_backend(monkeypatch)
    patient = _make_patient(tmp_path, clinical=True)

    with pytest.warns(RuntimeWarning, match="bitsandbytes"):
        record = _run_cli(
            monkeypatch, tmp_path, _write_config(tmp_path, TINY_CONFIG), patient,
            "--enable-vlm", embed_dim=16,
        )

    assert record["progression_probability"] is not None
    assert record["clinical_report"] is None
    assert "VLM report: not generated" in capsys.readouterr().out
