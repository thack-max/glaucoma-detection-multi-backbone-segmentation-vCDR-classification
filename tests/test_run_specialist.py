"""Tier 2 CLI (scripts/run_specialist.py): config wiring, checkpoints, VLM.

DINOv2 is replaced by a small deterministic stub so these tests stay offline.
"""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
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


def _fake_installed(monkeypatch, *, present=(), absent=()):
    """Make the reporter's backend check see packages as (not) installed.

    Scoped to the reporter module: transformers' own availability checks also
    use find_spec and then import the package, so a global fake breaks them.
    """
    import src.models.vlm.llama_vision_reporter as vlm_module

    real_find_spec = importlib.util.find_spec

    def _find_spec(name, *args, **kwargs):
        if name in absent:
            return None
        if name in present:
            return importlib.machinery.ModuleSpec(name, None)
        return real_find_spec(name, *args, **kwargs)

    fake = SimpleNamespace(util=SimpleNamespace(find_spec=_find_spec))
    # raising=False: the module only imports importlib once it pre-checks.
    monkeypatch.setattr(vlm_module, "importlib", fake, raising=False)


def _patch_mllama_from_pretrained(monkeypatch, exc: Exception):
    # Patch the class itself: transformers swaps the `transformers` module
    # object in sys.modules on lazy import, so patching that attribute is flaky.
    from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration

    def _from_pretrained(cls, *args, **kwargs):
        raise exc

    monkeypatch.setattr(
        MllamaForConditionalGeneration, "from_pretrained", classmethod(_from_pretrained),
    )


def _fail_mllama_load_on_missing_backend(monkeypatch):
    """Make the real LlamaVisionReporter fail the way transformers does with an unusable bitsandbytes."""
    # Installed (so the up-front check passes) but rejected by transformers,
    # e.g. bitsandbytes older than the 0.46.1 transformers 5.x requires.
    _fake_installed(monkeypatch, present=("accelerate", "bitsandbytes"))
    _patch_mllama_from_pretrained(
        monkeypatch,
        ImportError(
            "Using `bitsandbytes` 4-bit quantization requires bitsandbytes: "
            "`pip install -U bitsandbytes>=0.46.1`"
        ),
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


@pytest.mark.parametrize(
    ("load_in_4bit", "absent", "expected"),
    [
        (False, ("accelerate",), "accelerate"),
        (True, ("accelerate", "bitsandbytes"), "accelerate, bitsandbytes"),
        (True, ("bitsandbytes",), "bitsandbytes"),
    ],
)
def test_reporter_missing_backend_points_to_vlm_extra(monkeypatch, load_in_4bit, absent, expected):
    # transformers reports a missing accelerate as a ValueError from inside
    # from_pretrained; that must not become a licence/login hint.
    from src.models.vlm.llama_vision_reporter import LlamaVisionReporter

    _fake_installed(
        monkeypatch, absent=absent,
        present=tuple({"accelerate", "bitsandbytes"} - set(absent)),
    )
    _patch_mllama_from_pretrained(
        monkeypatch,
        ValueError(
            "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting "
            "`torch.set_default_device(device)` requires `accelerate`. You can install it "
            "with `pip install accelerate`"
        ),
    )
    reporter = LlamaVisionReporter(model_name="test/tiny-vlm", load_in_4bit=load_in_4bit)
    with pytest.raises(ImportError, match=r"missing " + expected + r"\..*\[vlm\]") as excinfo:
        reporter.generate_report()
    assert "licen" not in str(excinfo.value).lower()


def test_reporter_missing_bitsandbytes_offers_full_precision(monkeypatch):
    # The [vlm] extra installs bitsandbytes only where wheels exist (not on
    # Intel macOS), so the hint must also name the load_in_4bit: false route.
    from src.models.vlm.llama_vision_reporter import LlamaVisionReporter

    _fake_installed(monkeypatch, absent=("bitsandbytes",), present=("accelerate",))
    _patch_mllama_from_pretrained(monkeypatch, AssertionError("must not be reached"))
    reporter = LlamaVisionReporter(model_name="test/tiny-vlm", load_in_4bit=True)
    with pytest.raises(ImportError, match=r"load_in_4bit: false"):
        reporter.generate_report()


def test_reporter_licence_hint_only_for_hub_errors(monkeypatch):
    from src.models.vlm.llama_vision_reporter import LlamaVisionReporter

    _fake_installed(monkeypatch, present=("accelerate", "bitsandbytes"))
    reporter = LlamaVisionReporter(model_name="test/tiny-vlm", load_in_4bit=False)

    _patch_mllama_from_pretrained(monkeypatch, OSError("You are trying to access a gated repo."))
    with pytest.raises(RuntimeError, match="license"):
        reporter.generate_report()

    _patch_mllama_from_pretrained(monkeypatch, ValueError("unsupported dtype"))
    with pytest.raises(RuntimeError, match="unsupported dtype") as excinfo:
        reporter.generate_report()
    assert "license" not in str(excinfo.value)


class _FakeProcessor:
    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, text, return_tensors):
        self.prompts.append(text)
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return "REPORT TEXT"


class _FakeModel:
    device = torch.device("cpu")

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


def _loaded_reporter():
    from src.models.vlm.llama_vision_reporter import LlamaVisionReporter

    reporter = LlamaVisionReporter(model_name="test/tiny-vlm", load_in_4bit=False)
    reporter._model, reporter._processor = _FakeModel(), _FakeProcessor()
    return reporter


def test_reporter_prompt_handles_missing_values():
    # vCDR is None whenever there is no fundus image; nothing may assume a number.
    reporter = _loaded_reporter()
    report = reporter.generate_report(
        segmentation_results={"disc_mask": None, "cup_mask": None, "vcdr": None},
        progression_prediction={"probability": None, "class": None, "uncertainty": None},
        clinical_context={"iop": 21.0, "risk_score": None},
    )
    assert report.raw_text == "REPORT TEXT"
    prompt = reporter._processor.prompts[-1]
    assert "vCDR: N/A" in prompt
    assert "Risk Score: N/A" in prompt
    assert "Progression Prediction: Not available" in prompt


def test_reporter_prompt_includes_pipeline_progression():
    # SpecialistPipeline passes {"probability", "class", "uncertainty"}.
    reporter = _loaded_reporter()
    reporter.generate_report(
        segmentation_results={"vcdr": 0.61234},
        progression_prediction={"probability": 0.42, "class": "slow", "uncertainty": 0.1},
        clinical_context={"risk_score": 0.7},
    )
    prompt = reporter._processor.prompts[-1]
    assert "vCDR: 0.612" in prompt
    assert "Risk Score: 0.700" in prompt
    assert "Progression probability: 0.42 (slow) (uncertainty: 0.100)" in prompt


def test_enable_vlm_reports_for_patient_without_fundus(monkeypatch, tmp_path):
    # The patient has visits/ + clinical.json but no fundus.png, so vcdr is None.
    from src.models.vlm.llama_vision_reporter import LlamaVisionReporter

    processor = _FakeProcessor()

    def _fake_load(self):
        self._model, self._processor = _FakeModel(), processor

    monkeypatch.setattr(LlamaVisionReporter, "_ensure_loaded", _fake_load)
    patient = _make_patient(tmp_path, clinical=True)
    assert not (patient / "fundus.png").exists()

    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, TINY_CONFIG), patient,
        "--enable-vlm", embed_dim=16,
    )
    assert record["vcdr"] is None
    assert record["clinical_report"] == "REPORT TEXT"
    assert "vCDR: N/A" in processor.prompts[-1]


class _BuggyReporter:
    def generate_report(self, **kwargs):
        raise TypeError("unsupported format string passed to NoneType.__format__")


class _UnloadableReporter:
    def __init__(self, exc: Exception):
        self.exc = exc

    def generate_report(self, **kwargs):
        raise self.exc


def test_vlm_programming_error_propagates():
    pipeline = SpecialistPipeline(report_generator=_BuggyReporter(), device="cpu", enable_vlm=True)
    with pytest.raises(TypeError, match="NoneType"):
        pipeline.run(clinical_data={"iop": 20.0})


@pytest.mark.parametrize(
    "exc",
    [ImportError("no accelerate"), OSError("gated repo"), RuntimeError("Failed to load")],
    ids=["ImportError", "OSError", "RuntimeError"],
)
def test_vlm_load_failure_degrades_to_warning(exc):
    pipeline = SpecialistPipeline(
        report_generator=_UnloadableReporter(exc), device="cpu", enable_vlm=True,
    )
    with pytest.warns(RuntimeWarning, match="VLM report not generated"):
        output = pipeline.run(clinical_data={"iop": 20.0})
    assert output.clinical_report is None


# ---------------------------------------------------------------------------
# Zero-filled forecaster inputs must be reported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("clinical", "vf", "imputed"),
    [
        (False, False, ["clinical", "vf"]),
        (True, False, ["vf"]),
        (False, True, ["clinical"]),
        (True, True, []),
    ],
    ids=["none", "clinical-only", "vf-only", "both"],
)
def test_zero_filled_forecaster_inputs_are_reported(
    monkeypatch, tmp_path, recwarn, clinical, vf, imputed,
):
    config = copy.deepcopy(TINY_CONFIG)
    # VFGraphNet readout is 2 * output_dim (= 16 in TINY_CONFIG).
    config["progression"]["forecaster"].update(clinical_dim=3, vf_dim=16)
    patient = _make_patient(tmp_path, clinical=clinical, vf=vf)

    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, config), patient, embed_dim=16,
    )

    assert record["progression_probability"] is not None
    assert record["imputed_modalities"] == imputed
    zero_fill_warnings = [w for w in recwarn if "zero-filled" in str(w.message)]
    if imputed:
        assert len(zero_fill_warnings) == 1
        assert issubclass(zero_fill_warnings[0].category, RuntimeWarning)
        for name in imputed:
            assert name in str(zero_fill_warnings[0].message)
    else:
        assert not zero_fill_warnings


def test_forecaster_without_aux_slots_imputes_nothing(monkeypatch, tmp_path):
    # Slots the forecaster was not built with are not "missing".
    config = copy.deepcopy(TINY_CONFIG)
    config["progression"]["forecaster"].update(clinical_dim=0, vf_dim=0)
    patient = _make_patient(tmp_path)
    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, config), patient, embed_dim=16,
    )
    assert record["imputed_modalities"] == []


@pytest.mark.parametrize(
    ("clinical_json", "imputed"),
    [
        ({}, ["clinical.iop", "clinical.age", "clinical.sex"]),
        ({"age": 70}, ["clinical.iop", "clinical.sex"]),
        # Keys are lower-case: "IOP" is not read, so all three are zero-filled.
        ({"IOP": 22}, ["clinical.iop", "clinical.age", "clinical.sex"]),
        ({"iop": None, "age": 70, "sex": 1}, ["clinical.iop"]),
        ({"iop": 28.0, "age": 70, "sex": 1}, []),
    ],
    ids=["empty", "age-only", "wrong-case", "null-iop", "complete"],
)
def test_zero_filled_clinical_fields_are_reported(
    monkeypatch, tmp_path, recwarn, clinical_json, imputed,
):
    # A clinical.json that exists but lacks a field zero-fills that field just
    # like a missing file zero-fills all three: it must be reported too.
    config = copy.deepcopy(TINY_CONFIG)  # forecaster clinical_dim=3, vf_dim=0
    patient = _make_patient(tmp_path)
    with open(patient / "clinical.json", "w") as f:
        json.dump(clinical_json, f)

    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, config), patient, embed_dim=16,
    )

    assert record["progression_probability"] is not None
    assert record["imputed_modalities"] == imputed
    zero_fill_warnings = [w for w in recwarn if "zero-filled" in str(w.message)]
    if imputed:
        assert len(zero_fill_warnings) == 1
        for name in imputed:
            assert name in str(zero_fill_warnings[0].message)
    else:
        assert not zero_fill_warnings


def test_zero_filled_clinical_fields_fed_to_fusion_are_reported(monkeypatch, tmp_path):
    # Without a forecaster clinical slot, the fusion module still consumes the
    # clinical vector, so its zero-filled fields are reported as well.
    config = copy.deepcopy(TINY_CONFIG)
    config["progression"]["forecaster"].update(clinical_dim=0)
    patient = _make_patient(tmp_path)
    with open(patient / "clinical.json", "w") as f:
        json.dump({"age": 70}, f)

    record = _run_cli(
        monkeypatch, tmp_path, _write_config(tmp_path, config), patient, embed_dim=16,
    )
    assert record["imputed_modalities"] == ["clinical.iop", "clinical.sex"]


def test_non_numeric_clinical_field_names_the_field():
    pipeline = SpecialistPipeline(device="cpu")
    with pytest.raises(ValueError, match="clinical.sex"):
        pipeline._clinical_vector({"iop": 20.0, "age": 61, "sex": "female"})


def test_checkpoint_trained_on_random_encoder_warns(tmp_path, capsys):
    # A --allow-random-encoder smoke checkpoint loads, but must say so.
    from src.models.progression.evidential_forecaster import EvidentialForecaster

    forecaster = EvidentialForecaster(temporal_dim=8, clinical_dim=3, hidden_dim=8)
    path = tmp_path / "best.pth"
    torch.save({"forecaster": forecaster.state_dict(), "encoder_pretrained": False}, path)
    run_specialist._load_weights(forecaster, str(path), ["forecaster"], "forecaster")
    assert "RANDOMLY INITIALISED DINOv2" in capsys.readouterr().err

    torch.save({"forecaster": forecaster.state_dict(), "encoder_pretrained": True}, path)
    run_specialist._load_weights(forecaster, str(path), ["forecaster"], "forecaster")
    assert "RANDOMLY" not in capsys.readouterr().err
