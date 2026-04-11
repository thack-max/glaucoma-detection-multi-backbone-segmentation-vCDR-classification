"""
End-to-end smoke tests for the Tier 1 training + inference pipeline.

These tests are the first to exercise the *full* chain:

    scripts/make_synthetic_refuge.py
        -> scripts/train_segmentation.py
            -> checkpoint on disk
                -> scripts/run_screening.py
                    -> summary.json

The existing module-level unit tests (test_models.py, test_pipeline.py)
stub out everything above the nn.Module level, so they can't catch bugs
in:
  - the training script's argparse / YAML config wiring
  - checkpoint serialization format (``state["model"]`` key, etc.)
  - DataLoader compatibility with the FundusDataset manifest schema
  - ``run_screening.py`` loading the checkpoint that was just produced

These tests run for real on the synthetic REFUGE fixture from
``scripts/make_synthetic_refuge.py`` so there are no external
dependencies — a CI runner with only CPU torch can execute them end
to end.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.make_synthetic_longitudinal import generate as generate_longitudinal
from scripts.make_synthetic_refuge import generate
from src.models.segmentation.boundary_former import BoundaryFormer


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_refuge_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small synthetic REFUGE-layout dataset shared across the tests
    in this module — generated once so the fixture cost is paid once."""
    root = tmp_path_factory.mktemp("synthetic_refuge")
    generate(
        output=root,
        per_split=6,  # 3 normal + 3 glaucoma per split
        image_size=64,
        splits=["train", "val", "test"],
        seed=0,
    )
    return root


@pytest.fixture(scope="module")
def trained_segmenter_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_refuge_root: Path,
) -> Path:
    """Run the real ``train_segmentation.py`` CLI for one epoch against
    the synthetic dataset and return the path to the saved checkpoint.

    Shared across tests in this module so we only pay the 1-epoch
    training cost once.
    """
    output_dir = tmp_path_factory.mktemp("seg_ckpt")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_segmentation.py"),
            "--data-root", str(synthetic_refuge_root),
            "--epochs", "1",
            "--batch-size", "2",
            "--image-size", "64",
            "--base-channels", "8",
            "--growth-rate", "4",
            "--lr", "1e-3",
            "--device", "cpu",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"train_segmentation.py failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    checkpoint = output_dir / "best.pth"
    assert checkpoint.exists(), (
        f"Expected {checkpoint} to exist. stdout={result.stdout}"
    )
    return checkpoint


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainSegmentationSmoke:
    def test_checkpoint_is_loadable(
        self, trained_segmenter_checkpoint: Path,
    ) -> None:
        """The checkpoint must be a dict with ``model`` / ``epoch`` /
        ``metrics`` keys that ``run_screening.py`` expects."""
        state = torch.load(trained_segmenter_checkpoint, map_location="cpu")
        assert isinstance(state, dict)
        assert "model" in state
        assert "epoch" in state
        assert "metrics" in state
        assert set(state["metrics"].keys()) >= {
            "dice_od", "dice_oc", "iou_od", "iou_oc",
        }

    def test_checkpoint_loads_into_matching_model(
        self, trained_segmenter_checkpoint: Path,
    ) -> None:
        """A freshly built BoundaryFormer with the same hyper-parameters
        must accept the checkpoint via ``load_state_dict`` without
        unexpected or missing keys."""
        state = torch.load(trained_segmenter_checkpoint, map_location="cpu")
        model = BoundaryFormer(num_classes=3, base_channels=8, growth_rate=4)
        missing, unexpected = model.load_state_dict(state["model"], strict=True)
        assert missing == []
        assert unexpected == []

    def test_trained_segmenter_runs_inference(
        self, trained_segmenter_checkpoint: Path,
    ) -> None:
        """Load the checkpoint, run a forward pass on a dummy image,
        confirm we get back well-formed logits + boundary output. This
        is what ``run_screening.py`` does per image."""
        state = torch.load(trained_segmenter_checkpoint, map_location="cpu")
        model = BoundaryFormer(num_classes=3, base_channels=8, growth_rate=4)
        model.load_state_dict(state["model"])
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (1, 3, 64, 64)
        assert "boundary" in out
        assert out["boundary"].shape == (1, 1, 64, 64)


class TestRunScreeningCliWithTrainedCheckpoint:
    """Covers the full ``train → save → load → infer → summary`` loop
    through the actual ``run_screening.py`` CLI, in baseline mode so we
    don't need DINOv2 to be downloadable in CI."""

    def test_cli_consumes_trained_checkpoint(
        self,
        tmp_path: Path,
        synthetic_refuge_root: Path,
        trained_segmenter_checkpoint: Path,
    ) -> None:
        # Write a screening config that points at the trained checkpoint
        # and uses the same small architecture. run_screening.py loads
        # YAML directly so this is the real integration surface.
        config_path = tmp_path / "screening.yaml"
        config_path.write_text(
            "tier: screening\n"
            "pipeline:\n"
            "  device: cpu\n"
            "  baseline_mode: true\n"
            "  referral_threshold: 0.5\n"
            "  vcdr_alert_threshold: 0.65\n"
            "segmentation:\n"
            "  model: boundary_former\n"
            "  num_classes: 3\n"
            "  base_channels: 8\n"
            "  growth_rate: 4\n"
            f"  checkpoint: {trained_segmenter_checkpoint}\n"
            "  image_size: 64\n"
            "feature_extraction:\n"
            "  model: dinov2_vitb14\n"
            "  variant: dinov2_vitb14\n"
            "  embed_dim: 768\n"
            "  freeze_encoder: true\n"
            "  image_size: 64\n"
            "classifier:\n"
            "  type: linear\n"
            "  input_dim: 768\n"
            "output:\n"
            "  output_dir: outputs/screening/\n"
        )

        # Pick the first real image from the synthetic dataset so
        # ``_read_image`` handles a PNG the same way it would in prod.
        image_dir = synthetic_refuge_root / "images"
        images = sorted(image_dir.glob("*.png"))
        assert images, "synthetic fixture did not produce any images"
        test_image = images[0]

        output_dir = tmp_path / "screening_out"
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_screening.py"),
                "--image", str(test_image),
                "--output-dir", str(output_dir),
                "--baseline",
                "--config", str(config_path),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"run_screening.py failed:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        # Verify the checkpoint path was actually exercised, not skipped.
        assert "Loaded segmenter weights" in result.stdout, result.stdout

        summary_path = output_dir / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["succeeded"] == 1
        rec = summary["results"][0]
        # Baseline mode emits vcdr + risk_level on every record.
        assert "risk_score" in rec
        assert "vcdr" in rec
        assert rec["risk_level"] in {"low", "moderate", "high"}


@pytest.fixture(scope="module")
def synthetic_longitudinal_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialise a tiny longitudinal dataset under a module-scoped tmp
    path and return the root. Shared across the Tier 2 smoke tests."""
    root = tmp_path_factory.mktemp("synthetic_longitudinal")
    generate_longitudinal(
        output=root,
        num_patients=4,
        num_visits=4,
        fundus_size=32,
        oct_shape=(4, 8, 8),
        seed=0,
    )
    return root


class TestSyntheticLongitudinalFixture:
    """Basic contract checks for make_synthetic_longitudinal.py."""

    def test_manifest_matches_multimodal_schema(
        self, synthetic_longitudinal_root: Path,
    ) -> None:
        manifest = synthetic_longitudinal_root / "manifest.csv"
        assert manifest.exists()
        with open(manifest) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        expected_columns = {
            "patient_id", "visit_index", "timestamp_days",
            "fundus_path", "oct_path", "vf_path",
            "rnfl_thickness", "iop", "age", "sex",
            "group_id", "progression_label",
        }
        assert rows
        assert expected_columns.issubset(set(rows[0].keys()))

        # Every referenced artefact must actually exist.
        for row in rows:
            assert (synthetic_longitudinal_root / row["fundus_path"]).is_file()
            assert (synthetic_longitudinal_root / row["oct_path"]).is_file()
            assert (synthetic_longitudinal_root / row["vf_path"]).is_file()

    def test_multimodal_dataset_loads_fixture(
        self, synthetic_longitudinal_root: Path,
    ) -> None:
        """The fixture must round-trip through MultimodalGlaucomaDataset
        and produce tensors of the shape the training script expects."""
        from src.data.multimodal_dataset import MultimodalGlaucomaDataset

        ds = MultimodalGlaucomaDataset(
            synthetic_longitudinal_root,
            observation_window=4,
        )
        assert len(ds) == 4  # num_patients

        sample = ds[0]
        # fundus: [T, 3, H, W]
        assert sample["fundus"].shape == (4, 3, 32, 32)
        # oct: [T, 1, D, H, W]
        assert sample["oct"].shape == (4, 1, 4, 8, 8)
        # vf: [T, 54, 3]
        assert sample["vf"].shape == (4, 54, 3)
        # timestamps: [T]
        assert sample["timestamps"].shape == (4,)
        # Timestamps must be strictly monotone within a patient
        diffs = sample["timestamps"][1:] - sample["timestamps"][:-1]
        assert (diffs > 0).all()
        # Progression target present
        assert "progression_label" in sample
        assert "rnfl_target" in sample

    def test_progression_labels_balance_across_patients(
        self, synthetic_longitudinal_root: Path,
    ) -> None:
        with open(synthetic_longitudinal_root / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        labels_per_patient = {}
        for row in rows:
            labels_per_patient[row["patient_id"]] = int(row["progression_label"])
        labels = list(labels_per_patient.values())
        assert labels.count(0) == labels.count(1), (
            f"expected balanced progression labels, got {labels}"
        )

    def test_progressing_patients_have_lower_final_rnfl(
        self, synthetic_longitudinal_root: Path,
    ) -> None:
        """Essential for the upcoming progression training smoke test to
        actually have signal: the regression target must differ between
        progressing and stable patients."""
        with open(synthetic_longitudinal_root / "manifest.csv") as f:
            rows = list(csv.DictReader(f))

        by_patient: dict[str, list[tuple[float, float, int]]] = {}
        for row in rows:
            by_patient.setdefault(row["patient_id"], []).append(
                (
                    float(row["timestamp_days"]),
                    float(row["rnfl_thickness"]),
                    int(row["progression_label"]),
                ),
            )

        final_rnfl_progressing: list[float] = []
        final_rnfl_stable: list[float] = []
        for visits in by_patient.values():
            visits.sort(key=lambda v: v[0])
            final_rnfl = visits[-1][1]
            label = visits[-1][2]
            (final_rnfl_progressing if label == 1 else final_rnfl_stable).append(
                final_rnfl,
            )

        assert final_rnfl_progressing and final_rnfl_stable
        mean_prog = sum(final_rnfl_progressing) / len(final_rnfl_progressing)
        mean_stable = sum(final_rnfl_stable) / len(final_rnfl_stable)
        # Stable patients should end with a higher RNFL than progressing
        # patients (at least 1 µm separation). If this fails, the
        # synthetic generator's progression bias is no longer meaningful.
        assert mean_stable - mean_prog > 1.0, (
            f"stable mean {mean_stable:.2f} should exceed progressing "
            f"mean {mean_prog:.2f} by at least 1 µm"
        )


class TestTrainProgressionInProcess:
    """In-process smoke test for ``train_progression.py``.

    Like the classifier test, we monkey-patch ``DINOv2Encoder`` in the
    script module to keep the runtime in the seconds range. The point
    is to exercise the training loop + checkpoint path, not to verify
    DINOv2 features.
    """

    def test_progression_training_loop_produces_checkpoint(
        self,
        tmp_path: Path,
        synthetic_longitudinal_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch.nn as nn

        import scripts.train_progression as tp

        class _StubEncoder(nn.Module):
            """Deterministic stand-in for DINOv2Encoder.

            Emits 16-dim features per image. Training script flattens
            ``[B, T]`` to ``[B*T]`` and calls ``encoder(...)`` under
            ``no_grad``, so this just needs a fast forward.
            """

            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                self.embed_dim = 16
                self.proj = nn.Conv2d(3, self.embed_dim, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return nn.functional.adaptive_avg_pool2d(
                    self.proj(x), 1,
                ).flatten(1)

            def eval(self) -> "_StubEncoder":  # type: ignore[override]
                return super().eval()  # type: ignore[return-value]

            def to(self, *args, **kwargs) -> "_StubEncoder":  # type: ignore[override]
                return super().to(*args, **kwargs)  # type: ignore[return-value]

        monkeypatch.setattr(tp, "DINOv2Encoder", _StubEncoder)

        output_dir = tmp_path / "progression_ckpt"
        monkeypatch.setattr(
            sys, "argv",
            [
                "train_progression.py",
                "--data-root", str(synthetic_longitudinal_root),
                "--epochs", "1",
                "--batch-size", "2",
                "--lr", "1e-3",
                "--d-model", "16",
                "--d-state", "4",
                "--mamba-layers", "1",
                "--observation-window", "4",
                "--clinical-dim", "3",
                "--num-classes", "2",
                "--device", "cpu",
                "--output-dir", str(output_dir),
            ],
        )

        tp.main()

        checkpoint = output_dir / "best.pth"
        assert checkpoint.exists(), "train_progression.py did not emit best.pth"
        state = torch.load(checkpoint, map_location="cpu")
        assert "mamba" in state
        assert "forecaster" in state
        # The args dict is used to reconstruct the model at inference.
        assert state["args"]["d_model"] == 16
        assert state["args"]["observation_window"] == 4

    def test_encode_visits_handles_synthetic_fundus_shape(
        self,
        synthetic_longitudinal_root: Path,
    ) -> None:
        """The encode_visits helper flattens [B, T, 3, H, W] -> [B*T, ...]
        before calling the frozen encoder. Regression test that a real
        batch from the synthetic dataset runs through it cleanly."""
        from torch.utils.data import DataLoader

        from src.data.multimodal_dataset import MultimodalGlaucomaDataset
        from scripts.train_progression import encode_visits

        class _StubEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 8
                self.proj = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x):
                return torch.nn.functional.adaptive_avg_pool2d(
                    self.proj(x), 1,
                ).flatten(1)

        encoder = _StubEncoder()
        ds = MultimodalGlaucomaDataset(
            synthetic_longitudinal_root, observation_window=4,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        fundus = batch["fundus"]  # [B, T, 3, H, W]
        assert fundus.shape == (2, 4, 3, 32, 32)
        feats = encode_visits(encoder, fundus, device=torch.device("cpu"))
        assert feats.shape == (2, 4, 8)


class TestTrainScreeningClassifierInProcess:
    """In-process version of the classifier training path.

    We do NOT subprocess ``train_screening_classifier.py`` because that
    would download the real DINOv2 weights (slow) and the purpose of
    this smoke test is to verify the script's training loop, not the
    DINOv2 feature quality. The real DINOv2 path is already covered by
    ``test_models.py::TestDINOv2Encoder``.
    """

    def test_classifier_training_loop_produces_checkpoint(
        self,
        tmp_path: Path,
        synthetic_refuge_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch.nn as nn

        import scripts.train_screening_classifier as tsc

        class _StubEncoder(nn.Module):
            """Deterministic stand-in for DINOv2Encoder.

            Mirrors the two attributes the training script touches:
            ``embed_dim`` and ``forward(images) -> [B, embed_dim]``.
            """

            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                self.embed_dim = 16
                self.proj = nn.Conv2d(3, self.embed_dim, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return nn.functional.adaptive_avg_pool2d(
                    self.proj(x), 1,
                ).flatten(1)

            def eval(self) -> "_StubEncoder":  # type: ignore[override]
                return super().eval()  # type: ignore[return-value]

            def to(self, *args, **kwargs) -> "_StubEncoder":  # type: ignore[override]
                return super().to(*args, **kwargs)  # type: ignore[return-value]

        monkeypatch.setattr(tsc, "DINOv2Encoder", _StubEncoder)

        output = tmp_path / "classifier.pth"
        monkeypatch.setattr(
            sys, "argv",
            [
                "train_screening_classifier.py",
                "--data-root", str(synthetic_refuge_root),
                "--epochs", "1",
                "--batch-size", "2",
                "--lr", "1e-2",
                "--image-size", "64",
                "--device", "cpu",
                "--output", str(output),
            ],
        )

        tsc.main()

        assert output.exists(), "classifier checkpoint was not written"
        state = torch.load(output, map_location="cpu")
        assert "classifier" in state
        # The classifier head should be a LayerNorm + Dropout + Linear
        # chain whose Linear weight maps embed_dim -> 1.
        sd = state["classifier"]
        # Find the Linear layer and assert its shape matches the stub's
        # embed_dim (16). This catches silent dim drift in the script.
        linear_weight_keys = [k for k in sd if k.endswith("weight")
                              and sd[k].dim() == 2]
        assert linear_weight_keys
        linear_w = sd[linear_weight_keys[-1]]
        assert linear_w.shape == (1, 16), (
            f"classifier head expected Linear(16, 1), got {tuple(linear_w.shape)}"
        )
