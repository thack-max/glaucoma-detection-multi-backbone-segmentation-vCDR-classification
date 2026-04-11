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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

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
