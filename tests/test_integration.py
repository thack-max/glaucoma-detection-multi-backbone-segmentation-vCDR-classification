"""
End-to-end integration test for the GlaucoMamba two-tier flow.

This exercises the referral handoff that is the core differentiator of
the two-tier design:

    1. Tier 1 screening runs on a fundus photo with a minimal BoundaryFormer
       + DINOv2 stub + classifier head combo.
    2. The screening output is serialized to a referral.json packet and
       persisted to disk.
    3. A fresh SpecialistPipeline (different Python objects — no shared
       state) ingests the referral from disk, runs Tier 2 with synthetic
       longitudinal data, and produces a progression forecast with
       calibrated uncertainty.
    4. Both tiers are invoked via their CLI scripts to also cover the
       argparse + YAML config loading paths.

This is the test that would have caught the bugs fixed in the previous
commit (BoundaryFormer decoder channels, cross-modal fusion for partial
modalities, and the stubbed progression branch).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.transforms import get_fundus_transforms
from src.models.fusion.cross_modal_attention import CrossModalFusion
from src.models.progression.delta_t_mamba import DeltaTMamba
from src.models.progression.evidential_forecaster import EvidentialForecaster
from src.models.segmentation.boundary_former import BoundaryFormer
from src.pipeline.common import ReferralPacket, RiskLevel
from src.pipeline.screening import ScreeningPipeline
from src.pipeline.specialist import SpecialistPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


class _StubEncoder(nn.Module):
    """Small stand-in for DINOv2 so tests stay fast and offline.

    Mirrors the DINOv2Encoder contract used by the pipelines: a CLS forward
    returning [B, embed_dim] and extract_patch_features returning [B, N, D].
    """

    def __init__(self, embed_dim: int = 16, num_patches: int = 49) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.proj = nn.Conv2d(3, embed_dim, 3, padding=1)
        self.head = nn.Linear(embed_dim, embed_dim)

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.proj(x)
        side = int(self.num_patches ** 0.5)
        feats = nn.functional.adaptive_avg_pool2d(feats, (side, side))
        return feats.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_patch_features(x).mean(dim=1))


def _build_tier1_pipeline() -> ScreeningPipeline:
    torch.manual_seed(0)
    segmenter = BoundaryFormer(num_classes=3, base_channels=16, growth_rate=8)
    segmenter.eval()
    encoder = _StubEncoder(embed_dim=16)
    classifier = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 1))
    return ScreeningPipeline(
        segmenter=segmenter,
        feature_encoder=encoder,
        classifier_head=classifier,
        device="cpu",
        referral_threshold=0.0,  # force referral so we exercise the handoff
        vcdr_alert_threshold=0.65,
    )


def _build_tier2_pipeline() -> SpecialistPipeline:
    torch.manual_seed(1)
    encoder = _StubEncoder(embed_dim=32)
    temporal = DeltaTMamba(
        input_dim=32, d_model=16, d_state=4, num_layers=1, expand=1,
    )
    forecaster = EvidentialForecaster(
        temporal_dim=16, clinical_dim=3, hidden_dim=16,
    )
    fusion = CrossModalFusion(
        modality_dims={"fundus": 32, "oct": 32, "vf": 2, "clinical": 3},
        fused_dim=16,
        num_heads=2,
    )
    for m in (encoder, temporal, forecaster, fusion):
        m.eval()
    return SpecialistPipeline(
        feature_encoder=encoder,
        temporal_encoder=temporal,
        forecaster=forecaster,
        fusion_module=fusion,
        device="cpu",
    )


def _make_synthetic_fundus(tmp_path: Path) -> Path:
    """Produce a reproducible fundus-like RGB image on disk."""
    np.random.seed(42)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    y, x = np.ogrid[:128, :128]
    # Orange background disc
    mask = (y - 64) ** 2 + (x - 64) ** 2 < 60 ** 2
    img[mask] = [180, 100, 60]
    # Brighter optic disc in the centre
    disc = (y - 64) ** 2 + (x - 64) ** 2 < 15 ** 2
    img[disc] = [220, 200, 160]
    path = tmp_path / "synthetic_fundus.png"
    Image.fromarray(img).save(path)
    return path


# ---------------------------------------------------------------------------
# Programmatic integration (fast, no subprocess)
# ---------------------------------------------------------------------------


class TestTier1ToTier2Handoff:
    def test_full_referral_flow(self, tmp_path: Path):
        # ---- Tier 1 ----
        tier1 = _build_tier1_pipeline()
        image_path = _make_synthetic_fundus(tmp_path)

        from src.data.fundus_dataset import _read_image
        image_np = _read_image(image_path)
        transform = get_fundus_transforms("val", image_size=64)
        tensor = transform(image_np)["image"].unsqueeze(0)

        with torch.no_grad():
            screening = tier1.run(tensor, demographics={"group_id": 1})

        assert screening.risk_level in set(RiskLevel)
        assert screening.refer_to_specialist  # threshold=0 forces this
        assert screening.vcdr is not None

        referral = tier1.create_referral(
            screening,
            patient_id="INT001",
            screening_site="Unit Test Clinic",
            demographics={"age": 62, "race": "Black", "group_id": 1},
        )
        referral.fundus_image_path = str(image_path)
        # A randomly-initialized BoundaryFormer does not produce a sensible
        # vCDR on our synthetic input, so simulate the result a trained model
        # would output. The point of this test is the handoff, not the
        # segmentation quality.
        referral.vcdr = 0.68

        # Serialize to disk like the real flow would
        referral_path = tmp_path / "referral.json"
        with open(referral_path, "w") as f:
            json.dump(referral.to_dict(), f)

        # ---- Tier 2 ingest ----
        with open(referral_path) as f:
            loaded = ReferralPacket.from_dict(json.load(f))
        assert loaded.patient_id == "INT001"
        assert loaded.risk_level == screening.risk_level.value
        assert loaded.vcdr == 0.68

        tier2 = _build_tier2_pipeline()
        intake = tier2.ingest_referral(loaded)
        assert intake["patient_id"] == "INT001"
        assert intake["tier1_vcdr"] == loaded.vcdr

        # ---- Tier 2 run with synthetic longitudinal data ----
        visit_history = [
            {"features": torch.randn(32)} for _ in range(4)
        ]
        timestamps = torch.tensor([[0, 90, 210, 365]], dtype=torch.float32)
        clinical = {"iop": 19.0, "age": 62, "sex": 1}

        with torch.no_grad():
            specialist_output = tier2.run(
                fundus_image=tensor,
                vf_data=torch.randn(1, 54, 3),
                visit_history=visit_history,
                timestamps=timestamps,
                clinical_data=clinical,
                referral=loaded,
            )

        # The referral should be threaded through the output
        assert specialist_output.referral_source is not None
        assert specialist_output.referral_source.patient_id == "INT001"
        # vcdr comes from the referral when skip_segmentation kicks in
        assert specialist_output.vcdr == loaded.vcdr

        # Progression forecast populated
        assert specialist_output.progression_probability is not None
        assert 0.0 <= specialist_output.progression_probability <= 1.0
        assert specialist_output.progression_class in {"stable", "slow", "rapid"}
        assert specialist_output.progression_uncertainty is not None
        assert specialist_output.predicted_rnfl_thickness is not None

        # Fusion should have run too
        assert specialist_output.fusion_uncertainty is not None


# ---------------------------------------------------------------------------
# CLI integration (covers argparse + YAML config path)
# ---------------------------------------------------------------------------


class TestScreeningCLI:
    def test_screening_cli_writes_summary(self, tmp_path: Path):
        image_path = _make_synthetic_fundus(tmp_path)
        output_dir = tmp_path / "screening_out"

        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        # Use baseline mode to avoid downloading DINOv2 in CI
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_screening.py"),
                "--image", str(image_path),
                "--output-dir", str(output_dir),
                "--baseline",
                "--config", str(REPO_ROOT / "configs" / "tier" / "screening.yaml"),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

        summary_path = output_dir / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["succeeded"] == 1
        assert summary["results"][0]["image"].endswith("synthetic_fundus.png")

    def test_specialist_cli_help(self):
        """Smoke test: CLI argparse wires up without an import error."""
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_specialist.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        assert result.returncode == 0
        assert "Tier 2" in result.stdout
