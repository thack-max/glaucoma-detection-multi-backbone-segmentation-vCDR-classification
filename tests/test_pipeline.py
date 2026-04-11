"""Tests for the two-tier pipeline: common utilities, screening, and specialist."""

import torch
import pytest

from src.pipeline.common import (
    ReferralPacket,
    RiskLevel,
    classify_risk,
    compute_vcdr,
)
from src.pipeline.screening import ScreeningOutput, ScreeningPipeline
from src.pipeline.specialist import SpecialistOutput, SpecialistPipeline


# --- common.py tests ---


class TestComputeVCDR:
    def test_normal_ratio(self):
        disc = torch.zeros(100, 100)
        cup = torch.zeros(100, 100)
        disc[20:80, 20:80] = 1.0  # height = 60
        cup[35:65, 35:65] = 1.0   # height = 30
        vcdr = compute_vcdr(disc, cup)
        assert 0.49 < vcdr < 0.51  # ~0.5

    def test_empty_masks(self):
        disc = torch.zeros(100, 100)
        cup = torch.zeros(100, 100)
        assert compute_vcdr(disc, cup) == 0.0

    def test_no_cup(self):
        disc = torch.zeros(100, 100)
        disc[20:80, 20:80] = 1.0
        cup = torch.zeros(100, 100)
        assert compute_vcdr(disc, cup) == 0.0

    def test_large_cup(self):
        disc = torch.zeros(100, 100)
        cup = torch.zeros(100, 100)
        disc[10:90, 10:90] = 1.0  # height = 80
        cup[15:85, 15:85] = 1.0   # height = 70
        vcdr = compute_vcdr(disc, cup)
        assert vcdr > 0.8


class TestClassifyRisk:
    def test_low_risk(self):
        assert classify_risk(0.1, 0.3) == RiskLevel.LOW

    def test_moderate_risk(self):
        assert classify_risk(0.5, 0.4) == RiskLevel.MODERATE

    def test_high_risk_score(self):
        assert classify_risk(0.8, 0.4) == RiskLevel.HIGH

    def test_high_risk_vcdr_escalation(self):
        # Even moderate score escalates to HIGH if vCDR is very high
        assert classify_risk(0.4, 0.7) == RiskLevel.HIGH

    def test_custom_thresholds(self):
        assert classify_risk(0.2, 0.3, low_threshold=0.1, high_threshold=0.5) == RiskLevel.MODERATE


class TestReferralPacket:
    def test_roundtrip_serialization(self):
        packet = ReferralPacket(
            patient_id="P001",
            screening_site="Rural Clinic A",
            risk_score=0.73,
            risk_level="high",
            vcdr=0.62,
            referral_reason="Risk score 0.73 >= 0.5",
            urgency="urgent",
            demographics={"age": 55, "race": "Black"},
        )
        d = packet.to_dict()
        restored = ReferralPacket.from_dict(d)

        assert restored.patient_id == "P001"
        assert restored.risk_score == 0.73
        assert restored.vcdr == 0.62
        assert restored.urgency == "urgent"
        assert restored.demographics == {"age": 55, "race": "Black"}

    def test_from_dict_ignores_unknown_keys(self):
        d = {"patient_id": "P002", "unknown_field": "should_be_ignored"}
        packet = ReferralPacket.from_dict(d)
        assert packet.patient_id == "P002"

    def test_default_values(self):
        packet = ReferralPacket()
        assert packet.patient_id == ""
        assert packet.risk_score == 0.0
        assert packet.urgency == "routine"


# --- screening.py tests ---


class TestScreeningPipeline:
    def test_construction_no_models(self):
        pipeline = ScreeningPipeline(device="cpu")
        assert pipeline.device == "cpu"
        assert pipeline.baseline_mode is False

    def test_baseline_mode(self):
        pipeline = ScreeningPipeline(device="cpu", baseline_mode=True)
        assert pipeline.baseline_mode is True

    def test_create_referral(self):
        pipeline = ScreeningPipeline(device="cpu")
        output = ScreeningOutput(
            risk_score=0.8,
            risk_level=RiskLevel.HIGH,
            vcdr=0.65,
            refer_to_specialist=True,
            referral_reason="Risk score 0.80 >= 0.5",
            urgency="urgent",
        )
        referral = pipeline.create_referral(
            output,
            patient_id="P001",
            screening_site="Clinic A",
            demographics={"age": 60},
        )
        assert referral.patient_id == "P001"
        assert referral.risk_score == 0.8
        assert referral.risk_level == "high"
        assert referral.urgency == "urgent"


# --- specialist.py tests ---


class TestSpecialistPipeline:
    def test_construction_no_models(self):
        pipeline = SpecialistPipeline(device="cpu")
        assert pipeline.enable_vlm is False

    def test_vlm_disabled_by_default(self):
        pipeline = SpecialistPipeline(device="cpu")
        assert pipeline.enable_vlm is False

    def test_vlm_explicit_enable(self):
        pipeline = SpecialistPipeline(device="cpu", enable_vlm=True)
        assert pipeline.enable_vlm is True

    def test_ingest_referral(self):
        pipeline = SpecialistPipeline(device="cpu")
        referral = ReferralPacket(
            patient_id="P001",
            screening_site="Rural Clinic",
            risk_score=0.73,
            risk_level="high",
            vcdr=0.62,
            disc_mask_path="/data/P001/disc.png",
            urgency="urgent",
        )
        intake = pipeline.ingest_referral(referral)

        assert intake["patient_id"] == "P001"
        assert intake["tier1_risk_score"] == 0.73
        assert intake["has_fundus_segmentation"] is True
        assert intake["needs_oct"] is True
        assert intake["needs_longitudinal"] is True

    def test_progression_forecasting_end_to_end(self):
        """Full Delta-t Mamba -> Evidential forecaster integration."""
        from src.models.progression.delta_t_mamba import DeltaTMamba
        from src.models.progression.evidential_forecaster import EvidentialForecaster

        temporal = DeltaTMamba(
            input_dim=32, d_model=16, d_state=4, num_layers=1, expand=1,
        )
        forecaster = EvidentialForecaster(
            temporal_dim=16, clinical_dim=3, hidden_dim=16, num_classes=2,
        )
        pipeline = SpecialistPipeline(
            temporal_encoder=temporal,
            forecaster=forecaster,
            device="cpu",
        )
        pipeline.temporal_encoder.eval()
        pipeline.forecaster.eval()

        visit_history = [
            {"features": torch.randn(32)} for _ in range(4)
        ]
        timestamps = torch.tensor([[0, 90, 180, 365]], dtype=torch.float32)
        clinical = {"iop": 18.0, "age": 67.0, "sex": 1.0}

        with torch.no_grad():
            output = pipeline.run(
                visit_history=visit_history,
                timestamps=timestamps,
                clinical_data=clinical,
            )

        assert output.progression_probability is not None
        assert 0.0 <= output.progression_probability <= 1.0
        assert output.progression_class in {"stable", "slow", "rapid"}
        assert output.progression_uncertainty is not None
        assert output.predicted_rnfl_thickness is not None

    def test_progression_without_forecaster(self):
        """Missing forecaster should not crash the pipeline."""
        pipeline = SpecialistPipeline(device="cpu")
        output = pipeline.run(
            visit_history=[{"features": torch.randn(32)}],
            timestamps=torch.tensor([[0.0]]),
        )
        assert output.progression_probability is None

    def test_vf_embedding_is_populated_from_encode_graph(self):
        """Regression: the pipeline must expose the graph-level readout
        embedding (``[B, readout_dim]``) as ``vf_embedding`` in addition to
        the classification logits. Downstream fusion + forecaster configs
        declare ``vf`` / ``vf_dim`` as the readout, not the logits."""
        from src.models.progression.vf_graph_net import VFGraphNet

        vf_net = VFGraphNet(
            node_features=3, hidden_dim=32, num_gin_layers=2, output_dim=32,
        )
        vf_net.eval()
        pipeline = SpecialistPipeline(vf_graph_net=vf_net, device="cpu")

        vf = torch.randn(1, 54, 3)
        with torch.no_grad():
            out = pipeline.run(vf_data=vf)

        # Classification logits (clinical readout)
        assert out.vf_classification is not None
        assert out.vf_classification.shape == (1, 2)
        # Graph readout embedding (feature for fusion + forecaster) —
        # readout_dim = output_dim * 2 = 64
        assert out.vf_embedding is not None
        assert out.vf_embedding.shape == (1, 64)

    def test_forecaster_receives_vf_embedding_not_logits(self):
        """Regression for shape-mismatch crash: with a realistic specialist
        config (``vf_dim == VFGraphNet.readout_dim``) the forecaster must
        accept the VF embedding from the pipeline without blowing up."""
        from src.models.progression.delta_t_mamba import DeltaTMamba
        from src.models.progression.evidential_forecaster import (
            EvidentialForecaster,
        )
        from src.models.progression.vf_graph_net import VFGraphNet

        vf_net = VFGraphNet(
            node_features=3, hidden_dim=32, num_gin_layers=2, output_dim=32,
        )
        temporal = DeltaTMamba(
            input_dim=32, d_model=16, d_state=4, num_layers=1, expand=1,
        )
        # vf_dim here matches the graph readout (32 * 2 = 64), NOT num_classes.
        # Under the legacy pipeline wiring the forecaster would be fed the
        # [B, 2] classification logits and the first Linear(80, hidden) in
        # `shared` would raise RuntimeError.
        forecaster = EvidentialForecaster(
            temporal_dim=16,
            clinical_dim=3,
            vf_dim=64,
            hidden_dim=16,
            num_classes=2,
        )
        for m in (vf_net, temporal, forecaster):
            m.eval()

        pipeline = SpecialistPipeline(
            vf_graph_net=vf_net,
            temporal_encoder=temporal,
            forecaster=forecaster,
            device="cpu",
        )

        vf = torch.randn(1, 54, 3)
        visit_history = [{"features": torch.randn(32)} for _ in range(3)]
        timestamps = torch.tensor([[0.0, 90.0, 180.0]])
        clinical = {"iop": 18.0, "age": 67.0, "sex": 1.0}

        with torch.no_grad():
            out = pipeline.run(
                vf_data=vf,
                visit_history=visit_history,
                timestamps=timestamps,
                clinical_data=clinical,
            )

        assert out.progression_probability is not None
        assert 0.0 <= out.progression_probability <= 1.0
        # The embedding path should have been exercised (not the fallback).
        assert out.vf_embedding is not None
        assert out.vf_embedding.shape == (1, 64)

    def test_forecaster_falls_back_on_vf_shape_mismatch(self):
        """If a stub VFGraphNet produces an unexpected embedding shape
        (e.g. legacy configs still passing classification logits), the
        pipeline should fall back to temporal-only forecasting instead of
        aborting. Previously the try/except only caught TypeError."""
        from src.models.progression.delta_t_mamba import DeltaTMamba
        from src.models.progression.evidential_forecaster import (
            EvidentialForecaster,
        )

        class _BadVFNet(torch.nn.Module):
            """Stub that ONLY returns classification logits — no
            ``encode_graph``. The pipeline used to crash the forecaster
            by forwarding these [B, 2] logits into a vf_dim=64 head."""

            def forward(self, x):
                return torch.zeros(x.shape[0], 2)

        temporal = DeltaTMamba(
            input_dim=32, d_model=16, d_state=4, num_layers=1, expand=1,
        )
        forecaster = EvidentialForecaster(
            temporal_dim=16,
            clinical_dim=0,
            vf_dim=64,
            hidden_dim=16,
            num_classes=2,
        )
        for m in (temporal, forecaster):
            m.eval()

        pipeline = SpecialistPipeline(
            vf_graph_net=_BadVFNet(),
            temporal_encoder=temporal,
            forecaster=forecaster,
            device="cpu",
        )

        with torch.no_grad():
            out = pipeline.run(
                vf_data=torch.randn(1, 54, 3),
                visit_history=[{"features": torch.randn(32)} for _ in range(3)],
                timestamps=torch.tensor([[0.0, 90.0, 180.0]]),
            )

        # Fallback path: forecast still produced, just without vf_features.
        assert out.progression_probability is not None
