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
