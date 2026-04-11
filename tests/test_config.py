"""Tests for the typed config schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    ConfigError,
    ScreeningConfig,
    SpecialistConfig,
    load_screening_config,
    load_specialist_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shipped configs — they must parse cleanly
# ---------------------------------------------------------------------------


def test_shipped_screening_config_parses():
    cfg = load_screening_config(REPO_ROOT / "configs" / "tier" / "screening.yaml")
    assert cfg.tier == "screening"
    assert cfg.pipeline.device in {"cpu", "cuda", "mps"}
    assert 0.0 <= cfg.pipeline.referral_threshold <= 1.0
    assert cfg.segmentation.num_classes >= 2
    assert cfg.feature_extraction.embed_dim > 0


def test_shipped_specialist_config_parses():
    cfg = load_specialist_config(REPO_ROOT / "configs" / "tier" / "specialist.yaml")
    assert cfg.tier == "specialist"
    assert cfg.pipeline.device in {"cpu", "cuda", "mps"}
    assert "temporal_encoder" in cfg.progression


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_wrong_tier_rejected(tmp_path: Path):
    _write_yaml(
        tmp_path / "bad.yaml",
        {"tier": "specialist", "pipeline": {}, "segmentation": {},
         "feature_extraction": {}},
    )
    with pytest.raises(ConfigError, match="Expected tier=screening"):
        load_screening_config(tmp_path / "bad.yaml")


def test_invalid_device_rejected(tmp_path: Path):
    _write_yaml(
        tmp_path / "bad.yaml",
        {
            "tier": "screening",
            "pipeline": {"device": "tpu", "referral_threshold": 0.5,
                         "vcdr_alert_threshold": 0.65},
            "segmentation": {},
            "feature_extraction": {},
        },
    )
    with pytest.raises(ConfigError, match="device"):
        load_screening_config(tmp_path / "bad.yaml")


def test_threshold_out_of_range_rejected(tmp_path: Path):
    _write_yaml(
        tmp_path / "bad.yaml",
        {
            "tier": "screening",
            "pipeline": {"device": "cpu", "referral_threshold": 1.5,
                         "vcdr_alert_threshold": 0.65},
            "segmentation": {},
            "feature_extraction": {},
        },
    )
    with pytest.raises(ConfigError, match="referral_threshold"):
        load_screening_config(tmp_path / "bad.yaml")


def test_missing_required_section_rejected(tmp_path: Path):
    _write_yaml(
        tmp_path / "bad.yaml",
        {"tier": "screening", "pipeline": {}},
    )
    with pytest.raises(ConfigError, match="segmentation"):
        load_screening_config(tmp_path / "bad.yaml")


def test_specialist_missing_progression_rejected(tmp_path: Path):
    _write_yaml(
        tmp_path / "bad.yaml",
        {
            "tier": "specialist",
            "pipeline": {"device": "cuda"},
            "segmentation": {},
            "feature_extraction": {},
            "visual_field": {},
            "fusion": {},
        },
    )
    with pytest.raises(ConfigError, match="progression"):
        load_specialist_config(tmp_path / "bad.yaml")


def test_num_classes_too_small_rejected(tmp_path: Path):
    _write_yaml(
        tmp_path / "bad.yaml",
        {
            "tier": "screening",
            "pipeline": {"device": "cpu"},
            "segmentation": {"num_classes": 1},
            "feature_extraction": {},
        },
    )
    with pytest.raises(ConfigError, match="num_classes"):
        load_screening_config(tmp_path / "bad.yaml")
