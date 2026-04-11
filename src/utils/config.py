"""
Typed configuration schemas for Tier 1 + Tier 2 pipelines.

The run scripts previously reached into nested YAML keys with no schema
validation — a typo in `configs/tier/screening.yaml` would surface as an
opaque KeyError mid-run on a hospital deployment. These dataclass-backed
schemas parse a YAML config once, enforce types and ranges at the boundary,
and emit a helpful error message if something is missing or malformed.

Why this matters for the original goal:
Tier 1 is meant to run on rural clinic devices by non-ML staff, and Tier 2
is meant to run as a long-lived hospital service. Config typos must fail
LOUD and EARLY, not partway through a batch of patients.

We use plain stdlib dataclasses instead of pydantic to keep the install
footprint minimal — the CPU screening deployment target has no reason to
ship a pydantic runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file fails validation."""


def _require(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"{section_name}: missing required key '{key}'")
    return section[key]


def _validate_device(device: str) -> str:
    if device not in {"cpu", "cuda", "mps"}:
        raise ConfigError(f"pipeline.device must be cpu/cuda/mps, got '{device}'")
    return device


def _validate_unit_interval(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], got {value}")
    return value


# ---------------------------------------------------------------------------
# Tier 1 schema
# ---------------------------------------------------------------------------


@dataclass
class ScreeningPipelineSection:
    device: str = "cpu"
    baseline_mode: bool = False
    referral_threshold: float = 0.5
    vcdr_alert_threshold: float = 0.65


@dataclass
class SegmentationSection:
    model: str = "boundary_former"
    num_classes: int = 3
    base_channels: int = 64
    growth_rate: int = 32
    image_size: int = 512
    checkpoint: str | None = None
    polar: bool = False


@dataclass
class FeatureExtractionSection:
    model: str = "dinov2_vitb14"
    variant: str = "dinov2_vitb14"
    embed_dim: int = 768
    freeze_encoder: bool = True
    use_registers: bool = False
    image_size: int = 224


@dataclass
class ClassifierSection:
    type: str = "linear"
    input_dim: int = 768
    checkpoint: str | None = None


@dataclass
class FairnessSection:
    enabled: bool = True
    model: str = "equity_attention"
    num_groups: int = 3
    adversarial_weight: float = 0.1
    checkpoint: str | None = None


@dataclass
class ScreeningOutputSection:
    save_overlay: bool = True
    save_referral_json: bool = True
    save_attention_map: bool = True
    output_dir: str = "outputs/screening/"


@dataclass
class ScreeningConfig:
    tier: str
    pipeline: ScreeningPipelineSection
    segmentation: SegmentationSection
    feature_extraction: FeatureExtractionSection
    classifier: ClassifierSection
    fairness: FairnessSection
    output: ScreeningOutputSection
    explainability: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScreeningConfig:
        if data.get("tier") != "screening":
            raise ConfigError(f"Expected tier=screening, got tier={data.get('tier')!r}")

        pipe_data = _require(data, "pipeline", "root")
        seg_data = _require(data, "segmentation", "root")
        feat_data = _require(data, "feature_extraction", "root")

        pipeline = ScreeningPipelineSection(
            device=_validate_device(str(pipe_data.get("device", "cpu"))),
            baseline_mode=bool(pipe_data.get("baseline_mode", False)),
            referral_threshold=_validate_unit_interval(
                float(pipe_data.get("referral_threshold", 0.5)),
                "pipeline.referral_threshold",
            ),
            vcdr_alert_threshold=_validate_unit_interval(
                float(pipe_data.get("vcdr_alert_threshold", 0.65)),
                "pipeline.vcdr_alert_threshold",
            ),
        )

        segmentation = SegmentationSection(
            model=str(seg_data.get("model", "boundary_former")),
            num_classes=int(seg_data.get("num_classes", 3)),
            base_channels=int(seg_data.get("base_channels", 64)),
            growth_rate=int(seg_data.get("growth_rate", 32)),
            image_size=int(seg_data.get("image_size", 512)),
            checkpoint=seg_data.get("checkpoint"),
            polar=bool(seg_data.get("polar", False)),
        )
        if segmentation.num_classes < 2:
            raise ConfigError(
                f"segmentation.num_classes must be >= 2, got {segmentation.num_classes}"
            )

        feature_extraction = FeatureExtractionSection(
            model=str(feat_data.get("model", "dinov2_vitb14")),
            variant=str(feat_data.get("variant", "dinov2_vitb14")),
            embed_dim=int(feat_data.get("embed_dim", 768)),
            freeze_encoder=bool(feat_data.get("freeze_encoder", True)),
            use_registers=bool(feat_data.get("use_registers", False)),
            image_size=int(feat_data.get("image_size", 224)),
        )

        cls_data = data.get("classifier", {})
        classifier = ClassifierSection(
            type=str(cls_data.get("type", "linear")),
            input_dim=int(cls_data.get("input_dim", feature_extraction.embed_dim)),
            checkpoint=cls_data.get("checkpoint"),
        )

        fair_data = data.get("fairness", {})
        fairness = FairnessSection(
            enabled=bool(fair_data.get("enabled", True)),
            model=str(fair_data.get("model", "equity_attention")),
            num_groups=int(fair_data.get("num_groups", 3)),
            adversarial_weight=float(fair_data.get("adversarial_weight", 0.1)),
            checkpoint=fair_data.get("checkpoint"),
        )
        if fairness.num_groups < 1:
            raise ConfigError("fairness.num_groups must be >= 1")

        out_data = data.get("output", {})
        output = ScreeningOutputSection(
            save_overlay=bool(out_data.get("save_overlay", True)),
            save_referral_json=bool(out_data.get("save_referral_json", True)),
            save_attention_map=bool(out_data.get("save_attention_map", True)),
            output_dir=str(out_data.get("output_dir", "outputs/screening/")),
        )

        return cls(
            tier="screening",
            pipeline=pipeline,
            segmentation=segmentation,
            feature_extraction=feature_extraction,
            classifier=classifier,
            fairness=fairness,
            output=output,
            explainability=data.get("explainability", {}),
        )


# ---------------------------------------------------------------------------
# Tier 2 schema (lighter — most of Tier 2 is module dicts)
# ---------------------------------------------------------------------------


@dataclass
class SpecialistPipelineSection:
    device: str = "cuda"
    enable_vlm: bool = False


@dataclass
class SpecialistConfig:
    tier: str
    pipeline: SpecialistPipelineSection
    segmentation: dict[str, Any]
    feature_extraction: dict[str, Any]
    visual_field: dict[str, Any]
    progression: dict[str, Any]
    fusion: dict[str, Any]
    vlm: dict[str, Any] = field(default_factory=dict)
    fairness: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecialistConfig:
        if data.get("tier") != "specialist":
            raise ConfigError(f"Expected tier=specialist, got tier={data.get('tier')!r}")

        pipe_data = _require(data, "pipeline", "root")
        pipeline = SpecialistPipelineSection(
            device=_validate_device(str(pipe_data.get("device", "cuda"))),
            enable_vlm=bool(pipe_data.get("enable_vlm", False)),
        )

        required = ["segmentation", "feature_extraction", "visual_field", "progression", "fusion"]
        for key in required:
            _require(data, key, "root")

        return cls(
            tier="specialist",
            pipeline=pipeline,
            segmentation=data["segmentation"],
            feature_extraction=data["feature_extraction"],
            visual_field=data["visual_field"],
            progression=data["progression"],
            fusion=data["fusion"],
            vlm=data.get("vlm", {}),
            fairness=data.get("fairness", {}),
            output=data.get("output", {}),
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_screening_config(path: str | Path) -> ScreeningConfig:
    """Parse and validate a Tier 1 screening YAML config."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config at {path} did not parse to a dict")
    return ScreeningConfig.from_dict(data)


def load_specialist_config(path: str | Path) -> SpecialistConfig:
    """Parse and validate a Tier 2 specialist YAML config."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config at {path} did not parse to a dict")
    return SpecialistConfig.from_dict(data)
