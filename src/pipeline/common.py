"""
Shared utilities for the two-tier glaucoma pipeline.

Contains the referral handoff mechanism between Tier 1 (screening) and
Tier 2 (specialist), plus shared computation like vCDR calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import torch


class RiskLevel(Enum):
    """Traffic-light risk classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


def compute_vcdr(disc_mask: torch.Tensor, cup_mask: torch.Tensor) -> float:
    """Compute vertical cup-to-disc ratio from segmentation masks.

    Computed for clinical reference and reporting but is NOT the sole
    classification determinant (unlike the baseline PMC12469222 approach).

    Args:
        disc_mask: Binary disc mask [H, W].
        cup_mask: Binary cup mask [H, W].

    Returns:
        Vertical cup-to-disc ratio as a float.
    """
    disc_rows = disc_mask.any(dim=-1).nonzero(as_tuple=True)[0]
    cup_rows = cup_mask.any(dim=-1).nonzero(as_tuple=True)[0]

    if len(disc_rows) == 0 or len(cup_rows) == 0:
        return 0.0

    disc_height = disc_rows[-1].item() - disc_rows[0].item() + 1
    cup_height = cup_rows[-1].item() - cup_rows[0].item() + 1

    return cup_height / max(disc_height, 1)


def classify_risk(
    risk_score: float,
    vcdr: float,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
    vcdr_alert_threshold: float = 0.65,
) -> RiskLevel:
    """Classify risk level from the screening score and vCDR.

    Uses the DINOv2-based risk score as the primary determinant,
    with vCDR as a safety-net escalation (if vCDR is very high,
    escalate even if the risk score is moderate).

    Args:
        risk_score: Model-based glaucoma probability [0, 1].
        vcdr: Vertical cup-to-disc ratio.
        low_threshold: Score below this → LOW risk.
        high_threshold: Score above this → HIGH risk.
        vcdr_alert_threshold: vCDR above this escalates to HIGH regardless.

    Returns:
        RiskLevel enum value.
    """
    if risk_score >= high_threshold or vcdr >= vcdr_alert_threshold:
        return RiskLevel.HIGH
    if risk_score >= low_threshold:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


@dataclass
class ReferralPacket:
    """Data packet passed from Tier 1 screening to Tier 2 specialist intake.

    Contains all Tier 1 outputs so the specialist does not need to re-run
    fundus segmentation or re-extract CFP features. Serialized as JSON for
    cross-system portability (rural clinic → hospital PACS).
    """

    # Patient identification
    patient_id: str = ""
    screening_site: str = ""
    screening_date: str = field(default_factory=lambda: datetime.now().isoformat())

    # Tier 1 results
    risk_score: float = 0.0
    risk_level: str = "low"  # "low", "moderate", "high"
    vcdr: float = 0.0

    # Artifact paths (filled by the screening script, not the pipeline class)
    fundus_image_path: str = ""
    disc_mask_path: str = ""
    cup_mask_path: str = ""
    overlay_path: str = ""
    dinov2_embedding_path: str = ""

    # Referral details
    referral_reason: str = ""
    urgency: str = "routine"  # "routine", "urgent", "emergent"

    # Demographics for FIN / fairness tracking
    demographics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "patient_id": self.patient_id,
            "screening_site": self.screening_site,
            "screening_date": self.screening_date,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "vcdr": self.vcdr,
            "fundus_image_path": self.fundus_image_path,
            "disc_mask_path": self.disc_mask_path,
            "cup_mask_path": self.cup_mask_path,
            "overlay_path": self.overlay_path,
            "dinov2_embedding_path": self.dinov2_embedding_path,
            "referral_reason": self.referral_reason,
            "urgency": self.urgency,
            "demographics": self.demographics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReferralPacket:
        """Deserialize from a dictionary. Ignores unknown keys for forward compat."""
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
