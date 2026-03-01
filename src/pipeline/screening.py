"""
Tier 1: Screening pipeline for community and rural clinic deployment.

Input: Single 2D color fundus photograph.
Pipeline: FunduSAM segmentation -> RETFound-CFP classification -> structured output.
Target latency: <5s CPU, <1s GPU.
No OCT, no longitudinal data, no VLM.

The baseline approach (vCDR > 0.5 from PMC12469222) is retained as a comparison
mode via the baseline_mode flag but is NOT the primary classification method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from src.pipeline.common import ReferralPacket, RiskLevel, classify_risk, compute_vcdr


@dataclass
class ScreeningOutput:
    """Structured output from Tier 1 screening.

    Deterministic, auditable fields — no VLM hallucination risk.
    This is what the screening operator sees on their device.
    """

    # Segmentation results
    disc_mask: torch.Tensor | None = None
    cup_mask: torch.Tensor | None = None
    vcdr: float | None = None
    segmentation_overlay: Any = None  # numpy array from visualization.overlay_segmentation

    # Classification (RETFound-based)
    risk_score: float = 0.0  # Probability of glaucoma [0, 1]
    risk_level: RiskLevel = RiskLevel.LOW

    # Referral decision
    refer_to_specialist: bool = False
    referral_reason: str = ""
    urgency: str = "routine"

    # Baseline comparison
    baseline_vcdr_positive: bool | None = None  # vCDR > 0.5 per baseline method

    # Metadata
    latency_ms: float = 0.0


class ScreeningPipeline:
    """Tier 1 screening: single fundus photo to referral decision.

    Designed for low-resource settings. Uses only:
      - FunduSAM for optic disc/cup segmentation
      - RETFound-CFP for deep feature extraction and risk scoring
      - FIN for fairness normalization (optional)

    When baseline_mode=True, uses only vCDR > 0.5 threshold to reproduce
    the PMC12469222 methodology for direct comparison.
    """

    def __init__(
        self,
        fundusam: Any = None,
        retfound_cfp: Any = None,
        classifier_head: Any = None,
        fair_norm: Any = None,
        device: str = "cpu",
        referral_threshold: float = 0.5,
        vcdr_alert_threshold: float = 0.65,
        baseline_mode: bool = False,
    ) -> None:
        """
        Args:
            fundusam: FunduSAMSegmenter instance.
            retfound_cfp: RETFoundEncoder instance (CFP variant).
            classifier_head: nn.Module mapping [B, 1024] embedding to [B, 1] score.
                If None, falls back to vCDR-only mode.
            fair_norm: FairIdentityNormalization instance (optional).
            device: "cpu" (default for screening) or "cuda".
            referral_threshold: Risk score above which to recommend referral.
            vcdr_alert_threshold: vCDR above which to escalate regardless.
            baseline_mode: If True, classify solely on vCDR > 0.5.
        """
        self.fundusam = fundusam
        self.retfound_cfp = retfound_cfp
        self.classifier_head = classifier_head
        self.fair_norm = fair_norm
        self.device = device
        self.referral_threshold = referral_threshold
        self.vcdr_alert_threshold = vcdr_alert_threshold
        self.baseline_mode = baseline_mode

    def run(
        self,
        fundus_image: torch.Tensor,
        demographics: dict[str, Any] | None = None,
    ) -> ScreeningOutput:
        """Run Tier 1 screening on a single fundus image.

        Args:
            fundus_image: Color fundus photograph [1, 3, H, W].
            demographics: Optional dict with age, sex, race/group_id (for FIN).

        Returns:
            ScreeningOutput with risk assessment and referral recommendation.
        """
        start = time.perf_counter()
        output = ScreeningOutput()

        # Phase 2: Segmentation
        if self.fundusam is not None:
            seg = self.fundusam(fundus_image.to(self.device))
            output.disc_mask = seg[:, 1]
            output.cup_mask = seg[:, 2]
            output.vcdr = compute_vcdr(output.disc_mask[0], output.cup_mask[0])

        # Always compute baseline comparison
        if output.vcdr is not None:
            output.baseline_vcdr_positive = output.vcdr > 0.5

        if self.baseline_mode:
            # Pure baseline: classify solely on vCDR > 0.5 (PMC12469222)
            output.risk_score = output.vcdr or 0.0
            output.risk_level = RiskLevel.HIGH if output.baseline_vcdr_positive else RiskLevel.LOW
            output.refer_to_specialist = output.baseline_vcdr_positive or False
            if output.refer_to_specialist:
                output.referral_reason = "vCDR > 0.5 (baseline threshold)"
        else:
            # Primary method: RETFound features + classifier
            if self.retfound_cfp is not None:
                embedding = self.retfound_cfp(fundus_image.to(self.device))

                # Apply FIN if demographics available
                if self.fair_norm is not None and demographics and "group_id" in demographics:
                    group_id = torch.tensor([demographics["group_id"]], device=self.device)
                    embedding = self.fair_norm(embedding, group_id)

                if self.classifier_head is not None:
                    logits = self.classifier_head(embedding)
                    output.risk_score = float(torch.sigmoid(logits).item())
                else:
                    # No classifier head: fall back to vCDR as score
                    output.risk_score = output.vcdr or 0.0

            # Risk classification
            output.risk_level = classify_risk(
                output.risk_score,
                output.vcdr or 0.0,
                vcdr_alert_threshold=self.vcdr_alert_threshold,
            )
            output.refer_to_specialist = output.risk_level in (
                RiskLevel.MODERATE,
                RiskLevel.HIGH,
            )

            if output.refer_to_specialist:
                reasons = []
                if output.risk_score >= self.referral_threshold:
                    reasons.append(f"Risk score {output.risk_score:.2f} >= {self.referral_threshold}")
                if output.vcdr and output.vcdr >= self.vcdr_alert_threshold:
                    reasons.append(f"vCDR {output.vcdr:.3f} >= {self.vcdr_alert_threshold}")
                output.referral_reason = "; ".join(reasons)
                output.urgency = "urgent" if output.risk_level == RiskLevel.HIGH else "routine"

        output.latency_ms = (time.perf_counter() - start) * 1000
        return output

    def create_referral(
        self,
        screening_output: ScreeningOutput,
        patient_id: str = "",
        screening_site: str = "",
        demographics: dict[str, Any] | None = None,
    ) -> ReferralPacket:
        """Create a referral packet from screening results for Tier 2 intake.

        Artifact paths (masks, embeddings) should be set by the calling script
        after saving to disk, since path management is deployment-specific.
        """
        return ReferralPacket(
            patient_id=patient_id,
            screening_site=screening_site,
            risk_score=screening_output.risk_score,
            risk_level=screening_output.risk_level.value,
            vcdr=screening_output.vcdr or 0.0,
            referral_reason=screening_output.referral_reason,
            urgency=screening_output.urgency,
            demographics=demographics or {},
        )
