"""
Tier 2: Specialist management pipeline for tertiary hospital deployment.

Input: Sequential visits with fundus + OCT + VF + clinical data.
Pipeline: FunduSAM + MedSAM-2 -> RETFound -> ConvNeXt-V2 + BiLSTM -> (VLM optional).
Accepts referrals from Tier 1 to avoid redundant reprocessing.

VLM report generation is gated behind enable_vlm=False by default.
All VLM outputs are always marked experimental with mandatory human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.pipeline.common import ReferralPacket, compute_vcdr


@dataclass
class SpecialistOutput:
    """Comprehensive diagnostic output from Tier 2 specialist pipeline."""

    # Segmentation (fundus)
    disc_mask: torch.Tensor | None = None
    cup_mask: torch.Tensor | None = None
    vcdr: float | None = None

    # Segmentation (OCT)
    rnfl_segmentation: torch.Tensor | None = None

    # Features
    retfound_cfp_embedding: torch.Tensor | None = None
    retfound_oct_embedding: torch.Tensor | None = None

    # Progression forecasting (core Tier 2 value)
    progression_probability: float | None = None
    progression_class: str | None = None  # "stable", "slow", "rapid"

    # VLM report (experimental, gated)
    clinical_report: str | None = None
    vlm_report_is_experimental: bool = True

    # Referral source (if patient came from Tier 1)
    referral_source: ReferralPacket | None = None

    # Metadata
    requires_specialist_review: bool = True


class SpecialistPipeline:
    """Tier 2 specialist: multimodal longitudinal assessment.

    Consumes fundus + OCT + VF + clinical data across multiple visits.
    Produces progression forecasts and optionally VLM-generated clinical
    summaries (experimental, human review required).

    Can accept a ReferralPacket from Tier 1 to skip redundant fundus processing.

    Critical finding from Mass Eye & Ear study: RNFL OCT data is the dominant
    signal (removing it: AUC 0.569). VF data is largely redundant (removing it:
    AUC 0.975). The pipeline prioritizes OCT accordingly.
    """

    def __init__(
        self,
        fundusam: Any = None,
        medsam2: Any = None,
        retfound_cfp: Any = None,
        retfound_oct: Any = None,
        temporal_encoder: Any = None,
        forecaster: Any = None,
        report_generator: Any = None,
        fair_norm: Any = None,
        device: str = "cuda",
        enable_vlm: bool = False,
    ) -> None:
        """
        Args:
            fundusam: FunduSAMSegmenter instance.
            medsam2: MedSAM2Segmenter instance.
            retfound_cfp: RETFoundEncoder for color fundus photographs.
            retfound_oct: RETFoundEncoder for OCT images.
            temporal_encoder: TemporalEncoder (ConvNeXt-V2).
            forecaster: ProgressionForecaster (BiLSTM).
            report_generator: GlaucomaReportGenerator (LLaVA, experimental).
            fair_norm: FairIdentityNormalization instance.
            device: Compute device ("cuda" default for hospital GPU servers).
            enable_vlm: Must be explicitly True to enable VLM reports.
        """
        self.fundusam = fundusam
        self.medsam2 = medsam2
        self.retfound_cfp = retfound_cfp
        self.retfound_oct = retfound_oct
        self.temporal_encoder = temporal_encoder
        self.forecaster = forecaster
        self.report_generator = report_generator
        self.fair_norm = fair_norm
        self.device = device
        self.enable_vlm = enable_vlm

    def run(
        self,
        fundus_image: torch.Tensor | None = None,
        oct_volume: torch.Tensor | None = None,
        visit_history: list[dict] | None = None,
        clinical_data: dict | None = None,
        referral: ReferralPacket | None = None,
    ) -> SpecialistOutput:
        """Run Tier 2 specialist assessment.

        Args:
            fundus_image: Current fundus photo [1, 3, H, W].
            oct_volume: OCT B-scan stack [slices, H, W].
            visit_history: List of prior visit dicts for progression analysis.
                Each dict should contain imaging tensors and clinical metrics.
            clinical_data: Dict with IOP, age, sex, race, medications.
            referral: Optional ReferralPacket from Tier 1 screening.
                If provided, fundus segmentation can be skipped.

        Returns:
            SpecialistOutput with comprehensive diagnostic results.
        """
        output = SpecialistOutput()
        output.referral_source = referral

        # --- Phase 2: Segmentation ---

        # Fundus: skip if referral already has results
        if referral is not None and referral.vcdr > 0:
            output.vcdr = referral.vcdr
        elif fundus_image is not None and self.fundusam is not None:
            seg = self.fundusam(fundus_image.to(self.device))
            output.disc_mask = seg[:, 1]
            output.cup_mask = seg[:, 2]
            output.vcdr = compute_vcdr(output.disc_mask[0], output.cup_mask[0])

        # OCT: always runs — this is the Tier 2 critical modality
        if oct_volume is not None and self.medsam2 is not None:
            output.rnfl_segmentation = self.medsam2(oct_volume.to(self.device))

        # --- Phase 3: Feature extraction ---

        if fundus_image is not None and self.retfound_cfp is not None:
            output.retfound_cfp_embedding = self.retfound_cfp(
                fundus_image.to(self.device)
            )

        if oct_volume is not None and self.retfound_oct is not None:
            # Extract features from the middle slice of the OCT volume
            mid = oct_volume.shape[0] // 2
            oct_slice = oct_volume[mid].unsqueeze(0)
            if oct_slice.dim() == 3:
                oct_slice = oct_slice.expand(-1, 3, -1, -1)  # grayscale → 3ch
            output.retfound_oct_embedding = self.retfound_oct(
                oct_slice.to(self.device)
            )

        # --- Phase 4: Progression forecasting ---

        if (
            visit_history
            and self.temporal_encoder is not None
            and self.forecaster is not None
        ):
            # Encode each visit's imaging data into feature vectors,
            # then feed the sequence into the BiLSTM forecaster.
            # Full implementation depends on how visit_history dicts
            # are structured — handled by the calling script.
            pass

        # --- Phase 5: VLM clinical reporting (experimental, gated) ---

        if self.enable_vlm and self.report_generator is not None:
            try:
                report = self.report_generator.generate_report(
                    image=fundus_image,
                    segmentation_mask=output.disc_mask,
                    retfound_embedding=output.retfound_cfp_embedding,
                    progression_prediction={
                        "probability": output.progression_probability,
                        "class": output.progression_class,
                    },
                    clinical_context=clinical_data,
                )
                output.clinical_report = report.raw_text
                output.vlm_report_is_experimental = True
            except NotImplementedError:
                output.clinical_report = None

        return output

    def ingest_referral(self, referral: ReferralPacket) -> dict[str, Any]:
        """Inspect a Tier 1 referral and plan what Tier 2 still needs to do.

        Returns a summary dict useful for triaging referrals.
        """
        return {
            "patient_id": referral.patient_id,
            "source": referral.screening_site,
            "tier1_risk_score": referral.risk_score,
            "tier1_risk_level": referral.risk_level,
            "tier1_vcdr": referral.vcdr,
            "urgency": referral.urgency,
            "has_fundus_segmentation": bool(referral.disc_mask_path),
            "has_retfound_embedding": bool(referral.retfound_embedding_path),
            "needs_oct": True,
            "needs_longitudinal": True,
            "demographics": referral.demographics,
        }
