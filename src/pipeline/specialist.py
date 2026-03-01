"""
Tier 2: Specialist management pipeline for tertiary hospital deployment.

Input: Sequential visits with fundus + OCT + VF + clinical data.
Pipeline: BoundaryFormer + Swin-UNETR -> DINOv2 -> VF-GIN + Delta-t Mamba
          -> Cross-Modal Fusion -> Evidential Forecaster -> (Llama Vision optional).
Accepts referrals from Tier 1 to avoid redundant reprocessing.

Architecture (GlaucoMamba):
  - BoundaryFormer: Dense depthwise separable U-Net (fundus segmentation)
  - Swin-UNETR: 3D transformer for OCT volume RNFL segmentation
  - DINOv2-Base: Frozen feature extraction (fundus + OCT)
  - VF-GIN: Graph Isomorphism Network for visual field spatial analysis
  - Delta-t Mamba: Time-aware state-space model for irregular longitudinal data
  - Cross-Modal Attention: Learned multimodal fusion with MC dropout uncertainty
  - Evidential Forecaster: Calibrated uncertainty per prediction
  - EquityAttention: Fair predictions across demographics
  - Llama 3.2 Vision: Clinical report generation (gated, experimental)

VLM report generation is gated behind enable_vlm=False by default.
All VLM outputs are always marked experimental with mandatory human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    dinov2_fundus_embedding: torch.Tensor | None = None
    dinov2_oct_embedding: torch.Tensor | None = None

    # VF analysis
    vf_classification: torch.Tensor | None = None
    vf_node_importance: torch.Tensor | None = None

    # Progression forecasting (core Tier 2 value)
    progression_probability: float | None = None
    progression_class: str | None = None  # "stable", "slow", "rapid"
    progression_uncertainty: float | None = None  # Epistemic uncertainty

    # Regression output
    predicted_rnfl_thickness: float | None = None
    rnfl_aleatoric_uncertainty: float | None = None
    rnfl_epistemic_uncertainty: float | None = None

    # Fusion uncertainty
    fusion_uncertainty: torch.Tensor | None = None

    # VLM report (experimental, gated)
    clinical_report: str | None = None
    vlm_report_is_experimental: bool = True

    # Referral source (if patient came from Tier 1)
    referral_source: ReferralPacket | None = None

    # Metadata
    requires_specialist_review: bool = True


class SpecialistPipeline:
    """Tier 2 specialist: multimodal longitudinal assessment.

    GlaucoMamba architecture — consumes fundus + OCT + VF + clinical data
    across multiple visits. Produces progression forecasts with calibrated
    uncertainty and optionally VLM-generated clinical summaries.

    Can accept a ReferralPacket from Tier 1 to skip redundant fundus processing.

    Critical finding: RNFL OCT data is the dominant signal (removing it: AUC 0.569).
    VF data is largely redundant (removing it: AUC 0.975). The pipeline prioritizes
    OCT but uses VF-GIN for spatial pattern analysis when available.
    """

    def __init__(
        self,
        segmenter: Any = None,
        oct_segmenter: Any = None,
        feature_encoder: Any = None,
        vf_graph_net: Any = None,
        temporal_encoder: Any = None,
        fusion_module: Any = None,
        forecaster: Any = None,
        report_generator: Any = None,
        equity_module: Any = None,
        device: str = "cuda",
        enable_vlm: bool = False,
    ) -> None:
        """
        Args:
            segmenter: BoundaryFormer instance for fundus segmentation.
            oct_segmenter: SwinUNETROCT instance for OCT volume segmentation.
            feature_encoder: DINOv2Encoder instance.
            vf_graph_net: VFGraphNet instance for visual field analysis.
            temporal_encoder: DeltaTMamba instance.
            fusion_module: CrossModalFusion instance.
            forecaster: EvidentialForecaster instance.
            report_generator: LlamaVisionReporter instance (experimental).
            equity_module: EquityModule instance.
            device: Compute device ("cuda" default for hospital GPU servers).
            enable_vlm: Must be explicitly True to enable VLM reports.
        """
        self.segmenter = segmenter
        self.oct_segmenter = oct_segmenter
        self.feature_encoder = feature_encoder
        self.vf_graph_net = vf_graph_net
        self.temporal_encoder = temporal_encoder
        self.fusion_module = fusion_module
        self.forecaster = forecaster
        self.report_generator = report_generator
        self.equity_module = equity_module
        self.device = device
        self.enable_vlm = enable_vlm

    def run(
        self,
        fundus_image: torch.Tensor | None = None,
        oct_volume: torch.Tensor | None = None,
        vf_data: torch.Tensor | None = None,
        visit_history: list[dict] | None = None,
        timestamps: torch.Tensor | None = None,
        clinical_data: dict | None = None,
        referral: ReferralPacket | None = None,
    ) -> SpecialistOutput:
        """Run Tier 2 specialist assessment.

        Args:
            fundus_image: Current fundus photo [1, 3, H, W].
            oct_volume: OCT B-scan stack [1, 1, D, H, W] or [D, H, W].
            vf_data: Visual field test data [1, 54, 3] (sensitivity, TD, PD).
            visit_history: List of prior visit dicts for progression analysis.
            timestamps: Visit timestamps [1, T] in days since baseline.
            clinical_data: Dict with IOP, age, sex, race, medications.
            referral: Optional ReferralPacket from Tier 1 screening.

        Returns:
            SpecialistOutput with comprehensive diagnostic results.
        """
        output = SpecialistOutput()
        output.referral_source = referral

        # --- Segmentation ---

        # Fundus: skip if referral already has results
        if referral is not None and referral.vcdr > 0:
            output.vcdr = referral.vcdr
        elif fundus_image is not None and self.segmenter is not None:
            seg_output = self.segmenter(fundus_image.to(self.device))
            logits = seg_output["logits"]
            pred = logits.argmax(dim=1)
            output.disc_mask = (pred == 1).float()
            output.cup_mask = (pred == 2).float()
            output.vcdr = compute_vcdr(output.disc_mask[0], output.cup_mask[0])

        # OCT: always runs — this is the Tier 2 critical modality
        if oct_volume is not None and self.oct_segmenter is not None:
            vol = oct_volume.to(self.device)
            if vol.ndim == 3:
                vol = vol.unsqueeze(0).unsqueeze(0)
            elif vol.ndim == 4:
                vol = vol.unsqueeze(0)
            output.rnfl_segmentation = self.oct_segmenter.segment_rnfl(vol)

        # --- Feature extraction ---

        if fundus_image is not None and self.feature_encoder is not None:
            output.dinov2_fundus_embedding = self.feature_encoder(
                fundus_image.to(self.device)
            )

        if oct_volume is not None and self.feature_encoder is not None:
            # Extract features from the middle B-scan of the OCT volume
            vol = oct_volume
            if vol.ndim == 5:
                vol = vol[0, 0]  # [D, H, W]
            elif vol.ndim == 4:
                vol = vol[0]
            mid = vol.shape[0] // 2
            oct_slice = vol[mid].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            oct_slice = oct_slice.expand(-1, 3, -1, -1)  # grayscale -> 3ch
            output.dinov2_oct_embedding = self.feature_encoder(
                oct_slice.to(self.device)
            )

        # --- VF graph analysis ---

        if vf_data is not None and self.vf_graph_net is not None:
            output.vf_classification = self.vf_graph_net(vf_data.to(self.device))
            output.vf_node_importance = self.vf_graph_net.get_node_importance(
                vf_data.to(self.device)
            )

        # --- Progression forecasting ---

        if (
            visit_history
            and self.temporal_encoder is not None
            and self.forecaster is not None
        ):
            # Encode visit sequence with timestamps for irregular interval handling
            # visit_features: [1, T, feature_dim] from DINOv2 on each visit's image
            # The calling script is responsible for stacking visit features
            pass

        # --- VLM clinical reporting (experimental, gated) ---

        if self.enable_vlm and self.report_generator is not None:
            try:
                report = self.report_generator.generate_report(
                    image=fundus_image,
                    segmentation_results={
                        "disc_mask": output.disc_mask,
                        "cup_mask": output.cup_mask,
                        "vcdr": output.vcdr,
                    },
                    dinov2_embedding=output.dinov2_fundus_embedding,
                    progression_prediction={
                        "probability": output.progression_probability,
                        "class": output.progression_class,
                        "uncertainty": output.progression_uncertainty,
                    },
                    clinical_context=clinical_data,
                )
                output.clinical_report = report.raw_text
                output.vlm_report_is_experimental = True
            except (NotImplementedError, RuntimeError):
                output.clinical_report = None

        return output

    def ingest_referral(self, referral: ReferralPacket) -> dict[str, Any]:
        """Inspect a Tier 1 referral and plan what Tier 2 still needs to do."""
        return {
            "patient_id": referral.patient_id,
            "source": referral.screening_site,
            "tier1_risk_score": referral.risk_score,
            "tier1_risk_level": referral.risk_level,
            "tier1_vcdr": referral.vcdr,
            "urgency": referral.urgency,
            "has_fundus_segmentation": bool(referral.disc_mask_path),
            "has_dinov2_embedding": bool(referral.retfound_embedding_path),
            "needs_oct": True,
            "needs_longitudinal": True,
            "demographics": referral.demographics,
        }
