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

import warnings
from dataclasses import dataclass
from typing import Any

import torch

from src.pipeline.common import ReferralPacket, compute_vcdr

try:
    from src.models.vlm.llama_vision_reporter import VLMNotConfiguredError
except ImportError:  # pragma: no cover - fallback if VLM module refactors
    class VLMNotConfiguredError(RuntimeError):
        """Placeholder when the VLM module can't be imported at all."""


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
    # Graph readout embedding from VFGraphNet.encode_graph(), shape
    # [B, readout_dim]. This is the feature that downstream fusion and
    # forecasting modules consume — the two-class ``vf_classification``
    # logits are a clinical readout only, not a feature vector, and must
    # NOT be re-used as a [B, vf_dim] input to the fusion/forecaster heads.
    vf_embedding: torch.Tensor | None = None

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
        # Only emit the "VLM unavailable" warning the first time we hit
        # an unconfigured reporter so real patient batches don't spam
        # the operator log once per patient.
        self._vlm_warned: bool = False

        # If the caller explicitly enabled VLM but provided no generator
        # instance, flag that up front rather than silently disabling it.
        if enable_vlm and report_generator is None:
            warnings.warn(
                "SpecialistPipeline: enable_vlm=True but no report_generator "
                "was provided. VLM clinical reports will be disabled.",
                RuntimeWarning,
                stacklevel=2,
            )

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
            vf_on_device = vf_data.to(self.device)
            # Produce BOTH outputs:
            #   - vf_classification: [B, num_classes] logits for the clinical
            #     readout (normal vs glaucomatous VF).
            #   - vf_embedding: [B, readout_dim] graph-level embedding, the
            #     feature that fusion + forecaster heads were *configured*
            #     to consume. ``specialist.yaml`` declares ``vf: 256`` and
            #     ``vf_dim: 256`` — these expect the readout, not the
            #     classification head.
            output.vf_classification = self.vf_graph_net(vf_on_device)
            if hasattr(self.vf_graph_net, "encode_graph"):
                output.vf_embedding = self.vf_graph_net.encode_graph(vf_on_device)
            if hasattr(self.vf_graph_net, "get_node_importance"):
                output.vf_node_importance = self.vf_graph_net.get_node_importance(
                    vf_on_device
                )

        # --- Multimodal fusion (current visit) ---

        if self.fusion_module is not None:
            modality_feats: dict[str, torch.Tensor] = {}
            if output.dinov2_fundus_embedding is not None:
                modality_feats["fundus"] = output.dinov2_fundus_embedding
            if output.dinov2_oct_embedding is not None:
                modality_feats["oct"] = output.dinov2_oct_embedding
            if output.vf_embedding is not None:
                # Feed the graph-level readout embedding so the shape
                # matches the fusion module's ``vf`` modality_dim.
                modality_feats["vf"] = output.vf_embedding
            elif output.vf_classification is not None:
                # Back-compat path for stub VFGraphNets without encode_graph
                # (used by some unit tests).
                modality_feats["vf"] = output.vf_classification
            if clinical_data is not None:
                clin_vec = torch.tensor(
                    [
                        float(clinical_data.get("iop", 0.0)),
                        float(clinical_data.get("age", 0.0)),
                        float(clinical_data.get("sex", 0.0)),
                    ],
                    device=self.device,
                ).unsqueeze(0)
                modality_feats["clinical"] = clin_vec

            if modality_feats:
                try:
                    fused = self.fusion_module(modality_feats, return_uncertainty=True)
                    if isinstance(fused, dict):
                        output.fusion_uncertainty = fused.get("std")
                except (RuntimeError, KeyError):
                    # Fused modules may require a strict subset of modalities
                    pass

        # --- Progression forecasting ---

        if (
            visit_history
            and self.temporal_encoder is not None
            and self.forecaster is not None
        ):
            # Encode visit sequence with timestamps for irregular interval handling
            visit_features = self._stack_visit_features(visit_history)
            if visit_features is not None:
                if timestamps is None:
                    timestamps = self._default_timestamps(visit_features.shape[1])
                visit_features = visit_features.to(self.device)
                timestamps = timestamps.to(self.device)

                temporal_embedding = self.temporal_encoder(visit_features, timestamps)

                # Optional clinical + VF auxiliary features for the forecaster
                clinical_features = None
                if clinical_data is not None:
                    clinical_features = torch.tensor(
                        [
                            float(clinical_data.get("iop", 0.0)),
                            float(clinical_data.get("age", 0.0)),
                            float(clinical_data.get("sex", 0.0)),
                        ],
                        device=self.device,
                    ).unsqueeze(0)

                # Prefer the graph readout embedding for the forecaster's
                # ``vf_features`` slot. The classification logits are NOT a
                # feature vector and break any forecaster configured with
                # ``vf_dim > num_classes``.
                vf_features = None
                if output.vf_embedding is not None:
                    vf_features = output.vf_embedding
                elif output.vf_classification is not None:
                    vf_features = output.vf_classification

                forecast = self._forecast_with_fallback(
                    temporal_embedding=temporal_embedding,
                    clinical_features=clinical_features,
                    vf_features=vf_features,
                )

                self._populate_forecast(output, forecast)

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
            except VLMNotConfiguredError as exc:
                # Expected failure: VLM stack isn't installed or the
                # Llama weights aren't downloadable. Warn once per
                # pipeline instance and continue. The rest of the
                # specialist assessment is still valid.
                if not self._vlm_warned:
                    warnings.warn(
                        f"VLM report generation disabled: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._vlm_warned = True
                output.clinical_report = None
            except NotImplementedError:
                # Stub reporters used by tests.
                output.clinical_report = None

        return output

    # ------------------------------------------------------------------
    # Progression forecasting helpers
    # ------------------------------------------------------------------

    def _forecast_with_fallback(
        self,
        temporal_embedding: torch.Tensor,
        clinical_features: torch.Tensor | None,
        vf_features: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Run the forecaster, falling back gracefully when upstream
        modalities are malformed.

        The strict call path passes whichever ancillary features the
        pipeline managed to produce. If that raises because a feature has
        the wrong shape (typically: a legacy stub VFGraphNet returned
        classification logits instead of the graph readout), we retry
        with the bad modality dropped so the forecaster zero-pads
        internally. A final positional fallback handles forecasters
        whose ``forward`` does not accept kwargs at all.
        """
        try:
            return self.forecaster(
                temporal_embedding,
                clinical_features=clinical_features,
                vf_features=vf_features,
            )
        except ValueError:
            # Shape-mismatch path. Drop the vf channel and let the
            # forecaster zero-pad any configured-but-missing modality.
            try:
                return self.forecaster(
                    temporal_embedding,
                    clinical_features=clinical_features,
                    vf_features=None,
                )
            except TypeError:
                return self.forecaster(temporal_embedding)
        except TypeError:
            # Forecaster signature does not accept kwargs — positional only.
            return self.forecaster(temporal_embedding)

    def _stack_visit_features(
        self, visit_history: list[dict]
    ) -> torch.Tensor | None:
        """Stack per-visit features into a [1, T, D] tensor.

        Each visit dict may provide either:
          - 'features': precomputed DINOv2 embedding [D] or [1, D], or
          - 'fundus': a raw image tensor [1, 3, H, W] that we encode on the fly.
        """
        if not visit_history:
            return None

        feats: list[torch.Tensor] = []
        for visit in visit_history:
            if "features" in visit and visit["features"] is not None:
                f = visit["features"]
                if f.ndim == 1:
                    f = f.unsqueeze(0)
                feats.append(f)
                continue
            if self.feature_encoder is not None and "fundus" in visit:
                img = visit["fundus"].to(self.device)
                feats.append(self.feature_encoder(img))

        if not feats:
            return None
        # [T, D] -> [1, T, D]
        return torch.stack(feats, dim=1) if feats[0].ndim == 2 else torch.stack(feats).unsqueeze(0)

    @staticmethod
    def _default_timestamps(num_visits: int) -> torch.Tensor:
        """Assume uniform 90-day visit spacing when no timestamps are provided."""
        return torch.arange(num_visits, dtype=torch.float32).mul_(90.0).unsqueeze(0)

    @staticmethod
    def _populate_forecast(
        output: SpecialistOutput,
        forecast: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Write forecaster outputs into the SpecialistOutput dataclass."""
        cls = forecast.get("classification") if isinstance(forecast, dict) else None
        if cls is not None and "prob" in cls:
            prob = cls["prob"]
            # Probability of the progressing class (index 1)
            if prob.ndim == 2 and prob.shape[-1] >= 2:
                output.progression_probability = float(prob[0, 1].item())
                # Human-readable class label based on probability buckets
                p = output.progression_probability
                if p < 0.3:
                    output.progression_class = "stable"
                elif p < 0.7:
                    output.progression_class = "slow"
                else:
                    output.progression_class = "rapid"
            if "uncertainty" in cls and cls["uncertainty"] is not None:
                output.progression_uncertainty = float(cls["uncertainty"][0].item())

        reg = forecast.get("regression") if isinstance(forecast, dict) else None
        if reg is not None and "mu" in reg:
            output.predicted_rnfl_thickness = float(reg["mu"][0].item())
            if "aleatoric" in reg:
                output.rnfl_aleatoric_uncertainty = float(reg["aleatoric"][0].item())
            if "epistemic" in reg:
                output.rnfl_epistemic_uncertainty = float(reg["epistemic"][0].item())

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
