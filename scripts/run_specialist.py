"""
Tier 2 specialist assessment entry point.

Usage:
    python scripts/run_specialist.py --patient-dir path/to/patient_data/
    python scripts/run_specialist.py --referral referral.json --oct scan.npy
    python scripts/run_specialist.py --patient-dir data/ --enable-vlm

Loads all GlaucoMamba Tier 2 components (fundus segmenter, OCT segmenter,
DINOv2, VF-GIN, Delta-t Mamba, Evidential Forecaster, Cross-Modal Fusion)
from the config and runs multimodal progression assessment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.transforms import get_fundus_transforms
from src.data.fundus_dataset import _read_image
from src.pipeline.common import ReferralPacket
from src.pipeline.specialist import SpecialistPipeline
from src.utils.checkpoints import load_checkpoint, load_into

# The copy shipped with the package (configs/ ships beside scripts/), so the
# installed `glaucoma-specialist` command works from any directory.
PACKAGED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tier" / "specialist.yaml"
# The original cwd-relative default: a project directory's own edited copy.
LOCAL_CONFIG = Path("configs") / "tier" / "specialist.yaml"


def default_config() -> Path:
    """Default --config: ./configs/tier/specialist.yaml if present, else the packaged copy."""
    return LOCAL_CONFIG if LOCAL_CONFIG.is_file() else PACKAGED_CONFIG


class CheckpointLoadError(RuntimeError):
    """A configured checkpoint is missing or does not fit its module."""


def _safe_build(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except CheckpointLoadError:
        # The user asked for trained weights: never fall back to random ones.
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {fn.__name__}: {exc}")
        return None


def _load_weights(model, checkpoint: str | None, keys: list[str], name: str):
    """Strictly load `checkpoint`'s state dict (first of `keys` present) into `model`."""
    if not checkpoint:
        return model
    try:
        ckpt = load_checkpoint(checkpoint)
        if not load_into(model, ckpt, key_candidates=keys, strict=True):
            raise KeyError(
                f"none of {keys} in checkpoint (found {sorted(ckpt['state_dicts'])})"
            )
    except Exception as exc:  # noqa: BLE001
        raise CheckpointLoadError(f"{name}: cannot load {checkpoint}: {exc}") from exc
    print(f"  Loaded {name} weights: {checkpoint}")
    return model


def _load_segmenter(cfg: dict):
    from src.models.segmentation.boundary_former import BoundaryFormer
    model = BoundaryFormer(
        num_classes=cfg.get("num_classes", 3),
        base_channels=cfg.get("base_channels", 64),
        growth_rate=cfg.get("growth_rate", 32),
    )
    ckpt = cfg.get("checkpoint")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    return model.eval()


def _load_oct_segmenter(cfg: dict):
    from src.models.segmentation.swin_unetr_oct import SwinUNETROCT
    model = SwinUNETROCT(
        in_channels=cfg.get("in_channels", 1),
        num_classes=cfg.get("num_classes", 4),
        embed_dim=cfg.get("embed_dim", 48),
        depths=tuple(cfg.get("depths", [2, 2, 2, 2])),
        num_heads=tuple(cfg.get("num_heads", [3, 6, 12, 24])),
    )
    ckpt = cfg.get("checkpoint")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    return model.eval()


def _load_encoder(cfg: dict):
    from src.models.features.dinov2_encoder import DINOv2Encoder
    return DINOv2Encoder(
        variant=cfg.get("variant", "dinov2_vitb14"),
        freeze_encoder=cfg.get("freeze_encoder", True),
        use_registers=cfg.get("use_registers", False),
    ).eval()


def _load_vf_graph_net(cfg: dict):
    from src.models.progression.vf_graph_net import VFGraphNet
    model = VFGraphNet(
        node_features=cfg.get("node_features", 3),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_gin_layers=cfg.get("num_gin_layers", 4),
        output_dim=cfg.get("output_dim", 128),
    )
    # Key written by scripts/train_specialist.py
    return _load_weights(model, cfg.get("checkpoint"), ["vf_net"], "vf_graph_net").eval()


def _load_temporal_encoder(cfg: dict):
    from src.models.progression.delta_t_mamba import DeltaTMamba
    model = DeltaTMamba(
        input_dim=cfg.get("input_dim", 768),
        d_model=cfg.get("d_model", 512),
        d_state=cfg.get("d_state", 16),
        num_layers=cfg.get("num_layers", 4),
        expand=cfg.get("expand", 2),
        dropout=cfg.get("dropout", 0.2),
    )
    # Keys written by scripts/train_specialist.py and scripts/train_progression.py
    return _load_weights(
        model, cfg.get("checkpoint"), ["temporal", "mamba"], "temporal_encoder",
    ).eval()


def _load_forecaster(cfg: dict):
    from src.models.progression.evidential_forecaster import EvidentialForecaster
    model = EvidentialForecaster(
        temporal_dim=cfg.get("temporal_dim", 512),
        clinical_dim=cfg.get("clinical_dim", 0),
        vf_dim=cfg.get("vf_dim", 0),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.3),
        num_classes=cfg.get("num_classes", 2),
    )
    return _load_weights(model, cfg.get("checkpoint"), ["forecaster"], "forecaster").eval()


def _load_fusion(cfg: dict):
    from src.models.fusion.cross_modal_attention import CrossModalFusion
    model = CrossModalFusion(
        modality_dims=cfg.get("modality_dims", {"fundus": 768, "oct": 768}),
        fused_dim=cfg.get("fused_dim", 512),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.2),
        mc_samples=cfg.get("mc_samples", 10),
    )
    # Key written by scripts/train_specialist.py
    return _load_weights(model, cfg.get("checkpoint"), ["fusion"], "fusion").eval()


def _load_vlm_reporter(cfg: dict):
    """Llama 3.2 Vision reporter; weights load lazily on the first report."""
    from src.models.vlm.llama_vision_reporter import LlamaVisionReporter
    return LlamaVisionReporter(
        model_name=cfg.get("model", "meta-llama/Llama-3.2-11B-Vision-Instruct"),
        lora_rank=cfg.get("lora_rank", 16),
        max_new_tokens=cfg.get("max_new_tokens", 512),
        load_in_4bit=cfg.get("load_in_4bit", True),
    )


def _load_patient_dir(patient_dir: Path) -> dict:
    """Load whatever files are present in a patient data directory.

    Expected layout (all optional):
        patient_dir/
          fundus.png           # current visit fundus
          oct.npy              # OCT volume
          vf.npy               # VF 54x3 tensor
          clinical.json        # IOP, age, sex, etc.
          visits/              # longitudinal visits (fundus per visit)
          timestamps.json      # [day0, day_t1, ...]
    """
    data: dict = {"patient_id": patient_dir.name}

    fundus = patient_dir / "fundus.png"
    if fundus.exists():
        img = _read_image(fundus)
        transform = get_fundus_transforms("val", image_size=224)
        data["fundus"] = transform(img)["image"].unsqueeze(0)

    oct_path = patient_dir / "oct.npy"
    if oct_path.exists():
        data["oct"] = torch.from_numpy(np.load(oct_path)).float()

    vf_path = patient_dir / "vf.npy"
    if vf_path.exists():
        vf = torch.from_numpy(np.load(vf_path)).float()
        if vf.ndim == 2:
            vf = vf.unsqueeze(0)
        data["vf"] = vf

    clinical_path = patient_dir / "clinical.json"
    if clinical_path.exists():
        with open(clinical_path) as f:
            data["clinical"] = json.load(f)

    visits_dir = patient_dir / "visits"
    if visits_dir.exists():
        transform = get_fundus_transforms("val", image_size=224)
        history = []
        for p in sorted(visits_dir.glob("*.png")) + sorted(visits_dir.glob("*.jpg")):
            img = _read_image(p)
            history.append({"fundus": transform(img)["image"].unsqueeze(0)})
        if history:
            data["visit_history"] = history

    ts_path = patient_dir / "timestamps.json"
    if ts_path.exists():
        with open(ts_path) as f:
            ts = json.load(f)
        data["timestamps"] = torch.tensor([ts], dtype=torch.float32)

    return data


def _serialize_output(output, patient_id: str, referral: ReferralPacket | None) -> dict:
    return {
        "patient_id": patient_id,
        "referral_source": referral.to_dict() if referral is not None else None,
        "vcdr": output.vcdr,
        "progression_probability": output.progression_probability,
        "progression_class": output.progression_class,
        "progression_uncertainty": output.progression_uncertainty,
        "predicted_rnfl_thickness": output.predicted_rnfl_thickness,
        "rnfl_aleatoric_uncertainty": output.rnfl_aleatoric_uncertainty,
        "rnfl_epistemic_uncertainty": output.rnfl_epistemic_uncertainty,
        "clinical_report": output.clinical_report,
        "vlm_report_is_experimental": output.vlm_report_is_experimental,
        "requires_specialist_review": output.requires_specialist_review,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 2: Specialist Assessment")
    parser.add_argument("--patient-dir", type=str, help="Directory with patient multimodal data")
    parser.add_argument("--referral", type=str, help="Path to Tier 1 referral JSON")
    parser.add_argument("--oct", type=str, help="Path to OCT volume (.npy)")
    parser.add_argument("--config", type=str, default=str(default_config()))
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Always say which file was used: the default depends on the cwd.
    print(f"Config: {Path(args.config).resolve()}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config["pipeline"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable — falling back to CPU")
        device = "cpu"
    enable_vlm = args.enable_vlm or config["pipeline"].get("enable_vlm", False)
    output_dir = Path(args.output_dir or config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build models (best-effort; all optional) ---
    print("Building Tier 2 models...")
    segmenter = _safe_build(_load_segmenter, config.get("segmentation", {}).get("fundus", {}))
    oct_segmenter = _safe_build(_load_oct_segmenter, config.get("segmentation", {}).get("oct", {}))
    feature_encoder = _safe_build(_load_encoder, config.get("feature_extraction", {}))
    vf_graph_net = _safe_build(_load_vf_graph_net, config.get("visual_field", {}))
    temporal_encoder = _safe_build(_load_temporal_encoder, config.get("progression", {}).get("temporal_encoder", {}))
    forecaster = _safe_build(_load_forecaster, config.get("progression", {}).get("forecaster", {}))
    fusion_module = _safe_build(_load_fusion, config.get("fusion", {}))
    report_generator = None
    if enable_vlm:
        report_generator = _safe_build(_load_vlm_reporter, config.get("vlm", {}))

    for name, mod in [
        ("segmenter", segmenter),
        ("oct_segmenter", oct_segmenter),
        ("feature_encoder", feature_encoder),
        ("vf_graph_net", vf_graph_net),
        ("temporal_encoder", temporal_encoder),
        ("forecaster", forecaster),
        ("fusion_module", fusion_module),
    ]:
        if mod is not None:
            mod.to(device)

    pipeline = SpecialistPipeline(
        segmenter=segmenter,
        oct_segmenter=oct_segmenter,
        feature_encoder=feature_encoder,
        vf_graph_net=vf_graph_net,
        temporal_encoder=temporal_encoder,
        forecaster=forecaster,
        fusion_module=fusion_module,
        report_generator=report_generator,
        device=device,
        enable_vlm=enable_vlm,
    )

    print("\nTier 2 Specialist Pipeline")
    print(f"Device: {device}")
    print(f"VLM reports: {'ENABLED (experimental)' if enable_vlm else 'disabled'}")
    print(f"Output: {output_dir}")

    # --- Load referral (if provided) ---
    referral = None
    if args.referral:
        with open(args.referral) as f:
            referral = ReferralPacket.from_dict(json.load(f))
        print(f"\nReferral for patient: {referral.patient_id}")
        print(f"  Tier 1 risk: {referral.risk_level} "
              f"(score {referral.risk_score:.2f}, vCDR {referral.vcdr:.3f})")
        intake = pipeline.ingest_referral(referral)
        print(json.dumps(intake, indent=2))

    # --- Load patient data and run ---
    patient_id = "unknown"
    run_kwargs: dict = {}
    if args.patient_dir:
        patient_dir = Path(args.patient_dir)
        patient_id = patient_dir.name
        data = _load_patient_dir(patient_dir)
        run_kwargs["fundus_image"] = data.get("fundus")
        run_kwargs["oct_volume"] = data.get("oct")
        run_kwargs["vf_data"] = data.get("vf")
        run_kwargs["clinical_data"] = data.get("clinical")
        run_kwargs["visit_history"] = data.get("visit_history")
        run_kwargs["timestamps"] = data.get("timestamps")
    elif args.oct:
        run_kwargs["oct_volume"] = torch.from_numpy(np.load(args.oct)).float()

    if referral is not None:
        run_kwargs["referral"] = referral

    if not run_kwargs:
        print("\nProvide --patient-dir, --oct, or --referral. Use --help for usage.")
        return

    output = pipeline.run(**run_kwargs)
    record = _serialize_output(output, patient_id, referral)

    out_path = output_dir / f"{patient_id}_specialist.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"\nWrote {out_path}")
    if output.progression_probability is not None:
        print(
            f"Progression: {output.progression_class} "
            f"(p={output.progression_probability:.2f}, "
            f"uncertainty={output.progression_uncertainty:.3f})"
        )
    if enable_vlm and output.clinical_report is None:
        print("VLM report: not generated (model could not be loaded or generation failed)")


if __name__ == "__main__":
    main()
