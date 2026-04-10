"""
Joint Tier 2 training: Delta-t Mamba + Cross-Modal Fusion + Evidential Forecaster.

train_progression.py trains the temporal encoder + forecaster on DINOv2
fundus features alone. This script is the end-to-end version — all Tier 2
components learn together on multimodal longitudinal data.

Forward pass per patient:
  1. DINOv2 (frozen) encodes every visit's fundus image -> [B, T, 768]
  2. DINOv2 (frozen) encodes the middle OCT B-scan for the latest visit -> [B, 768]
  3. VF-GIN encodes the latest visit's visual field test -> [B, 256]
  4. Delta-t Mamba encodes the temporal sequence -> [B, d_model]
  5. Cross-Modal Fusion combines {fundus_current, oct, vf, clinical} -> [B, fused_dim]
  6. Concat temporal + fused -> evidential forecaster
  7. EvidentialLoss (classification + regression) backprops through everything.

Only Delta-t Mamba, VF-GIN, Cross-Modal Fusion, and Evidential Forecaster
receive gradient updates — DINOv2 stays frozen to keep the AUC 0.958 prior.

Usage:
    python scripts/train_specialist.py \
        --data-root data/longitudinal \
        --epochs 50 \
        --batch-size 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MultimodalGlaucomaDataset
from src.data.transforms import get_fundus_transforms
from src.models.features.dinov2_encoder import DINOv2Encoder
from src.models.fusion.cross_modal_attention import CrossModalFusion
from src.models.progression.delta_t_mamba import DeltaTMamba
from src.models.progression.evidential_forecaster import (
    EvidentialForecaster,
    EvidentialLoss,
)
from src.models.progression.vf_graph_net import VFGraphNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint Tier 2 specialist training")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--observation-window", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--mamba-layers", type=int, default=4)
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--vf-hidden-dim", type=int, default=128)
    parser.add_argument("--vf-output-dim", type=int, default=128)
    parser.add_argument("--forecaster-hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints/specialist")
    return parser.parse_args()


def _encode_fundus_visits(
    encoder: DINOv2Encoder,
    fundus: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Frozen DINOv2 over every visit. Returns [B, T, embed_dim]."""
    B, T, C, H, W = fundus.shape
    flat = fundus.view(B * T, C, H, W).to(device)
    with torch.no_grad():
        feats = encoder(flat)
    return feats.view(B, T, -1)


def _encode_oct_latest_visit(
    encoder: DINOv2Encoder,
    oct_volume: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Frozen DINOv2 on the middle B-scan of the last visit's OCT.

    Input shape: [B, T, C, D, H, W] or [B, T, D, H, W].
    """
    if oct_volume.ndim == 6:
        vol = oct_volume[:, -1, 0]  # [B, D, H, W]
    elif oct_volume.ndim == 5:
        vol = oct_volume[:, -1]  # [B, D, H, W]
    else:
        raise ValueError(f"Unexpected OCT shape: {oct_volume.shape}")

    mid = vol.shape[1] // 2
    b_scan = vol[:, mid]  # [B, H, W]
    b_scan = b_scan.unsqueeze(1).expand(-1, 3, -1, -1)  # gray -> 3ch
    with torch.no_grad():
        return encoder(b_scan.to(device))


def _forward_batch(
    batch: dict[str, torch.Tensor],
    encoder: DINOv2Encoder,
    vf_net: VFGraphNet,
    temporal: DeltaTMamba,
    fusion: CrossModalFusion,
    forecaster: EvidentialForecaster,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor | None]:
    """Run one batch forward. Returns (forecast_output, cls_targets, reg_targets)."""
    fundus = batch["fundus"]
    timestamps = batch["timestamps"].to(device)
    clinical_seq = batch["clinical"].to(device)

    # Per-visit fundus features
    visit_feats = _encode_fundus_visits(encoder, fundus, device)
    temporal_embed = temporal(visit_feats, timestamps)  # [B, d_model]

    # Multimodal fusion on the CURRENT (latest) visit
    modality_feats = {"fundus": visit_feats[:, -1]}

    if "oct" in batch:
        modality_feats["oct"] = _encode_oct_latest_visit(encoder, batch["oct"], device)

    if "vf" in batch:
        vf_input = batch["vf"][:, -1].to(device)  # [B, 54, 3] latest visit
        vf_logits = vf_net(vf_input)
        # Use the VF graph embedding as the feature (forecaster learns to project)
        modality_feats["vf"] = vf_net.encode_graph(vf_input)

    modality_feats["clinical"] = clinical_seq[:, -1]  # latest visit clinical [B, 3]

    fused = fusion(modality_feats)  # [B, fused_dim]

    # Stitch temporal + fused embeddings before forecasting
    combined = torch.cat([temporal_embed, fused], dim=-1)
    output = forecaster(combined)

    cls_targets = batch["progression_label"].long().to(device)
    reg_targets = batch["rnfl_target"].float().to(device) if "rnfl_target" in batch else None

    return output, cls_targets, reg_targets


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Models ---
    encoder = DINOv2Encoder(variant=args.dinov2_variant, freeze_encoder=True).to(device)
    encoder.eval()

    vf_net = VFGraphNet(
        node_features=3,
        hidden_dim=args.vf_hidden_dim,
        num_gin_layers=4,
        output_dim=args.vf_output_dim,
    ).to(device)

    temporal = DeltaTMamba(
        input_dim=encoder.embed_dim,
        d_model=args.d_model,
        d_state=args.d_state,
        num_layers=args.mamba_layers,
    ).to(device)

    fusion = CrossModalFusion(
        modality_dims={
            "fundus": encoder.embed_dim,
            "oct": encoder.embed_dim,
            "vf": vf_net.readout_dim,
            "clinical": 3,
        },
        fused_dim=args.fused_dim,
        num_heads=4,
    ).to(device)

    forecaster = EvidentialForecaster(
        temporal_dim=args.d_model + args.fused_dim,
        hidden_dim=args.forecaster_hidden,
        num_classes=2,
    ).to(device)

    loss_fn = EvidentialLoss(cls_weight=1.0, reg_weight=0.5, kl_weight=0.1)

    optimizer = torch.optim.AdamW(
        list(vf_net.parameters())
        + list(temporal.parameters())
        + list(fusion.parameters())
        + list(forecaster.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    # --- Data ---
    transform = get_fundus_transforms("train", image_size=224)
    dataset = MultimodalGlaucomaDataset(
        root=args.data_root,
        observation_window=args.observation_window,
        transform=transform,
        load_oct=True,
        load_vf=True,
    )

    print(f"DINOv2 (frozen):   {args.dinov2_variant} ({encoder.embed_dim} dim)")
    print(f"VF-GIN:            hidden={args.vf_hidden_dim} out={args.vf_output_dim}")
    print(f"Delta-t Mamba:     d_model={args.d_model} layers={args.mamba_layers}")
    print(f"Cross-Modal Fusion: fused_dim={args.fused_dim}")
    print(f"Observation window: {args.observation_window} visits")
    print(f"Dataset:           {len(dataset)} patients")
    print(f"Device:            {device}")

    if len(dataset) == 0:
        print("\nEmpty dataset. Use scripts/prepare_longitudinal.py to build a manifest.")
        return

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    history: list[dict] = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        vf_net.train()
        temporal.train()
        fusion.train()
        forecaster.train()

        total = 0.0
        n_batches = 0
        for batch in loader:
            output, cls_targets, reg_targets = _forward_batch(
                batch, encoder, vf_net, temporal, fusion, forecaster, device,
            )
            loss = loss_fn(output, cls_targets, reg_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            n_batches += 1

        epoch_loss = total / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch:03d} | loss {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "vf_net": vf_net.state_dict(),
                    "temporal": temporal.state_dict(),
                    "fusion": fusion.state_dict(),
                    "forecaster": forecaster.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "loss": epoch_loss,
                },
                output_dir / "best.pth",
            )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest loss:  {best_loss:.4f}")
    print(f"Checkpoint: {output_dir}/best.pth")


if __name__ == "__main__":
    main()
