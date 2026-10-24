"""
Training script for Tier 2 progression forecasting (GlaucoMamba).

Architecture:
  - DINOv2-Base (frozen): per-visit feature extraction
  - Delta-t Mamba: time-aware SSM encoding longitudinal visits
  - Evidential Forecaster: classification + regression with calibrated uncertainty

Usage:
    python scripts/train_progression.py --data-root data/longitudinal --epochs 50

Key findings driving this architecture:
  - BiLSTM (old pipeline) cannot handle irregular visit intervals — Delta-t Mamba
    natively encodes inter-visit time gaps into state transitions.
  - Binary classifiers (old pipeline) provide no calibration — Evidential DL
    separates epistemic vs aleatoric uncertainty so clinicians know when to
    trust the model.
  - RNFL OCT is the dominant signal (removing it: AUC 0.569), so we predict
    RNFL thickness as an auxiliary regression target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MultimodalGlaucomaDataset
from src.data.transforms import get_fundus_transforms
from src.models.features.dinov2_encoder import DINOv2Encoder, require_pretrained
from src.models.progression.delta_t_mamba import DeltaTMamba
from src.models.progression.evidential_forecaster import (
    EvidentialForecaster,
    EvidentialLoss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tier 2 progression forecaster")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--mamba-layers", type=int, default=4)
    parser.add_argument("--observation-window", type=int, default=4)
    parser.add_argument("--clinical-dim", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints/progression")
    parser.add_argument(
        "--allow-random-encoder", action="store_true",
        help="Train even if DINOv2's pretrained weights cannot be downloaded (smoke runs "
             "only: the features are random; the checkpoint records encoder_pretrained: false).",
    )
    return parser.parse_args()


def encode_visits(
    encoder: DINOv2Encoder,
    fundus: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run the frozen DINOv2 encoder over every visit in a batch.

    Args:
        encoder: Frozen DINOv2Encoder.
        fundus: [B, T, 3, H, W].

    Returns:
        Visit-level features [B, T, embed_dim].
    """
    B, T, C, H, W = fundus.shape
    flat = fundus.view(B * T, C, H, W).to(device)
    with torch.no_grad():
        feats = encoder(flat)  # [B*T, D]
    return feats.view(B, T, -1)


def train_one_epoch(
    mamba: DeltaTMamba,
    forecaster: EvidentialForecaster,
    loss_fn: EvidentialLoss,
    encoder: DINOv2Encoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    mamba.train()
    forecaster.train()
    total = 0.0
    steps = 0

    for batch in loader:
        fundus = batch["fundus"]  # [B, T, 3, H, W]
        timestamps = batch["timestamps"].to(device)
        clinical = batch["clinical"][:, -1].to(device)  # latest visit's clinical
        cls_targets = batch.get("progression_label")
        reg_targets = batch.get("rnfl_target")

        if cls_targets is None:
            continue
        cls_targets = cls_targets.to(device)
        reg_targets = reg_targets.to(device) if reg_targets is not None else None

        visit_feats = encode_visits(encoder, fundus, device)
        temporal = mamba(visit_feats, timestamps)
        output = forecaster(temporal, clinical_features=clinical)

        loss = loss_fn(output, cls_targets, reg_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        steps += 1

    return total / max(steps, 1)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Models ---
    encoder = DINOv2Encoder(variant=args.dinov2_variant, freeze_encoder=True).to(device)
    encoder.eval()
    encoder_pretrained = require_pretrained(encoder, args.allow_random_encoder)

    mamba = DeltaTMamba(
        input_dim=encoder.embed_dim,
        d_model=args.d_model,
        d_state=args.d_state,
        num_layers=args.mamba_layers,
    ).to(device)

    forecaster = EvidentialForecaster(
        temporal_dim=args.d_model,
        clinical_dim=args.clinical_dim,
        hidden_dim=256,
        num_classes=args.num_classes,
    ).to(device)

    loss_fn = EvidentialLoss(cls_weight=1.0, reg_weight=0.5, kl_weight=0.1)

    # --- Data ---
    transform = get_fundus_transforms("train", image_size=224)
    dataset = MultimodalGlaucomaDataset(
        root=args.data_root,
        observation_window=args.observation_window,
        transform=transform,
        load_oct=False,  # DINOv2 on fundus only for speed
        load_vf=False,
    )

    print(f"Delta-t Mamba: d_model={args.d_model} layers={args.mamba_layers}")
    print(f"DINOv2 encoder: {args.dinov2_variant} frozen ({encoder.embed_dim} dim)")
    print(f"Observation window: {args.observation_window} visits")
    print(f"Dataset: {len(dataset)} patients")
    print(f"Device: {device}")

    if len(dataset) == 0:
        print("\nEmpty dataset. See src/data/multimodal_dataset.py for manifest format.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(
        list(mamba.parameters()) + list(forecaster.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(
            mamba, forecaster, loss_fn, encoder, loader, optimizer, device,
        )
        print(f"Epoch {epoch:03d} | loss {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "mamba": mamba.state_dict(),
                    "forecaster": forecaster.state_dict(),
                    "args": vars(args),
                    "encoder_pretrained": encoder_pretrained,
                },
                output_dir / "best.pth",
            )


if __name__ == "__main__":
    main()
