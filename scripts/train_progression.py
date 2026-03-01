"""
Training script for Phase 4: Spatiotemporal progression forecasting.

Architecture: ConvNeXt-V2 (per-visit encoder) + BiLSTM (sequence model)
Reference: Mass Eye & Ear study (medRxiv, Nov 2025)

Usage:
    python scripts/train_progression.py --data-root data/longitudinal --epochs 50

Key findings from the reference study:
    - ConvNeXt-V2 + BiLSTM: AUC 0.97
    - Removing RNFL data: AUC drops to 0.569 (RNFL is critical)
    - Removing VF data: AUC 0.975 (VF is largely redundant)
    - Saliency maps localized to arcuate bundles (biologically plausible)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MultimodalGlaucomaDataset
from src.models.progression.temporal_encoder import TemporalEncoder
from src.models.progression.forecaster import ProgressionForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train progression forecaster")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="convnextv2_base")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--observation-window", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints/progression")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize models
    encoder = TemporalEncoder(
        backbone=args.backbone,
        pretrained=True,
    ).to(device)

    forecaster = ProgressionForecaster(
        input_dim=encoder.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.lstm_layers,
    ).to(device)

    # Combined parameters
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(forecaster.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Temporal Encoder: {args.backbone}")
    print(f"Forecaster: BiLSTM ({args.lstm_layers} layers, hidden={args.hidden_dim})")
    print(f"Observation window: {args.observation_window} visits")
    print(f"Device: {device}")

    # TODO: Load dataset and run training loop
    # dataset = MultimodalGlaucomaDataset(
    #     root=args.data_root,
    #     observation_window=args.observation_window,
    # )
    print("\nDataset loading requires longitudinal data in the expected format.")
    print("See src/data/multimodal_dataset.py for manifest.csv specification.")


if __name__ == "__main__":
    main()
