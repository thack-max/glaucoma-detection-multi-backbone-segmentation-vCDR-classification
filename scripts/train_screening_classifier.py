"""
Train a lightweight classifier head on frozen DINOv2-Base embeddings for Tier 1.

This trains a simple linear layer that maps 768-dim DINOv2 features to a
binary glaucoma risk score. Small enough to train on CPU in minutes.

Usage:
    python scripts/train_screening_classifier.py \
        --data-root data/refuge \
        --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.models.features.dinov2_encoder import DINOv2Encoder


class ClassifierHead(nn.Module):
    """Simple linear classifier on top of frozen DINOv2 embeddings."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tier 1 screening classifier")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="checkpoints/screening_classifier.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load frozen DINOv2 encoder
    encoder = DINOv2Encoder(
        variant=args.dinov2_variant,
        freeze_encoder=True,
    ).to(device)
    encoder.eval()

    embed_dim = encoder.embed_dim

    # Classifier head (only trainable parameters)
    classifier = ClassifierHead(input_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"DINOv2 encoder: {args.dinov2_variant} frozen ({sum(p.numel() for p in encoder.parameters()):,} params)")
    print(f"Classifier head: {sum(p.numel() for p in classifier.parameters()):,} trainable params")
    print(f"Embedding dim: {embed_dim}")
    print(f"Device: {device}")

    # TODO: Load dataset with labels, extract embeddings, train classifier
    # transform = get_fundus_transforms("train", image_size=224)
    # train_ds = FundusDataset(args.data_root, split="train", transform=transform)
    print(f"\nDataset required for training.")
    print(f"DINOv2 weights will auto-download from torch.hub on first use.")


if __name__ == "__main__":
    main()
