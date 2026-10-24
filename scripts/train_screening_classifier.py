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
from torch.utils.data import DataLoader

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.models.classifier_head import ClassifierHead
from src.models.features.dinov2_encoder import DINOv2Encoder, require_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tier 1 screening classifier")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="checkpoints/screening_classifier.pth")
    parser.add_argument(
        "--allow-random-encoder", action="store_true",
        help="Train even if DINOv2's pretrained weights cannot be downloaded (smoke runs "
             "only: the features are random; the checkpoint records encoder_pretrained: false).",
    )
    return parser.parse_args()


def _run_epoch(
    encoder: DINOv2Encoder,
    classifier: ClassifierHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, float]:
    if optimizer is None:
        classifier.eval()
    else:
        classifier.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch.get("label")
        if labels is None:
            continue
        labels = labels.float().to(device)

        with torch.no_grad():
            feats = encoder(images)

        logits = classifier(feats)
        loss = criterion(logits, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.shape[0]
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += int((preds == labels).sum().item())
        total += labels.shape[0]

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    encoder = DINOv2Encoder(
        variant=args.dinov2_variant,
        freeze_encoder=True,
    ).to(device)
    encoder.eval()
    encoder_pretrained = require_pretrained(encoder, args.allow_random_encoder)

    embed_dim = encoder.embed_dim
    classifier = ClassifierHead(input_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_transform = get_fundus_transforms("train", image_size=args.image_size)
    val_transform = get_fundus_transforms("val", image_size=args.image_size)
    train_ds = FundusDataset(args.data_root, split="train", transform=train_transform)
    val_ds = FundusDataset(args.data_root, split="val", transform=val_transform)

    print(f"DINOv2 encoder: {args.dinov2_variant} frozen ({embed_dim} dim)")
    print(f"Classifier: {sum(p.numel() for p in classifier.parameters()):,} trainable params")
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    if len(train_ds) == 0:
        print("\nEmpty training set. See src/data/fundus_dataset.py for manifest format.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2) if len(val_ds) else None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(encoder, classifier, train_loader, optimizer, criterion, device)
        msg = f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f}"

        if val_loader is not None:
            val_loss, val_acc = _run_epoch(encoder, classifier, val_loader, None, criterion, device)
            msg += f" | val loss {val_loss:.4f} acc {val_acc:.3f}"

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "classifier": classifier.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "encoder_pretrained": encoder_pretrained,
                    },
                    args.output,
                )

        print(msg)

    if val_loader is None:
        torch.save(
            {"classifier": classifier.state_dict(), "encoder_pretrained": encoder_pretrained},
            args.output,
        )
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
