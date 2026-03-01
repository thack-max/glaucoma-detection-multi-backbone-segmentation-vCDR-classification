"""
Training script for Phase 2: FunduSAM optic disc/cup segmentation.

Usage:
    python scripts/train_segmentation.py --data-root data/refuge --epochs 100

Prerequisites:
    1. Download SAM weights: https://github.com/facebookresearch/segment-anything
    2. Download REFUGE dataset: https://refuge.grand-challenge.org/
    3. Organize data per FundusDataset expected layout

Target metrics on REFUGE (from arXiv:2502.06220):
    Dice OD: 96.1%  |  Dice OC: 86.7%
    IoU OD: 88.2%   |  IoU OC: 78.9%
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.evaluation.metrics import dice_coefficient, iou_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FunduSAM segmentation")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints/fundusam")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        disc_masks = batch["disc_mask"].to(device)
        cup_masks = batch["cup_mask"].to(device)

        # Stack masks into multi-class target: [B, H, W] with values 0/1/2
        target = torch.zeros_like(disc_masks, dtype=torch.long)
        target[disc_masks > 0.5] = 1
        target[cup_masks > 0.5] = 2

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    dice_od, dice_oc, iou_od, iou_oc = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            disc_gt = batch["disc_mask"].to(device)
            cup_gt = batch["cup_mask"].to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            disc_pred = (preds >= 1).float()
            cup_pred = (preds == 2).float()

            for i in range(images.shape[0]):
                dice_od.append(dice_coefficient(disc_pred[i], disc_gt[i]))
                dice_oc.append(dice_coefficient(cup_pred[i], cup_gt[i]))
                iou_od.append(iou_score(disc_pred[i], disc_gt[i]))
                iou_oc.append(iou_score(cup_pred[i], cup_gt[i]))

    return {
        "dice_od": sum(dice_od) / len(dice_od),
        "dice_oc": sum(dice_oc) / len(dice_oc),
        "iou_od": sum(iou_od) / len(iou_od),
        "iou_oc": sum(iou_oc) / len(iou_oc),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_transform = get_fundus_transforms("train")
    val_transform = get_fundus_transforms("val")

    train_ds = FundusDataset(args.data_root, split="train", transform=train_transform)
    val_ds = FundusDataset(args.data_root, split="val", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # TODO: Replace with actual FunduSAM model initialization
    # model = FunduSAMSegmenter(sam_checkpoint=args.sam_checkpoint).to(device)
    print("FunduSAM model initialization requires SAM weights.")
    print("Download from: https://github.com/facebookresearch/segment-anything")
    print(f"Training data: {len(train_ds)} images, Validation: {len(val_ds)} images")


if __name__ == "__main__":
    main()
