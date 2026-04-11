"""
Training script for Tier 1/2 fundus segmentation with BoundaryFormer.

Usage:
    python scripts/train_segmentation.py --data-root data/refuge --epochs 100

Target metrics on REFUGE:
    Dice OD: ~96%   |  Dice OC: ~89%+   (improvement over FunduSAM's 86.7%)
    IoU OD:  ~89%   |  IoU OC:  ~81%+

Architecture:
    BoundaryFormer = Dense depthwise-separable U-Net with boundary attention
    in each decoder block, supervised with a combined focal + dice + boundary
    loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.evaluation.metrics import dice_coefficient, iou_score
from src.models.segmentation.boundary_former import (
    BoundaryFormer,
    BoundaryFocalDiceLoss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoundaryFormer segmentation")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--growth-rate", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints/boundary_former")
    return parser.parse_args()


def _build_targets(disc: torch.Tensor, cup: torch.Tensor) -> torch.Tensor:
    """Merge disc + cup masks into a multi-class target [B, H, W] with 0/1/2."""
    target = torch.zeros_like(disc, dtype=torch.long)
    target[disc > 0.5] = 1
    target[cup > 0.5] = 2
    return target


def train_epoch(
    model: BoundaryFormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: BoundaryFocalDiceLoss,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        target = _build_targets(
            batch["disc_mask"].to(device), batch["cup_mask"].to(device)
        )
        out = model(images)
        loss = loss_fn(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())

    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: BoundaryFormer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    dice_od, dice_oc, iou_od, iou_oc = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        disc_gt = batch["disc_mask"].to(device)
        cup_gt = batch["cup_mask"].to(device)

        out = model(images)
        preds = out["logits"].argmax(dim=1)

        disc_pred = (preds >= 1).float()
        cup_pred = (preds == 2).float()

        for i in range(images.shape[0]):
            dice_od.append(dice_coefficient(disc_pred[i], disc_gt[i]))
            dice_oc.append(dice_coefficient(cup_pred[i], cup_gt[i]))
            iou_od.append(iou_score(disc_pred[i], disc_gt[i]))
            iou_oc.append(iou_score(cup_pred[i], cup_gt[i]))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "dice_od": _mean(dice_od),
        "dice_oc": _mean(dice_oc),
        "iou_od": _mean(iou_od),
        "iou_oc": _mean(iou_oc),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = get_fundus_transforms("train", image_size=args.image_size)
    val_transform = get_fundus_transforms("val", image_size=args.image_size)

    train_ds = FundusDataset(args.data_root, split="train", transform=train_transform)
    val_ds = FundusDataset(args.data_root, split="val", transform=val_transform)

    model = BoundaryFormer(
        num_classes=3,
        base_channels=args.base_channels,
        growth_rate=args.growth_rate,
    ).to(device)

    loss_fn = BoundaryFocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"BoundaryFormer: base={args.base_channels} growth={args.growth_rate}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    if len(train_ds) == 0:
        print("\nEmpty training set. See src/data/fundus_dataset.py for manifest format.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    best_dice_oc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | loss {train_loss:.4f}", end="")
        if len(val_ds) > 0:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f" | Dice OD {val_metrics['dice_od']:.3f} OC {val_metrics['dice_oc']:.3f}"
                f" | IoU OD {val_metrics['iou_od']:.3f} OC {val_metrics['iou_oc']:.3f}"
            )
            if val_metrics["dice_oc"] > best_dice_oc:
                best_dice_oc = val_metrics["dice_oc"]
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics},
                    output_dir / "best.pth",
                )
        else:
            print()


if __name__ == "__main__":
    main()
