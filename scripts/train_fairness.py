"""
Train a fair Tier 1 screening classifier with adversarial demographic invariance.

Combines:
  - Frozen DINOv2 feature encoder
  - EquityAttention feature-weighting block
  - Linear classification head for glaucoma risk
  - Adversarial group predictor with a Gradient Reversal Layer

Training loss:
    L = BCE(classifier(features), glaucoma) + lambda * CE(group_pred(GRL(features)), group)

The GRL flips the gradient of the second term before it reaches the feature
extractor, so minimizing L simultaneously:
  - Trains the classifier to detect glaucoma
  - Trains the group predictor to guess demographic group from features
  - Pushes the EquityAttention features AWAY from group-predictive signal

Expected output: a model that matches the overall AUC of a non-fair baseline
while narrowing the per-group AUC gap — the ES-AUC / fairness gap metric
reported in Harvard-GF and FairDist.

Usage:
    python scripts/train_fairness.py \
        --data-root data/harvard_gf \
        --dinov2-variant dinov2_vitb14 \
        --epochs 30 \
        --lambda-adv 0.3 \
        --num-groups 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.evaluation.fairness_audit import FairnessAuditor
from src.models.classifier_head import ClassifierHead
from src.models.fairness.equity_attention import EquityModule
from src.models.features.dinov2_encoder import DINOv2Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial fairness training for Tier 1 screening")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    parser.add_argument("--num-groups", type=int, default=3)
    parser.add_argument("--lambda-adv", type=float, default=0.3,
                        help="Strength of the adversarial fairness term")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints/fairness")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Number of epochs to train the group predictor "
                             "without the GRL before adversarial training kicks in.")
    return parser.parse_args()


def _run_epoch(
    encoder: DINOv2Encoder,
    equity: EquityModule,
    classifier: ClassifierHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_adv: float,
    reverse_gradient: bool,
) -> dict[str, float]:
    """One train or eval pass. optimizer=None means eval mode."""
    training = optimizer is not None
    if training:
        equity.train()
        classifier.train()
    else:
        equity.eval()
        classifier.eval()

    bce = nn.BCEWithLogitsLoss()

    total_cls_loss = 0.0
    total_adv_loss = 0.0
    correct = 0
    total = 0
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_groups: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch.get("label")
        groups = batch.get("group_id")
        if labels is None:
            continue
        labels = labels.float().to(device)
        if groups is not None:
            groups = groups.long().to(device)

        with torch.no_grad():
            raw_feats = encoder(images)

        # EquityAttention + classification (with grad)
        if training:
            equity_feats = equity(raw_feats, groups)
            logits = classifier(equity_feats)
            cls_loss = bce(logits, labels)

            adv_loss = torch.tensor(0.0, device=device)
            if groups is not None and lambda_adv > 0:
                adv_loss = equity.adversarial_loss(
                    equity_feats, groups, reverse_gradient=reverse_gradient,
                )

            total_loss = cls_loss + lambda_adv * adv_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                equity_feats = equity(raw_feats, groups)
                logits = classifier(equity_feats)
                cls_loss = bce(logits, labels)
                adv_loss = torch.tensor(0.0, device=device)

        total_cls_loss += float(cls_loss.item()) * labels.shape[0]
        total_adv_loss += float(adv_loss.item()) * labels.shape[0]
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += int((preds == labels).sum().item())
        total += labels.shape[0]

        all_scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        if groups is not None:
            all_groups.extend(groups.detach().cpu().tolist())

    metrics: dict[str, float] = {
        "cls_loss": total_cls_loss / max(total, 1),
        "adv_loss": total_adv_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }

    # Fairness audit when group ids are available
    if all_groups and len(set(all_labels)) > 1:
        try:
            auditor = FairnessAuditor(max_acceptable_gap=1.0)  # don't warn during train
            report = auditor.audit(
                np.array(all_labels), np.array(all_scores), np.array(all_groups),
            )
            metrics["auc"] = report.overall_auc
            metrics["es_auc"] = report.es_auc
            metrics["auc_gap"] = report.auc_gap
        except Exception:
            pass

    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Models ---
    encoder = DINOv2Encoder(variant=args.dinov2_variant, freeze_encoder=True).to(device)
    encoder.eval()
    embed_dim = encoder.embed_dim

    equity = EquityModule(
        feature_dim=embed_dim,
        num_groups=args.num_groups,
        adversarial_weight=1.0,  # lambda_adv is applied at the outer loss
    ).to(device)
    classifier = ClassifierHead(input_dim=embed_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(equity.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    # --- Data ---
    train_transform = get_fundus_transforms("train", image_size=args.image_size)
    val_transform = get_fundus_transforms("val", image_size=args.image_size)

    train_ds = FundusDataset(args.data_root, split="train", transform=train_transform)
    val_ds = FundusDataset(args.data_root, split="val", transform=val_transform)

    print(f"DINOv2 encoder:      {args.dinov2_variant} (frozen, {embed_dim} dim)")
    print(f"EquityAttention:     feature_dim={embed_dim}, num_groups={args.num_groups}")
    print(f"lambda_adv:          {args.lambda_adv}")
    print(f"Warm-up epochs (no GRL): {args.warmup_epochs}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    if len(train_ds) == 0:
        print("\nEmpty training set. See scripts/prepare_*.py to build a manifest.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2) if len(val_ds) else None

    history: list[dict] = []
    best_es_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        reverse_gradient = epoch > args.warmup_epochs
        train_metrics = _run_epoch(
            encoder, equity, classifier, train_loader, optimizer, device,
            lambda_adv=args.lambda_adv, reverse_gradient=reverse_gradient,
        )

        msg = (
            f"Epoch {epoch:03d} | "
            f"cls {train_metrics['cls_loss']:.4f} "
            f"adv {train_metrics['adv_loss']:.4f} "
            f"acc {train_metrics['accuracy']:.3f}"
        )

        epoch_record = {"epoch": epoch, "grl_active": reverse_gradient, "train": train_metrics}

        if val_loader is not None:
            val_metrics = _run_epoch(
                encoder, equity, classifier, val_loader, None, device,
                lambda_adv=args.lambda_adv, reverse_gradient=False,
            )
            msg += f" | val cls {val_metrics['cls_loss']:.4f} acc {val_metrics['accuracy']:.3f}"
            if "es_auc" in val_metrics:
                msg += f" es_auc {val_metrics['es_auc']:.3f} gap {val_metrics['auc_gap']:.3f}"

                if val_metrics["es_auc"] > best_es_auc:
                    best_es_auc = val_metrics["es_auc"]
                    torch.save(
                        {
                            "equity": equity.state_dict(),
                            "classifier": classifier.state_dict(),
                            "val_metrics": val_metrics,
                            "args": vars(args),
                        },
                        output_dir / "best.pth",
                    )
            epoch_record["val"] = val_metrics

        history.append(epoch_record)
        print(msg)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val ES-AUC: {best_es_auc:.3f}")
    print(f"Checkpoint:      {output_dir}/best.pth")


if __name__ == "__main__":
    main()
