"""
Train an XGBoost screening classifier on DINOv2 + handcrafted features.

Pipeline:
  1. For each fundus image: run BoundaryFormer (or load mask) -> disc/cup masks
  2. Compute handcrafted features (vCDR, rim ratios, ISNT) from the masks
  3. Run frozen DINOv2 -> 768-dim embedding
  4. Concatenate -> [768 + 11] feature vector
  5. Train XGBoost -> binary glaucoma classifier
  6. Save model + report SHAP top-feature importances on validation set

Usage:
    python scripts/train_xgboost.py \
        --data-root data/refuge \
        --segmenter-checkpoint checkpoints/boundary_former/best.pth \
        --output checkpoints/xgboost_screening.joblib

If --segmenter-checkpoint is omitted, the script reads disc/cup masks from
the manifest's `disc_mask_path` and `cup_mask_path` columns directly. This
lets you train the XGBoost head on REFUGE ground truth even before you
have a trained BoundaryFormer.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.models.classification.feature_extraction import (
    extract_handcrafted_features,
    HANDCRAFTED_FEATURE_NAMES,
)
from src.models.classification.xgboost_classifier import XGBoostScreeningClassifier
from src.models.features.dinov2_encoder import DINOv2Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost screening classifier")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    parser.add_argument("--segmenter-checkpoint", type=str, default=None,
                        help="Optional BoundaryFormer checkpoint. If absent, "
                             "uses ground-truth masks from the manifest.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="checkpoints/xgboost_screening.joblib")
    return parser.parse_args()


def _load_segmenter(checkpoint_path: str | None, device: str):
    """Build BoundaryFormer + load checkpoint, or return None to use GT masks."""
    if not checkpoint_path:
        return None
    from src.models.segmentation.boundary_former import BoundaryFormer

    model = BoundaryFormer(num_classes=3, base_channels=64, growth_rate=32)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    return model.to(device)


def _segment(model, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run BoundaryFormer and return (disc_mask, cup_mask) [H, W]."""
    with torch.no_grad():
        out = model(image)
    pred = out["logits"].argmax(dim=1)
    disc = (pred >= 1).float()[0]
    cup = (pred == 2).float()[0]
    return disc, cup


def _build_features(
    encoder: DINOv2Encoder,
    segmenter,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the [N, 768+11] feature matrix and corresponding labels."""
    encoder.eval()
    embeddings: list[np.ndarray] = []
    handcrafted: list[np.ndarray] = []
    labels: list[int] = []

    for batch in loader:
        if "label" not in batch:
            continue
        images = batch["image"].to(device)

        with torch.no_grad():
            embs = encoder(images).cpu().numpy()  # [B, 768]

        for i in range(images.shape[0]):
            img_i = images[i:i + 1]

            if segmenter is not None:
                disc, cup = _segment(segmenter, img_i)
            else:
                # Use ground-truth masks from manifest
                if "disc_mask" not in batch or "cup_mask" not in batch:
                    continue
                disc = batch["disc_mask"][i]
                cup = batch["cup_mask"][i]

            feats = extract_handcrafted_features(disc, cup).to_array()
            embeddings.append(embs[i])
            handcrafted.append(feats)
            labels.append(int(batch["label"][i].item()))

    if not embeddings:
        return np.array([]).reshape(0, 0), np.array([])

    X = np.concatenate(
        [np.stack(embeddings), np.stack(handcrafted)], axis=1,
    )
    y = np.array(labels, dtype=np.int32)
    return X, y


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Build feature extractor + segmenter
    encoder = DINOv2Encoder(variant=args.dinov2_variant, freeze_encoder=True).to(device)
    segmenter = _load_segmenter(args.segmenter_checkpoint, args.device)

    # Build datasets
    transform = get_fundus_transforms("val", image_size=args.image_size)
    train_ds = FundusDataset(args.data_root, split="train", transform=transform)
    val_ds = FundusDataset(args.data_root, split="val", transform=transform)

    print(f"DINOv2 encoder: {args.dinov2_variant} (frozen, {encoder.embed_dim}-dim)")
    print(f"Segmenter: {'BoundaryFormer (loaded)' if segmenter else 'ground-truth masks from manifest'}")
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    if len(train_ds) == 0:
        print("\nEmpty training set. Run scripts/prepare_*.py first.")
        return

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0) if len(val_ds) else None

    print("\nExtracting features for training...")
    X_train, y_train = _build_features(encoder, segmenter, train_loader, device)
    print(f"  Train matrix: {X_train.shape}, glaucoma fraction: {y_train.mean():.3f}")

    eval_set = None
    if val_loader is not None and len(val_ds) > 0:
        print("Extracting features for validation...")
        X_val, y_val = _build_features(encoder, segmenter, val_loader, device)
        if len(X_val) > 0:
            eval_set = (X_val, y_val)
            print(f"  Val matrix:   {X_val.shape}, glaucoma fraction: {y_val.mean():.3f}")

    # Train
    print("\nTraining XGBoost...")
    clf = XGBoostScreeningClassifier(
        embed_dim=encoder.embed_dim,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    clf.fit(X_train, y_train, eval_set=eval_set)

    # Evaluate + save
    if eval_set is not None:
        from sklearn.metrics import roc_auc_score, accuracy_score

        proba = clf.predict_proba(eval_set[0])
        try:
            auc = roc_auc_score(eval_set[1], proba)
        except ValueError:
            auc = float("nan")
        acc = accuracy_score(eval_set[1], proba >= 0.5)
        print(f"\nValidation AUC: {auc:.4f} | accuracy: {acc:.4f}")

        # SHAP top-features summary on a few validation samples
        try:
            print("\nSHAP top-5 contributions on val samples:")
            for i in range(min(3, len(eval_set[0]))):
                expl = clf.explain(eval_set[0][i:i + 1])
                top = expl.top_contributions(k=5)
                tags = ", ".join(
                    f"{name}={val:+.3f}" for name, val in top
                    if name in HANDCRAFTED_FEATURE_NAMES
                )
                if not tags:
                    tags = ", ".join(f"{name}={val:+.3f}" for name, val in top)
                print(f"  sample {i}: risk={expl.risk_score:.3f} | {tags}")
        except ImportError:
            print("(SHAP not installed — install shap to see attribution)")

    clf.save(args.output)
    print(f"\nSaved classifier to {args.output}")


if __name__ == "__main__":
    main()
