"""
Evaluation metrics for segmentation, classification, and regression.

Segmentation metrics match those used by the baseline (PMC12469222):
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU)

Classification metrics:
  - AUC-ROC (primary metric across all referenced papers)
  - Equalized-Subgroup AUC (ES-AUC) from Harvard-GF fairness framework

Regression metrics (for VF MD prediction, per PMC12641456):
  - Mean Absolute Error (MAE)
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """Compute Dice Similarity Coefficient.

    Baseline reference values (MaskFormer on REFUGE):
      - OD: 93.83%
      - OC: 93.71%

    FunduSAM reference values (REFUGE):
      - OD: 96.1%
      - OC: 86.7%
    """
    pred_flat = pred.flatten().float()
    target_flat = target.flatten().float()

    intersection = (pred_flat * target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """Compute Intersection over Union.

    Baseline reference values (MaskFormer on REFUGE):
      - OD: 88.29%
      - OC: 91.09%

    FunduSAM reference values (REFUGE):
      - OD: 88.2%
      - OC: 78.9%
    """
    pred_flat = pred.flatten().float()
    target_flat = target.flatten().float()

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Compute Area Under the ROC Curve.

    Reference AUCs from verified papers:
      - RETFound (glaucoma): 0.83-0.95 depending on demographic
      - LLaVA fine-tuned (DIGS/ADAGES): 0.92
      - PaliGemma fine-tuned: 0.90
      - ConvNeXt-V2 + BiLSTM progression: 0.97
    """
    return float(roc_auc_score(labels, scores))


def compute_vcdr_mae(
    pred_vcdr: np.ndarray,
    true_vcdr: np.ndarray,
) -> float:
    """Mean Absolute Error for vCDR estimation.

    retinIA reference (Frontiers in Ophthalmology, 2025):
      AI MAE: 0.056
      First-year residents MAE: 0.105
    """
    return float(np.mean(np.abs(pred_vcdr - true_vcdr)))


def compute_vf_md_mae(
    pred_md: np.ndarray,
    true_md: np.ndarray,
) -> float:
    """Mean Absolute Error for Visual Field Mean Deviation prediction.

    LLaVA reference (PMC12641456):
      MAE: 1.79 dB (best, vision encoder + MLP fine-tuned)
      PaliGemma: 1.84 dB
      RETFound: 1.87 dB
    """
    return float(np.mean(np.abs(pred_md - true_md)))


def sensitivity_at_specificity(
    labels: np.ndarray,
    scores: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """Sensitivity (true positive rate) at a fixed specificity target.

    Finds the operating point on the ROC curve where specificity
    (1 - FPR) is at least `target_specificity`, then returns the
    corresponding TPR. This is the canonical reporting convention
    for glaucoma screening trials.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    specificity = 1.0 - fpr
    mask = specificity >= target_specificity
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def specificity_at_sensitivity(
    labels: np.ndarray,
    scores: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """Specificity at a fixed sensitivity target (secondary reporting metric)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    specificity = 1.0 - fpr
    mask = tpr >= target_sensitivity
    if not mask.any():
        return 0.0
    return float(specificity[mask].max())


def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return accuracy, sensitivity, specificity, PPV, NPV at a fixed threshold.

    Useful for direct baseline vs GlaucoMamba comparison where both systems
    emit a [0, 1] risk score.
    """
    preds = (scores >= threshold).astype(int)
    labels = labels.astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn

    return {
        "accuracy": (tp + tn) / max(total, 1),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
