"""
Handcrafted morphometric features from optic disc / cup segmentation masks.

These are the clinically interpretable features ophthalmologists actually
look at — vCDR, rim/disc area ratio, ISNT (inferior, superior, nasal,
temporal) sector ratios. Combining them with DINOv2 embeddings inside an
XGBoost classifier lets us:

  - Match or beat the deep-only classifier on small datasets where neural
    features overfit
  - Provide SHAP-based feature importance that names a clinically meaningful
    quantity (e.g. "vCDR contributes +0.23 to the glaucoma probability")
    rather than an opaque embedding dimension
  - Stay interpretable to clinicians who already trust vCDR / rim ratios

Reference for the ISNT rule: Jonas et al., Surv Ophthalmol 1999. Healthy
optic discs follow Inferior > Superior > Nasal > Temporal rim thickness.
Violation of the ISNT rule is a classical glaucoma sign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


HANDCRAFTED_FEATURE_NAMES = [
    "vcdr",
    "horizontal_cdr",
    "rim_disc_area_ratio",
    "cup_disc_area_ratio",
    "disc_area_norm",
    "cup_area_norm",
    "rim_thickness_inferior",
    "rim_thickness_superior",
    "rim_thickness_nasal",
    "rim_thickness_temporal",
    "isnt_violation",
]


@dataclass
class HandcraftedFeatures:
    """Clinically interpretable features extracted from a segmentation mask."""

    vcdr: float = 0.0
    horizontal_cdr: float = 0.0
    rim_disc_area_ratio: float = 0.0
    cup_disc_area_ratio: float = 0.0
    disc_area_norm: float = 0.0  # disc area / image area
    cup_area_norm: float = 0.0
    rim_thickness_inferior: float = 0.0
    rim_thickness_superior: float = 0.0
    rim_thickness_nasal: float = 0.0
    rim_thickness_temporal: float = 0.0
    isnt_violation: float = 0.0  # 1.0 if rim_I < rim_S or rim_S < rim_N or rim_N < rim_T

    def to_array(self) -> np.ndarray:
        """Return features as a numpy vector in HANDCRAFTED_FEATURE_NAMES order."""
        return np.array([asdict(self)[name] for name in HANDCRAFTED_FEATURE_NAMES], dtype=np.float32)


def _as_numpy_2d(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a torch tensor or numpy array to a 2D float array in [0, 1]."""
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().float().numpy()
    else:
        m = np.asarray(mask, dtype=np.float32)
    while m.ndim > 2:
        m = m.squeeze(0)
    return (m > 0.5).astype(np.float32)


def _vertical_extent(mask: np.ndarray) -> tuple[int, int]:
    """Return (top_row, bottom_row) of nonzero pixels, or (-1, -1) if empty."""
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return -1, -1
    return int(rows[0]), int(rows[-1])


def _horizontal_extent(mask: np.ndarray) -> tuple[int, int]:
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return -1, -1
    return int(cols[0]), int(cols[-1])


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    return float(ys.mean()), float(xs.mean())


def _rim_thickness_quadrant(
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    quadrant: str,
) -> float:
    """Approximate rim thickness in one quadrant relative to the disc centre.

    quadrant in {"inferior", "superior", "nasal", "temporal"}.
    "Nasal" and "temporal" are computed assuming a right eye orientation
    (nasal = left half of the image). For a robust orientation-agnostic
    classifier, this is fine because we report ISNT_violation as a derived
    flag rather than relying on the raw N/T values directly.
    """
    cen = _centroid(disc_mask)
    if cen is None:
        return 0.0
    cy, cx = cen
    H, W = disc_mask.shape

    if quadrant == "superior":
        y0, y1, x0, x1 = 0, int(cy), 0, W
    elif quadrant == "inferior":
        y0, y1, x0, x1 = int(cy), H, 0, W
    elif quadrant == "nasal":
        y0, y1, x0, x1 = 0, H, 0, int(cx)
    elif quadrant == "temporal":
        y0, y1, x0, x1 = 0, H, int(cx), W
    else:
        raise ValueError(f"Unknown quadrant: {quadrant}")

    disc_q = disc_mask[y0:y1, x0:x1]
    cup_q = cup_mask[y0:y1, x0:x1]
    rim_q = (disc_q - cup_q).clip(0, 1)
    disc_q_area = max(disc_q.sum(), 1.0)
    return float(rim_q.sum() / disc_q_area)


def extract_handcrafted_features(
    disc_mask: torch.Tensor | np.ndarray,
    cup_mask: torch.Tensor | np.ndarray,
) -> HandcraftedFeatures:
    """Compute morphometric features from an OD/OC segmentation pair.

    Args:
        disc_mask: Binary disc mask [H, W] (or [..., H, W] — leading dims
            are squeezed).
        cup_mask: Binary cup mask [H, W].

    Returns:
        HandcraftedFeatures populated with vCDR, area ratios, and per-quadrant
        rim ratios. Returns zeros when the disc mask is empty (avoids NaNs
        from propagating into XGBoost).
    """
    disc = _as_numpy_2d(disc_mask)
    cup = _as_numpy_2d(cup_mask)
    H, W = disc.shape
    image_area = H * W

    feats = HandcraftedFeatures()

    if disc.sum() == 0:
        return feats

    # vCDR (vertical) and horizontal CDR
    d_top, d_bot = _vertical_extent(disc)
    c_top, c_bot = _vertical_extent(cup)
    if d_top >= 0 and d_bot > d_top:
        disc_h = d_bot - d_top + 1
        cup_h = (c_bot - c_top + 1) if (c_top >= 0 and c_bot > c_top) else 0
        feats.vcdr = cup_h / max(disc_h, 1)

    d_left, d_right = _horizontal_extent(disc)
    c_left, c_right = _horizontal_extent(cup)
    if d_left >= 0 and d_right > d_left:
        disc_w = d_right - d_left + 1
        cup_w = (c_right - c_left + 1) if (c_left >= 0 and c_right > c_left) else 0
        feats.horizontal_cdr = cup_w / max(disc_w, 1)

    # Areas
    disc_area = float(disc.sum())
    cup_area = float(cup.sum())
    rim_area = max(disc_area - cup_area, 0.0)

    feats.cup_disc_area_ratio = cup_area / max(disc_area, 1.0)
    feats.rim_disc_area_ratio = rim_area / max(disc_area, 1.0)
    feats.disc_area_norm = disc_area / max(image_area, 1.0)
    feats.cup_area_norm = cup_area / max(image_area, 1.0)

    # Per-quadrant rim thickness
    feats.rim_thickness_inferior = _rim_thickness_quadrant(disc, cup, "inferior")
    feats.rim_thickness_superior = _rim_thickness_quadrant(disc, cup, "superior")
    feats.rim_thickness_nasal = _rim_thickness_quadrant(disc, cup, "nasal")
    feats.rim_thickness_temporal = _rim_thickness_quadrant(disc, cup, "temporal")

    # ISNT rule: healthy disc has Inferior >= Superior >= Nasal >= Temporal.
    # Allow a small tolerance so a perfectly concentric (no-violation) disc
    # isn't flagged as violating ISNT due to floating-point quadrant area
    # differences.
    rim_i = feats.rim_thickness_inferior
    rim_s = feats.rim_thickness_superior
    rim_n = feats.rim_thickness_nasal
    rim_t = feats.rim_thickness_temporal
    eps = 0.02  # 2% of the rim/disc-area ratio is below clinical resolution
    isnt_satisfied = (
        (rim_i + eps >= rim_s)
        and (rim_s + eps >= rim_n)
        and (rim_n + eps >= rim_t)
    )
    feats.isnt_violation = 0.0 if isnt_satisfied else 1.0

    return feats
