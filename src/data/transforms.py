"""
Data transforms for fundus and OCT imaging.

Includes:
  - Cartesian <-> polar coordinate conversion (FunduSAM-style preprocessing)
  - Standard fundus augmentation pipeline for training
  - Deterministic validation/inference pre-processing

Polar transforms are useful for optic disc/cup segmentation because they
convert the radially symmetric structure into rectangular coordinates,
making edge detection more amenable to CNN/transformer kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover - only used at inference time
    cv2 = None


# ---------------------------------------------------------------------------
# Polar coordinate conversion
# ---------------------------------------------------------------------------


def cartesian_to_polar(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Convert an image from cartesian to polar coordinates.

    Args:
        image: Grayscale [H, W] or RGB [H, W, 3] numpy array.
        center: (cx, cy) centre of the polar expansion.
        radius: Maximum radius in cartesian pixels.
        output_size: (out_h, out_w) size of the polar image.
                     out_h corresponds to radial samples,
                     out_w corresponds to angular samples.

    Returns:
        Polar image with the same number of channels as the input.
    """
    out_h, out_w = output_size
    cx, cy = center

    # Precompute sampling grid in polar space
    r = np.linspace(0, radius, out_h)[:, None]  # [out_h, 1]
    theta = np.linspace(0, 2 * np.pi, out_w, endpoint=False)[None, :]  # [1, out_w]

    # Map (r, theta) -> (x, y) in cartesian space
    xs = (cx + r * np.cos(theta)).astype(np.float32)
    ys = (cy + r * np.sin(theta)).astype(np.float32)

    if cv2 is not None:
        return cv2.remap(image, xs, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Pure numpy fallback: nearest-neighbor sampling
    H, W = image.shape[:2]
    xs_c = np.clip(np.round(xs).astype(np.int32), 0, W - 1)
    ys_c = np.clip(np.round(ys).astype(np.int32), 0, H - 1)
    if image.ndim == 2:
        return image[ys_c, xs_c]
    return image[ys_c, xs_c, :]


def polar_to_cartesian(
    polar_image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Inverse of cartesian_to_polar.

    Args:
        polar_image: Polar image [out_h, out_w] or [out_h, out_w, 3].
        center: (cx, cy) centre used in the forward transform.
        radius: Maximum radius in cartesian pixels.
        output_size: Target cartesian image size (H, W).

    Returns:
        Cartesian image with the same number of channels as the input.
    """
    H, W = output_size
    cx, cy = center
    out_h, out_w = polar_image.shape[:2]

    # Build cartesian grid and compute corresponding polar coordinates
    ys, xs = np.mgrid[0:H, 0:W]
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    theta[theta < 0] += 2 * np.pi

    # Normalize to polar sampling grid
    r_idx = (r / max(radius, 1e-6) * (out_h - 1)).astype(np.float32)
    theta_idx = (theta / (2 * np.pi) * out_w).astype(np.float32)

    if cv2 is not None:
        return cv2.remap(
            polar_image, theta_idx, r_idx,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

    r_c = np.clip(np.round(r_idx).astype(np.int32), 0, out_h - 1)
    t_c = np.clip(np.round(theta_idx).astype(np.int32), 0, out_w - 1)
    if polar_image.ndim == 2:
        return polar_image[r_c, t_c]
    return polar_image[r_c, t_c, :]


# ---------------------------------------------------------------------------
# Fundus image transform pipelines
# ---------------------------------------------------------------------------


@dataclass
class FundusTransform:
    """Minimal fundus transform: resize + normalize (+ optional flip/jitter).

    Designed to be dependency-light (numpy only) so that tests and the
    screening pipeline remain importable without torchvision. If torchvision
    is available it could be substituted by the caller.

    Polar preprocessing:
        Setting ``polar=True`` warps the resized image (and any masks) into
        polar coordinates about the image centre. This is the preprocessing
        trick used by the PMC12469222 baseline to simplify OD/OC contour
        segmentation — the radially symmetric optic nerve head becomes a
        rectangular strip, and disc/cup boundaries become near-horizontal.
        BoundaryFormer training can opt into this as a cheap augmentation.
    """

    image_size: int = 224
    train: bool = False
    # ImageNet-style normalization (matches DINOv2 pre-training).
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Polar preprocessing (opt-in)
    polar: bool = False
    polar_radius_frac: float = 0.5  # radius relative to image_size / 2

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor fallback
        H, W = arr.shape[:2]
        ys = (np.linspace(0, H - 1, self.image_size)).astype(np.int32)
        xs = (np.linspace(0, W - 1, self.image_size)).astype(np.int32)
        if arr.ndim == 2:
            return arr[ys[:, None], xs[None, :]]
        return arr[ys[:, None], xs[None, :], :]

    def __call__(
        self,
        image: np.ndarray,
        disc_mask: np.ndarray | None = None,
        cup_mask: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply the transform.

        Args:
            image: RGB fundus image [H, W, 3] uint8.
            disc_mask: Optional disc mask [H, W] uint8/bool.
            cup_mask: Optional cup mask [H, W] uint8/bool.

        Returns:
            Dict with 'image' [3, S, S] float tensor plus optional masks.
        """
        img = self._resize(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # Random horizontal flip for training augmentation
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            if disc_mask is not None:
                disc_mask = self._resize(disc_mask)
                disc_mask = disc_mask[:, ::-1].copy()
            if cup_mask is not None:
                cup_mask = self._resize(cup_mask)
                cup_mask = cup_mask[:, ::-1].copy()
        else:
            if disc_mask is not None:
                disc_mask = self._resize(disc_mask)
            if cup_mask is not None:
                cup_mask = self._resize(cup_mask)

        # Polar preprocessing (applied AFTER resize so the radius is consistent)
        if self.polar:
            center = (self.image_size // 2, self.image_size // 2)
            radius = self.polar_radius_frac * self.image_size
            out_size = (self.image_size, self.image_size)
            img = cartesian_to_polar(img, center, radius, out_size)
            if disc_mask is not None:
                disc_mask = cartesian_to_polar(disc_mask, center, radius, out_size)
            if cup_mask is not None:
                cup_mask = cartesian_to_polar(cup_mask, center, radius, out_size)

        # To tensor + normalize
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        img_t = (img_t - mean) / std

        out: dict[str, torch.Tensor] = {"image": img_t}

        if disc_mask is not None:
            out["disc_mask"] = torch.from_numpy((disc_mask > 0).astype(np.float32))
        if cup_mask is not None:
            out["cup_mask"] = torch.from_numpy((cup_mask > 0).astype(np.float32))

        return out


def get_fundus_transforms(
    split: str = "train",
    image_size: int = 224,
    polar: bool = False,
) -> FundusTransform:
    """Convenience constructor used by training/evaluation scripts.

    Args:
        split: "train", "val", or "test". Only "train" enables augmentation.
        image_size: Output square size in pixels.
        polar: If True, wrap the image into polar coordinates. Matches the
            PMC12469222 baseline's OD/OC preprocessing.
    """
    return FundusTransform(
        image_size=image_size,
        train=(split == "train"),
        polar=polar,
    )
