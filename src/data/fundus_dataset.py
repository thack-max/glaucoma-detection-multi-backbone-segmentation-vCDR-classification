"""
Fundus dataset for Tier 1 screening and segmentation training.

Supports REFUGE / ORIGA / G1020-style directory layouts via a manifest CSV.
The manifest has one row per image with the following columns:

    image_path,disc_mask_path,cup_mask_path,label,split,group_id

Fields:
  - image_path: path to a fundus RGB image (.png/.jpg).
  - disc_mask_path: path to the disc binary mask (optional).
  - cup_mask_path: path to the cup binary mask (optional).
  - label: 0 (normal) or 1 (glaucoma). Empty/NaN if unknown.
  - split: "train" | "val" | "test".
  - group_id: integer demographic group (for fairness auditing, optional).

The dataset does not require pandas — it uses the stdlib csv module so that
tests and CPU-only screening devices keep a minimal dependency footprint.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


@dataclass
class FundusSample:
    """One row from the fundus dataset manifest."""

    image_path: Path
    disc_mask_path: Path | None = None
    cup_mask_path: Path | None = None
    label: int | None = None
    group_id: int | None = None


def _read_image(path: Path) -> np.ndarray:
    """Read an image as a numpy array. Falls back to numpy if PIL is missing."""
    if Image is not None:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    # Minimal fallback — .npy files only.
    if path.suffix == ".npy":
        return np.load(path)
    raise ImportError("Install Pillow to read image files: pip install pillow")


def _read_mask(path: Path) -> np.ndarray:
    """Read a mask as a 2D uint8 numpy array."""
    if Image is not None:
        with Image.open(path) as im:
            arr = np.array(im.convert("L"))
        return (arr > 0).astype(np.uint8)
    if path.suffix == ".npy":
        return np.load(path).astype(np.uint8)
    raise ImportError("Install Pillow to read mask files: pip install pillow")


class FundusDataset(Dataset):
    """Generic fundus dataset backed by a manifest.csv.

    Layout:

        data_root/
          manifest.csv
          images/
          masks/disc/
          masks/cup/

    The manifest paths are resolved relative to data_root.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any | None = None,
        manifest_name: str = "manifest.csv",
    ) -> None:
        """
        Args:
            root: Dataset root directory.
            split: "train" / "val" / "test" — filters the manifest rows.
            transform: Callable applied to the raw sample dict.
            manifest_name: Manifest filename inside the root directory.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: list[FundusSample] = []

        manifest = self.root / manifest_name
        if not manifest.exists():
            return  # Allow empty datasets in tests

        with open(manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", split) != split:
                    continue

                def _resolve(key: str) -> Path | None:
                    val = row.get(key, "")
                    return (self.root / val) if val else None

                label_raw = row.get("label", "")
                group_raw = row.get("group_id", "")

                sample = FundusSample(
                    image_path=self.root / row["image_path"],
                    disc_mask_path=_resolve("disc_mask_path"),
                    cup_mask_path=_resolve("cup_mask_path"),
                    label=int(label_raw) if label_raw not in ("", "nan", "NaN") else None,
                    group_id=int(group_raw) if group_raw not in ("", "nan", "NaN") else None,
                )
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = _read_image(sample.image_path)

        disc_mask = _read_mask(sample.disc_mask_path) if sample.disc_mask_path else None
        cup_mask = _read_mask(sample.cup_mask_path) if sample.cup_mask_path else None

        if self.transform is not None:
            out = self.transform(image, disc_mask=disc_mask, cup_mask=cup_mask)
        else:
            out = {
                "image": torch.from_numpy(image).float().permute(2, 0, 1) / 255.0,
            }
            if disc_mask is not None:
                out["disc_mask"] = torch.from_numpy(disc_mask.astype(np.float32))
            if cup_mask is not None:
                out["cup_mask"] = torch.from_numpy(cup_mask.astype(np.float32))

        if sample.label is not None:
            out["label"] = torch.tensor(sample.label, dtype=torch.long)
        if sample.group_id is not None:
            out["group_id"] = torch.tensor(sample.group_id, dtype=torch.long)
        out["path"] = str(sample.image_path)

        return out
