"""
Prepare the G1020 fundus glaucoma dataset for GlaucoMamba training.

G1020 (Bajwa et al., IJCNN 2020) is a 1020-image dataset with 296 glaucoma
and 724 normal fundus photographs. The typical distribution layout is:

    G1020/
      Images_Square/*.jpg             # 1020 square-cropped fundus photos
      Masks/*.png                     # multi-class disc/cup masks
      G1020.csv                       # imageID, binaryLabels columns

G1020 does not come with an official train/val/test split. This script
produces a deterministic label-stratified random split.

Mask encoding varies between distributions. We try:
  0/1/2 class indices, and the REFUGE-style 0/128/255 contour format.

Usage:
    python scripts/prepare_g1020.py --source /data/G1020 --output data/g1020
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS = {".png", ".bmp", ".tif"}

LABEL_COL_CANDIDATES = [
    "binarylabels", "binarylabel", "label", "glaucoma", "class", "diagnosis",
]
IMAGE_COL_CANDIDATES = [
    "imageid", "image_id", "filename", "image", "name",
]


@dataclass
class G1020Row:
    stem: str
    image_path: Path
    mask_path: Path | None
    label: int | None
    split: str = "train"


def _find_dir(root: Path, names: list[str]) -> Path | None:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    return None


def _load_g1020_labels(source: Path) -> dict[str, int]:
    """Find and parse the G1020 metadata CSV."""
    candidates = [
        "G1020.csv", "g1020.csv", "labels.csv", "metadata.csv", "G1020_metadata.csv",
    ]
    csv_path = None
    for c in candidates:
        p = source / c
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        # Fall back to any top-level CSV
        for p in source.glob("*.csv"):
            csv_path = p
            break
    if csv_path is None:
        return {}

    labels: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.lower().strip(): (v or "").strip() for k, v in raw_row.items()}
            name_col = next((k for k in IMAGE_COL_CANDIDATES if k in row and row[k]), None)
            label_col = next((k for k in LABEL_COL_CANDIDATES if k in row and row[k] != ""), None)
            if name_col is None:
                continue

            stem = Path(row[name_col]).stem
            if label_col is None:
                labels[stem] = 0
                continue
            val = row[label_col].lower()
            labels[stem] = 1 if val in {"1", "glaucoma", "glaucomatous", "true"} else 0
    return labels


def _decode_mask(
    mask_path: Path,
    disc_out: Path,
    cup_out: Path,
) -> None:
    with Image.open(mask_path) as im:
        arr = np.array(im.convert("L"))

    unique = set(np.unique(arr).tolist())
    if unique.issubset({0, 1, 2, 3}):
        disc_mask = ((arr == 1) | (arr == 2)).astype(np.uint8) * 255
        cup_mask = (arr == 2).astype(np.uint8) * 255
    else:
        disc_mask = (arr <= 200).astype(np.uint8) * 255
        cup_mask = (arr <= 50).astype(np.uint8) * 255

    disc_out.parent.mkdir(parents=True, exist_ok=True)
    cup_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(disc_mask).save(disc_out)
    Image.fromarray(cup_mask).save(cup_out)


def _scan_g1020(source: Path) -> list[G1020Row]:
    images_dir = _find_dir(source, ["Images_Square", "Images", "images"])
    masks_dir = _find_dir(source, ["Masks", "masks", "Mask"])
    if images_dir is None:
        raise SystemExit(f"Could not find Images_Square/ under {source}")

    labels = _load_g1020_labels(source)
    rows: list[G1020Row] = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        mask_path = None
        if masks_dir is not None:
            for ext in MASK_EXTS:
                candidate = masks_dir / f"{img_path.stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

        rows.append(
            G1020Row(
                stem=img_path.stem,
                image_path=img_path,
                mask_path=mask_path,
                label=labels.get(img_path.stem),
            )
        )
    return rows


def _assign_splits(
    rows: list[G1020Row],
    ratio: tuple[float, float, float],
    seed: int,
) -> None:
    rng = random.Random(seed)
    buckets: dict[int | None, list[G1020Row]] = {}
    for row in rows:
        buckets.setdefault(row.label, []).append(row)

    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_tr = int(n * ratio[0])
        n_va = int(n * ratio[1])
        for i, row in enumerate(bucket):
            if i < n_tr:
                row.split = "train"
            elif i < n_tr + n_va:
                row.split = "val"
            else:
                row.split = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare G1020 manifest")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split-ratio", type=str, default="0.7,0.15,0.15")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--abs-image-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = _scan_g1020(source)
    if not rows:
        raise SystemExit(f"No images found under {source}")

    parts = tuple(float(x) for x in args.split_ratio.split(","))
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 0.02:
        raise SystemExit("--split-ratio must be three values summing to ~1.0")
    _assign_splits(rows, parts, args.split_seed)  # type: ignore[arg-type]

    images_out = output / "images"
    if not args.abs_image_paths:
        images_out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    with_masks = 0
    with_labels = 0

    for row in rows:
        if args.abs_image_paths:
            image_rel = str(row.image_path)
        else:
            link = images_out / f"{row.stem}{row.image_path.suffix.lower()}"
            if not link.exists():
                try:
                    link.symlink_to(row.image_path)
                except OSError:
                    import shutil

                    shutil.copy2(row.image_path, link)
            image_rel = str(link.relative_to(output))

        disc_rel = cup_rel = ""
        if row.mask_path is not None:
            disc_out = output / "masks" / "disc" / f"{row.stem}.png"
            cup_out = output / "masks" / "cup" / f"{row.stem}.png"
            _decode_mask(row.mask_path, disc_out, cup_out)
            disc_rel = str(disc_out.relative_to(output))
            cup_rel = str(cup_out.relative_to(output))
            with_masks += 1

        if row.label is not None:
            with_labels += 1

        manifest.append(
            {
                "image_path": image_rel,
                "disc_mask_path": disc_rel,
                "cup_mask_path": cup_rel,
                "label": str(row.label) if row.label is not None else "",
                "split": row.split,
                "group_id": "",
            }
        )

    manifest_path = output / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path", "disc_mask_path", "cup_mask_path",
                "label", "split", "group_id",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    counts = {"train": 0, "val": 0, "test": 0}
    glaucoma = 0
    for r in manifest:
        counts[r["split"]] += 1
        if r["label"] == "1":
            glaucoma += 1

    print(f"Wrote {len(manifest)} rows to {manifest_path}")
    print(f"  with masks:  {with_masks}")
    print(f"  with labels: {with_labels}")
    print(f"  glaucoma:    {glaucoma}")
    print(f"  splits:      train={counts['train']} val={counts['val']} test={counts['test']}")


if __name__ == "__main__":
    main()
