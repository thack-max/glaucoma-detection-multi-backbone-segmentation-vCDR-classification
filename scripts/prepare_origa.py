"""
Prepare the ORIGA fundus glaucoma dataset for GlaucoMamba training.

ORIGA-light is a 650-image fundus dataset (168 glaucoma / 482 normal) from
the Singapore Malay Eye Study. It does not come with an official train/val
split, so this script produces a deterministic random split.

Common layouts we try to handle:

    ORIGA/
      Images/*.jpg
      Masks/*.png                       # multi-class mask (0/1/2 or 0/128/255)
      OrigaList.csv                     # or .xlsx -- filename + glaucoma label

    ORIGA/
      Images_Square/
      Semantic_Segmentation/
      ImageList.csv

The semantic segmentation masks use one of two encodings that we auto-detect:

    Encoding A (contour-style, same as REFUGE):
        0 = cup, 128 = disc, 255 = background
    Encoding B (class-index style):
        0 = background, 1 = disc, 2 = cup    (or sometimes 0/255 with sep masks)

Usage:
    python scripts/prepare_origa.py --source /data/ORIGA --output data/origa
    python scripts/prepare_origa.py --source /data/ORIGA --output data/origa \
        --label-csv OrigaList.csv --split-seed 42
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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}


@dataclass
class OrigaRow:
    stem: str
    image_path: Path
    mask_path: Path | None
    label: int | None  # 0=normal, 1=glaucoma, None if unknown
    split: str = "train"


def _find_dir(root: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    return None


def _find_file(root: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = root / c
        if p.is_file():
            return p
    return None


def _load_label_table(source: Path, override: str | None) -> dict[str, int]:
    """Map image stem -> glaucoma label (0/1) from a CSV/XLSX file."""
    candidates = ["OrigaList.csv", "ImageList.csv", "origa_list.csv", "labels.csv"]
    if override:
        candidates = [override] + candidates

    label_file = _find_file(source, candidates)
    if label_file is None:
        return {}

    labels: dict[str, int] = {}
    with open(label_file, newline="") as f:
        reader = csv.DictReader(f)
        # Normalize column names to lowercase
        for raw_row in reader:
            row = {k.lower().strip(): (v or "").strip() for k, v in raw_row.items()}
            # Try common column names
            name_col = next(
                (k for k in ("filename", "image", "name", "imagename", "image_name")
                 if k in row and row[k]),
                None,
            )
            label_col = next(
                (k for k in ("glaucoma", "label", "diagnosis", "class")
                 if k in row and row[k] != ""),
                None,
            )
            if name_col is None:
                continue

            stem = Path(row[name_col]).stem
            if label_col is None:
                labels[stem] = 0
                continue

            val = row[label_col].lower()
            if val in {"1", "glaucoma", "glaucomatous", "true", "yes", "y"}:
                labels[stem] = 1
            else:
                labels[stem] = 0

    return labels


def _decode_origa_mask(
    mask_path: Path,
    disc_out: Path,
    cup_out: Path,
) -> None:
    """Split an ORIGA mask into disc + cup binary masks.

    Auto-detects whether the encoding is the REFUGE-style contour mask
    (0/128/255) or the class-index style (0/1/2).
    """
    with Image.open(mask_path) as im:
        arr = np.array(im.convert("L"))

    unique_vals = set(np.unique(arr).tolist())

    if unique_vals.issubset({0, 1, 2, 3}):
        # Class-index style: 0=bg, 1=disc, 2=cup
        disc_mask = ((arr == 1) | (arr == 2)).astype(np.uint8) * 255
        cup_mask = (arr == 2).astype(np.uint8) * 255
    else:
        # REFUGE-style contour mask: 0=cup, 128=disc, 255=bg
        disc_mask = (arr <= 200).astype(np.uint8) * 255
        cup_mask = (arr <= 50).astype(np.uint8) * 255

    disc_out.parent.mkdir(parents=True, exist_ok=True)
    cup_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(disc_mask).save(disc_out)
    Image.fromarray(cup_mask).save(cup_out)


def _scan_origa(source: Path) -> list[OrigaRow]:
    """Discover all image/mask pairs in an ORIGA source directory."""
    images_dir = _find_dir(source, ["Images", "Images_Square", "images"])
    masks_dir = _find_dir(
        source,
        [
            "Masks",
            "Mask",
            "Semantic_Segmentation",
            "Disc_Cup_Masks",
            "masks",
            "GroundTruth",
        ],
    )

    if images_dir is None:
        raise SystemExit(f"Could not find an Images/ directory under {source}")

    labels = _load_label_table(source, None)
    rows: list[OrigaRow] = []
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
            OrigaRow(
                stem=img_path.stem,
                image_path=img_path,
                mask_path=mask_path,
                label=labels.get(img_path.stem),
            )
        )

    return rows


def _assign_splits(
    rows: list[OrigaRow],
    split_ratio: tuple[float, float, float],
    seed: int,
) -> None:
    """Stratify by label (when known) and assign train/val/test splits."""
    rng = random.Random(seed)

    buckets: dict[int | None, list[OrigaRow]] = {0: [], 1: [], None: []}
    for row in rows:
        buckets.setdefault(row.label, []).append(row)

    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        for i, row in enumerate(bucket):
            if i < n_train:
                row.split = "train"
            elif i < n_train + n_val:
                row.split = "val"
            else:
                row.split = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ORIGA manifest")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label-csv", type=str, default=None)
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.7,0.15,0.15",
        help="train,val,test proportions (must sum to ~1.0)",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--abs-image-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = _scan_origa(source)
    if not rows:
        raise SystemExit(f"No images found under {source}")

    parts = tuple(float(x) for x in args.split_ratio.split(","))
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 0.02:
        raise SystemExit("--split-ratio must be three values summing to ~1.0")

    _assign_splits(rows, parts, args.split_seed)  # type: ignore[arg-type]

    images_out = output / "images"
    if not args.abs_image_paths:
        images_out.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
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
            _decode_origa_mask(row.mask_path, disc_out, cup_out)
            disc_rel = str(disc_out.relative_to(output))
            cup_rel = str(cup_out.relative_to(output))
            with_masks += 1

        if row.label is not None:
            with_labels += 1

        manifest_rows.append(
            {
                "image_path": image_rel,
                "disc_mask_path": disc_rel,
                "cup_mask_path": cup_rel,
                "label": str(row.label) if row.label is not None else "",
                "split": row.split,
                "group_id": "",
            }
        )

    manifest = output / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path", "disc_mask_path", "cup_mask_path",
                "label", "split", "group_id",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts = {"train": 0, "val": 0, "test": 0}
    glaucoma = 0
    for r in manifest_rows:
        counts[r["split"]] += 1
        if r["label"] == "1":
            glaucoma += 1

    print(f"Wrote {len(manifest_rows)} rows to {manifest}")
    print(f"  with masks:  {with_masks}")
    print(f"  with labels: {with_labels}")
    print(f"  glaucoma:    {glaucoma}")
    print(f"  splits:      train={counts['train']} val={counts['val']} test={counts['test']}")


if __name__ == "__main__":
    main()
