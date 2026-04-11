"""
Prepare the REFUGE fundus glaucoma dataset for GlaucoMamba training.

REFUGE (REtinal FUndus Glaucoma challengE) is distributed by the Grand
Challenge platform. After extracting the official zips, the layout looks like:

    REFUGE/
      Training400/
        Glaucoma/                     # 40 jpg images
        Non-Glaucoma/                 # 360 jpg images
      Training400-GT/
        Disc_Cup_Masks/
          Glaucoma/                   # bmp masks, 0=cup, 128=disc, 255=bg
          Non-Glaucoma/
        Glaucoma_label.xlsx           # or .csv
      REFUGE-Validation400/           # 400 jpg images (flat)
      REFUGE-Validation400-GT/
        Disc_Cup_Masks/
      Test400/                        # 400 jpg images
      Test400-GT/
        Disc_Cup_Masks/

This script walks any subset of that layout and writes:

    <output>/manifest.csv
    <output>/masks/disc/<stem>.png   (extracted from the multi-class mask)
    <output>/masks/cup/<stem>.png

The image paths in the manifest point directly at the source to avoid
duplicating the ~2 GB of fundus JPEGs. Extracted binary masks live next to
the manifest so the manifest stays self-contained.

Usage:
    python scripts/prepare_refuge.py --source /data/REFUGE --output data/refuge

Mask decoding (REFUGE convention):
    disc region = pixels where mask <= 128   (cup + disc are both disc)
    cup region  = pixels where mask == 0
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


# Directory patterns we recognize for each split.
SPLIT_LAYOUTS = {
    "train": {
        "images": ["Training400"],
        "masks": ["Training400-GT/Disc_Cup_Masks"],
    },
    "val": {
        "images": ["REFUGE-Validation400", "Validation400"],
        "masks": [
            "REFUGE-Validation400-GT/Disc_Cup_Masks",
            "Validation400-GT/Disc_Cup_Masks",
        ],
    },
    "test": {
        "images": ["Test400", "REFUGE-Test400"],
        "masks": [
            "Test400-GT/Disc_Cup_Masks",
            "REFUGE-Test400-GT/Disc_Cup_Masks",
        ],
    },
}


@dataclass
class RefugeRow:
    stem: str
    image_path: Path
    mask_path: Path | None
    label: int  # 0 = normal, 1 = glaucoma
    split: str


def _find_first_existing(root: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        p = root / candidate
        if p.exists():
            return p
    return None


def _scan_split(source: Path, split: str) -> list[RefugeRow]:
    """Discover image/mask pairs for a single split."""
    layout = SPLIT_LAYOUTS[split]
    images_dir = _find_first_existing(source, layout["images"])
    if images_dir is None:
        return []

    masks_dir = _find_first_existing(source, layout["masks"])

    rows: list[RefugeRow] = []
    # Glob recursively because REFUGE Training400 splits by label.
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue

        # Infer glaucoma label from ancestor folder name when available
        label = 0
        if any(part.lower() == "glaucoma" for part in img_path.relative_to(images_dir).parts):
            label = 1

        mask_path = None
        if masks_dir is not None:
            # Masks may live in a flat folder or in Glaucoma/ Non-Glaucoma/ subdirs.
            for ext in (".bmp", ".png", ".tif", ".tiff"):
                candidate = masks_dir / f"{img_path.stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
                if label == 1:
                    candidate = masks_dir / "Glaucoma" / f"{img_path.stem}{ext}"
                else:
                    candidate = masks_dir / "Non-Glaucoma" / f"{img_path.stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

        rows.append(
            RefugeRow(
                stem=img_path.stem,
                image_path=img_path,
                mask_path=mask_path,
                label=label,
                split=split,
            )
        )

    return rows


def _decode_refuge_mask(
    mask_path: Path,
    disc_out: Path,
    cup_out: Path,
) -> None:
    """Split a REFUGE multi-class mask into disc + cup binary masks.

    REFUGE encoding:
        pixel value 0   -> optic cup
        pixel value 128 -> optic disc annulus
        pixel value 255 -> background
    """
    with Image.open(mask_path) as im:
        arr = np.array(im.convert("L"))

    # Disc region is anything <=128 (cup + disc annulus); cup region is only 0.
    disc_mask = (arr <= 200).astype(np.uint8) * 255  # tolerate some JPEG noise
    cup_mask = (arr <= 50).astype(np.uint8) * 255

    disc_out.parent.mkdir(parents=True, exist_ok=True)
    cup_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(disc_mask).save(disc_out)
    Image.fromarray(cup_mask).save(cup_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare REFUGE manifest")
    parser.add_argument("--source", type=str, required=True, help="Root of extracted REFUGE download")
    parser.add_argument("--output", type=str, required=True, help="Output dataset directory")
    parser.add_argument(
        "--abs-image-paths",
        action="store_true",
        help="Write absolute image paths into the manifest (default: symlink into output/images/)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to emit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    images_out = output / "images"
    if not args.abs_image_paths:
        images_out.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    total_with_masks = 0
    total_glaucoma = 0
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        rows = _scan_split(source, split)
        print(f"{split}: {len(rows)} images discovered")

        for row in rows:
            # Resolve or symlink the image path
            if args.abs_image_paths:
                image_rel = str(row.image_path)
            else:
                link_path = images_out / f"{row.stem}{row.image_path.suffix.lower()}"
                if not link_path.exists():
                    try:
                        link_path.symlink_to(row.image_path)
                    except OSError:
                        # Fall back to copy if symlinks aren't supported
                        import shutil

                        shutil.copy2(row.image_path, link_path)
                image_rel = str(link_path.relative_to(output))

            disc_rel = ""
            cup_rel = ""
            if row.mask_path is not None:
                disc_out = output / "masks" / "disc" / f"{row.stem}.png"
                cup_out = output / "masks" / "cup" / f"{row.stem}.png"
                _decode_refuge_mask(row.mask_path, disc_out, cup_out)
                disc_rel = str(disc_out.relative_to(output))
                cup_rel = str(cup_out.relative_to(output))
                total_with_masks += 1

            if row.label == 1:
                total_glaucoma += 1

            manifest_rows.append(
                {
                    "image_path": image_rel,
                    "disc_mask_path": disc_rel,
                    "cup_mask_path": cup_rel,
                    "label": str(row.label),
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
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} rows to {manifest_path}")
    print(f"  with masks: {total_with_masks}")
    print(f"  glaucoma:   {total_glaucoma}")
    print(f"  normal:     {len(manifest_rows) - total_glaucoma}")


if __name__ == "__main__":
    main()
