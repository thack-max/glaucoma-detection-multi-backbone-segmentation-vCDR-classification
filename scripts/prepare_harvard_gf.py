"""
Prepare the Harvard Glaucoma Fairness (Harvard-GF) dataset manifest.

Harvard-GF (IEEE TMI 2024, PMC11251413) is a 3,300-subject dataset used
for fairness benchmarking across racial/ethnic groups. It is distributed
with each subject as a .npz file containing the fundus image plus
demographic and label metadata, organised as:

    Harvard-GF/
      Training/
        data_00001.npz
        data_00002.npz
        ...
      Validation/
      Test/
      data_summary.csv          # filename, race, gender, age, label

This script extracts each .npz, writes the image as PNG into an output
directory, and produces a `manifest.csv` consumable by FundusDataset
with `group_id` populated for fairness audits.

Group encoding (matches the Harvard-GF paper):
    0 = Asian
    1 = Black / African American
    2 = White / Caucasian

Usage:
    python scripts/prepare_harvard_gf.py \
        --source /data/Harvard-GF \
        --output data/harvard_gf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


SPLIT_DIRS = {
    "train": ["Training", "training", "train"],
    "val": ["Validation", "validation", "val"],
    "test": ["Test", "test"],
}

GROUP_MAP = {
    "asian": 0,
    "black": 1,
    "african american": 1,
    "african-american": 1,
    "white": 2,
    "caucasian": 2,
}


def _find_split_dir(source: Path, split: str) -> Path | None:
    for candidate in SPLIT_DIRS[split]:
        d = source / candidate
        if d.is_dir():
            return d
    return None


def _load_summary(source: Path) -> dict[str, dict[str, str]]:
    """Map filename stem -> {race, gender, age, label} from data_summary.csv.

    Harvard-GF distributes a flat CSV mapping each .npz file stem to
    its demographics. We try a few known column-name variants.
    """
    summary_paths = [source / "data_summary.csv", source / "summary.csv"]
    csv_path = next((p for p in summary_paths if p.exists()), None)
    if csv_path is None:
        return {}

    out: dict[str, dict[str, str]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_lower = {k.lower().strip(): (v or "").strip() for k, v in row.items()}
            stem = None
            for key in ("filename", "file", "name", "data_file", "id"):
                if key in row_lower and row_lower[key]:
                    stem = Path(row_lower[key]).stem
                    break
            if stem is None:
                continue
            out[stem] = row_lower
    return out


def _race_to_group(race: str) -> int | None:
    if not race:
        return None
    return GROUP_MAP.get(race.strip().lower(), None)


def _label_from_row(row: dict[str, str]) -> int | None:
    for key in ("label", "glaucoma", "diagnosis"):
        if key in row and row[key] != "":
            val = row[key].lower()
            if val in {"1", "glaucoma", "yes", "true"}:
                return 1
            if val in {"0", "normal", "healthy", "no", "false"}:
                return 0
    return None


def _extract_npz(npz_path: Path, image_out: Path) -> dict[str, str] | None:
    """Load an Harvard-GF .npz and write the image as PNG.

    Returns metadata extracted from the .npz, or None if no image found.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None

    img_keys = [k for k in data.files if k.lower() in {"image", "img", "fundus", "slo_fundus"}]
    if not img_keys:
        return None

    img = data[img_keys[0]]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    image_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(image_out)

    # Try to pull demographics directly from the .npz if the summary CSV
    # is absent.
    meta: dict[str, str] = {}
    for key in data.files:
        if key.lower() in {"race", "ethnicity", "gender", "sex", "age", "label", "glaucoma"}:
            meta[key.lower()] = str(data[key].item() if hasattr(data[key], "item") else data[key])
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Harvard-GF manifest")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="If >0, process at most N files per split (useful for smoke tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(source)
    print(f"Loaded summary CSV with {len(summary)} entries" if summary else "No summary CSV found")

    images_root = output / "images"
    images_root.mkdir(exist_ok=True)

    rows: list[dict[str, str]] = []
    counts = {"train": 0, "val": 0, "test": 0}
    glaucoma = 0
    with_group = 0

    for split in ("train", "val", "test"):
        split_dir = _find_split_dir(source, split)
        if split_dir is None:
            print(f"  [skip] {split}: no directory found")
            continue

        npz_files = sorted(split_dir.glob("*.npz"))
        if args.limit_per_split > 0:
            npz_files = npz_files[: args.limit_per_split]
        print(f"  {split}: {len(npz_files)} files")

        for npz_path in npz_files:
            stem = npz_path.stem
            image_out = images_root / split / f"{stem}.png"
            meta = _extract_npz(npz_path, image_out)
            if meta is None:
                continue

            csv_meta = summary.get(stem, {})
            combined = {**meta, **csv_meta}  # CSV entries override .npz metadata

            label = _label_from_row(combined)
            group_id = _race_to_group(
                combined.get("race") or combined.get("ethnicity", "")
            )

            counts[split] += 1
            if label == 1:
                glaucoma += 1
            if group_id is not None:
                with_group += 1

            rows.append(
                {
                    "image_path": str(image_out.relative_to(output)),
                    "disc_mask_path": "",
                    "cup_mask_path": "",
                    "label": str(label) if label is not None else "",
                    "split": split,
                    "group_id": str(group_id) if group_id is not None else "",
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
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {manifest}")
    print(f"  train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"  glaucoma:    {glaucoma}")
    print(f"  with group:  {with_group}")


if __name__ == "__main__":
    main()
