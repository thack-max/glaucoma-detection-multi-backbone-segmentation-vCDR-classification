"""
Generate a small synthetic REFUGE-like fundus dataset for smoke tests.

The goal is to unblock end-to-end training + inference without needing to
download the real REFUGE corpus. Every artefact this script writes matches
the layout ``FundusDataset`` expects after ``scripts/prepare_refuge.py``:

    <output>/
      manifest.csv        (image_path, disc_mask_path, cup_mask_path,
                           label, split, group_id)
      images/<stem>.png   RGB fundus-like images
      masks/disc/<stem>.png   binary disc masks (255/0)
      masks/cup/<stem>.png    binary cup masks  (255/0)

Each sample is procedurally generated:
  - a warm orange retinal background with radial falloff + Gaussian noise
  - a bright circular optic disc at a slightly randomised centre
  - a darker circular optic cup inside the disc, offset so the vCDR is
    clinically plausible (~0.25 for normals, ~0.7 for glaucoma)
  - a brighter vessel-like streak to give the classifier something to
    latch on to beyond the vCDR

The generator is DETERMINISTIC for a given ``--seed`` so training smoke
tests produce reproducible manifests. Labels are balanced per split.

Usage:
    # Default: 12 images per split (train/val/test), 96x96 RGB
    python scripts/make_synthetic_refuge.py --output tests/fixtures/synthetic_refuge

    # Smaller / faster
    python scripts/make_synthetic_refuge.py \\
        --output /tmp/fake_refuge --per-split 4 --image-size 64
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


# Fundus background colour (approx. retinal orange in RGB)
BG_COLOR = np.array([180, 100, 60], dtype=np.float32)
# Optic disc base colour — brighter, yellowish
DISC_COLOR = np.array([230, 200, 140], dtype=np.float32)
# Cup is paler still
CUP_COLOR = np.array([250, 240, 210], dtype=np.float32)


@dataclass
class SyntheticSample:
    stem: str
    split: str
    label: int
    group_id: int


def _make_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Retinal-orange field with smooth radial falloff and speckle noise."""
    ys, xs = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = size / 2.0, size / 2.0
    # Radial falloff darkens the edges
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / (size / 2.0)
    falloff = np.clip(1.0 - 0.5 * r, 0.0, 1.0)[..., None]
    img = BG_COLOR[None, None, :] * falloff
    # Gentle multiplicative noise
    img *= 0.9 + 0.2 * rng.random((size, size, 1))
    return img


def _draw_disc_and_cup(
    img: np.ndarray,
    disc_radius: float,
    cup_radius: float,
    center: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint the optic disc + cup, return binary disc/cup masks."""
    size = img.shape[0]
    ys, xs = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = center
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    disc_mask = (r <= disc_radius).astype(np.uint8)
    cup_mask = (r <= cup_radius).astype(np.uint8)

    disc_px = disc_mask.astype(bool) & ~cup_mask.astype(bool)
    cup_px = cup_mask.astype(bool)

    # Soft vignette inside the disc so it's not a flat circle
    disc_shade = np.clip(1.0 - 0.3 * r / max(disc_radius, 1.0), 0.7, 1.0)[..., None]

    img[disc_px] = DISC_COLOR * disc_shade[disc_px]
    img[cup_px] = CUP_COLOR
    # A subtle bright ridge along a random arc for the classifier to find
    angle = rng.uniform(0, 2 * np.pi)
    dx = np.cos(angle) * disc_radius * 0.9
    dy = np.sin(angle) * disc_radius * 0.9
    ridge_r = np.sqrt((ys - (cy + dy)) ** 2 + (xs - (cx + dx)) ** 2)
    ridge = (ridge_r <= disc_radius * 0.15).astype(np.float32)[..., None]
    img += ridge * 25.0

    return disc_mask, cup_mask


def _render_sample(
    size: int,
    label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one RGB fundus image and its disc/cup masks.

    Glaucoma cases (label=1) get a thinner rim → larger vCDR.
    """
    img = _make_background(size, rng)

    # Disc takes ~35% of the image height. Cup radius depends on label.
    disc_radius = size * rng.uniform(0.30, 0.38) / 2.0
    if label == 1:
        cup_frac = rng.uniform(0.68, 0.82)  # large cup → high vCDR
    else:
        cup_frac = rng.uniform(0.22, 0.40)
    cup_radius = disc_radius * cup_frac

    # Slightly off-centre so tests exercise non-trivial geometry
    center = (
        size / 2.0 + rng.uniform(-size * 0.03, size * 0.03),
        size / 2.0 + rng.uniform(-size * 0.03, size * 0.03),
    )
    disc_mask, cup_mask = _draw_disc_and_cup(img, disc_radius, cup_radius, center, rng)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, disc_mask, cup_mask


def _split_plan(per_split: int, splits: list[str]) -> list[SyntheticSample]:
    """Produce a label-balanced plan across all requested splits."""
    samples: list[SyntheticSample] = []
    for split in splits:
        for i in range(per_split):
            label = i % 2  # alternate to guarantee both classes per split
            group_id = i % 3  # cycle 0/1/2 for fairness audit coverage
            samples.append(
                SyntheticSample(
                    stem=f"synthetic_{split}_{i:03d}",
                    split=split,
                    label=label,
                    group_id=group_id,
                )
            )
    return samples


def generate(
    output: Path,
    per_split: int,
    image_size: int,
    splits: list[str],
    seed: int,
) -> Path:
    """Materialise the synthetic dataset under ``output``. Returns the
    path to the written ``manifest.csv``.

    This is importable from tests, so the whole fixture can be built
    in-process without shelling out to the CLI.
    """
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "masks" / "disc").mkdir(parents=True, exist_ok=True)
    (output / "masks" / "cup").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    plan = _split_plan(per_split, splits)

    rows: list[dict[str, str]] = []
    for sample in plan:
        img, disc_mask, cup_mask = _render_sample(image_size, sample.label, rng)

        image_path = output / "images" / f"{sample.stem}.png"
        disc_path = output / "masks" / "disc" / f"{sample.stem}.png"
        cup_path = output / "masks" / "cup" / f"{sample.stem}.png"

        Image.fromarray(img).save(image_path)
        Image.fromarray(disc_mask * 255).save(disc_path)
        Image.fromarray(cup_mask * 255).save(cup_path)

        rows.append(
            {
                "image_path": str(image_path.relative_to(output)),
                "disc_mask_path": str(disc_path.relative_to(output)),
                "cup_mask_path": str(cup_path.relative_to(output)),
                "label": str(sample.label),
                "split": sample.split,
                "group_id": str(sample.group_id),
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

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic REFUGE-like dataset for smoke tests",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/fixtures/synthetic_refuge",
        help="Output directory (default: tests/fixtures/synthetic_refuge)",
    )
    parser.add_argument(
        "--per-split",
        type=int,
        default=12,
        help="Number of images per split (default: 12)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Square image side length in pixels (default: 96)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to emit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    manifest = generate(
        output=output,
        per_split=args.per_split,
        image_size=args.image_size,
        splits=splits,
        seed=args.seed,
    )
    print(f"Wrote {args.per_split * len(splits)} synthetic samples")
    print(f"  manifest: {manifest}")
    print(f"  images:   {output / 'images'}")
    print(f"  masks:    {output / 'masks'}")


if __name__ == "__main__":
    main()
