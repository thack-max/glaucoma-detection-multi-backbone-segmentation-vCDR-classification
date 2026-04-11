"""
Generate a small synthetic longitudinal multimodal dataset for Tier 2 smoke tests.

Builds a minimal corpus compatible with ``MultimodalGlaucomaDataset``:

    <output>/
      manifest.csv            # patient_id,visit_index,timestamp_days,
                              # fundus_path,oct_path,vf_path,
                              # rnfl_thickness,iop,age,sex,
                              # group_id,progression_label
      fundus/<pid>_v<ix>.png  # small RGB fundus-like image per visit
      oct/<pid>_v<ix>.npy     # small OCT volume [1, D, H, W]
      vf/<pid>_v<ix>.npy      # VF measurements [54, 3]

Each patient has ``num_visits`` visits with irregular spacing (90 / 180 /
360 days plus noise, so Delta-t Mamba actually has non-uniform deltas to
process). Half of the patients are labelled progressing, and their
simulated RNFL thickness trajectory declines faster than the non-
progressing half. This makes the training loss monotone-ish over a few
epochs so smoke tests can assert that it decreases.

The generator is deterministic for a fixed ``--seed``.

Usage:
    python scripts/make_synthetic_longitudinal.py \\
        --output tests/fixtures/synthetic_longitudinal \\
        --num-patients 4 --num-visits 4
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


MANIFEST_COLUMNS = [
    "patient_id", "visit_index", "timestamp_days",
    "fundus_path", "oct_path", "vf_path",
    "rnfl_thickness", "iop", "age", "sex",
    "group_id", "progression_label",
]


@dataclass
class SyntheticVisit:
    patient_id: str
    visit_index: int
    timestamp_days: float
    fundus_path: str
    oct_path: str
    vf_path: str
    rnfl_thickness: float
    iop: float
    age: float
    sex: int
    group_id: int
    progression_label: int


def _render_fundus(size: int, rng: np.random.Generator) -> np.ndarray:
    """Tiny fundus-like RGB image. Good enough for DataLoader tests."""
    ys, xs = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = size / 2, size / 2
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    # Orange background with radial falloff
    bg = np.stack(
        [
            np.clip(200 - 1.5 * r, 40, 255),
            np.clip(120 - 1.0 * r, 20, 220),
            np.clip(70 - 0.8 * r, 10, 180),
        ],
        axis=-1,
    )
    bg += rng.integers(-10, 10, size=bg.shape, endpoint=False).astype(np.float32)
    # Bright circular disc in the centre
    disc = (r <= size * 0.18).astype(np.float32)[..., None]
    bg = bg * (1 - disc) + np.array([[230, 200, 140]]) * disc
    return np.clip(bg, 0, 255).astype(np.uint8)


def _render_oct_volume(
    depth: int,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tiny synthetic OCT volume ``[1, depth, height, width]``.

    Structure: a bright horizontal band simulating the RNFL layer, with
    depth-dependent noise. Not realistic — just enough geometry for the
    DataLoader and Swin-UNETR smoke tests to have non-trivial content.
    """
    volume = rng.normal(20.0, 5.0, size=(depth, height, width)).astype(np.float32)
    rnfl_row = height // 3
    volume[:, rnfl_row : rnfl_row + 2, :] += 60.0  # "bright" RNFL band
    return volume[np.newaxis, ...]  # add channel dim -> [1, D, H, W]


def _render_vf(rng: np.random.Generator, progression_stage: float) -> np.ndarray:
    """54-point VF measurements as a ``[54, 3]`` array.

    Columns: sensitivity (dB), total deviation (dB), pattern deviation (dB).
    ``progression_stage`` in [0, 1] adds a downward bias to sensitivity
    so later visits of progressing patients look visibly worse.
    """
    sensitivity = rng.normal(28.0 - 4.0 * progression_stage, 2.0, size=54)
    total_dev = sensitivity - 28.0
    pattern_dev = rng.normal(0.0, 1.5, size=54)
    return np.stack([sensitivity, total_dev, pattern_dev], axis=-1).astype(np.float32)


def _visit_timestamps(
    num_visits: int,
    rng: np.random.Generator,
) -> list[float]:
    """Irregular spacing: base 90/180/360 days with +/- 20d jitter."""
    gaps = [0.0]
    base = [90.0, 180.0, 180.0, 360.0]
    for i in range(num_visits - 1):
        gap = base[i % len(base)] + rng.uniform(-20.0, 20.0)
        gaps.append(gaps[-1] + gap)
    return gaps


def generate(
    output: Path,
    num_patients: int,
    num_visits: int,
    fundus_size: int,
    oct_shape: tuple[int, int, int],
    seed: int,
) -> Path:
    """Materialise the synthetic longitudinal dataset under ``output``.

    Returns the path to the written ``manifest.csv``. Importable by
    tests so the fixture can be built into ``tmp_path`` with no
    subprocess.
    """
    output.mkdir(parents=True, exist_ok=True)
    (output / "fundus").mkdir(exist_ok=True)
    (output / "oct").mkdir(exist_ok=True)
    (output / "vf").mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)

    rows: list[dict[str, str]] = []
    for pid_idx in range(num_patients):
        pid = f"P{pid_idx:03d}"
        # Alternate progression labels so each split/class is represented.
        progression_label = pid_idx % 2
        group_id = pid_idx % 3
        age = float(55 + rng.integers(-5, 15))
        sex = int(rng.integers(0, 2))

        timestamps = _visit_timestamps(num_visits, rng)
        # RNFL thickness declines; progressing patients decline faster
        # so the regression target has real signal.
        rnfl_start = float(rng.uniform(85.0, 100.0))
        rnfl_decline_per_day = (
            -0.02 if progression_label == 1 else -0.005
        )

        for v in range(num_visits):
            ts = timestamps[v]
            rnfl = rnfl_start + rnfl_decline_per_day * ts + rng.normal(0, 1.5)
            iop = float(np.clip(16.0 + rng.normal(0, 2.0), 8.0, 30.0))

            stem = f"{pid}_v{v:02d}"

            fundus_rel = f"fundus/{stem}.png"
            oct_rel = f"oct/{stem}.npy"
            vf_rel = f"vf/{stem}.npy"

            # Write fundus PNG
            Image.fromarray(_render_fundus(fundus_size, rng)).save(output / fundus_rel)
            # Write OCT volume as .npy
            np.save(
                output / oct_rel,
                _render_oct_volume(*oct_shape, rng=rng),
            )
            # Write VF as .npy; progression_stage scales with timestamp
            stage = 0.0
            if progression_label == 1:
                stage = min(1.0, ts / (num_visits * 180.0))
            np.save(
                output / vf_rel,
                _render_vf(rng, progression_stage=stage),
            )

            rows.append(
                {
                    "patient_id": pid,
                    "visit_index": str(v),
                    "timestamp_days": f"{ts:.1f}",
                    "fundus_path": fundus_rel,
                    "oct_path": oct_rel,
                    "vf_path": vf_rel,
                    "rnfl_thickness": f"{rnfl:.2f}",
                    "iop": f"{iop:.2f}",
                    "age": f"{age:.1f}",
                    "sex": str(sex),
                    "group_id": str(group_id),
                    "progression_label": str(progression_label),
                }
            )

    manifest = output / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic longitudinal multimodal dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/fixtures/synthetic_longitudinal",
    )
    parser.add_argument("--num-patients", type=int, default=4)
    parser.add_argument("--num-visits", type=int, default=4)
    parser.add_argument("--fundus-size", type=int, default=64)
    parser.add_argument(
        "--oct-shape",
        type=str,
        default="8,16,16",
        help="Comma-separated D,H,W for synthetic OCT volumes",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    d, h, w = (int(x) for x in args.oct_shape.split(","))
    manifest = generate(
        output=output,
        num_patients=args.num_patients,
        num_visits=args.num_visits,
        fundus_size=args.fundus_size,
        oct_shape=(d, h, w),
        seed=args.seed,
    )
    print(
        f"Wrote {args.num_patients} patients x {args.num_visits} visits"
        f" = {args.num_patients * args.num_visits} rows"
    )
    print(f"  manifest: {manifest}")


if __name__ == "__main__":
    main()
