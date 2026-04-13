"""Tests for the dataset prep scripts (prepare_*.py)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.fundus_dataset import FundusDataset
from src.data.multimodal_dataset import MultimodalGlaucomaDataset
from src.data.transforms import get_fundus_transforms


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    """Run a prep script via `python scripts/<name>.py` and return its result."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _make_refuge_mask(path: Path) -> None:
    """Write a REFUGE-style 0/128/255 mask."""
    arr = np.full((64, 64), 255, dtype=np.uint8)
    arr[10:50, 10:50] = 128  # disc
    arr[20:40, 20:40] = 0    # cup
    Image.fromarray(arr).save(path)


def _make_classindex_mask(path: Path) -> None:
    """Write a 0/1/2 class-index mask."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[10:50, 10:50] = 1
    arr[20:40, 20:40] = 2
    Image.fromarray(arr).save(path)


def _make_rgb_image(path: Path, color: tuple[int, int, int] = (100, 50, 50)) -> None:
    img = Image.new("RGB", (64, 64), color)
    img.save(path)


# ---------------------------------------------------------------------------
# REFUGE
# ---------------------------------------------------------------------------


class TestPrepareRefuge:
    def test_end_to_end(self, tmp_path: Path):
        src = tmp_path / "src"
        out = tmp_path / "out"

        tr_g = src / "Training400" / "Glaucoma"
        tr_n = src / "Training400" / "Non-Glaucoma"
        gt_g = src / "Training400-GT" / "Disc_Cup_Masks" / "Glaucoma"
        gt_n = src / "Training400-GT" / "Disc_Cup_Masks" / "Non-Glaucoma"
        val_img = src / "REFUGE-Validation400"
        val_gt = src / "REFUGE-Validation400-GT" / "Disc_Cup_Masks"
        for p in [tr_g, tr_n, gt_g, gt_n, val_img, val_gt]:
            p.mkdir(parents=True)

        for i in range(2):
            _make_rgb_image(tr_g / f"g{i}.jpg")
            _make_refuge_mask(gt_g / f"g{i}.bmp")
        for i in range(3):
            _make_rgb_image(tr_n / f"n{i}.jpg")
            _make_refuge_mask(gt_n / f"n{i}.bmp")
        for i in range(2):
            _make_rgb_image(val_img / f"V{i}.jpg")
            _make_refuge_mask(val_gt / f"V{i}.bmp")

        result = _run_script("prepare_refuge.py", "--source", str(src), "--output", str(out))
        assert result.returncode == 0, result.stderr

        with open(out / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 7  # 5 train + 2 val
        splits = {r["split"] for r in rows}
        assert splits == {"train", "val"}

        glaucoma_rows = [r for r in rows if r["label"] == "1"]
        assert len(glaucoma_rows) == 2  # g0, g1 in Training400/Glaucoma

        # All rows should have masks
        assert all(r["disc_mask_path"] and r["cup_mask_path"] for r in rows)

        # Disc mask should be non-empty and larger than the cup mask
        disc = np.array(Image.open(out / rows[0]["disc_mask_path"]))
        cup = np.array(Image.open(out / rows[0]["cup_mask_path"]))
        assert disc.sum() > cup.sum() > 0

        # FundusDataset can consume the manifest
        ds = FundusDataset(out, split="train", transform=get_fundus_transforms("val", 32))
        assert len(ds) == 5
        sample = ds[0]
        assert "image" in sample and "disc_mask" in sample and "label" in sample

    def test_missing_source_exits(self, tmp_path: Path):
        result = _run_script(
            "prepare_refuge.py",
            "--source", str(tmp_path / "nonexistent"),
            "--output", str(tmp_path / "out"),
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# ORIGA
# ---------------------------------------------------------------------------


class TestPrepareOriga:
    def test_end_to_end_with_refuge_style_masks(self, tmp_path: Path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        (src / "Images").mkdir(parents=True)
        (src / "Masks").mkdir(parents=True)

        for i in range(4):
            _make_rgb_image(src / "Images" / f"img{i}.jpg")
            _make_refuge_mask(src / "Masks" / f"img{i}.png")

        # Label CSV
        with open(src / "OrigaList.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "Glaucoma"])
            writer.writeheader()
            for i in range(4):
                writer.writerow({"Filename": f"img{i}.jpg", "Glaucoma": str(i % 2)})

        result = _run_script(
            "prepare_origa.py",
            "--source", str(src),
            "--output", str(out),
            "--split-ratio", "0.5,0.25,0.25",
            "--split-seed", "0",
        )
        assert result.returncode == 0, result.stderr

        with open(out / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

        labels = {r["label"] for r in rows}
        assert labels == {"0", "1"}

        splits = {r["split"] for r in rows}
        assert splits.issubset({"train", "val", "test"})

    def test_class_index_masks_decoded(self, tmp_path: Path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        (src / "Images").mkdir(parents=True)
        (src / "Masks").mkdir(parents=True)

        _make_rgb_image(src / "Images" / "a.jpg")
        _make_classindex_mask(src / "Masks" / "a.png")

        result = _run_script(
            "prepare_origa.py",
            "--source", str(src),
            "--output", str(out),
        )
        assert result.returncode == 0, result.stderr

        disc = np.array(Image.open(out / "masks" / "disc" / "a.png"))
        cup = np.array(Image.open(out / "masks" / "cup" / "a.png"))
        assert disc.sum() > cup.sum() > 0


# ---------------------------------------------------------------------------
# G1020
# ---------------------------------------------------------------------------


class TestPrepareG1020:
    def test_end_to_end(self, tmp_path: Path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        (src / "Images_Square").mkdir(parents=True)
        (src / "Masks").mkdir(parents=True)

        for i in range(5):
            _make_rgb_image(src / "Images_Square" / f"image_{i}.jpg")
            _make_classindex_mask(src / "Masks" / f"image_{i}.png")

        # G1020.csv uses imageID + binaryLabels columns
        with open(src / "G1020.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["imageID", "binaryLabels"])
            writer.writeheader()
            for i in range(5):
                writer.writerow({"imageID": f"image_{i}.jpg", "binaryLabels": str(i % 2)})

        result = _run_script(
            "prepare_g1020.py",
            "--source", str(src),
            "--output", str(out),
            "--split-ratio", "0.6,0.2,0.2",
        )
        assert result.returncode == 0, result.stderr

        with open(out / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert all(r["disc_mask_path"] for r in rows)

        # Labels preserved
        labels = [r["label"] for r in rows]
        assert labels.count("0") == 3
        assert labels.count("1") == 2


# ---------------------------------------------------------------------------
# Longitudinal
# ---------------------------------------------------------------------------


class TestPrepareLongitudinal:
    def test_progression_label_from_slope(self, tmp_path: Path):
        metadata = tmp_path / "metadata.csv"
        data_root = tmp_path / "data"
        out = tmp_path / "out"
        data_root.mkdir()

        with open(metadata, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "patient_id", "visit_date", "fundus_path", "oct_path",
                    "vf_path", "rnfl_thickness", "iop", "age", "sex", "race",
                ],
            )
            writer.writeheader()
            # Patient 1: rapidly declining RNFL (progressing)
            for i, (date, rnfl) in enumerate([
                ("2022-01-01", 95.0),
                ("2022-07-01", 90.0),
                ("2023-01-01", 85.0),
                ("2023-07-01", 80.0),
            ]):
                writer.writerow({
                    "patient_id": "P001",
                    "visit_date": date,
                    "fundus_path": f"P001/visit{i}.jpg",
                    "oct_path": f"P001/oct{i}.npy",
                    "vf_path": f"P001/vf{i}.npy",
                    "rnfl_thickness": str(rnfl),
                    "iop": "18.0",
                    "age": "65",
                    "sex": "M",
                    "race": "Black",
                })
            # Patient 2: stable RNFL (not progressing)
            for i, (date, rnfl) in enumerate([
                ("2022-01-01", 90.0),
                ("2022-07-01", 89.5),
                ("2023-01-01", 90.2),
            ]):
                writer.writerow({
                    "patient_id": "P002",
                    "visit_date": date,
                    "fundus_path": f"P002/visit{i}.jpg",
                    "oct_path": "",
                    "vf_path": "",
                    "rnfl_thickness": str(rnfl),
                    "iop": "15.0",
                    "age": "70",
                    "sex": "F",
                    "race": "Asian",
                })

        result = _run_script(
            "prepare_longitudinal.py",
            "--metadata", str(metadata),
            "--data-root", str(data_root),
            "--output", str(out),
            "--progression-slope", "-1.5",
        )
        assert result.returncode == 0, result.stderr

        with open(out / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 7  # 4 + 3 visits

        p001_rows = [r for r in rows if r["patient_id"] == "P001"]
        p002_rows = [r for r in rows if r["patient_id"] == "P002"]

        # Progression labels stored on every visit of the patient
        assert p001_rows[0]["progression_label"] == "1"  # declining
        assert p002_rows[0]["progression_label"] == "0"  # stable

        # Group IDs mapped from race
        assert p001_rows[0]["group_id"] == "1"  # Black
        assert p002_rows[0]["group_id"] == "0"  # Asian

        # Timestamps are days since the patient's baseline
        assert float(p001_rows[0]["timestamp_days"]) == 0.0
        assert float(p001_rows[-1]["timestamp_days"]) > 400  # ~547 days

        # MultimodalGlaucomaDataset can parse the output (with load_oct/vf off
        # since our paths don't resolve to real files)
        ds = MultimodalGlaucomaDataset(
            out, observation_window=4, load_oct=False, load_vf=False,
        )
        assert len(ds) == 2

    def test_min_visits_filter(self, tmp_path: Path):
        metadata = tmp_path / "metadata.csv"
        data_root = tmp_path / "data"
        out = tmp_path / "out"
        data_root.mkdir()

        with open(metadata, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["patient_id", "visit_date", "rnfl_thickness"]
            )
            writer.writeheader()
            writer.writerow({"patient_id": "P1", "visit_date": "2022-01-01", "rnfl_thickness": "90"})
            for i in range(3):
                writer.writerow({
                    "patient_id": "P2",
                    "visit_date": f"2022-0{i + 1}-01",
                    "rnfl_thickness": str(90 - i),
                })

        result = _run_script(
            "prepare_longitudinal.py",
            "--metadata", str(metadata),
            "--data-root", str(data_root),
            "--output", str(out),
            "--min-visits", "2",
        )
        assert result.returncode == 0, result.stderr

        with open(out / "manifest.csv") as f:
            rows = list(csv.DictReader(f))

        patients = {r["patient_id"] for r in rows}
        assert patients == {"P2"}


class TestPrepareHarvardGF:
    def test_end_to_end_with_summary_csv(self, tmp_path: Path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        (src / "Training").mkdir(parents=True)
        (src / "Validation").mkdir(parents=True)

        # Synthesize 4 .npz files (2 train, 2 val) with image arrays
        rng = np.random.default_rng(0)
        for i in range(2):
            np.savez(
                src / "Training" / f"data_{i:04d}.npz",
                image=rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
            )
        for i in range(2):
            np.savez(
                src / "Validation" / f"data_{i + 100:04d}.npz",
                image=rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
            )

        with open(src / "data_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "race", "gender", "label"],
            )
            writer.writeheader()
            writer.writerow({"filename": "data_0000.npz", "race": "Asian", "gender": "F", "label": "0"})
            writer.writerow({"filename": "data_0001.npz", "race": "Black", "gender": "M", "label": "1"})
            writer.writerow({"filename": "data_0100.npz", "race": "White", "gender": "F", "label": "0"})
            writer.writerow({"filename": "data_0101.npz", "race": "Black", "gender": "M", "label": "1"})

        result = _run_script(
            "prepare_harvard_gf.py",
            "--source", str(src),
            "--output", str(out),
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

        with open(out / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

        # group_id mapped from race
        group_ids = {r["group_id"] for r in rows}
        assert group_ids == {"0", "1", "2"}

        # Labels propagated
        labels = sorted(r["label"] for r in rows)
        assert labels == ["0", "0", "1", "1"]

        # Splits assigned correctly
        splits = {r["split"] for r in rows}
        assert splits == {"train", "val"}

        # Images extracted
        for row in rows:
            assert (out / row["image_path"]).exists()


