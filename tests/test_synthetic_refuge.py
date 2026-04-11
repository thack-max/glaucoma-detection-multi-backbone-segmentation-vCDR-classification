"""
Tests for ``scripts/make_synthetic_refuge.py``.

The synthetic REFUGE fixture is the single bridge that lets the rest of
the Tier 1 test suite exercise the full training + inference path
without needing the real dataset. These tests lock in its contract:
  - layout matches ``scripts/prepare_refuge.py`` / FundusDataset
  - labels are balanced per split
  - vCDR ordering matches the label (glaucoma -> larger vCDR)
  - generation is deterministic for a fixed seed
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from scripts.make_synthetic_refuge import generate
from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.pipeline.common import compute_vcdr


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def test_generate_writes_expected_layout(tmp_path: Path) -> None:
    manifest = generate(
        output=tmp_path,
        per_split=6,
        image_size=64,
        splits=["train", "val", "test"],
        seed=0,
    )
    assert manifest == tmp_path / "manifest.csv"
    assert (tmp_path / "images").is_dir()
    assert (tmp_path / "masks" / "disc").is_dir()
    assert (tmp_path / "masks" / "cup").is_dir()

    rows = _read_manifest(manifest)
    assert len(rows) == 6 * 3

    # Every referenced artefact must actually exist on disk.
    for row in rows:
        assert (tmp_path / row["image_path"]).is_file()
        assert (tmp_path / row["disc_mask_path"]).is_file()
        assert (tmp_path / row["cup_mask_path"]).is_file()


def test_generate_balances_labels_per_split(tmp_path: Path) -> None:
    generate(
        output=tmp_path,
        per_split=8,
        image_size=32,
        splits=["train", "val", "test"],
        seed=1,
    )
    rows = _read_manifest(tmp_path / "manifest.csv")
    per_split: dict[str, list[int]] = {}
    for row in rows:
        per_split.setdefault(row["split"], []).append(int(row["label"]))
    for split, labels in per_split.items():
        assert labels.count(0) == labels.count(1), f"split {split} unbalanced"


def test_generate_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    generate(output=out_a, per_split=4, image_size=32, splits=["train"], seed=42)
    generate(output=out_b, per_split=4, image_size=32, splits=["train"], seed=42)

    with open(out_a / "manifest.csv") as f:
        rows_a = f.read()
    with open(out_b / "manifest.csv") as f:
        rows_b = f.read()
    assert rows_a == rows_b

    # And the actual image bytes should match for the same seed.
    from PIL import Image

    sample = rows_a.splitlines()[1].split(",")[0]  # first data row
    img_a = Image.open(out_a / sample).tobytes()
    img_b = Image.open(out_b / sample).tobytes()
    assert img_a == img_b


def test_synthetic_dataset_loads_via_fundus_dataset(tmp_path: Path) -> None:
    """The generator's manifest schema must match ``FundusDataset``."""
    generate(
        output=tmp_path,
        per_split=4,
        image_size=64,
        splits=["train"],
        seed=0,
    )
    transform = get_fundus_transforms("val", image_size=64)
    dataset = FundusDataset(tmp_path, split="train", transform=transform)
    assert len(dataset) == 4

    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["disc_mask"].shape == (64, 64)
    assert sample["cup_mask"].shape == (64, 64)
    assert "label" in sample
    assert sample["label"].dtype == torch.long
    assert "group_id" in sample


def test_vcdr_ordering_matches_label(tmp_path: Path) -> None:
    """Glaucoma-labelled synthetic cases must have a larger vCDR than
    normal cases, so that downstream classifiers and the baseline
    vCDR > 0.5 mode both produce sane signal. If this ever breaks, the
    smoke training tests lose their ability to learn anything."""
    generate(
        output=tmp_path,
        per_split=10,
        image_size=96,
        splits=["train"],
        seed=7,
    )
    dataset = FundusDataset(tmp_path, split="train", transform=None)

    normal_vcdrs: list[float] = []
    glaucoma_vcdrs: list[float] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        vcdr = compute_vcdr(sample["disc_mask"], sample["cup_mask"])
        if int(sample["label"].item()) == 1:
            glaucoma_vcdrs.append(vcdr)
        else:
            normal_vcdrs.append(vcdr)

    assert normal_vcdrs and glaucoma_vcdrs
    # Mean should separate clearly; individual samples have jitter.
    mean_normal = sum(normal_vcdrs) / len(normal_vcdrs)
    mean_glaucoma = sum(glaucoma_vcdrs) / len(glaucoma_vcdrs)
    assert mean_glaucoma > mean_normal + 0.2, (
        f"glaucoma vCDR ({mean_glaucoma:.3f}) should exceed "
        f"normal vCDR ({mean_normal:.3f}) by a clear margin"
    )
    assert mean_normal < 0.5  # baseline-mode negatives
    assert mean_glaucoma > 0.5  # baseline-mode positives
