"""Tests for the src/data module (datasets + fundus transforms)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.transforms import FundusTransform, get_fundus_transforms
from src.data.fundus_dataset import FundusDataset


# ---------------------------------------------------------------------------
# FundusTransform
# ---------------------------------------------------------------------------


def test_fundus_transform_output_shape():
    transform = get_fundus_transforms("val", image_size=64)
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    out = transform(image)
    assert out["image"].shape == (3, 64, 64)
    assert out["image"].dtype == torch.float32


def test_fundus_transform_with_masks():
    transform = get_fundus_transforms("val", image_size=32)
    image = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    disc = np.zeros((96, 96), dtype=np.uint8)
    disc[20:60, 20:60] = 1
    cup = np.zeros((96, 96), dtype=np.uint8)
    cup[30:50, 30:50] = 1

    out = transform(image, disc_mask=disc, cup_mask=cup)
    assert out["disc_mask"].shape == (32, 32)
    assert out["cup_mask"].shape == (32, 32)


def test_fundus_transform_normalize():
    transform = get_fundus_transforms("val", image_size=16)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = transform(image)
    # After normalization, the mean should be roughly (128/255 - mean)/std
    assert torch.isfinite(out["image"]).all()


def test_train_mode_augmentation_runs():
    transform = FundusTransform(image_size=16, train=True)
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    for _ in range(5):
        out = transform(image)
        assert out["image"].shape == (3, 16, 16)


# ---------------------------------------------------------------------------
# FundusDataset
# ---------------------------------------------------------------------------


def test_fundus_dataset_empty(tmp_path: Path):
    """Dataset with no manifest should be empty."""
    ds = FundusDataset(tmp_path, split="train")
    assert len(ds) == 0


def test_fundus_dataset_with_manifest(tmp_path: Path):
    """Dataset loads entries from a manifest.csv file."""
    # Create fake images
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()

    for i in range(3):
        img = Image.fromarray(
            (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        )
        img.save(tmp_path / "images" / f"img{i}.png")
        mask = Image.fromarray(
            (np.random.rand(32, 32) > 0.5).astype(np.uint8) * 255
        )
        mask.save(tmp_path / "masks" / f"disc{i}.png")

    # Write manifest
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path", "disc_mask_path", "cup_mask_path",
                "label", "split", "group_id",
            ],
        )
        writer.writeheader()
        for i in range(3):
            writer.writerow({
                "image_path": f"images/img{i}.png",
                "disc_mask_path": f"masks/disc{i}.png",
                "cup_mask_path": "",
                "label": str(i % 2),
                "split": "train" if i < 2 else "val",
                "group_id": "0",
            })

    train_ds = FundusDataset(
        tmp_path, split="train", transform=get_fundus_transforms("val", 16),
    )
    val_ds = FundusDataset(
        tmp_path, split="val", transform=get_fundus_transforms("val", 16),
    )

    assert len(train_ds) == 2
    assert len(val_ds) == 1

    item = train_ds[0]
    assert "image" in item
    assert item["image"].shape == (3, 16, 16)
    assert "label" in item
    assert "disc_mask" in item


# ---------------------------------------------------------------------------
# MultimodalGlaucomaDataset
# ---------------------------------------------------------------------------


def test_multimodal_dataset_empty(tmp_path: Path):
    from src.data.multimodal_dataset import MultimodalGlaucomaDataset
    ds = MultimodalGlaucomaDataset(tmp_path, observation_window=4)
    assert len(ds) == 0


def test_multimodal_dataset_trajectory_padding(tmp_path: Path):
    """Visits are padded/truncated to the observation window."""
    from src.data.multimodal_dataset import MultimodalGlaucomaDataset

    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id", "visit_index", "timestamp_days",
                "fundus_path", "oct_path", "vf_path",
                "rnfl_thickness", "iop", "age", "sex",
                "group_id", "progression_label",
            ],
        )
        writer.writeheader()
        # Patient 1 has 2 visits — should be padded to 4
        for i, day in enumerate([0, 180]):
            writer.writerow({
                "patient_id": "P001",
                "visit_index": i,
                "timestamp_days": day,
                "fundus_path": "",
                "oct_path": "",
                "vf_path": "",
                "rnfl_thickness": "80.0",
                "iop": "15.0",
                "age": "65.0",
                "sex": "1",
                "group_id": "0",
                "progression_label": "1",
            })
        # Patient 2 has 6 visits — should be truncated to 4
        for i in range(6):
            writer.writerow({
                "patient_id": "P002",
                "visit_index": i,
                "timestamp_days": i * 90,
                "fundus_path": "",
                "oct_path": "",
                "vf_path": "",
                "rnfl_thickness": "90.0",
                "iop": "18.0",
                "age": "70.0",
                "sex": "0",
                "group_id": "1",
                "progression_label": "0",
            })

    ds = MultimodalGlaucomaDataset(
        tmp_path, observation_window=4, load_oct=False, load_vf=False,
    )
    assert len(ds) == 2

    item = ds[0]
    assert item["fundus"].shape[0] == 4  # T=4 visits
    assert item["timestamps"].shape == (4,)
    assert item["clinical"].shape == (4, 3)
    assert "progression_label" in item
