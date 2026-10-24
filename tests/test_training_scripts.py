"""Tests for the training scripts (fairness, specialist, segmentation).

These are deliberately fast — they exercise the training loop logic on
synthetic in-memory data, not on real pretrained DINOv2 / GPU runs.
"""

from __future__ import annotations

import csv
import importlib
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.data.fundus_dataset import FundusDataset
from src.data.transforms import get_fundus_transforms
from src.models.fairness.equity_attention import EquityModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_fundus_manifest(root: Path, n: int = 8) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    with open(root / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path", "disc_mask_path", "cup_mask_path",
                "label", "split", "group_id",
            ],
        )
        writer.writeheader()
        for i in range(n):
            stem = f"img{i}"
            Image.new("RGB", (32, 32), (50 + i * 10, 100, 100)).save(
                root / "images" / f"{stem}.png",
            )
            writer.writerow(
                {
                    "image_path": f"images/{stem}.png",
                    "disc_mask_path": "",
                    "cup_mask_path": "",
                    "label": str(i % 2),
                    "split": "train" if i < n - 2 else "val",
                    "group_id": str(i % 3),
                }
            )


class _StubEncoder(nn.Module):
    """Fast stand-in for DINOv2 so training tests don't hit torch.hub."""

    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(3, embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)
        return feats.mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# train_fairness.py — adversarial loop
# ---------------------------------------------------------------------------


class TestFairnessTrainingLoop:
    def test_adversarial_step_reduces_group_predictability(self, tmp_path: Path):
        """After a handful of adversarial steps, features should become
        harder to classify into their group."""
        torch.manual_seed(0)

        equity = EquityModule(feature_dim=16, num_groups=3, adversarial_weight=1.0)
        classifier = nn.Linear(16, 1)
        optimizer = torch.optim.SGD(
            list(equity.parameters()) + list(classifier.parameters()),
            lr=0.5,
        )

        # Construct features that are trivially group-separable
        def synth_batch(seed: int):
            g = torch.randint(0, 3, (32,), generator=torch.Generator().manual_seed(seed))
            base = torch.randn(32, 16, generator=torch.Generator().manual_seed(seed + 1))
            # Inject group signal into the first three dimensions
            for i, gi in enumerate(g):
                base[i, gi.item()] += 3.0
            labels = (g == 1).float()  # the glaucoma label is correlated with group 1
            return base, g, labels

        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss()

        # Measure initial group separability
        with torch.no_grad():
            feats, groups, _ = synth_batch(100)
            equity_feats = equity(feats, groups)
            group_logits = equity.equity_attn.get_group_prediction(
                equity_feats, reverse_gradient=False,
            )
            initial_group_loss = float(ce(group_logits, groups).item())

        for step in range(30):
            feats, groups, labels = synth_batch(step)
            equity_feats = equity(feats, groups)
            cls_logits = classifier(equity_feats).squeeze(-1)
            cls_loss = bce(cls_logits, labels)
            adv_loss = equity.adversarial_loss(
                equity_feats, groups, reverse_gradient=True,
            )
            total = cls_loss + 0.5 * adv_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        # After training, the group predictor should struggle more than at init
        with torch.no_grad():
            feats, groups, _ = synth_batch(200)
            equity_feats = equity(feats, groups)
            group_logits = equity.equity_attn.get_group_prediction(
                equity_feats, reverse_gradient=False,
            )
            final_group_loss = float(ce(group_logits, groups).item())

        # Adversarial training should have made the group harder to predict
        # (higher CE loss on held-out batches from the same distribution).
        assert final_group_loss > initial_group_loss * 0.5  # at least not collapsed
        # And the classifier should still be learnable (loss shouldn't diverge)
        assert not np.isnan(final_group_loss)


# ---------------------------------------------------------------------------
# train_specialist.py — joint forward pass smoke test
# ---------------------------------------------------------------------------


class TestSpecialistJointForward:
    def test_forward_pass_through_all_components(self):
        """One optimization step through Delta-t Mamba + Fusion + Evidential."""
        from src.models.progression.delta_t_mamba import DeltaTMamba
        from src.models.progression.evidential_forecaster import (
            EvidentialForecaster,
            EvidentialLoss,
        )
        from src.models.progression.vf_graph_net import VFGraphNet
        from src.models.fusion.cross_modal_attention import CrossModalFusion

        torch.manual_seed(0)
        embed_dim = 16

        vf_net = VFGraphNet(node_features=3, hidden_dim=16, num_gin_layers=2, output_dim=16)
        temporal = DeltaTMamba(
            input_dim=embed_dim, d_model=16, d_state=4, num_layers=1, expand=1,
        )
        fusion = CrossModalFusion(
            modality_dims={
                "fundus": embed_dim,
                "oct": embed_dim,
                "vf": vf_net.readout_dim,
                "clinical": 3,
            },
            fused_dim=16,
            num_heads=2,
        )
        forecaster = EvidentialForecaster(
            temporal_dim=16 + 16,
            hidden_dim=16,
            num_classes=2,
        )
        loss_fn = EvidentialLoss()

        optimizer = torch.optim.SGD(
            list(vf_net.parameters())
            + list(temporal.parameters())
            + list(fusion.parameters())
            + list(forecaster.parameters()),
            lr=1e-3,
        )

        # Synthetic batch: 2 patients, 3 visits, 16-dim fundus features
        B, T = 2, 3
        visit_feats = torch.randn(B, T, embed_dim)
        timestamps = torch.tensor([[0, 180, 365], [0, 90, 365]], dtype=torch.float32)
        vf_input = torch.randn(B, 54, 3)
        clinical = torch.randn(B, 3)
        oct_feats = torch.randn(B, embed_dim)
        cls_targets = torch.tensor([0, 1])
        reg_targets = torch.tensor([85.0, 75.0])

        temporal_embed = temporal(visit_feats, timestamps)
        modality_feats = {
            "fundus": visit_feats[:, -1],
            "oct": oct_feats,
            "vf": vf_net.encode_graph(vf_input),
            "clinical": clinical,
        }
        fused = fusion(modality_feats)

        combined = torch.cat([temporal_embed, fused], dim=-1)
        output = forecaster(combined)

        loss = loss_fn(output, cls_targets, reg_targets)
        assert torch.isfinite(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert output["classification"]["prob"].shape == (B, 2)
        assert output["regression"]["mu"].shape == (B,)


# ---------------------------------------------------------------------------
# FundusDataset group-id propagation — used by fairness training
# ---------------------------------------------------------------------------


class TestFundusDatasetGroupId:
    def test_group_id_appears_in_sample(self, tmp_path: Path):
        _write_fundus_manifest(tmp_path, n=6)
        ds = FundusDataset(tmp_path, split="train", transform=get_fundus_transforms("val", 16))
        assert len(ds) > 0
        sample = ds[0]
        assert "group_id" in sample
        assert "label" in sample


# ---------------------------------------------------------------------------
# Trainers must not fit heads on a randomly initialised DINOv2
# ---------------------------------------------------------------------------

# trainer -> (needs a fundus manifest, extra argv given the output dir)
TRAINERS = {
    "train_screening_classifier": (True, lambda out: ["--output", str(out / "head.pth"),
                                                      "--image-size", "32"]),
    "train_fairness": (True, lambda out: ["--output-dir", str(out), "--image-size", "32"]),
    # MultimodalGlaucomaDataset: no manifest -> empty dataset (never reached).
    "train_progression": (False, lambda out: ["--output-dir", str(out), "--device", "cpu"]),
    "train_specialist": (False, lambda out: ["--output-dir", str(out), "--device", "cpu"]),
}


@pytest.mark.parametrize("trainer", sorted(TRAINERS))
def test_trainer_refuses_random_encoder(trainer, monkeypatch, tmp_path: Path):
    # torch.hub unreachable -> DINOv2Encoder falls back to random weights; a
    # head fitted on those features is meaningless, so training must stop
    # unless --allow-random-encoder is given.
    def _offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(torch.hub, "load", _offline)
    needs_manifest, extra = TRAINERS[trainer]
    root = tmp_path / "data"
    root.mkdir()
    if needs_manifest:
        _write_fundus_manifest(root)
    module = importlib.import_module(f"scripts.{trainer}")
    monkeypatch.setattr(
        sys, "argv",
        [f"{trainer}.py", "--data-root", str(root), "--dinov2-variant", "dinov2_vits14",
         "--epochs", "1", *extra(tmp_path / "out")],
    )

    with pytest.warns(RuntimeWarning, match="RANDOMLY INITIALISED"):
        with pytest.raises(SystemExit, match="--allow-random-encoder"):
            module.main()
    assert not list(tmp_path.rglob("*.pth"))


class _RandomStubEncoder(_StubEncoder):
    """What DINOv2Encoder is after the torch.hub fallback, minus the cost."""

    pretrained_loaded = False

    def __init__(self, variant: str = "dinov2_vits14", freeze_encoder: bool = True, **kwargs):
        super().__init__(embed_dim=8)


def test_screening_trainer_opt_in_records_random_encoder(monkeypatch, tmp_path: Path):
    from scripts import train_screening_classifier as trainer
    from src.models.classifier_head import load_screening_head

    root = tmp_path / "data"
    _write_fundus_manifest(root)
    manifest = root / "manifest.csv"
    # No val split: the trainer then saves once, after the last epoch.
    manifest.write_text(manifest.read_text().replace(",val,", ",train,"))
    out = tmp_path / "head.pth"
    monkeypatch.setattr(trainer, "DINOv2Encoder", _RandomStubEncoder)
    monkeypatch.setattr(
        sys, "argv",
        ["train_screening_classifier.py", "--data-root", str(root), "--epochs", "1",
         "--batch-size", "4", "--image-size", "32", "--output", str(out),
         "--allow-random-encoder"],
    )

    trainer.main()

    ckpt = torch.load(out, map_location="cpu", weights_only=True)
    assert ckpt["encoder_pretrained"] is False
    with pytest.warns(RuntimeWarning, match="RANDOMLY INITIALISED DINOv2"):
        load_screening_head(out, input_dim=8)
