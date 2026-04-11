"""End-to-end test for the baseline vs GlaucoMamba A/B evaluator."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_manifest(root: Path, n: int = 4) -> None:
    """Create a tiny FundusDataset layout with labels + group_ids."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks" / "disc").mkdir(parents=True, exist_ok=True)
    (root / "masks" / "cup").mkdir(parents=True, exist_ok=True)

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
            Image.new("RGB", (64, 64), (100 + i * 20, 80, 60)).save(
                root / "images" / f"{stem}.png",
            )
            disc = np.zeros((64, 64), dtype=np.uint8)
            disc[20:44, 20:44] = 255
            Image.fromarray(disc).save(root / "masks" / "disc" / f"{stem}.png")
            cup = np.zeros((64, 64), dtype=np.uint8)
            cup[28:36, 28:36] = 255
            Image.fromarray(cup).save(root / "masks" / "cup" / f"{stem}.png")

            writer.writerow(
                {
                    "image_path": f"images/{stem}.png",
                    "disc_mask_path": f"masks/disc/{stem}.png",
                    "cup_mask_path": f"masks/cup/{stem}.png",
                    "label": str(i % 2),
                    "split": "test",
                    "group_id": str(i % 2),
                }
            )


class TestEvaluateABEvaluation:
    def test_ab_evaluation_produces_artifacts(self, tmp_path: Path):
        """Full CLI run: informational tables + A/B metrics with no checkpoints."""
        data_root = tmp_path / "data"
        output_dir = tmp_path / "eval"
        _build_manifest(data_root, n=6)

        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate.py"),
                "--tier", "screening",
                "--data-root", str(data_root),
                "--split", "test",
                "--output-dir", str(output_dir),
                "--device", "cpu",
                "--image-size", "64",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        # Should succeed even without checkpoints (falls back to random init)
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

        # Informational table printed
        assert "TIER 1: SCREENING METRICS" in result.stdout
        # A/B header printed
        assert "BASELINE (vCDR > 0.5) vs GLAUCOMAMBA" in result.stdout

        # Artifacts written
        csv_path = output_dir / "baseline_vs_glaucomamba.csv"
        metrics_path = output_dir / "metrics.json"
        assert csv_path.exists()
        assert metrics_path.exists()

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        # Every row has the expected columns populated
        for row in rows:
            assert row["label"] in {"0", "1"}
            assert float(row["baseline_score"]) >= 0.0
            assert float(row["glaucomamba_score"]) >= 0.0

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "baseline" in metrics
        assert "glaucomamba" in metrics
        assert "delta" in metrics
        assert metrics["baseline"]["n"] == 6
        assert metrics["glaucomamba"]["n"] == 6

        # Fairness report written because we injected group_id
        fairness_path = output_dir / "fairness_report.json"
        assert fairness_path.exists()
        with open(fairness_path) as f:
            report = json.load(f)
        assert "per_group_auc" in report
        assert "es_auc" in report

    def test_informational_only_no_data(self):
        """Running without --data-root still prints the comparison tables."""
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate.py"),
                "--tier", "both",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        assert result.returncode == 0
        assert "TIER 1: SCREENING METRICS" in result.stdout
        assert "TIER 2: SPECIALIST METRICS" in result.stdout
        assert "BASELINE (vCDR > 0.5) vs GLAUCOMAMBA" not in result.stdout
