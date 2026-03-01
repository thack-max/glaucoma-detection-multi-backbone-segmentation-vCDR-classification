"""Tests for evaluation metrics."""

import numpy as np
import torch
import pytest

from src.evaluation.metrics import dice_coefficient, iou_score, compute_auc


def test_dice_perfect_overlap():
    pred = torch.ones(64, 64)
    target = torch.ones(64, 64)
    assert dice_coefficient(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_dice_no_overlap():
    pred = torch.zeros(64, 64)
    target = torch.ones(64, 64)
    assert dice_coefficient(pred, target) == pytest.approx(0.0, abs=1e-3)


def test_dice_partial_overlap():
    pred = torch.zeros(64, 64)
    target = torch.zeros(64, 64)
    pred[:32, :] = 1.0
    target[:32, :] = 1.0
    assert dice_coefficient(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_iou_perfect_overlap():
    pred = torch.ones(64, 64)
    target = torch.ones(64, 64)
    assert iou_score(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_iou_no_overlap():
    pred = torch.zeros(64, 64)
    target = torch.ones(64, 64)
    assert iou_score(pred, target) == pytest.approx(0.0, abs=1e-3)


def test_dice_iou_relationship():
    """Dice = 2*IoU / (1+IoU) for binary masks."""
    pred = torch.zeros(100, 100)
    target = torch.zeros(100, 100)
    pred[:60, :60] = 1.0
    target[:50, :50] = 1.0

    d = dice_coefficient(pred, target)
    i = iou_score(pred, target)
    expected_dice = 2 * i / (1 + i)
    assert d == pytest.approx(expected_dice, abs=1e-3)


def test_auc_perfect():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auc(labels, scores) == pytest.approx(1.0)


def test_auc_random():
    np.random.seed(42)
    labels = np.array([0] * 500 + [1] * 500)
    scores = np.random.rand(1000)
    auc = compute_auc(labels, scores)
    assert 0.4 < auc < 0.6  # Should be near 0.5 for random
