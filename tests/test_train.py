"""Tests for the training module."""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from src.train import compute_metrics, focal_binary_objective


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.01, 0.02, 0.03, 0.98, 0.99])
        metrics = compute_metrics(y_true, y_score)
        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["best_f1"] == 1.0

    def test_random_predictions_low_auc(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_score = rng.random(1000)
        metrics = compute_metrics(y_true, y_score)
        assert 0.4 <= metrics["roc_auc"] <= 0.6

    def test_metric_keys(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_score = rng.random(200)
        metrics = compute_metrics(y_true, y_score)
        assert set(metrics.keys()) == {"roc_auc", "pr_auc", "recall_at_1pct_fpr", "best_f1"}

    def test_values_bounded(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_score = rng.random(200)
        for v in compute_metrics(y_true, y_score).values():
            assert 0.0 <= v <= 1.0


class TestFocalLoss:
    def test_gradient_shape(self):
        rng = np.random.default_rng(42)
        n = 100
        y = rng.integers(0, 2, n).astype(np.float32)
        preds = rng.standard_normal(n)
        dtrain = xgb.DMatrix(rng.random((n, 5)), label=y)
        grad, hess = focal_binary_objective(preds, dtrain, gamma=2.0, alpha=0.25)
        assert grad.shape == (n,)
        assert hess.shape == (n,)

    def test_hessian_positive(self):
        rng = np.random.default_rng(42)
        n = 100
        y = rng.integers(0, 2, n).astype(np.float32)
        preds = rng.standard_normal(n)
        dtrain = xgb.DMatrix(rng.random((n, 5)), label=y)
        _, hess = focal_binary_objective(preds, dtrain)
        assert (hess > 0).all()

    def test_no_nans(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, n).astype(np.float32)
        preds = rng.standard_normal(n) * 5
        dtrain = xgb.DMatrix(rng.random((n, 5)), label=y)
        grad, hess = focal_binary_objective(preds, dtrain)
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
