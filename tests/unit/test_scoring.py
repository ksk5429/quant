"""Tests for scoring functions and statistical significance."""
from __future__ import annotations

import pytest
import numpy as np
from src.prediction.calibration import compute_brier, compute_ece, compute_log_loss
from src.prediction.advanced_scoring import (
    brier_decomposition, brier_skill_score, paired_brier_test, comprehensive_evaluate,
)


class TestBrier:
    def test_perfect_prediction(self):
        assert compute_brier([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0

    def test_worst_prediction(self):
        assert compute_brier([0.0, 1.0], [1.0, 0.0]) == 1.0

    def test_uniform_prediction(self):
        b = compute_brier([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0])
        assert abs(b - 0.25) < 0.001


class TestBrierDecomposition:
    def test_components_sum(self):
        preds = [0.8, 0.2, 0.6, 0.9, 0.1, 0.7]
        outs = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        d = brier_decomposition(preds, outs)
        # Brier ≈ reliability - resolution + uncertainty
        reconstructed = d.reliability - d.resolution + d.uncertainty
        assert abs(d.brier - reconstructed) < 0.01

    def test_reliability_zero_for_perfect(self):
        # Perfect calibration: predicted 0.5, observed 50%
        d = brier_decomposition([0.5] * 100, [1.0] * 50 + [0.0] * 50)
        assert d.reliability < 0.01


class TestBSS:
    def test_positive_when_better(self):
        assert brier_skill_score(0.15, 0.25) > 0

    def test_negative_when_worse(self):
        assert brier_skill_score(0.30, 0.25) < 0

    def test_zero_when_equal(self):
        assert brier_skill_score(0.25, 0.25) == 0.0


class TestPairedBrierTest:
    def test_significant_when_clearly_better(self):
        np.random.seed(42)
        # Our predictions are clearly better than market (0.5)
        ours = [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.1]
        market = [0.5] * 10
        outcomes = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        result = paired_brier_test(ours, market, outcomes, n_bootstrap=1000)
        assert result.bss > 0
        assert result.significant == True

    def test_not_significant_when_similar(self):
        ours = [0.5, 0.5, 0.5, 0.5, 0.5]
        market = [0.5, 0.5, 0.5, 0.5, 0.5]
        outcomes = [1.0, 0.0, 1.0, 0.0, 1.0]
        result = paired_brier_test(ours, market, outcomes, n_bootstrap=1000)
        assert result.bss == 0.0


class TestComprehensiveEvaluate:
    def test_returns_all_fields(self):
        preds = [0.8, 0.3, 0.6, 0.9, 0.2]
        outs = [1.0, 0.0, 1.0, 1.0, 0.0]
        m = comprehensive_evaluate(preds, outs)
        assert m.n_samples == 5
        assert 0 <= m.brier <= 1
        assert 0 <= m.ece <= 1
        assert 0 <= m.accuracy <= 1
        assert m.reliability >= 0
        assert m.resolution >= 0


class TestECE:
    def test_perfect_calibration(self):
        # Each bin has correct frequency
        ece = compute_ece([0.5] * 100, [1.0] * 50 + [0.0] * 50)
        assert ece < 0.1

    def test_overconfident(self):
        # Predict 0.9 but only 50% correct
        ece = compute_ece([0.9] * 20, [1.0] * 10 + [0.0] * 10)
        assert ece > 0.3
