"""Tests for probability calibration module."""

import numpy as np
import pytest

from src.prediction.calibration import ProbabilityCalibrator, CalibrationMetrics


class TestCalibratorCreation:
    def test_isotonic_calibrator(self):
        cal = ProbabilityCalibrator(method="isotonic")
        assert cal.method == "isotonic"
        assert not cal.is_fitted

    def test_platt_calibrator(self):
        cal = ProbabilityCalibrator(method="platt")
        assert cal.method == "platt"

    def test_temperature_calibrator(self):
        cal = ProbabilityCalibrator(method="temperature")
        assert cal.method == "temperature"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            ProbabilityCalibrator(method="unknown")


class TestCalibration:
    def _generate_overconfident_data(self, n: int = 200):
        """Generate data where predictions are systematically overconfident."""
        rng = np.random.RandomState(42)
        true_probs = rng.uniform(0.1, 0.9, n)
        # Overconfident: push predictions away from 0.5
        predictions = 0.5 + (true_probs - 0.5) * 1.5
        predictions = np.clip(predictions, 0.01, 0.99)
        outcomes = (rng.uniform(0, 1, n) < true_probs).astype(float)
        return predictions.tolist(), outcomes.tolist()

    def test_isotonic_fit_and_calibrate(self):
        preds, outs = self._generate_overconfident_data(500)
        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(preds, outs)

        assert cal.is_fitted
        assert cal.training_size == 500

        # Calibrated value should be in [0, 1]
        result = cal.calibrate(0.85)
        assert 0.0 <= result <= 1.0

    def test_platt_fit_and_calibrate(self):
        preds, outs = self._generate_overconfident_data(200)
        cal = ProbabilityCalibrator(method="platt")
        cal.fit(preds, outs)

        result = cal.calibrate(0.7)
        assert 0.0 <= result <= 1.0

    def test_temperature_fit_and_calibrate(self):
        preds, outs = self._generate_overconfident_data(200)
        cal = ProbabilityCalibrator(method="temperature")
        cal.fit(preds, outs)

        result = cal.calibrate(0.8)
        assert 0.0 <= result <= 1.0

    def test_unfitted_returns_raw(self):
        cal = ProbabilityCalibrator(method="isotonic")
        assert cal.calibrate(0.75) == 0.75

    def test_calibrate_batch(self):
        preds, outs = self._generate_overconfident_data(300)
        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(preds, outs)

        batch = cal.calibrate_batch([0.3, 0.5, 0.7, 0.9])
        assert len(batch) == 4
        assert all(0.0 <= p <= 1.0 for p in batch)


class TestCalibrationMetrics:
    def test_evaluate_returns_metrics(self):
        cal = ProbabilityCalibrator(method="isotonic")
        preds = [0.8, 0.3, 0.9, 0.1, 0.6]
        outs = [1.0, 0.0, 1.0, 0.0, 1.0]

        metrics = cal.evaluate(preds, outs)

        assert isinstance(metrics, CalibrationMetrics)
        assert 0.0 <= metrics.brier_score <= 1.0
        assert 0.0 <= metrics.ece <= 1.0
        assert 0.0 <= metrics.mce <= 1.0
        assert metrics.log_loss > 0
        assert metrics.n_samples == 5

    def test_perfect_predictions(self):
        cal = ProbabilityCalibrator(method="isotonic")
        preds = [1.0, 0.0, 1.0, 0.0]
        outs = [1.0, 0.0, 1.0, 0.0]

        metrics = cal.evaluate(preds, outs)
        assert metrics.brier_score < 0.01

    def test_is_well_calibrated(self):
        metrics = CalibrationMetrics(
            brier_score=0.15, ece=0.05, mce=0.10,
            log_loss=0.4, n_samples=100,
        )
        assert metrics.is_well_calibrated(brier_target=0.18)
        assert not metrics.is_well_calibrated(brier_target=0.10)
