"""Extended calibration tests including Hypothesis property-based tests."""
from __future__ import annotations

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from src.prediction.calibration import ProbabilityCalibrator


class TestAutoSelect:
    def test_temperature_for_small_n(self):
        cal = ProbabilityCalibrator(method="auto")
        cal.fit([0.6, 0.4, 0.7], [1.0, 0.0, 1.0])
        assert cal.active_method == "temperature"

    def test_histogram_for_medium_n(self):
        cal = ProbabilityCalibrator(method="auto")
        np.random.seed(42)
        preds = np.random.uniform(0.2, 0.8, 300).tolist()
        outs = (np.random.random(300) < np.array(preds)).astype(float).tolist()
        cal.fit(preds, outs)
        assert cal.active_method in ("histogram", "beta")


class TestCalibrationOutput:
    def test_output_in_range(self):
        cal = ProbabilityCalibrator(method="temperature")
        cal.fit([0.3, 0.7, 0.5, 0.9, 0.1], [0.0, 1.0, 1.0, 1.0, 0.0])
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            result = cal.calibrate(p)
            assert 0.01 <= result <= 0.99, f"calibrate({p}) = {result}"

    def test_batch_matches_individual(self):
        cal = ProbabilityCalibrator(method="temperature")
        cal.fit([0.3, 0.7, 0.5], [0.0, 1.0, 1.0])
        inputs = [0.2, 0.5, 0.8]
        batch = cal.calibrate_batch(inputs)
        individual = [cal.calibrate(p) for p in inputs]
        for b, i in zip(batch, individual):
            assert abs(b - i) < 0.001


class TestRollingWindow:
    def test_eviction(self):
        cal = ProbabilityCalibrator(method="temperature", max_history=10)
        for i in range(20):
            cal.fit([0.5], [float(i % 2)])
        assert cal.training_size == 10


class TestConformalResiduals:
    def test_holdout_split(self):
        cal = ProbabilityCalibrator(method="temperature")
        cal.fit(list(np.linspace(0.1, 0.9, 50)), [1.0] * 25 + [0.0] * 25)
        holdout_p, holdout_o = cal.get_conformal_residuals(holdout_fraction=0.3)
        assert len(holdout_p) > 0
        assert len(holdout_p) < 50

    def test_empty_when_too_few(self):
        cal = ProbabilityCalibrator(method="temperature")
        cal.fit([0.5, 0.6], [1.0, 0.0])
        holdout_p, holdout_o = cal.get_conformal_residuals()
        assert len(holdout_p) == 0


# ── Hypothesis Property-Based Tests ──

class TestPropertyBased:
    @given(st.floats(min_value=0.01, max_value=0.99))
    @settings(max_examples=200)
    def test_calibrate_always_valid(self, p):
        """Calibrator output is always in [0.01, 0.99]."""
        cal = ProbabilityCalibrator(method="temperature")
        cal.fit([0.3, 0.7, 0.5], [0.0, 1.0, 1.0])
        result = cal.calibrate(p)
        assert 0.01 <= result <= 0.99

    @given(st.lists(
        st.floats(min_value=0.05, max_value=0.95),
        min_size=3, max_size=20,
    ))
    @settings(max_examples=100)
    def test_calibrate_batch_same_length(self, probabilities):
        """Batch calibration returns same number of values."""
        cal = ProbabilityCalibrator(method="temperature")
        cal.fit([0.3, 0.7, 0.5], [0.0, 1.0, 1.0])
        result = cal.calibrate_batch(probabilities)
        assert len(result) == len(probabilities)
