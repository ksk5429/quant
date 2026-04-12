"""Tests for Fish prediction aggregation."""
from __future__ import annotations

import pytest
from src.mirofish.llm_fish import FishPrediction, aggregate_predictions


def make_preds(probs, confs=None):
    if confs is None:
        confs = [0.7] * len(probs)
    return [FishPrediction(f"fish_{i}", p, c) for i, (p, c) in enumerate(zip(probs, confs))]


class TestTrimmedMean:
    def test_removes_outlier(self):
        preds = make_preds([0.7, 0.7, 0.7, 0.7, 0.1])
        result = aggregate_predictions(preds, extremize=1.0, trim=True)
        # 0.1 should be trimmed, mean of [0.7,0.7,0.7,0.7] = 0.7
        assert abs(result["raw_probability"] - 0.7) < 0.02

    def test_no_trim_when_few_fish(self):
        preds = make_preds([0.3, 0.7])
        result = aggregate_predictions(preds, extremize=1.0, trim=True)
        # <5 Fish, no trimming, mean = 0.5
        assert abs(result["raw_probability"] - 0.5) < 0.02


class TestConfidenceWeighting:
    def test_high_confidence_dominates(self):
        preds = [
            FishPrediction("high", 0.80, 0.95),
            FishPrediction("low", 0.20, 0.10),
        ]
        result = aggregate_predictions(preds, extremize=1.0, trim=False)
        assert result["raw_probability"] > 0.65

    def test_equal_confidence_gives_mean(self):
        preds = make_preds([0.3, 0.7], confs=[0.5, 0.5])
        result = aggregate_predictions(preds, extremize=1.0, trim=False)
        assert abs(result["raw_probability"] - 0.5) < 0.02


class TestExtremization:
    def test_pushes_away_from_half(self):
        preds = make_preds([0.65, 0.65, 0.65, 0.65, 0.65])
        result = aggregate_predictions(preds, extremize=1.5, trim=False)
        assert result["extremized_probability"] > result["raw_probability"]

    def test_suppressed_on_high_spread(self):
        preds = make_preds([0.2, 0.8, 0.3, 0.7, 0.5])
        result = aggregate_predictions(preds, extremize=1.5, trim=False)
        # High spread → extremization should be suppressed
        diff = abs(result["extremized_probability"] - result["raw_probability"])
        assert diff < 0.05  # minimal extremization

    def test_no_extremize_when_factor_is_one(self):
        preds = make_preds([0.7, 0.7, 0.7, 0.7, 0.7])
        result = aggregate_predictions(preds, extremize=1.0, trim=False)
        assert abs(result["extremized_probability"] - result["raw_probability"]) < 0.001


class TestDisagreement:
    def test_flags_high_spread(self):
        preds = make_preds([0.2, 0.8, 0.3, 0.7, 0.5])
        result = aggregate_predictions(preds, extremize=1.0, trim=False)
        assert result["disagreement_flag"] == True

    def test_no_flag_low_spread(self):
        preds = make_preds([0.69, 0.71, 0.70, 0.70, 0.70])
        result = aggregate_predictions(preds, extremize=1.0, trim=False)
        assert result["disagreement_flag"] == False


class TestHealthCheck:
    def test_healthy_swarm(self):
        preds = make_preds([0.6, 0.7])
        result = aggregate_predictions(preds)
        assert result["swarm_healthy"] == True
        assert result["n_failed"] == 0

    def test_unhealthy_swarm(self):
        preds = [
            FishPrediction("a", 0.5, 0.1, reasoning="CLI error"),
            FishPrediction("b", 0.5, 0.1, reasoning="timeout after 120s"),
        ]
        result = aggregate_predictions(preds)
        assert result["swarm_healthy"] == False
        assert result["n_failed"] == 2
