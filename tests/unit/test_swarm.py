"""Tests for the Swarm orchestrator — ensemble prediction and aggregation."""

import asyncio

import numpy as np
import pytest

from src.mirofish.swarm import Swarm, SwarmPrediction
from src.mirofish.fish import FishPersona


class TestSwarmCreation:
    def test_default_swarm(self):
        swarm = Swarm(num_fish=5)
        assert len(swarm.fish) == 5
        assert swarm.bus.subscriber_count == 5

    def test_custom_personas(self):
        personas = [FishPersona.FINANCIAL_QUANT, FishPersona.CONTRARIAN_THINKER]
        swarm = Swarm(num_fish=2, personas=personas)
        assert swarm.fish[0].persona == FishPersona.FINANCIAL_QUANT
        assert swarm.fish[1].persona == FishPersona.CONTRARIAN_THINKER

    def test_swarm_has_god_node(self):
        swarm = Swarm(num_fish=3)
        assert swarm.god is not None


class TestSwarmPrediction:
    @pytest.mark.asyncio
    async def test_analyze_market_returns_prediction(self):
        swarm = Swarm(num_fish=5)
        prediction = await swarm.analyze_market(
            market_id="test-001",
            market_question="Will Bitcoin exceed $100k by Dec 2026?",
        )

        assert isinstance(prediction, SwarmPrediction)
        assert 0.0 <= prediction.probability <= 1.0
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.market_id == "test-001"
        assert len(prediction.fish_analyses) == 5

    @pytest.mark.asyncio
    async def test_prediction_includes_spread(self):
        swarm = Swarm(num_fish=7)
        prediction = await swarm.analyze_market(
            market_id="test-002",
            market_question="Will the Fed cut rates?",
        )

        assert prediction.spread >= 0.0
        assert prediction.std_dev >= 0.0

    @pytest.mark.asyncio
    async def test_market_price_creates_edge(self):
        swarm = Swarm(num_fish=3)
        prediction = await swarm.analyze_market(
            market_id="test-003",
            market_question="Test question?",
            current_price=0.65,
        )

        assert prediction.market_price == 0.65
        assert prediction.edge is not None
        assert prediction.edge == pytest.approx(
            prediction.probability - 0.65, abs=1e-4
        )

    @pytest.mark.asyncio
    async def test_prediction_stored_in_history(self):
        swarm = Swarm(num_fish=3)
        assert len(swarm.prediction_history) == 0

        await swarm.analyze_market("m1", "Q1?")
        assert len(swarm.prediction_history) == 1

        await swarm.analyze_market("m2", "Q2?")
        assert len(swarm.prediction_history) == 2


class TestSwarmAggregation:
    def test_bayesian_weighted_within_range(self):
        swarm = Swarm(num_fish=5)
        probabilities = np.array([0.3, 0.5, 0.7, 0.6, 0.4])
        confidences = np.array([0.8, 0.5, 0.9, 0.6, 0.7])

        result = swarm._bayesian_weighted_aggregate(probabilities, confidences)
        assert 0.0 <= result <= 1.0

    def test_high_confidence_fish_has_more_weight(self):
        swarm = Swarm(num_fish=2)
        # Fish 1: low prob, high confidence
        # Fish 2: high prob, low confidence
        probs = np.array([0.2, 0.8])
        confs = np.array([0.9, 0.1])

        result = swarm._bayesian_weighted_aggregate(probs, confs)
        # Should be closer to 0.2 (high confidence) than 0.8 (low confidence)
        assert result < 0.5

    def test_equal_confidence_gives_mean(self):
        swarm = Swarm(num_fish=3)
        probs = np.array([0.3, 0.5, 0.7])
        confs = np.array([0.5, 0.5, 0.5])

        result = swarm._bayesian_weighted_aggregate(probs, confs)
        assert result == pytest.approx(0.5, abs=0.01)


class TestSwarmOutcomeRecording:
    @pytest.mark.asyncio
    async def test_record_outcome(self):
        swarm = Swarm(num_fish=3)
        await swarm.analyze_market("m1", "Test?")

        scores = swarm.record_outcome("m1", 1.0)
        assert "ensemble" in scores
        assert 0.0 <= scores["ensemble"] <= 1.0

    @pytest.mark.asyncio
    async def test_performance_summary(self):
        swarm = Swarm(num_fish=3)
        await swarm.analyze_market("m1", "Q1?")
        swarm.record_outcome("m1", 1.0)

        summary = swarm.performance_summary
        assert summary["total_predictions"] == 1
        assert summary["fish_count"] == 3
