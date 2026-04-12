"""Tests for Fish agent — the core unit of the Mirofish swarm."""

import asyncio

import pytest

from src.mirofish.fish import Fish, FishPersona, FishAnalysis


class TestFishCreation:
    def test_fish_creates_with_persona(self):
        fish = Fish(persona=FishPersona.FINANCIAL_QUANT)
        assert fish.persona == FishPersona.FINANCIAL_QUANT
        assert "financial_quant" in fish.fish_id

    def test_fish_has_unique_id(self):
        f1 = Fish(persona=FishPersona.CONTRARIAN_THINKER)
        f2 = Fish(persona=FishPersona.CONTRARIAN_THINKER)
        assert f1.fish_id != f2.fish_id

    def test_fish_custom_id(self):
        fish = Fish(persona=FishPersona.DOMAIN_EXPERT, fish_id="test-fish-001")
        assert fish.fish_id == "test-fish-001"

    def test_all_personas_have_system_prompts(self):
        for persona in FishPersona:
            fish = Fish(persona=persona)
            assert len(fish.system_prompt) > 50


class TestFishAnalysis:
    @pytest.mark.asyncio
    async def test_stub_analysis_produces_valid_output(self):
        fish = Fish(persona=FishPersona.BAYESIAN_STATISTICIAN)
        analysis = await fish.analyze(
            market_id="test-market-001",
            market_question="Will Bitcoin exceed $100k by December 2026?",
        )

        assert isinstance(analysis, FishAnalysis)
        assert 0.0 <= analysis.probability <= 1.0
        assert 0.0 <= analysis.confidence <= 1.0
        assert analysis.market_id == "test-market-001"
        assert analysis.persona == FishPersona.BAYESIAN_STATISTICIAN
        assert len(analysis.reasoning_steps) > 0

    @pytest.mark.asyncio
    async def test_different_personas_give_different_results(self):
        """Persona diversity should produce varied probability estimates."""
        question = "Will the Fed cut rates in 2026?"
        results = {}

        for persona in FishPersona:
            fish = Fish(persona=persona)
            analysis = await fish.analyze(
                market_id="test-fed",
                market_question=question,
            )
            results[persona] = analysis.probability

        # At least some variation across personas
        probs = list(results.values())
        assert max(probs) - min(probs) > 0.05, "Personas should produce varied estimates"

    @pytest.mark.asyncio
    async def test_deterministic_stub_results(self):
        """Same persona + same question should give same stub result."""
        fish1 = Fish(persona=FishPersona.FINANCIAL_QUANT, fish_id="fixed-id")
        fish2 = Fish(persona=FishPersona.FINANCIAL_QUANT, fish_id="fixed-id")

        a1 = await fish1.analyze("m1", "Test question?")
        a2 = await fish2.analyze("m1", "Test question?")

        assert a1.probability == a2.probability

    @pytest.mark.asyncio
    async def test_analysis_records_in_history(self):
        fish = Fish(persona=FishPersona.GEOPOLITICAL_ANALYST)
        assert len(fish.history) == 0

        await fish.analyze("m1", "Question 1?")
        assert len(fish.history) == 1

        await fish.analyze("m2", "Question 2?")
        assert len(fish.history) == 2


class TestFishBrierScore:
    @pytest.mark.asyncio
    async def test_brier_score_calculation(self):
        fish = Fish(persona=FishPersona.CALIBRATION_SPECIALIST)
        await fish.analyze("m1", "Test?")

        # Get the predicted probability
        predicted = fish.history[0].probability

        # Record actual outcome
        brier = fish.record_outcome("m1", 1.0)
        assert brier is not None
        assert brier == pytest.approx((predicted - 1.0) ** 2, abs=1e-6)

    @pytest.mark.asyncio
    async def test_brier_score_no_matching_market(self):
        fish = Fish(persona=FishPersona.FINANCIAL_QUANT)
        brier = fish.record_outcome("nonexistent", 1.0)
        assert brier is None

    @pytest.mark.asyncio
    async def test_average_brier_score(self):
        fish = Fish(persona=FishPersona.BAYESIAN_STATISTICIAN)

        await fish.analyze("m1", "Q1?")
        await fish.analyze("m2", "Q2?")

        fish.record_outcome("m1", 1.0)
        fish.record_outcome("m2", 0.0)

        avg = fish.average_brier_score
        assert avg is not None
        assert 0.0 <= avg <= 1.0
