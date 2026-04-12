"""Swarm — the orchestrator that manages all Fish agents and aggregates predictions.

The Swarm:
1. Spawns Fish with diverse personas
2. Distributes markets to Fish for analysis
3. Coordinates communication via the MessageBus
4. Aggregates individual predictions using Bayesian confidence-weighted fusion
5. Tracks ensemble performance via Brier scores

Architecture follows PolySwarm (arXiv:2604.03888):
- Market prices are WITHHELD from Fish during inference (preserve independence)
- Aggregation uses confidence-weighted Bayesian fusion
- Diversity across personas is the primary accuracy driver
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.mirofish.fish import Fish, FishPersona, FishAnalysis, PERSONA_SYSTEM_PROMPTS
from src.mirofish.god_node import GodNode
from src.mirofish.message_bus import MessageBus, Message, MessageType


class SwarmPrediction(BaseModel):
    """Aggregated prediction from the entire swarm for one market."""

    market_id: str
    market_question: str

    # Aggregated probability
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Individual Fish analyses
    fish_analyses: list[FishAnalysis] = Field(default_factory=list)

    # Aggregation metadata
    aggregation_method: str = "bayesian_weighted"
    spread: float = 0.0  # Max - min across Fish (measures disagreement)
    std_dev: float = 0.0  # Standard deviation of Fish predictions

    # Market context (added post-aggregation)
    market_price: float | None = None
    edge: float | None = None  # Our probability - market price

    # Timing
    timestamp: float = Field(default_factory=time.time)
    total_latency_ms: float = 0.0


class Swarm:
    """The Mirofish swarm — an ensemble of diverse LLM Fish agents.

    Usage:
        swarm = Swarm(num_fish=7, llm_client=anthropic_client)
        prediction = await swarm.analyze_market(
            market_id="abc123",
            market_question="Will BTC exceed $100k by Dec 2026?",
            market_description="Bitcoin price prediction...",
        )
        print(f"Swarm prediction: {prediction.probability:.3f}")
    """

    def __init__(
        self,
        num_fish: int = 7,
        personas: list[FishPersona] | None = None,
        llm_client: Any = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_concurrent: int = 5,
        aggregation_method: str = "bayesian_weighted",
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.aggregation_method = aggregation_method

        # Initialize MessageBus
        self.bus = MessageBus()

        # Spawn Fish with diverse personas
        if personas is None:
            # Cycle through all persona types
            all_personas = list(FishPersona)
            personas = [all_personas[i % len(all_personas)] for i in range(num_fish)]

        self.fish: list[Fish] = []
        for persona in personas:
            fish = Fish(
                persona=persona,
                llm_client=llm_client,
                model=model,
                temperature=temperature,
            )
            self.fish.append(fish)
            self.bus.subscribe(fish.fish_id, self._make_fish_handler(fish))

        # Initialize GOD node
        self.god = GodNode(
            message_bus=self.bus,
            llm_client=llm_client,
            model=model,
        )

        # Performance tracking
        self.prediction_history: list[SwarmPrediction] = []

        logger.info(
            f"Swarm initialized: {len(self.fish)} Fish, "
            f"personas={[f.persona.value for f in self.fish]}"
        )

    def _make_fish_handler(self, fish: Fish):
        """Create a message handler for a Fish agent."""
        async def handler(message: Message) -> None:
            if message.msg_type == MessageType.EVENT_INJECTION:
                logger.info(
                    f"[{fish.fish_id}] Received event: "
                    f"{message.payload.get('event', '')[:60]}..."
                )
            elif message.msg_type == MessageType.REANALYSIS_REQUEST:
                logger.info(f"[{fish.fish_id}] Reanalysis requested")
        return handler

    async def analyze_market(
        self,
        market_id: str,
        market_question: str,
        market_description: str = "",
        current_price: float | None = None,
        news_context: list[str] | None = None,
    ) -> SwarmPrediction:
        """Run the full swarm analysis pipeline for a single market.

        Steps:
        1. All Fish analyze the market concurrently (price WITHHELD)
        2. Fish share cross-market signals via MessageBus
        3. Aggregate predictions using confidence-weighted Bayesian fusion
        4. Compare with market price to compute edge
        """
        start_time = time.monotonic()

        # Step 1: Concurrent Fish analysis (with semaphore for rate limiting)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_analyze(fish: Fish) -> FishAnalysis:
            async with semaphore:
                return await fish.analyze(
                    market_id=market_id,
                    market_question=market_question,
                    market_description=market_description,
                    # NOTE: current_price intentionally NOT passed to Fish
                    news_context=news_context,
                )

        analyses = await asyncio.gather(
            *[bounded_analyze(f) for f in self.fish],
            return_exceptions=True,
        )

        # Filter out failed analyses
        valid_analyses: list[FishAnalysis] = [
            a for a in analyses if isinstance(a, FishAnalysis)
        ]
        failed = len(analyses) - len(valid_analyses)
        if failed > 0:
            logger.warning(f"Swarm: {failed}/{len(analyses)} Fish failed for {market_id}")

        if not valid_analyses:
            logger.error(f"Swarm: All Fish failed for {market_id}")
            return SwarmPrediction(
                market_id=market_id,
                market_question=market_question,
                probability=0.5,
                confidence=0.0,
                aggregation_method=self.aggregation_method,
            )

        # Step 2: Aggregate predictions
        prediction = self._aggregate(valid_analyses, market_id, market_question)

        # Step 3: Add market context
        if current_price is not None:
            prediction.market_price = current_price
            prediction.edge = prediction.probability - current_price

        prediction.total_latency_ms = (time.monotonic() - start_time) * 1000
        self.prediction_history.append(prediction)

        logger.info(
            f"Swarm prediction for '{market_question[:50]}...': "
            f"P={prediction.probability:.3f} (conf={prediction.confidence:.2f}, "
            f"spread={prediction.spread:.3f})"
        )

        return prediction

    def _aggregate(
        self,
        analyses: list[FishAnalysis],
        market_id: str,
        market_question: str,
    ) -> SwarmPrediction:
        """Aggregate individual Fish predictions into a swarm prediction."""
        probabilities = np.array([a.probability for a in analyses])
        confidences = np.array([a.confidence for a in analyses])

        if self.aggregation_method == "bayesian_weighted":
            agg_prob = self._bayesian_weighted_aggregate(probabilities, confidences)
        elif self.aggregation_method == "mean":
            agg_prob = float(np.mean(probabilities))
        elif self.aggregation_method == "median":
            agg_prob = float(np.median(probabilities))
        else:
            agg_prob = self._bayesian_weighted_aggregate(probabilities, confidences)

        # Ensemble confidence = mean confidence weighted by agreement
        spread = float(np.max(probabilities) - np.min(probabilities))
        std_dev = float(np.std(probabilities))
        agreement_bonus = max(0, 1.0 - spread * 2)  # Higher when Fish agree
        agg_confidence = float(np.mean(confidences)) * (0.7 + 0.3 * agreement_bonus)

        return SwarmPrediction(
            market_id=market_id,
            market_question=market_question,
            probability=round(float(np.clip(agg_prob, 0.0, 1.0)), 4),
            confidence=round(float(np.clip(agg_confidence, 0.0, 1.0)), 4),
            fish_analyses=analyses,
            aggregation_method=self.aggregation_method,
            spread=round(spread, 4),
            std_dev=round(std_dev, 4),
        )

    def _bayesian_weighted_aggregate(
        self, probabilities: np.ndarray, confidences: np.ndarray
    ) -> float:
        """Confidence-weighted Bayesian aggregation (PolySwarm methodology).

        p_swarm = sum(w_i * p_i) / sum(w_i)

        where w_i = confidence_i * (1 + historical_accuracy_i)

        This ensures that Fish who are both confident AND historically accurate
        have more influence on the ensemble prediction.
        """
        # Weight = confidence * historical accuracy bonus
        weights = confidences.copy()

        # Boost weights for Fish with good track records
        for i, fish in enumerate(self.fish[:len(probabilities)]):
            brier = fish.average_brier_score
            if brier is not None:
                # Lower Brier = better. Convert to accuracy bonus (0 to 1)
                accuracy_bonus = max(0, 1.0 - brier * 2)
                weights[i] *= (1.0 + accuracy_bonus)

        # Weighted average
        total_weight = np.sum(weights)
        if total_weight == 0:
            return float(np.mean(probabilities))

        return float(np.sum(weights * probabilities) / total_weight)

    async def inject_event(self, event_description: str, urgency: str = "normal"):
        """Inject a real-world event via the GOD node."""
        return await self.god.inject_event(event_description, urgency=urgency)

    def record_outcome(self, market_id: str, actual_outcome: float) -> dict[str, float]:
        """Record the actual outcome for a market and compute Brier scores."""
        scores = {}
        for fish in self.fish:
            brier = fish.record_outcome(market_id, actual_outcome)
            if brier is not None:
                scores[fish.fish_id] = brier

        # Ensemble Brier score
        matching = [p for p in self.prediction_history if p.market_id == market_id]
        if matching:
            ensemble_brier = (matching[-1].probability - actual_outcome) ** 2
            scores["ensemble"] = ensemble_brier
            logger.info(f"Swarm Brier score for {market_id}: {ensemble_brier:.4f}")

        return scores

    @property
    def performance_summary(self) -> dict[str, Any]:
        """Summary of swarm performance across all resolved predictions."""
        fish_briers = {
            f.fish_id: f.average_brier_score
            for f in self.fish
            if f.average_brier_score is not None
        }
        return {
            "total_predictions": len(self.prediction_history),
            "fish_count": len(self.fish),
            "fish_brier_scores": fish_briers,
            "best_fish": min(fish_briers, key=fish_briers.get) if fish_briers else None,
            "worst_fish": max(fish_briers, key=fish_briers.get) if fish_briers else None,
        }
