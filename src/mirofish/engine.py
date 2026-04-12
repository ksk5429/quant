"""Mirofish Prediction Engine v3 — DEPRECATED. Use engine_v4.py instead.

v4 supersedes v3 with: SwarmRouter, volatility adjustment, conformal
intervals, DrawdownMonitor, ModelCompetition, asymmetric extremization.

DEPRECATED: 2026-04-12. Import from engine_v4 for new code:
    from src.mirofish.engine_v4 import PredictionEngineV4

Original v3 architecture (retained for reference):
  1. Researcher gathers context (1 call, stronger model)
  2. Multi-round Delphi: Fish predict independently, then see anonymized
     peer estimates and update across N rounds until convergence or max rounds
  3. Extremized aggregation with trimmed mean
  4. Calibration (isotonic/platt if fitted)
  5. Edge detection against market price
  6. Kelly position sizing → portfolio

Multi-Round Delphi Protocol:
  Round 1: Each Fish analyzes independently (price-withheld)
  Round 2+: Each Fish sees the swarm median and spread from previous round
            and updates its estimate. Fish that changed significantly get
            extra weight (they had new insight). Fish that didn't change
            get reduced weight (they're anchored).
  Convergence: Stop when the swarm standard deviation drops below threshold
               or max rounds reached.

This is the production pipeline that converts information → calibrated
probability → position → profit.

Usage:
    engine = PredictionEngine(model="haiku", max_rounds=3)
    result = await engine.analyze_market(question, description, market_price)
    # result includes: probability, portfolio position, reasoning
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.mirofish.llm_fish import (
    CLIFish, FishPrediction, aggregate_predictions,
    parse_fish_response, PERSONA_PROMPTS, DEFAULT_PERSONAS,
)
from src.mirofish.researcher import ResearcherFish, ResearchContext
from src.prediction.calibration import ProbabilityCalibrator
from src.risk.portfolio import (
    EdgeDetector, KellyPositionSizer, MarketSignal, Portfolio, Position,
)


@dataclass
class RoundResult:
    """Result from a single Delphi round."""
    round_number: int
    predictions: list[FishPrediction]
    raw_probability: float
    std_dev: float
    spread: float
    mean_confidence: float
    elapsed_s: float


@dataclass
class EngineResult:
    """Complete result from the prediction engine pipeline."""
    market_id: str
    question: str

    # Research phase
    research_context: ResearchContext | None
    research_elapsed_s: float

    # Delphi rounds
    rounds: list[RoundResult]
    n_rounds: int
    converged: bool

    # Final prediction
    raw_probability: float
    extremized_probability: float
    calibrated_probability: float

    # Fish details
    final_predictions: list[FishPrediction]
    spread: float
    std_dev: float
    mean_confidence: float
    effective_confidence: float
    disagreement_flag: bool

    # Position (if market price provided)
    position: Position | None
    edge: float

    # Timing
    total_elapsed_s: float


class PredictionEngine:
    """The full Mirofish prediction pipeline.

    Integrates Researcher, multi-round Delphi, calibration, and portfolio
    sizing into a single coherent system optimized for profit.
    """

    def __init__(
        self,
        model: str = "haiku",
        researcher_model: str = "sonnet",
        personas: list[str] | None = None,
        max_concurrent: int = 3,
        max_rounds: int = 3,
        convergence_threshold: float = 0.03,
        extremize: float = 1.5,
        use_researcher: bool = True,
        calibrator: ProbabilityCalibrator | None = None,
        kelly_fraction: float = 0.25,
        bankroll_usd: float = 1000.0,
        claude_bin: str = "",
    ) -> None:
        self.model = model
        self.researcher_model = researcher_model
        self.personas = personas or DEFAULT_PERSONAS
        self.max_concurrent = max_concurrent
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.extremize = extremize
        self.use_researcher = use_researcher
        self.calibrator = calibrator
        self.claude_bin = claude_bin

        # Edge detection
        self._edge_detector = EdgeDetector(
            min_edge=0.05, min_confidence=0.40, max_spread=0.35,
        )

        # Position sizing
        self._sizer = KellyPositionSizer(
            kelly_fraction=kelly_fraction,
            max_position_pct=0.05,
            bankroll_usd=bankroll_usd,
        )

        # Researcher (uses stronger model, runs once)
        if use_researcher:
            self._researcher = ResearcherFish(
                model=researcher_model, claude_bin=claude_bin,
            )

        # Fish agents
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._fish = [
            CLIFish(persona=p, model=model, claude_bin=claude_bin)
            for p in self.personas
        ]

        logger.info(
            f"Engine v3: {len(self._fish)} Fish, model={model}, "
            f"max_rounds={max_rounds}, convergence={convergence_threshold}, "
            f"researcher={'ON' if use_researcher else 'OFF'}"
        )

    async def analyze_market(
        self,
        question: str,
        description: str = "",
        outcomes: list[str] | None = None,
        market_price: float | None = None,
        market_id: str = "",
    ) -> EngineResult:
        """Run the full prediction pipeline on a single market.

        Args:
            question: Market question.
            description: Market description/context.
            outcomes: Possible outcomes (first = target).
            market_price: Current market price (for edge detection).
            market_id: Market identifier.

        Returns:
            EngineResult with prediction, position, and full reasoning.
        """
        t_start = time.monotonic()
        outcomes = outcomes or ["Yes", "No"]

        # ── Phase 1: Research ──
        research_ctx = None
        research_elapsed = 0.0

        if self.use_researcher:
            t_res = time.monotonic()
            research_ctx = await self._researcher.research(question, description)
            research_elapsed = time.monotonic() - t_res
            logger.info(f"Research: {research_elapsed:.1f}s, {len(research_ctx.key_facts)} facts")

        briefing = research_ctx.to_briefing() if research_ctx else ""

        # ── Phase 2: Multi-round Delphi ──
        rounds: list[RoundResult] = []
        converged = False
        prev_predictions: list[FishPrediction] = []

        for round_num in range(1, self.max_rounds + 1):
            t_round = time.monotonic()

            if round_num == 1:
                # Round 1: independent analysis
                predictions = await self._run_round(
                    question, description, outcomes, briefing,
                    round_number=1,
                )
            else:
                # Round 2+: Delphi update with peer context
                peer_context = self._build_peer_context(prev_predictions, round_num - 1)
                predictions = await self._run_round(
                    question, description, outcomes, briefing,
                    round_number=round_num,
                    peer_context=peer_context,
                )

            # Aggregate this round
            agg = aggregate_predictions(predictions, extremize=1.0, trim=False)
            std_dev = agg["std_dev"]

            round_result = RoundResult(
                round_number=round_num,
                predictions=predictions,
                raw_probability=agg["raw_probability"],
                std_dev=std_dev,
                spread=agg["spread"],
                mean_confidence=agg["mean_confidence"],
                elapsed_s=round(time.monotonic() - t_round, 1),
            )
            rounds.append(round_result)

            logger.info(
                f"Round {round_num}: P={agg['raw_probability']:.3f}, "
                f"std={std_dev:.3f}, spread={agg['spread']:.2f}, "
                f"{round_result.elapsed_s:.0f}s"
            )

            # Check convergence
            if round_num >= 2 and std_dev < self.convergence_threshold:
                converged = True
                logger.info(
                    f"Converged at round {round_num} (std={std_dev:.3f} "
                    f"< {self.convergence_threshold})"
                )
                break

            # Check if updating is changing anything
            if round_num >= 2:
                prev_std = rounds[-2].std_dev
                improvement = prev_std - std_dev
                if improvement < 0.005:
                    logger.info(
                        f"Delphi stalled at round {round_num} "
                        f"(improvement={improvement:.4f}), stopping"
                    )
                    break

            prev_predictions = predictions

        # ── Phase 3: Final aggregation with extremization ──
        final_preds = rounds[-1].predictions
        final_agg = aggregate_predictions(
            final_preds, extremize=self.extremize, trim=True,
        )

        raw_prob = final_agg["raw_probability"]
        ext_prob = final_agg["extremized_probability"]

        # ── Phase 4: Calibration ──
        if self.calibrator and self.calibrator.is_fitted:
            cal_prob = self.calibrator.calibrate(ext_prob)
        else:
            cal_prob = ext_prob

        # ── Phase 5: Edge detection + position sizing ──
        # Block trading if swarm is unhealthy (too many Fish failed)
        position = None
        edge = 0.0
        swarm_healthy = final_agg.get("swarm_healthy", True)

        if not swarm_healthy:
            logger.error("Swarm unhealthy — skipping position sizing")
        elif market_price is not None:
            edge = abs(cal_prob - market_price)
            signal = MarketSignal(
                market_id=market_id,
                question=question,
                market_price=market_price,
                swarm_probability=cal_prob,
                confidence=final_agg["effective_confidence"],
                spread=final_agg["spread"],
                disagreement_flag=final_agg["disagreement_flag"],
            )

            edges = self._edge_detector.detect_edges([signal])
            if edges:
                position = self._sizer.size_position(edges[0])

        total_elapsed = time.monotonic() - t_start

        result = EngineResult(
            market_id=market_id,
            question=question,
            research_context=research_ctx,
            research_elapsed_s=round(research_elapsed, 1),
            rounds=rounds,
            n_rounds=len(rounds),
            converged=converged,
            raw_probability=raw_prob,
            extremized_probability=ext_prob,
            calibrated_probability=cal_prob,
            final_predictions=final_preds,
            spread=final_agg["spread"],
            std_dev=final_agg["std_dev"],
            mean_confidence=final_agg["mean_confidence"],
            effective_confidence=final_agg["effective_confidence"],
            disagreement_flag=final_agg["disagreement_flag"],
            position=position,
            edge=round(edge, 4),
            total_elapsed_s=round(total_elapsed, 1),
        )

        self._log_result(result)
        return result

    async def _run_round(
        self,
        question: str,
        description: str,
        outcomes: list[str],
        briefing: str,
        round_number: int = 1,
        peer_context: str = "",
    ) -> list[FishPrediction]:
        """Run all Fish for a single Delphi round."""

        async def run_one(fish: CLIFish) -> FishPrediction:
            async with self._semaphore:
                prompt = self._build_round_prompt(
                    question, description, outcomes, briefing,
                    fish.persona, round_number, peer_context,
                )

                t0 = time.monotonic()
                try:
                    proc = await asyncio.create_subprocess_exec(
                        fish.claude_bin, "-p",
                        "--model", fish.model,
                        "--output-format", "text",
                        "--no-session-persistence",
                        "--tools", "",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env={**os.environ, "CLAUDECODE": ""},
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=prompt.encode("utf-8")),
                        timeout=120.0,
                    )
                    raw = stdout.decode("utf-8", errors="replace").strip()

                    if proc.returncode != 0:
                        return FishPrediction(
                            persona=fish.persona, probability=0.5,
                            confidence=0.1, reasoning="CLI error",
                        )

                    pred = parse_fish_response(raw, fish.persona)
                    pred.model = f"cli-{fish.model}"
                    pred.elapsed_s = round(time.monotonic() - t0, 1)
                    return pred

                except asyncio.TimeoutError:
                    return FishPrediction(
                        persona=fish.persona, probability=0.5,
                        confidence=0.1, reasoning="timeout",
                        elapsed_s=120.0,
                    )
                except Exception as e:
                    return FishPrediction(
                        persona=fish.persona, probability=0.5,
                        confidence=0.1, reasoning=str(e),
                    )

        tasks = [run_one(f) for f in self._fish]
        return list(await asyncio.gather(*tasks))

    def _build_round_prompt(
        self,
        question: str,
        description: str,
        outcomes: list[str],
        briefing: str,
        persona: str,
        round_number: int,
        peer_context: str,
    ) -> str:
        """Build prompt for a specific Delphi round."""
        system = PERSONA_PROMPTS.get(
            persona,
            f"You are a {persona.replace('_', ' ')} analyzing prediction markets.",
        )

        desc_block = f"\nCONTEXT:\n{description[:1000]}\n" if description else ""
        brief_block = f"\n{briefing}\n" if briefing else ""

        if round_number == 1:
            round_instruction = (
                "This is your FIRST analysis. Form your own independent judgment."
            )
        else:
            round_instruction = (
                f"This is ROUND {round_number} of a Delphi process.\n"
                f"{peer_context}\n"
                "Consider the group estimates above. You may update your "
                "estimate if you find the group's reasoning compelling, or "
                "maintain your position if you believe you have stronger "
                "evidence. Explain what changed (or why you maintained "
                "your estimate)."
            )

        return f"""{system}

QUESTION: {question}
{desc_block}{brief_block}
OUTCOMES: {', '.join(outcomes)}
TARGET: "{outcomes[0]}"

{round_instruction}

Think step by step, then respond with ONLY this JSON:
{{"steps": ["<step 1>", "<step 2>", "<step 3>"], "probability": <0.01-0.99>, "confidence": <0.1-1.0>, "reasoning": "<1-2 sentences>"}}

CONSTRAINTS:
- Do NOT know the market price. Independent judgment only.
- Avoid round numbers (0.50, 0.70). Be precise: 0.23, 0.67, 0.83.
- Confidence = certainty in YOUR estimate, not in the outcome."""

    def _build_peer_context(
        self, predictions: list[FishPrediction], prev_round: int,
    ) -> str:
        """Build anonymized peer estimate summary for Delphi rounds."""
        probs = [p.probability for p in predictions]
        median = float(np.median(probs))
        mean = float(np.mean(probs))
        std = float(np.std(probs))
        low = float(np.min(probs))
        high = float(np.max(probs))

        return (
            f"GROUP ESTIMATES FROM ROUND {prev_round} "
            f"(anonymized, {len(predictions)} analysts):\n"
            f"  Median: {median:.3f}\n"
            f"  Mean:   {mean:.3f}\n"
            f"  Range:  {low:.3f} to {high:.3f}\n"
            f"  Std:    {std:.3f}"
        )

    def _log_result(self, result: EngineResult) -> None:
        """Log the final result."""
        pos_str = ""
        if result.position:
            pos_str = (
                f" → {result.position.side} ${result.position.position_size_usd:.0f} "
                f"(edge={result.position.edge:.1%})"
            )

        logger.info(
            f"Engine: P={result.calibrated_probability:.3f} "
            f"(raw={result.raw_probability:.3f}, ext={result.extremized_probability:.3f}) "
            f"rounds={result.n_rounds} conv={'Y' if result.converged else 'N'} "
            f"spread={result.spread:.2f} "
            f"time={result.total_elapsed_s:.0f}s{pos_str}"
        )

    def print_result(self, result: EngineResult) -> None:
        """Print a detailed result report."""
        print(f"\n{'='*72}")
        print(f"MIROFISH ENGINE v3 — {result.question[:60]}")
        print(f"{'='*72}")

        if result.research_context:
            print(f"\n[RESEARCH] ({result.research_elapsed_s:.0f}s)")
            print(f"  Base rate: {result.research_context.base_rate[:80]}")
            print(f"  Facts: {len(result.research_context.key_facts)}")
            print(f"  Contrarian: {result.research_context.contrarian_case[:80]}")

        print(f"\n[DELPHI] {result.n_rounds} rounds, "
              f"{'converged' if result.converged else 'max rounds reached'}")
        for r in result.rounds:
            print(f"  Round {r.round_number}: P={r.raw_probability:.3f} "
                  f"std={r.std_dev:.3f} spread={r.spread:.2f} ({r.elapsed_s:.0f}s)")

        print(f"\n[PREDICTION]")
        print(f"  Raw:         {result.raw_probability:.4f}")
        print(f"  Extremized:  {result.extremized_probability:.4f}")
        print(f"  Calibrated:  {result.calibrated_probability:.4f}")
        print(f"  Confidence:  {result.effective_confidence:.2f}")
        print(f"  Spread:      {result.spread:.2f}")
        if result.disagreement_flag:
            print(f"  WARNING: High disagreement among Fish")

        print(f"\n[FISH ESTIMATES]")
        for fp in sorted(result.final_predictions, key=lambda x: x.probability):
            print(f"  {fp.persona:<25s} P={fp.probability:.3f} "
                  f"conf={fp.confidence:.2f}")

        if result.position:
            p = result.position
            print(f"\n[POSITION]")
            print(f"  Side:   {p.side}")
            print(f"  Edge:   {p.edge:.1%}")
            print(f"  Size:   ${p.position_size_usd:.2f}")
            print(f"  EV/$:   {p.expected_value:+.1%}")
        elif result.edge > 0:
            print(f"\n[NO POSITION] Edge={result.edge:.1%} (below threshold or high disagreement)")

        print(f"\n  Total time: {result.total_elapsed_s:.0f}s")
        print(f"{'='*72}")
