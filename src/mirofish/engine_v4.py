"""Mirofish Prediction Engine v4 — ultimate integrated pipeline.

Integrates all insights from trending GitHub repos:
- SwarmRouter (kyegomez/swarms): adaptive Fish count per market category
- Multi-model competition (NoFx): consecutive failure detection + safety mode
- Investor persona pattern (ai-hedge-fund): named forecaster personas
- External dataset (prediction-market-analysis): 170K+ resolved markets
- Tree-structured reasoning (Tree-GRPO): step-level scoring
- DeerFlow-inspired dynamic sub-agent spawning

Full pipeline:
  Market → Classify → Route → [Researcher] → Multi-Round Delphi
  → Aggregate (extremize+trim) → Calibrate (netcal auto) → Volatility Check
  → Edge Detect → Kelly Size → Portfolio → Track (MLflow)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.mirofish.llm_fish import (
    CLIFish, FishPrediction, aggregate_predictions,
    parse_fish_response, PERSONA_PROMPTS, DEFAULT_PERSONAS,
)
from src.mirofish.researcher import ResearcherFish
from src.mirofish.swarm_router import (
    SwarmConfig, classify_market, route_swarm, ModelCompetition,
)
from src.prediction.calibration import ProbabilityCalibrator
from src.prediction.advanced_scoring import (
    comprehensive_evaluate, brier_decomposition, conformal_prediction_interval,
)
from src.prediction.volatility import estimate_volatility, VolatilityEstimate
from src.risk.portfolio import (
    EdgeDetector, KellyPositionSizer, MarketSignal, Position, DrawdownMonitor,
)


@dataclass
class EngineV4Result:
    """Complete result from the v4 prediction engine."""
    # Market info
    market_id: str
    question: str
    category: str
    difficulty: str

    # Swarm configuration used
    n_fish: int
    n_rounds: int
    personas_used: list[str]
    model: str
    researcher_used: bool

    # Predictions
    raw_probability: float
    extremized_probability: float
    calibrated_probability: float

    # Confidence and disagreement
    spread: float
    std_dev: float
    effective_confidence: float
    disagreement_flag: bool
    swarm_healthy: bool

    # Volatility (if price history available)
    volatility: VolatilityEstimate | None = None

    # Position recommendation
    position: Position | None = None
    edge: float = 0.0

    # Conformal interval
    prediction_interval: tuple[float, float] | None = None

    # Timing
    total_elapsed_s: float = 0.0
    research_elapsed_s: float = 0.0

    # Fish details
    fish_predictions: list[FishPrediction] = field(default_factory=list)


class PredictionEngineV4:
    """Ultimate prediction engine integrating all optimizations.

    Key differences from v3:
    - SwarmRouter picks optimal persona set per market category
    - ModelCompetition tracks per-model accuracy, safety mode on 3 misses
    - Volatility-adjusted Kelly sizing
    - Conformal prediction intervals
    - External dataset-trained calibrator
    """

    def __init__(
        self,
        model: str = "haiku",
        researcher_model: str = "sonnet",
        max_concurrent: int = 3,
        available_modes: list[str] | None = None,
        calibrator: ProbabilityCalibrator | None = None,
        kelly_fraction: float = 0.25,
        bankroll_usd: float = 1000.0,
        claude_bin: str = "",
        db: "DatabaseManager | None" = None,
    ) -> None:
        self.model = model
        self.researcher_model = researcher_model
        self.max_concurrent = max_concurrent
        self.available_modes = available_modes or ["cli"]
        self.claude_bin = claude_bin
        self.db = db

        self.calibrator = calibrator or ProbabilityCalibrator(method="auto")
        self._seed_calibrator()
        self._edge_detector = EdgeDetector()
        self._sizer = KellyPositionSizer(
            kelly_fraction=kelly_fraction, bankroll_usd=bankroll_usd,
        )
        self._drawdown = DrawdownMonitor(max_drawdown_pct=0.15)
        self._competition = ModelCompetition()

        self._researcher = ResearcherFish(model=researcher_model, claude_bin=claude_bin)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Cumulative stats
        self._markets_analyzed = 0
        self._total_elapsed = 0.0

        logger.info(
            f"Engine v4: model={model}, concurrent={max_concurrent}, "
            f"modes={available_modes}"
        )

    def _seed_calibrator(self) -> None:
        """Seed the calibrator from DB first, then fall back to JSON files.

        Priority:
        1. Database calibration_data table (if db is connected)
        2. Retrodiction JSON files (fallback for first run before DB exists)
        """
        # Try DB first
        if self.db is not None:
            try:
                preds, outs = self.db.get_calibration_data(limit=5000)
                if preds:
                    self.calibrator.fit(preds, outs)
                    logger.info(
                        f"Calibrator seeded from DB: {len(preds)} samples, "
                        f"method={self.calibrator.active_method}"
                    )
                    return
            except Exception as e:
                logger.warning(f"DB calibration load failed: {e}")

        # Fall back to JSON files
        retro_dir = Path("data/retrodiction")
        if not retro_dir.exists():
            return

        preds, outs = [], []
        for fp in sorted(retro_dir.glob("retro_v2_*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                for p in data.get("predictions", []):
                    prob = p.get("extremized_probability", p.get("raw_probability"))
                    gt = p.get("ground_truth")
                    if prob is not None and gt is not None:
                        preds.append(float(prob))
                        outs.append(float(gt))
            except Exception:
                continue

        if preds:
            self.calibrator.fit(preds, outs)
            logger.info(
                f"Calibrator seeded from JSON: {len(preds)} samples, "
                f"method={self.calibrator.active_method}"
            )

    async def analyze(
        self,
        question: str,
        description: str = "",
        outcomes: list[str] | None = None,
        market_price: float | None = None,
        market_id: str = "",
        volume_usd: float = 0,
        price_history: list[float] | None = None,
    ) -> EngineV4Result:
        """Run the full v4 pipeline on a single market."""
        t_start = time.monotonic()
        outcomes = outcomes or ["Yes", "No"]

        # ── Step 1: Classify and route ──
        classification = classify_market(question, description, volume_usd)
        config = route_swarm(
            question, description, volume_usd, outcomes, self.available_modes,
        )

        # Override model if specified at engine level
        config.model = self.model
        config.researcher_model = self.researcher_model

        # ── Step 1b: 3-Fish pre-screen (skip unknowable markets) ──
        # Run 3 fast Fish. If all agree the probability is near 0.5
        # (spread < 0.10, all within 0.35-0.65), the market is likely
        # unknowable to the LLM. Skip the full swarm to save compute
        # and avoid guaranteed-bad predictions that inflate Brier.
        prescreen_skipped = False
        if config.use_researcher and len(config.personas) > 5:
            screen_fish = [
                CLIFish(persona=p, model=config.model, claude_bin=self.claude_bin)
                for p in ["base_rate_anchor", "calibrator", "contrarian"]
            ]
            screen_preds = await self._run_round(
                screen_fish, question, description, outcomes, "", 1, "",
            )
            screen_probs = [p.probability for p in screen_preds if p.confidence > 0.1]
            if screen_probs:
                all_near_50 = all(0.35 < p < 0.65 for p in screen_probs)
                screen_spread = max(screen_probs) - min(screen_probs)
                if all_near_50 and screen_spread < 0.10:
                    prescreen_skipped = True
                    logger.info(
                        f"Pre-screen: all 3 Fish near 0.50 "
                        f"(spread={screen_spread:.2f}). Market likely unknowable."
                    )
                    # Use pre-screen results directly instead of full swarm
                    agg = aggregate_predictions(screen_preds, extremize=1.0, trim=False)
                    elapsed = time.monotonic() - t_start
                    return EngineV4Result(
                        market_id=market_id, question=question,
                        category=classification["category"],
                        difficulty=classification["difficulty"],
                        n_fish=3, n_rounds=1,
                        personas_used=["base_rate_anchor", "calibrator", "contrarian"],
                        model=config.model, researcher_used=False,
                        raw_probability=agg["raw_probability"],
                        extremized_probability=agg["raw_probability"],  # no extremize
                        calibrated_probability=agg["raw_probability"],
                        spread=agg["spread"], std_dev=agg["std_dev"],
                        effective_confidence=agg["effective_confidence"],
                        disagreement_flag=True,  # high uncertainty
                        swarm_healthy=agg.get("swarm_healthy", True),
                        total_elapsed_s=round(elapsed, 1),
                        fish_predictions=screen_preds,
                    )

        # ── Step 2: Research (if routed) ──
        briefing = ""
        research_elapsed = 0.0
        if config.use_researcher:
            t_res = time.monotonic()
            ctx = await self._researcher.research(question, description)
            research_elapsed = time.monotonic() - t_res
            briefing = ctx.to_briefing()

        # ── Step 3: Multi-round Delphi ──
        fish_instances = [
            CLIFish(persona=p, model=config.model, claude_bin=self.claude_bin)
            for p in config.personas
        ]

        all_round_preds = []
        prev_peer_context = ""

        for round_num in range(1, config.max_rounds + 1):
            round_preds = await self._run_round(
                fish_instances, question, description, outcomes,
                briefing, round_num, prev_peer_context,
            )
            all_round_preds.append(round_preds)

            # Check convergence
            probs = [p.probability for p in round_preds]
            std = float(np.std(probs))
            if round_num >= 2 and std < 0.03:
                break

            # Build peer context for next round
            if round_num < config.max_rounds:
                median = float(np.median(probs))
                mean = float(np.mean(probs))
                spread = float(max(probs) - min(probs))
                prev_peer_context = (
                    f"GROUP ESTIMATES FROM ROUND {round_num} "
                    f"({len(round_preds)} analysts):\n"
                    f"  Median: {median:.3f}, Mean: {mean:.3f}, "
                    f"Spread: {spread:.3f}"
                )

        # ── Step 4: Final aggregation ──
        final_preds = all_round_preds[-1]
        agg = aggregate_predictions(
            final_preds, extremize=config.extremize, trim=True,
        )

        raw_prob = agg["raw_probability"]
        ext_prob = agg["extremized_probability"]

        # ── Step 5: Calibration ──
        if self.calibrator.is_fitted:
            cal_prob = self.calibrator.calibrate(ext_prob)
        else:
            cal_prob = ext_prob

        # ── Step 6: Conformal interval (using held-out split to avoid leakage) ──
        interval = None
        if self.calibrator.training_size >= 30:
            holdout_preds, holdout_outs = self.calibrator.get_conformal_residuals()
            if len(holdout_preds) >= 10:
                interval = conformal_prediction_interval(
                    holdout_preds, holdout_outs, cal_prob, alpha=0.1,
                )

        # ── Step 7: Volatility check ──
        vol_estimate = None
        kelly_vol_adj = 1.0
        if price_history and len(price_history) >= 5:
            vol_estimate = estimate_volatility(price_history)
            kelly_vol_adj = vol_estimate.kelly_adjustment

        # ── Step 8: Edge detection + Kelly sizing ──
        position = None
        edge = 0.0
        swarm_healthy = agg.get("swarm_healthy", True)

        if self._drawdown.halted:
            logger.warning("CIRCUIT BREAKER ACTIVE — no trading")
        elif not swarm_healthy:
            logger.error("Swarm unhealthy — no trading")
        elif market_price is not None:
            edge = abs(cal_prob - market_price)
            signal = MarketSignal(
                market_id=market_id,
                question=question,
                market_price=market_price,
                swarm_probability=cal_prob,
                confidence=agg["effective_confidence"],
                spread=agg["spread"],
                disagreement_flag=agg["disagreement_flag"],
                volume_usd=volume_usd,
                category=classification["category"],
            )
            edges = self._edge_detector.detect_edges([signal])
            if edges:
                position = self._sizer.size_position(edges[0])
                # Apply volatility adjustment
                if position and kelly_vol_adj < 1.0:
                    position.position_size_usd = round(
                        position.position_size_usd * kelly_vol_adj, 2
                    )
                    position.reason += f" (vol_adj={kelly_vol_adj})"

        elapsed = time.monotonic() - t_start
        self._markets_analyzed += 1
        self._total_elapsed += elapsed

        result = EngineV4Result(
            market_id=market_id,
            question=question,
            category=classification["category"],
            difficulty=classification["difficulty"],
            n_fish=len(config.personas),
            n_rounds=len(all_round_preds),
            personas_used=config.personas,
            model=config.model,
            researcher_used=config.use_researcher,
            raw_probability=raw_prob,
            extremized_probability=ext_prob,
            calibrated_probability=cal_prob,
            spread=agg["spread"],
            std_dev=agg["std_dev"],
            effective_confidence=agg["effective_confidence"],
            disagreement_flag=agg["disagreement_flag"],
            swarm_healthy=swarm_healthy,
            volatility=vol_estimate,
            position=position,
            edge=round(edge, 4),
            prediction_interval=interval,
            total_elapsed_s=round(elapsed, 1),
            research_elapsed_s=round(research_elapsed, 1),
            fish_predictions=final_preds,
        )

        # ── Log to database ──
        if self.db is not None:
            try:
                self.db.log_prediction(result, market_price=market_price)
            except Exception as e:
                logger.warning(f"DB log_prediction failed: {e}")

        return result

    async def _run_round(
        self,
        fish_list: list[CLIFish],
        question: str,
        description: str,
        outcomes: list[str],
        briefing: str,
        round_number: int,
        peer_context: str,
    ) -> list[FishPrediction]:
        """Run all Fish for a single Delphi round."""
        from src.mirofish.llm_fish import build_fish_prompt

        async def run_one(fish: CLIFish) -> FishPrediction:
            async with self._semaphore:
                # Build round-specific prompt
                system = PERSONA_PROMPTS.get(fish.persona, "")
                desc_block = f"\nCONTEXT:\n{description[:1000]}\n" if description else ""
                brief_block = f"\n{briefing}\n" if briefing else ""

                round_inst = (
                    "This is your FIRST analysis. Form independent judgment."
                    if round_number == 1 else
                    f"ROUND {round_number} of Delphi process.\n{peer_context}\n"
                    "Update or maintain your estimate with explanation."
                )

                prompt = f"""{system}

QUESTION: {question}
{desc_block}{brief_block}
OUTCOMES: {', '.join(outcomes)}
TARGET: "{outcomes[0]}"

{round_inst}

Respond with ONLY this JSON:
{{"steps": ["<step1>", "<step2>", "<step3>"], "probability": <0.01-0.99>, "confidence": <0.1-1.0>, "reasoning": "<1-2 sentences>"}}

CONSTRAINTS:
- No market price known. Independent judgment only.
- Avoid round numbers. Be precise: 0.23, 0.67, 0.83."""

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
                        confidence=0.1, reasoning=str(e)[:100],
                    )

        tasks = [run_one(f) for f in fish_list]
        return list(await asyncio.gather(*tasks))

    def record_outcome(
        self,
        market_id: str,
        prediction: float,
        outcome: float,
        position: Position | None = None,
        market_price: float | None = None,
    ) -> float:
        """Record a resolved market for tracking and calibration.

        Args:
            market_id: Market identifier.
            prediction: Our calibrated probability estimate.
            outcome: Actual outcome (1.0 or 0.0).
            position: The position we took (if any). Required for real P&L.
            market_price: Market price when position was entered.

        Returns:
            Brier score for this prediction.
        """
        brier = (prediction - outcome) ** 2
        self._competition.record(self.model, brier)

        # Feed calibration pipeline (in-memory + DB)
        self.calibrator.fit([prediction], [outcome])
        if self.db is not None:
            try:
                self.db.log_calibration_point(prediction, outcome, market_id)
                self.db.log_resolution(market_id, outcome)
            except Exception as e:
                logger.warning(f"DB record_outcome failed: {e}")

        # Track real P&L only when a position was actually taken
        if position and market_price is not None:
            if position.side == "YES":
                pnl = (outcome - market_price) * position.position_size_usd / market_price
            else:
                pnl = ((1 - outcome) - (1 - market_price)) * position.position_size_usd / (1 - market_price)
            self._drawdown.record_pnl(pnl)
            self._drawdown.check_halt(self._sizer.bankroll)

        return brier

    def print_result(self, r: EngineV4Result) -> None:
        """Print a comprehensive result report."""
        print(f"\n{'='*72}")
        print(f"MIROFISH v4 — {r.question[:55]}")
        print(f"{'='*72}")
        print(f"Category: {r.category} | Difficulty: {r.difficulty}")
        print(f"Fish: {r.n_fish} | Rounds: {r.n_rounds} | Model: {r.model}")
        print(f"Researcher: {'ON' if r.researcher_used else 'OFF'}")

        print(f"\nPrediction: {r.calibrated_probability:.4f}")
        print(f"  Raw:         {r.raw_probability:.4f}")
        print(f"  Extremized:  {r.extremized_probability:.4f}")
        print(f"  Calibrated:  {r.calibrated_probability:.4f}")
        if r.prediction_interval:
            print(f"  90% interval: [{r.prediction_interval[0]:.3f}, {r.prediction_interval[1]:.3f}]")

        print(f"\nConfidence: {r.effective_confidence:.2f} | Spread: {r.spread:.2f}")
        if r.disagreement_flag:
            print("  WARNING: High Fish disagreement")
        if not r.swarm_healthy:
            print("  ALERT: Swarm unhealthy — multiple Fish failed")

        if r.volatility:
            print(f"\nVolatility: {r.volatility.regime} (ratio={r.volatility.vol_ratio:.2f})")
            print(f"  Kelly adj: {r.volatility.kelly_adjustment}")

        if r.position:
            p = r.position
            print(f"\nPOSITION: {p.side} ${p.position_size_usd:.2f} (edge={p.edge:.1%})")
        elif r.edge > 0:
            print(f"\nNO POSITION (edge={r.edge:.1%}, below threshold or filtered)")

        print(f"\nFish estimates:")
        for fp in sorted(r.fish_predictions, key=lambda x: x.probability):
            print(f"  {fp.persona:<25s} P={fp.probability:.3f} conf={fp.confidence:.2f}")

        print(f"\nTime: {r.total_elapsed_s:.0f}s (research: {r.research_elapsed_s:.0f}s)")
        print(f"{'='*72}")

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "markets_analyzed": self._markets_analyzed,
            "total_elapsed_s": round(self._total_elapsed, 1),
            "avg_time_per_market": round(
                self._total_elapsed / max(1, self._markets_analyzed), 1
            ),
            "calibrator_fitted": self.calibrator.is_fitted,
            "calibrator_samples": self.calibrator.training_size,
            "drawdown_halted": self._drawdown.halted,
            "model_competition": self._competition.summary(),
        }
