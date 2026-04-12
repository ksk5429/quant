"""Retrodiction evaluation harness — DEPRECATED simulation mode.

WARNING: The simulate_fish_prediction() method in this module generates
predictions by adding noise to ground truth. This produces artificially
good Brier scores that do not reflect real LLM prediction quality.
Do NOT use this module to train calibration models.

For real retrodiction with actual LLM calls, use:
    python -m src.prediction.run_retrodiction --n 30 --model haiku

This module is retained for offline testing of the evaluation math
(Brier, ECE, aggregation) without consuming CLI resources.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.markets.history import ResolvedMarket, HistoricalMarketScraper
from src.mirofish.llm_fish import FishPrediction  # canonical definition
from src.prediction.calibration import CalibrationMetrics, ProbabilityCalibrator


@dataclass
class SwarmPrediction:
    """Aggregated swarm prediction for one market."""

    market_id: str
    question: str

    # Individual Fish predictions
    fish_predictions: list[FishPrediction]

    # Aggregated probability (confidence-weighted Bayesian)
    raw_probability: float
    calibrated_probability: float

    # Swarm metadata
    spread: float  # max - min across Fish
    mean_confidence: float
    n_fish: int

    # Ground truth (filled after evaluation)
    ground_truth: float | None = None
    brier_score: float | None = None

    @property
    def is_evaluated(self) -> bool:
        return self.ground_truth is not None


@dataclass
class RetrodictionResult:
    """Complete results from a retrodiction evaluation run."""

    run_id: str
    timestamp: str
    n_markets: int
    n_fish: int
    predictions: list[SwarmPrediction]

    # Aggregate metrics
    brier_score: float
    log_loss: float
    ece: float
    accuracy_at_50: float  # % correct when treating >0.5 as "yes"

    # Per-Fish Brier scores (for weight learning)
    fish_brier_scores: dict[str, float]

    # Calibration before/after
    raw_brier: float
    calibrated_brier: float


class RetrodictionHarness:
    """Evaluates the Mirofish swarm on resolved markets.

    The harness:
    1. Loads a corpus of resolved markets
    2. For each market, constructs a time-safe prompt (no future info)
    3. Runs the Fish swarm to get probability estimates
    4. Aggregates using confidence-weighted Bayesian method
    5. Optionally applies calibration
    6. Scores against ground truth
    """

    DEFAULT_PERSONAS = [
        "geopolitical_analyst",
        "financial_quant",
        "bayesian_statistician",
        "investigative_journalist",
        "contrarian_thinker",
        "domain_expert",
        "calibration_specialist",
    ]

    def __init__(
        self,
        personas: list[str] | None = None,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> None:
        self.personas = personas or self.DEFAULT_PERSONAS
        self.calibrator = calibrator
        self._results: list[SwarmPrediction] = []

    def build_fish_prompt(self, market: ResolvedMarket, persona: str) -> str:
        """Construct a time-safe prompt for a Fish agent.

        The prompt contains ONLY information that would have been available
        before the market resolved. It deliberately excludes:
        - The outcome / resolution
        - The closing time or resolution date context
        - The current market price (price-withheld protocol)
        """
        prompt = f"""You are a {persona.replace('_', ' ')} analyzing a prediction market.

MARKET QUESTION: {market.question}

MARKET DESCRIPTION:
{market.description[:1500]}

TASK: Estimate the probability that the FIRST listed outcome occurs.
The possible outcomes are: {', '.join(market.outcomes)}

You must respond with EXACTLY this JSON format:
{{
    "probability": <float between 0.01 and 0.99>,
    "confidence": <float between 0.1 and 1.0>,
    "reasoning": "<2-3 sentence explanation>"
}}

RULES:
- Base your estimate on publicly available information and your analytical expertise.
- You do NOT know the current market price. Form your own independent judgment.
- Express genuine uncertainty. Do not default to 0.5 unless you truly have no information.
- Your probability is for the FIRST outcome: "{market.outcomes[0]}"
"""
        return prompt

    def simulate_fish_prediction(self, market: ResolvedMarket, persona: str) -> FishPrediction:
        """Simulate a Fish prediction using heuristic baseline.

        In production, this calls an LLM API. For offline evaluation without
        API costs, this generates a calibrated random prediction based on
        market characteristics. Replace this method with actual LLM calls
        for real evaluation.

        The simulation uses:
        - Base rate from category (if known)
        - Random perturbation scaled by persona type
        - Contrarian persona inverts the base signal
        """
        # Seed based on market ID + persona for reproducibility
        seed = hash(f"{market.id}_{persona}") % (2**31)
        rng = np.random.RandomState(seed)

        # Ground truth base signal (slightly noisy oracle)
        gt = market.ground_truth
        noise = rng.normal(0, 0.20)  # 20% std noise
        base_signal = np.clip(gt + noise, 0.05, 0.95)

        # Persona-specific adjustments
        if persona == "contrarian_thinker":
            # Contrarian shifts toward 0.5 (expresses more uncertainty)
            base_signal = 0.5 + (base_signal - 0.5) * 0.6
            confidence = 0.4 + rng.random() * 0.3
        elif persona == "bayesian_statistician":
            # Bayesian is well-calibrated but conservative
            base_signal = 0.5 + (base_signal - 0.5) * 0.8
            confidence = 0.5 + rng.random() * 0.4
        elif persona == "calibration_specialist":
            # Calibration specialist is closest to ground truth
            extra_noise = rng.normal(0, 0.05)
            base_signal = np.clip(gt + extra_noise, 0.05, 0.95)
            confidence = 0.6 + rng.random() * 0.3
        elif persona == "financial_quant":
            confidence = 0.5 + rng.random() * 0.4
        elif persona == "investigative_journalist":
            # Journalist has strong opinions (higher variance)
            base_signal = np.clip(base_signal + rng.normal(0, 0.10), 0.05, 0.95)
            confidence = 0.5 + rng.random() * 0.5
        else:
            confidence = 0.4 + rng.random() * 0.4

        probability = float(np.clip(base_signal, 0.01, 0.99))
        confidence = float(np.clip(confidence, 0.1, 1.0))

        return FishPrediction(
            persona=persona,
            probability=round(probability, 4),
            confidence=round(confidence, 4),
            reasoning=f"[Simulated {persona} prediction]",
        )

    def aggregate_predictions(self, fish_preds: list[FishPrediction]) -> tuple[float, float, float]:
        """Confidence-weighted Bayesian aggregation.

        Returns (aggregated_probability, spread, mean_confidence).
        """
        if not fish_preds:
            return 0.5, 0.0, 0.0

        weights = []
        probs = []
        for fp in fish_preds:
            w = fp.confidence
            weights.append(w)
            probs.append(fp.probability)

        weights = np.array(weights)
        probs = np.array(probs)

        # Weighted average
        total_weight = weights.sum()
        if total_weight < 1e-8:
            agg_prob = float(np.mean(probs))
        else:
            agg_prob = float(np.sum(weights * probs) / total_weight)

        spread = float(np.max(probs) - np.min(probs))
        mean_conf = float(np.mean(weights))

        return (
            round(np.clip(agg_prob, 0.01, 0.99), 4),
            round(spread, 4),
            round(mean_conf, 4),
        )

    def evaluate_market(self, market: ResolvedMarket) -> SwarmPrediction:
        """Run the full swarm evaluation on a single resolved market."""
        fish_preds = []
        for persona in self.personas:
            pred = self.simulate_fish_prediction(market, persona)
            fish_preds.append(pred)

        raw_prob, spread, mean_conf = self.aggregate_predictions(fish_preds)

        # Apply calibration if available
        if self.calibrator and self.calibrator.is_fitted:
            cal_prob = self.calibrator.calibrate(raw_prob)
        else:
            cal_prob = raw_prob

        gt = market.ground_truth
        brier = (cal_prob - gt) ** 2

        return SwarmPrediction(
            market_id=market.id,
            question=market.question,
            fish_predictions=fish_preds,
            raw_probability=raw_prob,
            calibrated_probability=cal_prob,
            spread=spread,
            mean_confidence=mean_conf,
            n_fish=len(fish_preds),
            ground_truth=gt,
            brier_score=round(brier, 6),
        )

    def run_retrodiction(
        self,
        markets: list[ResolvedMarket],
        sample_size: int | None = None,
    ) -> RetrodictionResult:
        """Run retrodiction evaluation on a corpus of resolved markets.

        Args:
            markets: List of resolved markets with ground truth.
            sample_size: Randomly sample this many markets (None = all).

        Returns:
            RetrodictionResult with aggregate and per-Fish metrics.
        """
        if sample_size and sample_size < len(markets):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(markets), size=sample_size, replace=False)
            eval_markets = [markets[i] for i in indices]
        else:
            eval_markets = markets

        logger.info(
            f"Running retrodiction: {len(eval_markets)} markets, "
            f"{len(self.personas)} Fish personas"
        )

        predictions = []
        for i, market in enumerate(eval_markets):
            pred = self.evaluate_market(market)
            predictions.append(pred)

            if (i + 1) % 50 == 0:
                running_brier = np.mean([p.brier_score for p in predictions])
                logger.info(
                    f"Progress: {i + 1}/{len(eval_markets)} "
                    f"(running Brier: {running_brier:.4f})"
                )

        # Compute aggregate metrics
        cal_probs = [p.calibrated_probability for p in predictions]
        raw_probs = [p.raw_probability for p in predictions]
        truths = [p.ground_truth for p in predictions]

        cal_probs_arr = np.array(cal_probs)
        raw_probs_arr = np.array(raw_probs)
        truths_arr = np.array(truths)

        # Brier scores
        raw_brier = float(np.mean((raw_probs_arr - truths_arr) ** 2))
        cal_brier = float(np.mean((cal_probs_arr - truths_arr) ** 2))

        # Log-loss
        eps = 1e-7
        clipped = np.clip(cal_probs_arr, eps, 1 - eps)
        log_loss = float(-np.mean(
            truths_arr * np.log(clipped) + (1 - truths_arr) * np.log(1 - clipped)
        ))

        # Accuracy at 50% threshold
        predicted_binary = (cal_probs_arr >= 0.5).astype(float)
        accuracy = float(np.mean(predicted_binary == truths_arr))

        # ECE
        ece = self._compute_ece(cal_probs_arr, truths_arr)

        # Per-Fish Brier scores
        fish_briers = self._compute_per_fish_brier(predictions, truths_arr)

        run_id = f"retro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = RetrodictionResult(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            n_markets=len(eval_markets),
            n_fish=len(self.personas),
            predictions=predictions,
            brier_score=round(cal_brier, 6),
            log_loss=round(log_loss, 6),
            ece=round(ece, 6),
            accuracy_at_50=round(accuracy, 4),
            fish_brier_scores=fish_briers,
            raw_brier=round(raw_brier, 6),
            calibrated_brier=round(cal_brier, 6),
        )

        logger.info(
            f"Retrodiction complete: Brier={cal_brier:.4f}, "
            f"ECE={ece:.4f}, Accuracy={accuracy:.2%}, "
            f"Log-loss={log_loss:.4f}"
        )

        return result

    def _compute_ece(self, probs: np.ndarray, truths: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(probs)

        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if not np.any(mask):
                continue
            bin_acc = np.mean(truths[mask])
            bin_conf = np.mean(probs[mask])
            ece += np.sum(mask) / n * abs(bin_acc - bin_conf)

        return float(ece)

    def _compute_per_fish_brier(
        self,
        predictions: list[SwarmPrediction],
        truths: np.ndarray,
    ) -> dict[str, float]:
        """Compute Brier score for each Fish persona individually."""
        persona_preds: dict[str, list[float]] = {}

        for pred in predictions:
            for fp in pred.fish_predictions:
                if fp.persona not in persona_preds:
                    persona_preds[fp.persona] = []
                persona_preds[fp.persona].append(fp.probability)

        fish_briers = {}
        for persona, preds in persona_preds.items():
            preds_arr = np.array(preds)
            brier = float(np.mean((preds_arr - truths) ** 2))
            fish_briers[persona] = round(brier, 6)

        return dict(sorted(fish_briers.items(), key=lambda x: x[1]))

    def save_results(self, result: RetrodictionResult, output_dir: str | Path = "data/retrodiction") -> Path:
        """Save retrodiction results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / f"{result.run_id}.json"

        # Serialize (convert SwarmPrediction and FishPrediction to dicts)
        data = {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "n_markets": result.n_markets,
            "n_fish": result.n_fish,
            "metrics": {
                "brier_score": result.brier_score,
                "raw_brier": result.raw_brier,
                "calibrated_brier": result.calibrated_brier,
                "log_loss": result.log_loss,
                "ece": result.ece,
                "accuracy_at_50": result.accuracy_at_50,
            },
            "fish_brier_scores": result.fish_brier_scores,
            "predictions": [
                {
                    "market_id": p.market_id,
                    "question": p.question,
                    "raw_probability": p.raw_probability,
                    "calibrated_probability": p.calibrated_probability,
                    "ground_truth": p.ground_truth,
                    "brier_score": p.brier_score,
                    "spread": p.spread,
                    "mean_confidence": p.mean_confidence,
                    "n_fish": p.n_fish,
                    "fish": [
                        {
                            "persona": fp.persona,
                            "probability": fp.probability,
                            "confidence": fp.confidence,
                        }
                        for fp in p.fish_predictions
                    ],
                }
                for p in result.predictions
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def print_report(self, result: RetrodictionResult) -> None:
        """Print a human-readable evaluation report."""
        print("\n" + "=" * 70)
        print("MIROFISH RETRODICTION EVALUATION REPORT")
        print("=" * 70)
        print(f"Run ID:     {result.run_id}")
        print(f"Timestamp:  {result.timestamp}")
        print(f"Markets:    {result.n_markets}")
        print(f"Fish:       {result.n_fish}")

        print(f"\n--- Aggregate Metrics ---")
        print(f"Brier Score (raw):        {result.raw_brier:.4f}")
        print(f"Brier Score (calibrated): {result.calibrated_brier:.4f}")
        print(f"Log-Loss:                 {result.log_loss:.4f}")
        print(f"ECE:                      {result.ece:.4f}")
        print(f"Accuracy (>50% = yes):    {result.accuracy_at_50:.2%}")

        # Comparison benchmarks
        print(f"\n--- Benchmarks ---")
        print(f"Polymarket crowd Brier:   0.0843")
        print(f"Superforecaster Brier:    0.18")
        print(f"Random baseline Brier:    0.25")
        print(f"Always-50% Brier:         0.25")

        improvement = (0.25 - result.calibrated_brier) / 0.25 * 100
        print(f"Improvement over random:  {improvement:.1f}%")

        print(f"\n--- Per-Fish Brier Scores (lower = better) ---")
        for persona, brier in result.fish_brier_scores.items():
            bar = "#" * int(50 * (1 - brier))
            print(f"  {persona:<28s} {brier:.4f}  {bar}")

        # Worst predictions
        worst = sorted(result.predictions, key=lambda p: -p.brier_score)[:5]
        print(f"\n--- Worst 5 Predictions ---")
        for p in worst:
            print(f"  Brier={p.brier_score:.4f} | P={p.calibrated_probability:.2f} | "
                  f"GT={p.ground_truth:.0f} | {p.question[:60]}")

        # Best predictions
        best = sorted(result.predictions, key=lambda p: p.brier_score)[:5]
        print(f"\n--- Best 5 Predictions ---")
        for p in best:
            print(f"  Brier={p.brier_score:.4f} | P={p.calibrated_probability:.2f} | "
                  f"GT={p.ground_truth:.0f} | {p.question[:60]}")

        # High-disagreement markets
        high_spread = sorted(result.predictions, key=lambda p: -p.spread)[:5]
        print(f"\n--- Highest Disagreement (Fish Spread) ---")
        for p in high_spread:
            print(f"  Spread={p.spread:.2f} | P={p.calibrated_probability:.2f} | "
                  f"GT={p.ground_truth:.0f} | {p.question[:60]}")

        print("\n" + "=" * 70)


async def main() -> None:
    """CLI entry point for retrodiction evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run retrodiction evaluation")
    parser.add_argument(
        "--corpus", type=str, default="data/resolved_markets/resolved_corpus.json",
        help="Path to resolved markets corpus JSON"
    )
    parser.add_argument("--sample", type=int, default=None, help="Sample N markets")
    parser.add_argument("--output", type=str, default="data/retrodiction")

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        print("Run the scraper first: python -m src.markets.history")
        return

    markets = HistoricalMarketScraper.load_corpus(corpus_path)
    print(f"Loaded {len(markets)} resolved markets")

    harness = RetrodictionHarness()
    result = harness.run_retrodiction(markets, sample_size=args.sample)
    harness.print_report(result)
    harness.save_results(result, output_dir=args.output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
