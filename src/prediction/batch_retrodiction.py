"""Batch retrodiction runner with DB persistence and statistical reporting.

Phase 2 deliverable: runs N markets through the v4 engine,
stores all results in the database, and produces a comprehensive
statistical report with bootstrap confidence intervals.

Usage:
    with DatabaseManager() as db:
        engine = PredictionEngineV4(model="haiku", db=db)
        batch = BatchRetrodiction(engine, db)
        await batch.run(markets, concurrent=3, resume=True)
        batch.report()
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

from src.db.manager import DatabaseManager
from src.markets.history import ResolvedMarket
from src.prediction.calibration import compute_brier, compute_ece, compute_log_loss
from src.prediction.advanced_scoring import (
    brier_decomposition, comprehensive_evaluate,
)


@dataclass
class RetrodictionReport:
    """Comprehensive retrodiction evaluation report."""
    n_markets: int
    n_correct: int
    accuracy: float

    # Scoring
    brier: float
    brier_ci_lower: float
    brier_ci_upper: float
    log_loss: float
    ece: float

    # Brier decomposition
    reliability: float
    resolution: float
    uncertainty: float

    # Brier Skill Score vs RANDOM (uniform 0.5 baseline, NOT crowd prices)
    bss_vs_random: float
    bss_p_value: float
    bss_significant: bool

    # Per-category breakdown
    category_stats: dict[str, dict[str, float]]

    # Per-Fish rankings
    fish_brier: dict[str, float]

    # Calibration data
    calibration_bins: list[dict[str, float]]

    # Timing
    total_elapsed_s: float
    avg_elapsed_per_market: float


class BatchRetrodiction:
    """Runs batch retrodiction with DB persistence and resumption."""

    def __init__(
        self,
        engine: Any,  # PredictionEngineV4
        db: DatabaseManager,
    ) -> None:
        self.engine = engine
        self.db = db
        self._results: list[dict] = []

    async def run(
        self,
        markets: list[ResolvedMarket],
        concurrent: int = 3,
        resume: bool = True,
    ) -> None:
        """Run retrodiction on resolved markets.

        Args:
            markets: list of resolved markets with ground truth.
            concurrent: max concurrent Fish processes.
            resume: if True, skip markets already in DB.
        """
        t_start = time.monotonic()

        # Check which markets are already evaluated
        if resume:
            done_ids = set()
            rows = self.db.conn.execute(
                "SELECT DISTINCT market_id FROM predictions"
            ).fetchall()
            done_ids = {r["market_id"] for r in rows}
            remaining = [m for m in markets if m.id not in done_ids]
            logger.info(
                f"Resume mode: {len(done_ids)} already done, "
                f"{len(remaining)} remaining out of {len(markets)}"
            )
        else:
            remaining = markets

        running_brier = []

        for i, market in enumerate(remaining, 1):
            logger.info(f"[{i}/{len(remaining)}] {market.question[:65]}")

            try:
                result = await self.engine.analyze(
                    question=market.question,
                    description=market.description,
                    outcomes=market.outcomes,
                    market_id=market.id,
                )

                gt = market.ground_truth
                cal_p = result.calibrated_probability
                brier = (cal_p - gt) ** 2
                correct = (cal_p >= 0.5 and gt == 1.0) or (cal_p < 0.5 and gt == 0.0)
                running_brier.append(brier)

                # Record outcome in DB
                self.engine.record_outcome(
                    market_id=market.id,
                    prediction=cal_p,
                    outcome=gt,
                )

                self._results.append({
                    "market_id": market.id,
                    "question": market.question,
                    "category": result.category,
                    "prediction": cal_p,
                    "raw_probability": result.raw_probability,
                    "market_price": market.outcome_prices[0] if market.outcome_prices else None,
                    "ground_truth": gt,
                    "brier": brier,
                    "correct": correct,
                    "spread": result.spread,
                    "n_fish": result.n_fish,
                    "fish_predictions": [
                        {"persona": fp.persona, "probability": fp.probability}
                        for fp in result.fish_predictions
                    ],
                })

                avg_b = np.mean(running_brier)
                sym = "+" if correct else "X"
                logger.info(
                    f"  [{sym}] P={cal_p:.3f} GT={gt:.0f} "
                    f"Brier={brier:.4f} running={avg_b:.4f}"
                )

                # Checkpoint every 25 markets
                if i % 25 == 0:
                    self._save_checkpoint(i)

            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue

            # Print summary every 10
            if i % 10 == 0 and running_brier:
                logger.info(
                    f"  --- Progress: {i}/{len(remaining)} | "
                    f"Brier={np.mean(running_brier):.4f} | "
                    f"Accuracy={sum(1 for r in self._results if r['correct'])/len(self._results):.1%}"
                )

        elapsed = time.monotonic() - t_start
        logger.info(
            f"Batch retrodiction complete: {len(self._results)} markets "
            f"in {elapsed:.0f}s ({elapsed/max(len(self._results),1):.1f}s/market)"
        )

    def report(self) -> RetrodictionReport:
        """Generate comprehensive statistical report."""
        if not self._results:
            # Load from DB if no in-memory results
            self._load_from_db()

        if not self._results:
            logger.error("No results to report")
            raise ValueError("No retrodiction results available. Run batch first.")

        predictions = [r["prediction"] for r in self._results]
        outcomes = [r["ground_truth"] for r in self._results]
        n = len(predictions)

        # Basic metrics
        brier = compute_brier(predictions, outcomes)
        log_loss = compute_log_loss(predictions, outcomes)
        ece = compute_ece(predictions, outcomes)
        n_correct = sum(1 for r in self._results if r["correct"])
        accuracy = n_correct / n

        # Brier decomposition
        decomp = brier_decomposition(predictions, outcomes)

        # Bootstrap CI on Brier
        ci_lower, ci_upper, p_value, bss = self._bootstrap_analysis(
            predictions, outcomes
        )

        # Per-category breakdown
        category_stats = self._per_category_breakdown()

        # Per-Fish Brier
        fish_brier = self._per_fish_brier(outcomes)

        # Calibration bins
        cal_bins = self._calibration_bins(predictions, outcomes)

        report = RetrodictionReport(
            n_markets=n,
            n_correct=n_correct,
            accuracy=round(accuracy, 4),
            brier=round(brier, 6),
            brier_ci_lower=round(ci_lower, 6),
            brier_ci_upper=round(ci_upper, 6),
            log_loss=round(log_loss, 6),
            ece=round(ece, 6),
            reliability=decomp.reliability,
            resolution=decomp.resolution,
            uncertainty=decomp.uncertainty,
            bss_vs_random=round(bss, 4),
            bss_p_value=round(p_value, 4),
            bss_significant=p_value < 0.10,
            category_stats=category_stats,
            fish_brier=fish_brier,
            calibration_bins=cal_bins,
            total_elapsed_s=0,
            avg_elapsed_per_market=0,
        )

        self._print_report(report)
        return report

    def _bootstrap_analysis(
        self,
        predictions: list[float],
        outcomes: list[float],
        n_bootstrap: int = 10000,
    ) -> tuple[float, float, float, float]:
        """Bootstrap CI on Brier and BSS vs market (uniform 0.5)."""
        preds = np.array(predictions)
        outs = np.array(outcomes)
        n = len(preds)

        observed_brier = float(np.mean((preds - outs) ** 2))
        market_brier = 0.25  # uniform guess baseline

        # BSS = 1 - our/reference
        bss = 1.0 - observed_brier / market_brier

        # Bootstrap
        rng = np.random.RandomState(42)
        boot_briers = np.zeros(n_bootstrap)
        boot_bss = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_preds = preds[idx]
            boot_outs = outs[idx]
            bb = float(np.mean((boot_preds - boot_outs) ** 2))
            boot_briers[b] = bb
            boot_bss[b] = 1.0 - bb / market_brier

        ci_lower = float(np.percentile(boot_briers, 2.5))
        ci_upper = float(np.percentile(boot_briers, 97.5))

        # p-value: fraction of bootstrap samples where BSS <= 0
        p_value = float(np.mean(boot_bss <= 0))

        return ci_lower, ci_upper, p_value, bss

    def _per_category_breakdown(self) -> dict[str, dict[str, float]]:
        """Per-category Brier, accuracy, count."""
        categories: dict[str, list[dict]] = {}
        for r in self._results:
            cat = r.get("category", "general") or "general"
            categories.setdefault(cat, []).append(r)

        stats = {}
        for cat, results in categories.items():
            preds = [r["prediction"] for r in results]
            outs = [r["ground_truth"] for r in results]
            n_c = sum(1 for r in results if r["correct"])
            stats[cat] = {
                "n": len(results),
                "brier": round(compute_brier(preds, outs), 4),
                "accuracy": round(n_c / len(results), 4) if results else 0,
            }
        return dict(sorted(stats.items(), key=lambda x: x[1]["brier"]))

    def _per_fish_brier(self, outcomes: list[float]) -> dict[str, float]:
        """Per-Fish individual Brier scores."""
        fish_preds: dict[str, list[float]] = {}
        for r in self._results:
            for fp in r.get("fish_predictions", []):
                persona = fp["persona"]
                fish_preds.setdefault(persona, []).append(fp["probability"])

        fish_brier = {}
        outs = np.array(outcomes)
        for persona, preds in fish_preds.items():
            if len(preds) == len(outs):
                fish_brier[persona] = round(
                    float(np.mean((np.array(preds) - outs) ** 2)), 4
                )
        return dict(sorted(fish_brier.items(), key=lambda x: x[1]))

    def _calibration_bins(
        self, predictions: list[float], outcomes: list[float], n_bins: int = 10,
    ) -> list[dict[str, float]]:
        """10-bin calibration data for reliability diagram."""
        preds = np.array(predictions)
        outs = np.array(outcomes)
        edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for i in range(n_bins):
            mask = (preds >= edges[i]) & (preds < edges[i + 1])
            count = int(np.sum(mask))
            if count > 0:
                bins.append({
                    "bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                    "count": count,
                    "mean_pred": round(float(np.mean(preds[mask])), 4),
                    "actual_freq": round(float(np.mean(outs[mask])), 4),
                    "error": round(abs(float(np.mean(preds[mask])) - float(np.mean(outs[mask]))), 4),
                })
        return bins

    def _load_from_db(self) -> None:
        """Load results from database if none in memory."""
        rows = self.db.conn.execute("""
            SELECT p.market_id, p.question, p.category,
                   p.calibrated_probability, p.raw_probability, p.spread, p.n_fish,
                   p.fish_predictions, p.market_price,
                   r.outcome, r.brier_score
            FROM predictions p
            JOIN resolutions r ON p.market_id = r.market_id
            ORDER BY p.timestamp
        """).fetchall()

        for row in rows:
            gt = row["outcome"]
            cal_p = row["calibrated_probability"]
            correct = (cal_p >= 0.5 and gt == 1.0) or (cal_p < 0.5 and gt == 0.0)
            fish = json.loads(row["fish_predictions"]) if row["fish_predictions"] else []

            self._results.append({
                "market_id": row["market_id"],
                "question": row["question"],
                "category": row["category"],
                "prediction": cal_p,
                "raw_probability": row["raw_probability"],
                "market_price": row["market_price"],
                "ground_truth": gt,
                "brier": (cal_p - gt) ** 2,
                "correct": correct,
                "spread": row["spread"],
                "n_fish": row["n_fish"],
                "fish_predictions": fish,
            })

        logger.info(f"Loaded {len(self._results)} results from DB")

    def _save_checkpoint(self, n_complete: int) -> None:
        """Save checkpoint to data/retrodiction/."""
        out_dir = Path("data/retrodiction")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"checkpoint_{ts}_{n_complete}.json"
        path.write_text(
            json.dumps({"n": n_complete, "results": self._results[-25:]}, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Checkpoint saved: {path}")

    def _print_report(self, r: RetrodictionReport) -> None:
        """Print comprehensive report."""
        print(f"\n{'='*72}")
        print(f"K-FISH RETRODICTION REPORT")
        print(f"{'='*72}")
        print(f"Markets evaluated: {r.n_markets}")
        print(f"Correct: {r.n_correct}/{r.n_markets} ({r.accuracy:.1%})")

        print(f"\n{'─'*40} Scoring {'─'*24}")
        print(f"Brier Score:     {r.brier:.4f}  [{r.brier_ci_lower:.4f}, {r.brier_ci_upper:.4f}] 95% CI")
        print(f"Log-Loss:        {r.log_loss:.4f}")
        print(f"ECE:             {r.ece:.4f}")

        print(f"\n{'─'*40} Brier Decomposition {'─'*11}")
        print(f"Reliability:     {r.reliability:.4f}  (lower = better calibrated)")
        print(f"Resolution:      {r.resolution:.4f}  (higher = better discrimination)")
        print(f"Uncertainty:     {r.uncertainty:.4f}  (fixed for this dataset)")

        print(f"\n{'─'*40} Brier Skill Score {'─'*13}")
        print(f"BSS vs random:   {r.bss_vs_random:+.4f}  {'BEATS random' if r.bss_vs_random > 0 else 'LOSES to random'}")
        print(f"p-value:         {r.bss_p_value:.4f}  {'SIGNIFICANT' if r.bss_significant else 'not significant'}")

        if r.category_stats:
            print(f"\n{'─'*40} Per-Category {'─'*19}")
            print(f"{'Category':<15} {'N':>5} {'Brier':>8} {'Accuracy':>10}")
            for cat, s in r.category_stats.items():
                print(f"{cat:<15} {s['n']:>5} {s['brier']:>8.4f} {s['accuracy']:>9.1%}")

        if r.fish_brier:
            print(f"\n{'─'*40} Per-Fish Brier {'─'*17}")
            for persona, b in r.fish_brier.items():
                bar = "#" * int(40 * max(0, 1 - b / 0.35))
                print(f"  {persona:<25s} {b:.4f}  {bar}")

        if r.calibration_bins:
            print(f"\n{'─'*40} Calibration Curve {'─'*13}")
            print(f"{'Bin':<10} {'N':>5} {'Predicted':>10} {'Actual':>10} {'Error':>8}")
            for b in r.calibration_bins:
                print(f"{b['bin']:<10} {b['count']:>5} {b['mean_pred']:>10.4f} {b['actual_freq']:>10.4f} {b['error']:>8.4f}")

        print(f"{'='*72}")
