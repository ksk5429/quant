"""Retrodiction expansion pipeline — scrape, evaluate, store as parquet.

Builds a calibration training corpus from resolved Polymarket markets
by running each through the full engine_v4 pipeline and recording:
- Per-Fish raw probabilities
- Aggregated swarm probability
- Calibrated probability
- Resolved outcome (ground truth)
- Market price at close
- Bias regime classification

Output: data/retrodiction_corpus.parquet (columnar, fast to query)

Usage:
    python -m src.prediction.retrodiction_pipeline --n 200 --model haiku
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

from src.db.manager import DatabaseManager
from src.markets.history import HistoricalMarketScraper, ResolvedMarket
from src.mirofish.engine_v4 import PredictionEngineV4


def select_diverse_markets(
    markets: list[ResolvedMarket], n: int, seed: int = 42,
) -> list[ResolvedMarket]:
    """Stratified sample: mix of outcomes, volumes, categories."""
    rng = np.random.RandomState(seed)
    yes_wins = [m for m in markets if m.winning_index == 0]
    no_wins = [m for m in markets if m.winning_index == 1]

    n_yes = min(int(n * 0.4), len(yes_wins))
    n_no = min(n - n_yes, len(no_wins))
    if n_yes + n_no < n:
        extra = n - n_yes - n_no
        if len(yes_wins) > n_yes:
            n_yes += min(extra, len(yes_wins) - n_yes)

    yes_sorted = sorted(yes_wins, key=lambda m: -m.volume_usd)
    no_sorted = sorted(no_wins, key=lambda m: -m.volume_usd)

    yes_top = yes_sorted[:n_yes // 2]
    yes_rest = yes_sorted[n_yes // 2:]
    rng.shuffle(yes_rest)
    yes_sample = yes_top + yes_rest[:n_yes - len(yes_top)]

    no_top = no_sorted[:n_no // 2]
    no_rest = no_sorted[n_no // 2:]
    rng.shuffle(no_rest)
    no_sample = no_top + no_rest[:n_no - len(no_top)]

    selected = yes_sample + no_sample
    rng.shuffle(selected)
    return selected[:n]


async def run_retrodiction_pipeline(
    n_markets: int = 200,
    model: str = "haiku",
    concurrent: int = 3,
    corpus_path: str = "data/resolved_markets/resolved_corpus_5000.json",
    output_path: str = "data/retrodiction_corpus.parquet",
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full retrodiction pipeline and store as parquet.

    Returns DataFrame with columns:
        market_id, question, category, ground_truth, volume_usd,
        fish_<persona> (9 columns), swarm_raw, swarm_extremized,
        swarm_calibrated, market_price, spread, std_dev,
        n_fish, n_rounds, correct, brier
    """
    # Load corpus
    corpus = Path(corpus_path)
    if not corpus.exists():
        # Fall back to smaller corpus
        corpus = Path("data/resolved_markets/resolved_corpus_full.json")
    if not corpus.exists():
        raise FileNotFoundError(f"No corpus found at {corpus_path}")

    markets_all = HistoricalMarketScraper.load_corpus(str(corpus))
    logger.info(f"Loaded {len(markets_all)} markets from {corpus}")

    selected = select_diverse_markets(markets_all, n_markets, seed)
    logger.info(f"Selected {len(selected)} markets for retrodiction")

    # Initialize v5 engine with DB persistence
    with DatabaseManager("data/kfish.db") as db:
        engine = PredictionEngineV4(
            model=model,
            max_concurrent=concurrent,
            db=db,
        )

        rows = []
        running_brier = []
        t_start = time.monotonic()

        for i, market in enumerate(selected, 1):
            logger.info(f"[{i}/{len(selected)}] {market.question[:65]}")

            try:
                result = await engine.analyze(
                    question=market.question,
                    description=market.description,
                    outcomes=market.outcomes,
                    market_id=market.id,
                    volume_usd=market.volume_usd,
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue

            gt = market.ground_truth
            raw_p = result.raw_probability
            ext_p = result.extremized_probability
            cal_p = result.calibrated_probability

            brier = (cal_p - gt) ** 2
            correct = (cal_p >= 0.5 and gt == 1.0) or (cal_p < 0.5 and gt == 0.0)
            running_brier.append(brier)

            # Record outcome for calibrator training
            engine.record_outcome(
                market_id=market.id,
                prediction=cal_p,
                outcome=gt,
            )

            # Build row
            row = {
                "market_id": market.id,
                "question": market.question,
                "category": result.category,
                "ground_truth": gt,
                "volume_usd": market.volume_usd,
                "winning_outcome": market.winning_outcome,
                "swarm_raw": raw_p,
                "swarm_extremized": ext_p,
                "swarm_calibrated": cal_p,
                "market_price": market.outcome_prices[0] if market.outcome_prices else None,
                "spread": result.spread,
                "std_dev": result.std_dev,
                "mean_confidence": result.effective_confidence,
                "n_fish": result.n_fish,
                "n_rounds": result.n_rounds,
                "researcher_used": result.researcher_used,
                "correct": correct,
                "brier": round(brier, 6),
            }

            # Per-Fish columns
            for fp in result.fish_predictions:
                row[f"fish_{fp.persona}"] = fp.probability

            rows.append(row)

        if i % 10 == 0:
            avg_b = np.mean(running_brier)
            acc = sum(1 for r in rows if r["correct"]) / len(rows)
            logger.info(f"  Progress: {i}/{len(selected)} Brier={avg_b:.4f} Acc={acc:.1%}")

    elapsed = time.monotonic() - t_start

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Save as parquet
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Saved {len(df)} rows to {out} ({out.stat().st_size / 1024:.0f} KB)")

    # Print summary
    if len(df) > 0:
        brier_mean = df["brier"].mean()
        acc = df["correct"].mean()
        print(f"\n{'='*60}")
        print(f"RETRODICTION PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Markets:  {len(df)}")
        print(f"Brier:    {brier_mean:.4f}")
        print(f"Accuracy: {acc:.1%}")
        print(f"Time:     {elapsed:.0f}s ({elapsed/len(df):.1f}s/market)")
        print(f"Output:   {out}")
        print(f"{'='*60}")

    return df


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrodiction expansion pipeline")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--concurrent", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=str, default="data/resolved_markets/resolved_corpus_5000.json")
    parser.add_argument("--output", type=str, default="data/retrodiction_corpus.parquet")
    args = parser.parse_args()

    await run_retrodiction_pipeline(
        n_markets=args.n, model=args.model, concurrent=args.concurrent,
        corpus_path=args.corpus, output_path=args.output, seed=args.seed,
    )


if __name__ == "__main__":
    asyncio.run(main())
