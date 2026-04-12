"""Large-scale retrodiction runner using v2 Fish swarm.

Runs the 9-Fish CLI swarm on N resolved markets from the corpus,
saves all predictions for calibration pipeline training.

Usage:
    python src/prediction/run_retrodiction.py --n 30 --model haiku --concurrent 3
    python src/prediction/run_retrodiction.py --n 100 --model sonnet --concurrent 2
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Fix encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

from src.markets.history import HistoricalMarketScraper, ResolvedMarket
from src.mirofish.llm_fish import FishSwarm, aggregate_predictions
from src.prediction.calibration import ProbabilityCalibrator


def select_diverse_markets(
    markets: list[ResolvedMarket],
    n: int,
    seed: int = 42,
) -> list[ResolvedMarket]:
    """Select a diverse sample of markets for evaluation.

    Stratified sampling: ensures mix of outcomes (Yes/No wins),
    volume tiers, and question types.
    """
    rng = np.random.RandomState(seed)

    # Separate by outcome
    yes_wins = [m for m in markets if m.winning_index == 0]
    no_wins = [m for m in markets if m.winning_index == 1]

    # Target ~40% yes wins (matches realistic base rate better than 50/50)
    n_yes = min(int(n * 0.4), len(yes_wins))
    n_no = min(n - n_yes, len(no_wins))

    # If not enough of one category, fill from other
    if n_yes + n_no < n:
        remaining = n - n_yes - n_no
        if len(yes_wins) > n_yes:
            n_yes += min(remaining, len(yes_wins) - n_yes)
            remaining = n - n_yes - n_no
        if len(no_wins) > n_no:
            n_no += min(remaining, len(no_wins) - n_no)

    # Stratify by volume (mix of high and low volume markets)
    yes_sorted = sorted(yes_wins, key=lambda m: -m.volume_usd)
    no_sorted = sorted(no_wins, key=lambda m: -m.volume_usd)

    # Take half from top-volume, half random from rest
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

    logger.info(
        f"Selected {len(selected)} markets: "
        f"{sum(1 for m in selected if m.winning_index == 0)} yes-wins, "
        f"{sum(1 for m in selected if m.winning_index == 1)} no-wins"
    )
    return selected[:n]


async def run_retrodiction(
    markets: list[ResolvedMarket],
    model: str = "haiku",
    max_concurrent: int = 3,
    personas: list[str] | None = None,
) -> dict:
    """Run the v2 swarm on resolved markets and score against ground truth."""

    swarm = FishSwarm(
        mode="cli",
        model=model,
        personas=personas,
        max_concurrent=max_concurrent,
        extremize=1.5,
        trim=True,
    )

    results = []
    running_brier = []
    t_start = time.monotonic()

    for i, market in enumerate(markets):
        logger.info(f"[{i+1}/{len(markets)}] {market.question[:70]}")

        try:
            result = await swarm.predict(
                question=market.question,
                description=market.description,
                outcomes=market.outcomes,
            )
        except Exception as e:
            logger.error(f"  Swarm error: {e}")
            continue

        gt = market.ground_truth
        raw_p = result["raw_probability"]
        ext_p = result["extremized_probability"]

        brier_raw = (raw_p - gt) ** 2
        brier_ext = (ext_p - gt) ** 2
        correct = (ext_p >= 0.5 and gt == 1.0) or (ext_p < 0.5 and gt == 0.0)
        running_brier.append(brier_ext)

        entry = {
            "market_id": market.id,
            "question": market.question,
            "outcomes": market.outcomes,
            "ground_truth": gt,
            "winning_outcome": market.winning_outcome,
            "volume_usd": market.volume_usd,
            "raw_probability": raw_p,
            "extremized_probability": ext_p,
            "brier_raw": round(brier_raw, 6),
            "brier_ext": round(brier_ext, 6),
            "correct": correct,
            "spread": result["spread"],
            "std_dev": result["std_dev"],
            "mean_confidence": result["mean_confidence"],
            "effective_confidence": result["effective_confidence"],
            "disagreement_flag": result["disagreement_flag"],
            "n_fish": result["n_fish"],
            "wall_time_s": result["wall_time_s"],
            "fish": [
                {
                    "persona": fp.persona,
                    "probability": fp.probability,
                    "confidence": fp.confidence,
                    "reasoning": fp.reasoning[:200],
                    "elapsed_s": fp.elapsed_s,
                }
                for fp in result["fish_predictions"]
            ],
        }
        results.append(entry)

        avg_b = np.mean(running_brier)
        sym = "+" if correct else "X"
        logger.info(
            f"  [{sym}] P(ext)={ext_p:.3f} GT={gt:.0f} "
            f"Brier={brier_ext:.4f} running={avg_b:.4f} "
            f"spread={result['spread']:.2f} {result['wall_time_s']:.0f}s"
        )

    elapsed = time.monotonic() - t_start

    # ── Aggregate metrics ──
    if not results:
        return {"error": "no results"}

    ext_probs = np.array([r["extremized_probability"] for r in results])
    raw_probs = np.array([r["raw_probability"] for r in results])
    truths = np.array([r["ground_truth"] for r in results])

    brier_ext = float(np.mean((ext_probs - truths) ** 2))
    brier_raw = float(np.mean((raw_probs - truths) ** 2))

    eps = 1e-7
    clipped = np.clip(ext_probs, eps, 1 - eps)
    log_loss = float(-np.mean(truths * np.log(clipped) + (1 - truths) * np.log(1 - clipped)))

    accuracy = float(np.mean((ext_probs >= 0.5).astype(float) == truths))

    # ECE
    n_bins = 10
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for j in range(n_bins):
        mask = (ext_probs >= edges[j]) & (ext_probs < edges[j + 1])
        if np.any(mask):
            ece += np.sum(mask) / len(ext_probs) * abs(np.mean(truths[mask]) - np.mean(ext_probs[mask]))

    # Per-Fish Brier
    fish_briers: dict[str, list[float]] = {}
    for r in results:
        gt_val = r["ground_truth"]
        for fp in r["fish"]:
            fish_briers.setdefault(fp["persona"], []).append(
                (fp["probability"] - gt_val) ** 2
            )
    fish_avg = {k: round(float(np.mean(v)), 4) for k, v in fish_briers.items()}
    fish_avg = dict(sorted(fish_avg.items(), key=lambda x: x[1]))

    run_id = f"retro_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "n_markets": len(results),
        "n_fish": results[0]["n_fish"] if results else 0,
        "personas": swarm.personas,
        "wall_time_s": round(elapsed, 1),
        "metrics": {
            "brier_extremized": round(brier_ext, 6),
            "brier_raw": round(brier_raw, 6),
            "log_loss": round(log_loss, 6),
            "ece": round(ece, 6),
            "accuracy": round(accuracy, 4),
        },
        "fish_brier_scores": fish_avg,
        "predictions": results,
    }

    # Save
    out_dir = Path("data/retrodiction")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {out_path}")

    # ── Print report ──
    print("\n" + "=" * 72)
    print("MIROFISH v2 RETRODICTION REPORT")
    print("=" * 72)
    print(f"Run:       {run_id}")
    print(f"Model:     {model}")
    print(f"Markets:   {len(results)}")
    print(f"Fish:      {results[0]['n_fish'] if results else 0} ({len(swarm.personas)} personas)")
    print(f"Time:      {elapsed:.0f}s ({elapsed/len(results):.1f}s/market)")
    print(f"Cost:      $0.00 (Max subscription)")

    print(f"\n{'─'*40} Metrics {'─'*24}")
    print(f"Brier (extremized):  {brier_ext:.4f}")
    print(f"Brier (raw avg):     {brier_raw:.4f}")
    print(f"Log-loss:            {log_loss:.4f}")
    print(f"ECE:                 {ece:.4f}")
    print(f"Accuracy (>50%):     {accuracy:.1%}")
    print(f"\n{'─'*40} Benchmarks {'─'*21}")
    print(f"Polymarket crowd:    0.0843")
    print(f"Superforecaster:     0.18")
    print(f"Random (always 50%): 0.25")

    print(f"\n{'─'*40} Per-Fish Brier {'─'*17}")
    for persona, b in fish_avg.items():
        bar = "#" * int(40 * max(0, 1 - b / 0.3))
        print(f"  {persona:<25s} {b:.4f}  {bar}")

    n_correct = sum(1 for r in results if r["correct"])
    n_disagree = sum(1 for r in results if r["disagreement_flag"])
    print(f"\n{'─'*40} Summary {'─'*24}")
    print(f"Correct:       {n_correct}/{len(results)} ({n_correct/len(results):.0%})")
    print(f"Disagreements: {n_disagree}/{len(results)} ({n_disagree/len(results):.0%})")

    # Calibration data output
    cal_path = out_dir / f"{run_id}_calibration_data.json"
    cal_data = [
        {"predicted": r["extremized_probability"], "actual": r["ground_truth"]}
        for r in results
    ]
    cal_path.write_text(json.dumps(cal_data, indent=2), encoding="utf-8")
    print(f"\nCalibration data: {cal_path}")
    print("=" * 72)

    return summary


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mirofish v2 Retrodiction")
    parser.add_argument("--n", type=int, default=30, help="Number of markets")
    parser.add_argument("--model", type=str, default="haiku", help="CLI model")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent Fish")
    parser.add_argument("--corpus", type=str, default="data/resolved_markets/resolved_corpus_full.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        print("Run scraper first: python -m src.markets.history --limit 2500")
        return

    markets = HistoricalMarketScraper.load_corpus(corpus_path)
    print(f"Loaded {len(markets)} resolved markets")

    selected = select_diverse_markets(markets, n=args.n, seed=args.seed)
    await run_retrodiction(selected, model=args.model, max_concurrent=args.concurrent)


if __name__ == "__main__":
    asyncio.run(main())
