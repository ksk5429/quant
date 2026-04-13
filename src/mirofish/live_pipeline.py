"""Live prediction pipeline — scan, analyze, size, recommend.

Connects: MarketScanner → PredictionEngineV4 → Portfolio → Report

This is the production entry point. It:
1. Scans Polymarket for active high-volume markets
2. Routes each market to the appropriate swarm configuration
3. Runs the multi-round Delphi protocol
4. Calibrates, detects edges, sizes positions
5. Outputs a ranked portfolio recommendation
6. Logs everything to MLflow

Usage:
    python -m src.mirofish.live_pipeline --top 10 --model haiku
    python -m src.mirofish.live_pipeline --top 5 --model sonnet --concurrent 2
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

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

from src.markets.scanner import MarketScanner, ActiveMarket
from src.mirofish.engine_v4 import PredictionEngineV4, EngineV4Result
from src.risk.portfolio import KellyPositionSizer, MarketSignal, EdgeDetector


async def run_live_pipeline(
    top_n: int = 10,
    model: str = "haiku",
    max_concurrent: int = 3,
    min_volume: float = 100_000,
    min_liquidity: float = 20_000,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Run the full live pipeline: scan → analyze → size → report."""
    t_start = time.monotonic()

    # ── Step 1: Scan ──
    logger.info(f"Scanning Polymarket (min_vol=${min_volume:,.0f})...")
    scanner = MarketScanner(
        min_volume_usd=min_volume,
        min_liquidity_usd=min_liquidity,
    )
    candidates = await scanner.scan()
    top_markets = candidates[:top_n]
    logger.info(f"Selected {len(top_markets)} markets for analysis")

    # ── Step 2: Initialize engine ──
    engine = PredictionEngineV4(
        model=model,
        researcher_model="sonnet" if model == "haiku" else model,
        max_concurrent=max_concurrent,
        bankroll_usd=bankroll,
    )

    # ── Step 3: Analyze each market ──
    results: list[EngineV4Result] = []
    for i, market in enumerate(top_markets, 1):
        logger.info(f"[{i}/{len(top_markets)}] {market.question[:60]}")
        try:
            result = await engine.analyze(
                question=market.question,
                description=market.description,
                outcomes=market.outcomes,
                market_price=market.yes_price,
                market_id=market.id,
                volume_usd=market.volume_usd,
            )
            results.append((result, market))  # keep result paired with its market

            # Quick summary
            arrow = "+" if result.position else "-"
            logger.info(
                f"  [{arrow}] P={result.calibrated_probability:.3f} "
                f"mkt={market.yes_price:.3f} edge={result.edge:.3f} "
                f"({result.total_elapsed_s:.0f}s)"
            )
        except Exception as e:
            logger.error(f"  Analysis failed: {e}")

    # ── Step 4: Build portfolio ──
    signals = []
    for r, m in results:  # paired tuples — no misalignment possible
        if not r.swarm_healthy:
            continue
        signals.append(MarketSignal(
            market_id=r.market_id,
            question=r.question,
            market_price=m.yes_price,
            swarm_probability=r.calibrated_probability,
            confidence=r.effective_confidence,
            spread=r.spread,
            disagreement_flag=r.disagreement_flag,
            volume_usd=m.volume_usd,
            category=r.category,
        ))

    edge_detector = EdgeDetector()
    sizer = KellyPositionSizer(
        kelly_fraction=0.25,
        max_position_pct=0.05,
        bankroll_usd=bankroll,
    )

    tradeable = edge_detector.detect_edges(signals)
    portfolio = sizer.build_portfolio(tradeable) if tradeable else None

    elapsed = time.monotonic() - t_start

    # ── Step 5: Report ──
    print(f"\n{'='*80}")
    print(f"MIROFISH LIVE PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"Markets scanned:   {len(candidates)}")
    print(f"Markets analyzed:  {len(results)}")
    print(f"With edge:         {len(tradeable)}")
    print(f"Total time:        {elapsed:.0f}s ({elapsed/max(len(results),1):.0f}s/market)")
    print(f"Calibrator fitted: {engine.calibrator.is_fitted} ({engine.calibrator.training_size} samples)")

    print(f"\n{'─'*80}")
    print(f"{'#':>3} {'Cat':>10} {'Our P':>6} {'Mkt P':>6} {'Edge':>6} {'Spread':>6} {'Action':<20} Question")
    print(f"{'─'*80}")

    for i, (r, m) in enumerate(results, 1):
        edge = abs(r.calibrated_probability - m.yes_price)
        pos = r.position
        action = f"{pos.side} ${pos.position_size_usd:.0f}" if pos else "—"
        flag = "!" if r.disagreement_flag else " "
        print(
            f"{i:>3} {r.category:>10} "
            f"{r.calibrated_probability:>5.0%} {m.yes_price:>5.0%} "
            f"{edge:>5.1%} {r.spread:>5.2f}{flag} "
            f"{action:<20s} {r.question[:35]}"
        )

    if portfolio and portfolio.positions:
        print(f"\n{'─'*80}")
        print(f"PORTFOLIO RECOMMENDATION (bankroll=${bankroll:,.0f})")
        print(f"{'─'*80}")
        for p in portfolio.positions:
            print(f"  {p.side:<4} ${p.position_size_usd:>7.2f}  edge={p.edge:.1%}  {p.question[:45]}")
        print(f"\n  Total exposure: ${portfolio.total_exposure_usd:,.2f} ({portfolio.exposure_pct:.0%})")
        print(f"  Expected EV:    ${portfolio.total_expected_value:,.2f}")
    else:
        print(f"\n  No positions recommended (no sufficient edge found)")

    print(f"{'='*80}")

    # ── Save results ──
    out_dir = Path("data/live_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"live_{ts}.json"

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "n_scanned": len(candidates),
        "n_analyzed": len(results),
        "n_with_edge": len(tradeable),
        "elapsed_s": round(elapsed, 1),
        "model": model,
        "bankroll": bankroll,
        "predictions": [
            {
                "market_id": r.market_id,
                "question": r.question,
                "category": r.category,
                "our_probability": r.calibrated_probability,
                "market_price": m.yes_price,
                "edge": round(abs(r.calibrated_probability - m.yes_price), 4),
                "spread": r.spread,
                "confidence": r.effective_confidence,
                "n_fish": r.n_fish,
                "n_rounds": r.n_rounds,
                "position": {
                    "side": r.position.side,
                    "size_usd": r.position.position_size_usd,
                    "edge": r.position.edge,
                } if r.position else None,
            }
            for r, m in results
        ],
    }
    out_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Results saved: {out_path}")

    return save_data


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mirofish Live Pipeline")
    parser.add_argument("--top", type=int, default=10, help="Top N markets to analyze")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--concurrent", type=int, default=3)
    parser.add_argument("--min-volume", type=float, default=100000)
    parser.add_argument("--bankroll", type=float, default=1000)
    args = parser.parse_args()

    await run_live_pipeline(
        top_n=args.top,
        model=args.model,
        max_concurrent=args.concurrent,
        min_volume=args.min_volume,
        bankroll=args.bankroll,
    )


if __name__ == "__main__":
    asyncio.run(main())
