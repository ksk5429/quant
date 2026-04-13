"""Edge threshold optimizer — data-driven minimum edge for profitable trading.

Replaces the hardcoded 7% edge threshold with an empirically derived
value computed from retrodiction data.

Method:
1. Load retrodiction corpus (parquet or JSON)
2. For each market, compute: edge = |calibrated_prob - market_price|
3. Sweep threshold from 0% to 30% in 0.5% steps
4. At each threshold: compute cumulative Kelly returns after tx costs
5. Find the minimum threshold where returns turn positive
6. Store optimal threshold in config/default.yaml

Usage:
    optimizer = ThresholdOptimizer(tx_cost=0.02)
    result = optimizer.optimize(retrodiction_data)
    print(result.optimal_threshold)  # e.g., 0.058
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ThresholdOptimizationResult:
    """Result of edge threshold optimization."""
    optimal_threshold: float
    expected_return_at_optimal: float
    n_markets_at_optimal: int
    n_trades_at_optimal: int
    sweep_data: list[dict]  # threshold → return curve
    tx_cost: float
    kelly_fraction: float


class ThresholdOptimizer:
    """Find the minimum edge threshold where Kelly returns turn positive."""

    def __init__(
        self,
        tx_cost: float = 0.02,
        kelly_fraction: float = 0.25,
        min_threshold: float = 0.00,
        max_threshold: float = 0.30,
        step: float = 0.005,
    ) -> None:
        self.tx_cost = tx_cost
        self.kelly_fraction = kelly_fraction
        self.thresholds = np.arange(min_threshold, max_threshold + step, step)

    def optimize(
        self,
        data: pd.DataFrame | list[dict] | str,
    ) -> ThresholdOptimizationResult:
        """Find optimal edge threshold from retrodiction data.

        Data must have columns/keys:
        - swarm_calibrated (or swarm_extremized): our probability
        - market_price: Polymarket closing price
        - ground_truth: 1.0 or 0.0
        """
        # Load data
        if isinstance(data, str):
            path = Path(data)
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        # Get our probability (prefer calibrated, fall back to extremized)
        if "swarm_calibrated" in df.columns:
            our_prob = df["swarm_calibrated"].values
        elif "swarm_extremized" in df.columns:
            our_prob = df["swarm_extremized"].values
        elif "extremized_probability" in df.columns:
            our_prob = df["extremized_probability"].values
        else:
            raise ValueError("Data must have swarm_calibrated or swarm_extremized column")

        market_price = df["market_price"].values if "market_price" in df.columns else np.full(len(our_prob), 0.5)
        ground_truth = df["ground_truth"].values

        # Filter out rows with missing data
        valid = ~(np.isnan(our_prob) | np.isnan(ground_truth))
        if "market_price" in df.columns:
            valid &= ~np.isnan(market_price)
        our_prob = our_prob[valid]
        market_price = market_price[valid]
        ground_truth = ground_truth[valid]

        n_total = len(our_prob)
        edges = np.abs(our_prob - market_price)

        sweep_data = []
        best_threshold = 0.07  # fallback
        best_return = -999

        for threshold in self.thresholds:
            # Which markets pass the threshold?
            mask = edges >= (threshold + self.tx_cost)
            n_trades = int(np.sum(mask))

            if n_trades == 0:
                sweep_data.append({
                    "threshold": round(float(threshold), 4),
                    "n_trades": 0,
                    "cum_return": 0.0,
                    "win_rate": 0.0,
                    "avg_edge": 0.0,
                })
                continue

            # Compute Kelly returns for trades that pass threshold
            trade_probs = our_prob[mask]
            trade_prices = market_price[mask]
            trade_outcomes = ground_truth[mask]
            trade_edges = edges[mask]

            total_return = 0.0
            n_wins = 0

            for p_est, p_mkt, outcome in zip(trade_probs, trade_prices, trade_outcomes):
                # Determine side
                if p_est > p_mkt:
                    cost = p_mkt
                    p_win = p_est
                else:
                    cost = 1.0 - p_mkt
                    p_win = 1.0 - p_est

                if cost < 0.02 or cost > 0.98:
                    continue

                # Kelly bet fraction
                b = (1.0 - cost) / cost
                q = 1.0 - p_win
                kelly_raw = (p_win * b - q) / b
                if kelly_raw <= 0:
                    continue

                bet_frac = kelly_raw * self.kelly_fraction
                bet_frac = min(bet_frac, 0.05)  # cap at 5%

                # Simulate P&L
                if p_est > p_mkt:
                    pnl = (outcome / cost - 1.0) * bet_frac if outcome == 1.0 else -bet_frac
                else:
                    pnl = ((1 - outcome) / cost - 1.0) * bet_frac if outcome == 0.0 else -bet_frac

                # Subtract transaction cost
                pnl -= self.tx_cost * bet_frac

                total_return += pnl
                if pnl > 0:
                    n_wins += 1

            avg_return = total_return / max(n_trades, 1)
            win_rate = n_wins / max(n_trades, 1)
            avg_edge = float(np.mean(trade_edges))

            sweep_data.append({
                "threshold": round(float(threshold), 4),
                "n_trades": n_trades,
                "cum_return": round(total_return, 6),
                "avg_return": round(avg_return, 6),
                "win_rate": round(win_rate, 4),
                "avg_edge": round(avg_edge, 4),
            })

            if total_return > best_return and n_trades >= 5:
                best_return = total_return
                best_threshold = float(threshold)

        # Find minimum threshold where returns turn positive
        positive_thresholds = [
            s for s in sweep_data
            if s["cum_return"] > 0 and s["n_trades"] >= 5
        ]
        if positive_thresholds:
            optimal = min(positive_thresholds, key=lambda s: s["threshold"])
            optimal_threshold = optimal["threshold"]
        else:
            optimal_threshold = best_threshold

        # Count trades at optimal
        n_at_optimal = sum(1 for s in sweep_data if s["threshold"] == optimal_threshold)
        trades_at_opt = next(
            (s for s in sweep_data if abs(s["threshold"] - optimal_threshold) < 0.001),
            {"n_trades": 0, "cum_return": 0}
        )

        result = ThresholdOptimizationResult(
            optimal_threshold=round(optimal_threshold, 4),
            expected_return_at_optimal=round(trades_at_opt["cum_return"], 4),
            n_markets_at_optimal=n_total,
            n_trades_at_optimal=trades_at_opt["n_trades"],
            sweep_data=sweep_data,
            tx_cost=self.tx_cost,
            kelly_fraction=self.kelly_fraction,
        )

        logger.info(
            f"Threshold optimization: optimal={optimal_threshold:.1%} "
            f"({trades_at_opt['n_trades']} trades, "
            f"return={trades_at_opt['cum_return']:+.4f})"
        )

        return result

    def update_config(self, threshold: float, config_path: str = "config/default.yaml") -> None:
        """Update the edge threshold in config/default.yaml."""
        import yaml

        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Update or add the threshold
        if "risk" not in config:
            config["risk"] = {}
        config["risk"]["min_edge"] = round(threshold, 4)
        config["risk"]["min_edge_source"] = "empirical (retrodiction_pipeline)"
        config["risk"]["min_edge_updated"] = datetime.now().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Config updated: risk.min_edge = {threshold:.4f}")

    def print_sweep(self, result: ThresholdOptimizationResult) -> None:
        """Print the threshold sweep results."""
        print(f"\n{'='*60}")
        print(f"EDGE THRESHOLD OPTIMIZATION")
        print(f"{'='*60}")
        print(f"TX cost:        {result.tx_cost:.1%}")
        print(f"Kelly fraction: {result.kelly_fraction}")
        print(f"Optimal:        {result.optimal_threshold:.1%}")
        print(f"Trades at opt:  {result.n_trades_at_optimal}")
        print(f"Return at opt:  {result.expected_return_at_optimal:+.4f}")

        print(f"\n{'Threshold':>10} {'Trades':>7} {'Return':>10} {'WinRate':>8} {'AvgEdge':>8}")
        print(f"{'─'*45}")
        for s in result.sweep_data:
            if s["n_trades"] == 0:
                continue
            marker = " ◄" if abs(s["threshold"] - result.optimal_threshold) < 0.001 else ""
            print(
                f"{s['threshold']:>9.1%} {s['n_trades']:>7} "
                f"{s['cum_return']:>+9.4f} {s.get('win_rate', 0):>7.0%} "
                f"{s.get('avg_edge', 0):>7.1%}{marker}"
            )
        print(f"{'='*60}")


# Need datetime for config update
from datetime import datetime
