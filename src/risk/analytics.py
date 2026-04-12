"""Portfolio analytics module.

Uses quantstats and jquantstats for comprehensive risk metrics,
performance attribution, and Monte Carlo simulation.

Generates:
- Sharpe, Sortino, Calmar ratios
- Max drawdown analysis
- Monthly returns heatmap
- Monte Carlo simulation of Kelly strategies
- Performance comparison against benchmarks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger

try:
    import quantstats as qs
    HAS_QS = True
except ImportError:
    HAS_QS = False

try:
    import jquantstats as jqs
    HAS_JQS = True
except ImportError:
    HAS_JQS = False


@dataclass
class PerformanceMetrics:
    """Core performance metrics for a prediction market portfolio."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    n_trades: int
    n_wins: int
    n_losses: int


def compute_performance(
    pnl_series: Sequence[float],
    risk_free_rate: float = 0.04,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics from a P&L series.

    Args:
        pnl_series: list of per-trade P&L values (positive = profit)
        risk_free_rate: annualized risk-free rate for Sharpe computation
    """
    pnl = np.array(pnl_series)
    n = len(pnl)

    if n == 0:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
            max_drawdown_duration_days=0, win_rate=0, profit_factor=0,
            avg_win=0, avg_loss=0, n_trades=0, n_wins=0, n_losses=0,
        )

    # Cumulative returns
    cumulative = np.cumsum(pnl)
    total_return = float(cumulative[-1])

    # Win/loss stats
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n if n > 0 else 0
    avg_win = float(np.mean(wins)) if n_wins > 0 else 0
    avg_loss = float(np.mean(losses)) if n_losses > 0 else 0

    # Profit factor
    gross_profit = float(np.sum(wins)) if n_wins > 0 else 0
    gross_loss = float(np.abs(np.sum(losses))) if n_losses > 0 else 1e-8
    profit_factor = gross_profit / gross_loss

    # Sharpe ratio (annualized, assuming daily trades)
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl))
    daily_rf = risk_free_rate / 252
    sharpe = (mean_pnl - daily_rf) / std_pnl * np.sqrt(252) if std_pnl > 1e-8 else 0

    # Sortino (downside deviation only)
    downside = pnl[pnl < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-8
    sortino = (mean_pnl - daily_rf) / downside_std * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdowns = peak - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    # Max drawdown duration
    dd_durations = []
    current_dd_start = None
    for i, dd in enumerate(drawdowns):
        if dd > 0 and current_dd_start is None:
            current_dd_start = i
        elif dd == 0 and current_dd_start is not None:
            dd_durations.append(i - current_dd_start)
            current_dd_start = None
    if current_dd_start is not None:
        dd_durations.append(len(drawdowns) - current_dd_start)
    max_dd_duration = max(dd_durations) if dd_durations else 0

    # Calmar ratio
    annualized = mean_pnl * 252
    calmar = annualized / max_dd if max_dd > 1e-8 else 0

    return PerformanceMetrics(
        total_return=round(total_return, 2),
        annualized_return=round(annualized, 2),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        max_drawdown=round(max_dd, 4),
        max_drawdown_duration_days=max_dd_duration,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        avg_win=round(avg_win, 4),
        avg_loss=round(avg_loss, 4),
        n_trades=n,
        n_wins=n_wins,
        n_losses=n_losses,
    )


def monte_carlo_kelly(
    trade_history: Sequence[dict[str, float]] | None = None,
    edge_distribution: Sequence[float] | None = None,
    kelly_fraction: float = 0.25,
    bankroll: float = 1000.0,
    n_trades: int = 100,
    n_simulations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo simulation of Kelly-sized prediction market trades.

    Accepts either:
    - trade_history: list of dicts with {predicted_prob, market_price, outcome}
      for accurate simulation using actual market structure
    - edge_distribution: list of signed edges (legacy, less accurate)

    Args:
        trade_history: list of {predicted_prob, market_price, outcome} dicts
        edge_distribution: legacy signed edges (fallback)
        kelly_fraction: fraction of Kelly to use
        bankroll: starting capital
        n_trades: trades per simulation path
        n_simulations: number of paths to simulate
        seed: random seed

    Returns:
        Dict with percentile outcomes, ruin probability, expected growth.
    """
    rng = np.random.RandomState(seed)

    # Build trade parameters from history or edges
    if trade_history:
        trades = [
            {
                "p_est": t["predicted_prob"],
                "p_mkt": t["market_price"],
                "outcome": t["outcome"],
            }
            for t in trade_history
        ]
    elif edge_distribution is not None:
        # Legacy fallback: convert edges to synthetic trade params
        trades = []
        for e in edge_distribution:
            p_mkt = 0.5
            p_est = 0.5 + e
            outcome = 1.0 if e > 0 else 0.0
            trades.append({"p_est": p_est, "p_mkt": p_mkt, "outcome": outcome})
    else:
        raise ValueError("Provide either trade_history or edge_distribution")

    final_values = np.zeros(n_simulations)

    for sim in range(n_simulations):
        capital = bankroll
        for _ in range(n_trades):
            t = trades[rng.randint(0, len(trades))]
            p_est = t["p_est"]
            p_mkt = t["p_mkt"]

            # Determine side and Kelly sizing
            if p_est > p_mkt:
                # Buy YES at market_price
                cost = p_mkt
                p_win = p_est  # our estimated win probability
            else:
                # Buy NO at (1 - market_price)
                cost = 1.0 - p_mkt
                p_win = 1.0 - p_est

            if cost <= 0.01 or cost >= 0.99:
                continue

            b = (1.0 - cost) / cost  # payout ratio
            q = 1.0 - p_win
            kelly_raw = (p_win * b - q) / b
            if kelly_raw <= 0:
                continue

            bet_size = min(kelly_raw * kelly_fraction * capital, capital * 0.05)

            # Simulate outcome using actual win probability
            if rng.random() < p_win:
                capital += bet_size * b  # win: gain payout
            else:
                capital -= bet_size  # lose: lose stake

            capital = max(capital, 0)

        final_values[sim] = capital

    percentiles = {
        "p5": round(float(np.percentile(final_values, 5)), 2),
        "p25": round(float(np.percentile(final_values, 25)), 2),
        "p50": round(float(np.percentile(final_values, 50)), 2),
        "p75": round(float(np.percentile(final_values, 75)), 2),
        "p95": round(float(np.percentile(final_values, 95)), 2),
    }

    ruin_prob = float(np.mean(final_values < bankroll * 0.1))
    expected_growth = float(np.mean(final_values) / bankroll - 1)
    median_growth = float(np.median(final_values) / bankroll - 1)

    return {
        "percentiles": percentiles,
        "ruin_probability": round(ruin_prob, 4),
        "expected_growth": round(expected_growth, 4),
        "median_growth": round(median_growth, 4),
        "bankroll": bankroll,
        "kelly_fraction": kelly_fraction,
        "n_trades": n_trades,
        "n_simulations": n_simulations,
    }


def generate_tearsheet(
    pnl_series: Sequence[float],
    output_path: str | Path = "data/reports/tearsheet.html",
    title: str = "Mirofish Portfolio",
) -> Path | None:
    """Generate an HTML performance tearsheet using quantstats.

    Returns the path to the generated HTML file, or None if quantstats
    is not available.
    """
    if not HAS_QS:
        logger.warning("quantstats not available for tearsheet generation")
        return None

    try:
        import pandas as pd
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Convert P&L to returns series
        returns = pd.Series(pnl_series, name="Strategy")
        dates = pd.date_range("2024-01-01", periods=len(returns), freq="D")
        returns.index = dates

        qs.reports.html(returns, output=str(output), title=title)
        logger.info(f"Tearsheet saved: {output}")
        return output

    except Exception as e:
        logger.error(f"Tearsheet generation failed: {e}")
        return None
