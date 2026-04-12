"""Probability volatility and time-series analysis module.

Uses statsforecast for GARCH/ARCH volatility modeling on market
probability time series. Detects when a market is in a high-volatility
regime (breaking news, rapid repricing) vs. stable regime.

This directly affects position sizing:
- High volatility → reduce position size (prices moving fast, edge uncertain)
- Low volatility → full Kelly (stable edge, higher confidence)
- Volatility trend → early entry signal (market starting to move)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from loguru import logger

try:
    from statsforecast import StatsForecast
    from statsforecast.models import GARCH, ARCH, AutoARIMA
    HAS_STATSFORECAST = True
except ImportError:
    HAS_STATSFORECAST = False
    logger.warning("statsforecast not installed. pip install statsforecast")


@dataclass
class VolatilityEstimate:
    """Volatility regime analysis for a market's probability history."""
    current_vol: float           # current estimated volatility
    historical_mean_vol: float   # average volatility over window
    vol_ratio: float             # current / historical (>1 = high vol)
    regime: str                  # "low", "normal", "high", "extreme"
    kelly_adjustment: float      # multiply Kelly size by this (0.25-1.0)
    trend_direction: str         # "stable", "rising", "falling"
    n_observations: int


def estimate_volatility(
    price_history: Sequence[float],
    window: int = 20,
) -> VolatilityEstimate:
    """Estimate probability volatility from price history.

    Uses rolling standard deviation of log-returns as a simple volatility
    estimator. Falls back to this when statsforecast GARCH is not available
    or when the history is too short.

    Args:
        price_history: sequence of probability prices (0 to 1) over time
        window: rolling window for volatility estimation
    """
    prices = np.array(price_history, dtype=np.float64)
    prices = np.clip(prices, 0.01, 0.99)
    n = len(prices)

    if n < 5:
        return VolatilityEstimate(
            current_vol=0.0, historical_mean_vol=0.0, vol_ratio=1.0,
            regime="unknown", kelly_adjustment=1.0,
            trend_direction="stable", n_observations=n,
        )

    # Log-odds returns (more appropriate for probability time series)
    log_odds = np.log(prices / (1 - prices))
    returns = np.diff(log_odds)

    if len(returns) < 3:
        return VolatilityEstimate(
            current_vol=0.0, historical_mean_vol=0.0, vol_ratio=1.0,
            regime="unknown", kelly_adjustment=1.0,
            trend_direction="stable", n_observations=n,
        )

    # Rolling volatility
    effective_window = min(window, len(returns))
    current_vol = float(np.std(returns[-effective_window:]))
    historical_vol = float(np.std(returns))

    # Volatility ratio
    if historical_vol > 1e-8:
        vol_ratio = current_vol / historical_vol
    else:
        vol_ratio = 1.0

    # Regime classification
    if vol_ratio < 0.5:
        regime = "low"
        kelly_adj = 1.0
    elif vol_ratio < 1.5:
        regime = "normal"
        kelly_adj = 1.0
    elif vol_ratio < 3.0:
        regime = "high"
        kelly_adj = 0.5
    else:
        regime = "extreme"
        kelly_adj = 0.25

    # Trend direction (is volatility increasing or decreasing?)
    if len(returns) >= 10:
        recent_vol = np.std(returns[-5:])
        older_vol = np.std(returns[-10:-5])
        if recent_vol > older_vol * 1.3:
            trend = "rising"
        elif recent_vol < older_vol * 0.7:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return VolatilityEstimate(
        current_vol=round(current_vol, 6),
        historical_mean_vol=round(historical_vol, 6),
        vol_ratio=round(vol_ratio, 4),
        regime=regime,
        kelly_adjustment=kelly_adj,
        trend_direction=trend,
        n_observations=n,
    )


def garch_forecast(
    price_history: Sequence[float],
    horizon: int = 5,
) -> dict | None:
    """Fit GARCH(1,1) model and forecast volatility.

    Uses statsforecast for fast GARCH estimation. Returns forecasted
    volatility for the next `horizon` periods.

    Returns None if statsforecast is not available or data is insufficient.
    """
    if not HAS_STATSFORECAST:
        return None

    prices = np.array(price_history, dtype=np.float64)
    prices = np.clip(prices, 0.01, 0.99)

    if len(prices) < 30:
        return None

    log_odds = np.log(prices / (1 - prices))
    returns = np.diff(log_odds)

    try:
        import pandas as pd

        df = pd.DataFrame({
            "unique_id": ["market"] * len(returns),
            "ds": pd.date_range("2024-01-01", periods=len(returns), freq="D"),
            "y": returns,
        })

        sf = StatsForecast(
            models=[GARCH(1, 1)],
            freq="D",
            n_jobs=1,
        )
        sf.fit(df)
        forecast = sf.predict(h=horizon)

        return {
            "forecasted_vol": forecast["GARCH"].values.tolist(),
            "horizon": horizon,
            "model": "GARCH(1,1)",
        }

    except Exception as e:
        logger.warning(f"GARCH fitting failed: {e}")
        return None
