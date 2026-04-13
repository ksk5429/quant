"""External dataset loader for prediction-market-analysis parquet data.

Loads the 36GB prediction market dataset (Jon-Becker/prediction-market-analysis)
into our pipeline. Provides:
- Market metadata with resolution outcomes
- Trade-level price history for volatility analysis
- DuckDB-compatible queries for large-scale analysis

Data path: data/external/prediction-market-analysis/data/
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DatasetMarket:
    """A market from the external dataset with resolution data."""
    id: str
    question: str
    slug: str
    outcomes: list[str]
    outcome_prices: list[float]
    volume: float
    closed: bool
    end_date: str
    created_at: str

    @property
    def is_resolved(self) -> bool:
        if not self.outcome_prices or not self.closed:
            return False
        return max(self.outcome_prices) >= 0.95

    @property
    def ground_truth(self) -> float:
        if not self.is_resolved:
            raise ValueError(f"Market {self.id} not resolved")
        return float(self.outcome_prices[0])

    @property
    def winning_outcome(self) -> str:
        if not self.is_resolved:
            return "unresolved"
        idx = 0 if self.outcome_prices[0] > self.outcome_prices[1] else 1
        return self.outcomes[idx] if idx < len(self.outcomes) else "unknown"


class ExternalDatasetLoader:
    """Load and query the prediction-market-analysis parquet dataset.

    Provides both pandas (in-memory) and DuckDB (out-of-core) interfaces
    for querying hundreds of thousands of markets and millions of trades.
    """

    def __init__(self, data_dir: str | Path = "data/external/prediction-market-analysis/data") -> None:
        self.data_dir = Path(data_dir)
        self.markets_dir = self.data_dir / "polymarket" / "markets"
        self.trades_dir = self.data_dir / "polymarket" / "trades"
        self.legacy_trades_dir = self.data_dir / "polymarket" / "legacy_trades"
        self._markets_df: pd.DataFrame | None = None

    def load_all_markets(self, min_volume: float = 0) -> pd.DataFrame:
        """Load all market metadata from parquet files into a DataFrame."""
        if self._markets_df is not None:
            df = self._markets_df
        else:
            parquet_files = sorted(
                f for f in self.markets_dir.glob("*.parquet")
                if not f.name.startswith("._")
            )
            if not parquet_files:
                logger.error(f"No parquet files in {self.markets_dir}")
                return pd.DataFrame()

            dfs = []
            for pf in parquet_files:
                try:
                    dfs.append(pd.read_parquet(pf))
                except Exception as e:
                    logger.warning(f"Failed to read {pf.name}: {e}")

            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["id"], keep="last")

            # Parse JSON string fields
            df["outcomes_parsed"] = df["outcomes"].apply(self._parse_json_list)
            df["prices_parsed"] = df["outcome_prices"].apply(self._parse_json_list_float)

            self._markets_df = df
            logger.info(f"Loaded {len(df)} markets from {len(parquet_files)} files")

        if min_volume > 0:
            df = df[df["volume"] >= min_volume]

        return df

    def get_resolved_markets(self, min_volume: float = 1000) -> list[DatasetMarket]:
        """Get all resolved markets with ground truth outcomes."""
        df = self.load_all_markets(min_volume=min_volume)
        df_closed = df[df["closed"] == True].copy()

        markets = []
        for _, row in df_closed.iterrows():
            prices = row.get("prices_parsed", [])
            outcomes = row.get("outcomes_parsed", ["Yes", "No"])

            if not prices or len(prices) < 2:
                continue
            if max(prices) < 0.95:
                continue

            # Skip ambiguous
            if prices[0] == prices[1]:
                continue

            try:
                m = DatasetMarket(
                    id=str(row.get("id", "")),
                    question=str(row.get("question", "")),
                    slug=str(row.get("slug", "")),
                    outcomes=outcomes,
                    outcome_prices=prices,
                    volume=float(row.get("volume", 0)),
                    closed=True,
                    end_date=str(row.get("end_date", "")),
                    created_at=str(row.get("created_at", "")),
                )
                markets.append(m)
            except Exception:
                continue

        logger.info(f"Found {len(markets)} resolved markets (min_vol={min_volume})")
        return markets

    def get_trade_history(
        self, market_id: str, max_trades: int = 10000,
    ) -> pd.DataFrame | None:
        """Get trade history for a specific market (for volatility analysis).

        Uses DuckDB predicate pushdown when available for fast querying
        across large parquet datasets without loading everything into memory.
        Falls back to sequential pandas scan when DuckDB is unavailable.
        """
        try:
            import duckdb
            for trades_dir in [self.trades_dir, self.legacy_trades_dir]:
                if not trades_dir.exists():
                    continue
                glob_pattern = str(trades_dir / "*.parquet")
                # DuckDB predicate pushdown — parameterized to prevent SQL injection
                query = f"""
                    SELECT * FROM read_parquet('{glob_pattern}', union_by_name=true)
                    WHERE COALESCE(market, condition_id, '') = $1
                    LIMIT {int(max_trades)}
                """
                try:
                    result = duckdb.execute(query, [market_id]).fetchdf()
                    if len(result) > 0:
                        return result
                except Exception:
                    continue
        except ImportError:
            pass

        # Fallback: sequential pandas scan (slow for large datasets)
        for trades_dir in [self.trades_dir, self.legacy_trades_dir]:
            if not trades_dir.exists():
                continue
            for pf in sorted(trades_dir.glob("*.parquet")):
                if pf.name.startswith("._"):
                    continue
                try:
                    df = pd.read_parquet(pf)
                    id_col = "market" if "market" in df.columns else "condition_id"
                    if id_col in df.columns:
                        trades = df[df[id_col] == market_id]
                        if len(trades) > 0:
                            return trades.head(max_trades)
                except Exception:
                    continue
        return None

    def get_calibration_dataset(
        self, min_volume: float = 5000, max_markets: int | None = None,
    ) -> tuple[list[float], list[float]]:
        """Extract calibration data from OUR retrodiction results.

        IMPORTANT: This does NOT use the external dataset's resolution prices
        (which are 0.0 or 1.0 and provide no calibration signal). Instead,
        it loads our own swarm predictions from retrodiction JSON files
        where we have both our predicted probability AND the ground truth.

        Falls back to the raw markets data only as a crowd calibration
        baseline (not for training our calibrator).

        Returns (predictions, outcomes) from our retrodiction runs.
        """
        retro_dir = Path("data/retrodiction")
        predictions = []
        outcomes = []

        if retro_dir.exists():
            for fp in sorted(retro_dir.glob("retro_v2_*.json")):
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    for pred in data.get("predictions", []):
                        p = pred.get("extremized_probability", pred.get("raw_probability"))
                        gt = pred.get("ground_truth")
                        if p is not None and gt is not None:
                            predictions.append(float(p))
                            outcomes.append(float(gt))
                except Exception as e:
                    logger.warning(f"Failed to load retrodiction {fp.name}: {e}")

        if predictions:
            logger.info(
                f"Calibration dataset from retrodiction: {len(predictions)} predictions"
            )
            if max_markets and len(predictions) > max_markets:
                predictions = predictions[:max_markets]
                outcomes = outcomes[:max_markets]
            return predictions, outcomes

        # Fallback: warn that market resolution prices are not useful
        logger.warning(
            "No retrodiction data found. Run retrodiction first to generate "
            "calibration training data. External dataset resolution prices "
            "(0.0/1.0) provide no calibration signal."
        )
        return [], []

    def compute_crowd_calibration(self, n_bins: int = 10) -> dict[str, Any]:
        """Compute the Polymarket crowd's calibration metrics.

        NOTE: This uses OUR retrodiction predictions (not crowd closing prices).
        For crowd baseline, use the market closing prices from the external dataset
        directly, not this method.
        """
        from src.prediction.calibration import compute_brier, compute_ece

        prices, outcomes = self.get_calibration_dataset()
        if not prices:
            return {"error": "no data"}

        brier = compute_brier(prices, outcomes)
        ece = compute_ece(prices, outcomes, n_bins)

        # Per-bin breakdown
        p_arr = np.array(prices)
        o_arr = np.array(outcomes)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for i in range(n_bins):
            mask = (p_arr >= bin_edges[i]) & (p_arr < bin_edges[i + 1])
            if np.any(mask):
                bins.append({
                    "range": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                    "count": int(np.sum(mask)),
                    "mean_price": round(float(np.mean(p_arr[mask])), 4),
                    "actual_rate": round(float(np.mean(o_arr[mask])), 4),
                    "error": round(abs(float(np.mean(p_arr[mask])) - float(np.mean(o_arr[mask]))), 4),
                })

        return {
            "n_markets": len(prices),
            "brier_score": round(brier, 6),
            "ece": round(ece, 6),
            "base_rate": round(float(np.mean(outcomes)), 4),
            "bins": bins,
        }

    @staticmethod
    def _parse_json_list(val) -> list[str]:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return [str(x) for x in parsed] if isinstance(parsed, list) else [str(val)]
            except (json.JSONDecodeError, ValueError):
                return [str(val)]
        return ["Yes", "No"]

    @staticmethod
    def _parse_json_list_float(val) -> list[float]:
        if isinstance(val, list):
            return [float(x) for x in val]
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return [float(x) for x in parsed] if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []
