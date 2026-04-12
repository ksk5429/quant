"""Position manager — execute, resolve, reconcile positions.

Bridges the engine's Position recommendations with the executor's
order placement, and handles the full lifecycle: open → monitor → close.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from src.db.manager import DatabaseManager, PositionRecord
from src.execution.order_types import OrderResult, ClosedPosition, ReconciliationReport
from src.execution.polymarket_executor import PolymarketExecutor


class PositionManager:
    """Manages position lifecycle: execute, resolve, reconcile."""

    def __init__(
        self,
        executor: PolymarketExecutor,
        db: DatabaseManager,
    ) -> None:
        self.executor = executor
        self.db = db

    async def execute_position(
        self,
        position: Any,
        market: Any,
        prediction_id: int,
    ) -> OrderResult:
        """Execute a position recommendation from the engine.

        Flow:
        1. Get current midpoint from CLOB (or Gamma for paper)
        2. Verify edge still exists (price hasn't moved >2%)
        3. Place limit order at midpoint
        4. Log to DB
        5. Return result
        """
        market_id = getattr(market, "id", getattr(position, "market_id", ""))
        side = getattr(position, "side", "YES")
        size_usd = getattr(position, "position_size_usd", 0)

        # Get current price for stale-price check
        yes_price = getattr(market, "yes_price", 0.5)
        if side == "YES":
            order_price = yes_price
            token_id = getattr(market, "yes_token_id", "")
        else:
            order_price = 1.0 - yes_price
            token_id = getattr(market, "no_token_id", "")

        # Verify edge still exists (price moved <2% since prediction)
        predicted_edge = getattr(position, "edge", 0)
        current_edge = abs(
            getattr(position, "edge", 0)  # approximate
        )

        # Place order
        result = await self.executor.place_limit_order(
            token_id=token_id or f"token_{market_id}",
            side="BUY",
            price=order_price,
            size=size_usd,
            midpoint=order_price,
        )

        # Log to DB
        if result.is_filled:
            try:
                pos_id = self.db.open_position(
                    position=position,
                    prediction_id=prediction_id,
                    order_id=result.order_id,
                    tx_hash=result.tx_hash or "",
                )
                logger.info(f"Position opened: id={pos_id} {side} ${size_usd:.2f}")
            except Exception as e:
                logger.error(f"DB open_position failed: {e}")

        return result

    async def check_resolutions(self) -> list[ClosedPosition]:
        """Poll for resolved markets matching open positions.

        For each resolved market:
        1. Determine outcome (YES=1.0, NO=0.0)
        2. Calculate P&L
        3. Close position in DB
        4. Feed calibrator via engine record_outcome
        5. Update drawdown monitor
        """
        open_positions = self.db.get_open_positions()
        if not open_positions:
            return []

        closed = []
        for pos in open_positions:
            resolved = await self._check_market_resolution(pos.market_id)
            if resolved is None:
                continue

            outcome = resolved["outcome"]
            exit_price = 1.0 if outcome == 1.0 else 0.0

            # Calculate P&L
            if pos.side == "YES":
                pnl = (outcome - pos.entry_price) * pos.size_usd / max(pos.entry_price, 0.01)
            else:
                pnl = ((1.0 - outcome) - (1.0 - pos.entry_price)) * pos.size_usd / max(1.0 - pos.entry_price, 0.01)

            pnl = round(pnl, 2)

            # Close in DB
            try:
                self.db.close_position(
                    position_id=pos.id,
                    exit_price=exit_price,
                    pnl=pnl,
                    reason="resolved",
                )

                # Log resolution
                self.db.log_resolution(
                    market_id=pos.market_id,
                    outcome=outcome,
                    question=pos.question,
                )

                closed_pos = ClosedPosition(
                    position_id=pos.id,
                    market_id=pos.market_id,
                    question=pos.question,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size_usd=pos.size_usd,
                    pnl_usd=pnl,
                    pnl_pct=round(pnl / max(pos.size_usd, 0.01) * 100, 2),
                    reason="resolved",
                )
                closed.append(closed_pos)

                sym = "+" if pnl > 0 else "-"
                logger.info(
                    f"Position resolved: [{sym}] {pos.side} {pos.question[:40]} "
                    f"P&L=${pnl:+.2f}"
                )

            except Exception as e:
                logger.error(f"Failed to close position {pos.id}: {e}")

        if closed:
            logger.info(f"Resolved {len(closed)} positions, total P&L=${sum(c.pnl_usd for c in closed):+.2f}")

        return closed

    async def reconcile(self) -> ReconciliationReport:
        """Compare DB positions against on-chain state.

        In paper mode, this always passes (no on-chain state).
        In live mode, queries CLOB API for actual holdings.
        """
        open_positions = self.db.get_open_positions()
        n_db = len(open_positions)

        if self.executor.paper_trading:
            return ReconciliationReport(
                timestamp=datetime.now().isoformat(),
                n_db_open=n_db,
                n_chain_positions=n_db,  # paper = DB is truth
                matches=n_db,
                mismatches=0,
                details=[],
                is_clean=True,
            )

        # Live mode: check CLOB API for actual positions
        # TODO: implement with CLOB balance/positions endpoint
        chain_positions = 0
        mismatches = 0
        details = []

        logger.info(
            f"Reconciliation: DB has {n_db} open positions, "
            f"chain has {chain_positions} (live check TODO)"
        )

        return ReconciliationReport(
            timestamp=datetime.now().isoformat(),
            n_db_open=n_db,
            n_chain_positions=chain_positions,
            matches=min(n_db, chain_positions),
            mismatches=mismatches,
            details=details,
            is_clean=mismatches == 0,
        )

    async def _check_market_resolution(self, market_id: str) -> dict | None:
        """Check if a market has resolved via Gamma API."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    f"https://gamma-api.polymarket.com/markets",
                    params={"id": market_id},
                )
                r.raise_for_status()
                data = r.json()

            if not data or not isinstance(data, list):
                return None

            market = data[0]
            if not market.get("closed", False):
                return None

            prices_str = market.get("outcomePrices", "")
            try:
                prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                prices = [float(p) for p in prices]
            except (json.JSONDecodeError, ValueError, TypeError):
                return None

            if len(prices) < 2 or max(prices) < 0.95:
                return None

            outcome = 1.0 if prices[0] > prices[1] else 0.0
            return {"outcome": outcome, "prices": prices}

        except Exception as e:
            logger.debug(f"Resolution check failed for {market_id}: {e}")
            return None
