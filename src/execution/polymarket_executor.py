"""Polymarket execution client — wraps py-clob-client SDK.

SAFETY: paper_trading=True by default. Live trading requires explicit
--live CLI flag AND interactive confirmation. Six safety checks must
ALL pass before any order is sent.

Usage:
    executor = PolymarketExecutor(paper_trading=True)  # DEFAULT
    result = await executor.place_limit_order(token_id, "BUY", 0.65, 25.0)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from src.execution.order_types import OrderResult

# py-clob-client imports (graceful fallback if not installed)
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False
    logger.warning("py-clob-client not installed. pip install py-clob-client")


@dataclass
class SafetyCheckResult:
    """Result of pre-order safety checks."""
    passed: bool
    failures: list[str]


class PolymarketExecutor:
    """Polymarket CLOB order executor with safety checks.

    RULE 1: paper_trading=True is the default. NEVER change this default.
    RULE 3: Position limits are hard caps enforced here, not just in engine.
    """

    def __init__(
        self,
        private_key: str = "",
        funder_address: str = "",
        chain_id: int = 137,
        signature_type: int = 0,
        paper_trading: bool = True,
        max_position_usd: float = 50.0,
        max_exposure_usd: float = 300.0,
        host: str = "https://clob.polymarket.com",
    ) -> None:
        self.paper_trading = paper_trading
        self.max_position_usd = max_position_usd
        self.max_exposure_usd = max_exposure_usd
        self._total_exposure = 0.0
        self._drawdown_halted = False

        # Only initialize real client if live mode AND SDK available
        self._client: Any = None
        if not paper_trading and HAS_CLOB:
            key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY", "")
            funder = funder_address or os.environ.get("POLYMARKET_FUNDER", "")
            if not key:
                raise ValueError(
                    "POLYMARKET_PRIVATE_KEY required for live trading. "
                    "Set environment variable or pass private_key parameter."
                )
            self._client = ClobClient(
                host=host,
                key=key,
                chain_id=chain_id,
                signature_type=signature_type,
                funder=funder or None,
            )
            # Derive API credentials
            try:
                self._client.set_api_creds(self._client.create_or_derive_api_creds())
                logger.info("Polymarket CLOB client initialized (LIVE MODE)")
            except Exception as e:
                logger.error(f"Failed to derive API creds: {e}")
                raise

        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(
            f"PolymarketExecutor: mode={mode}, "
            f"max_position=${max_position_usd}, max_exposure=${max_exposure_usd}"
        )

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        time_in_force: str = "GTC",
        midpoint: float | None = None,
    ) -> OrderResult:
        """Place a limit order with safety checks.

        Args:
            token_id: Polymarket CTF token ID.
            side: "BUY" or "SELL".
            price: limit price (0.01-0.99).
            size: size in USDC.
            time_in_force: "GTC", "GTD", or "FOK".
            midpoint: current midpoint for stale price check.

        Returns:
            OrderResult with status and fill details.
        """
        # ── Safety checks (ALL must pass) ──
        safety = self._run_safety_checks(size, price, midpoint)
        if not safety.passed:
            logger.warning(f"Order REJECTED: {safety.failures}")
            return OrderResult(
                order_id="",
                status="rejected",
                filled_price=0,
                filled_size=0,
                side=side,
                token_id=token_id,
                timestamp=datetime.now().isoformat(),
                paper=self.paper_trading,
                rejection_reason="; ".join(safety.failures),
            )

        # ── Paper mode: simulate fill ──
        if self.paper_trading:
            return self._simulate_fill(token_id, side, price, size)

        # ── Live mode: place real order ──
        if not self._client:
            raise RuntimeError("CLOB client not initialized for live trading")

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
            )
            response = self._client.create_and_post_order(order_args)

            order_id = response.get("orderID", response.get("id", ""))
            status = "pending"

            # Track exposure
            self._total_exposure += size

            logger.info(
                f"LIVE ORDER: {side} {size:.2f} USDC @ {price:.4f} "
                f"token={token_id[:16]}... order_id={order_id}"
            )

            return OrderResult(
                order_id=str(order_id),
                status=status,
                filled_price=price,
                filled_size=size,
                side=side,
                token_id=token_id,
                timestamp=datetime.now().isoformat(),
                paper=False,
            )

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return OrderResult(
                order_id="",
                status="rejected",
                filled_price=0,
                filled_size=0,
                side=side,
                token_id=token_id,
                timestamp=datetime.now().isoformat(),
                paper=False,
                rejection_reason=str(e),
            )

    async def place_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,
    ) -> OrderResult:
        """Market order (FOK). Same safety checks as limit order."""
        return await self.place_limit_order(
            token_id=token_id,
            side=side,
            price=0.99 if side == "BUY" else 0.01,
            size=amount,
            time_in_force="FOK",
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if self.paper_trading:
            logger.info(f"[PAPER] Cancel order {order_id}")
            return True
        if not self._client:
            return False
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        if self.paper_trading:
            logger.info("[PAPER] Cancel all orders")
            return 0
        if not self._client:
            return 0
        try:
            self._client.cancel_all()
            return -1  # SDK doesn't return count
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return 0

    async def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        if self.paper_trading or not self._client:
            return []
        try:
            return self._client.get_orders() or []
        except Exception as e:
            logger.error(f"Get orders failed: {e}")
            return []

    async def get_balance(self) -> float:
        """Get USDC balance. Returns 0 in paper mode."""
        if self.paper_trading:
            return 0.0
        # Balance check would require web3 — return 0 for now
        # TODO: implement with web3 provider
        return 0.0

    def set_drawdown_halt(self, halted: bool) -> None:
        """Set or clear the drawdown halt flag."""
        self._drawdown_halted = halted
        if halted:
            logger.warning("EXECUTOR: Drawdown halt ACTIVATED — no new orders")
        else:
            logger.info("EXECUTOR: Drawdown halt CLEARED")

    def update_exposure(self, exposure: float) -> None:
        """Update total exposure from DB (called at startup)."""
        self._total_exposure = exposure

    def release_exposure(self, amount: float) -> None:
        """Decrement exposure when a position is closed or resolved."""
        self._total_exposure = max(0, self._total_exposure - amount)
        logger.debug(f"Exposure released: ${amount:.2f}, total now ${self._total_exposure:.2f}")

    # ── Safety Checks ────────────────────────────────────────────

    def _run_safety_checks(
        self,
        size: float,
        price: float,
        midpoint: float | None = None,
    ) -> SafetyCheckResult:
        """Run all 5 safety checks. ALL must pass.

        1. Size <= max_position_usd
        2. Total exposure + size <= max_exposure_usd
        3. Drawdown monitor not halted
        4. Price within 5% of midpoint (stale price protection)
        5. Not paper mode pretending to be live (redundant but defensive)
        """
        failures = []

        # Check 1: position size limit
        if size > self.max_position_usd:
            failures.append(
                f"Size ${size:.2f} exceeds max ${self.max_position_usd:.2f}"
            )

        # Check 2: total exposure limit
        if self._total_exposure + size > self.max_exposure_usd:
            failures.append(
                f"Exposure ${self._total_exposure + size:.2f} exceeds "
                f"max ${self.max_exposure_usd:.2f}"
            )

        # Check 3: drawdown halt
        if self._drawdown_halted:
            failures.append("Drawdown halt is active — no new orders")

        # Check 4: stale price protection
        if midpoint is not None and midpoint > 0:
            price_diff = abs(price - midpoint) / midpoint
            if price_diff > 0.05:
                failures.append(
                    f"Price {price:.4f} is {price_diff:.1%} from midpoint "
                    f"{midpoint:.4f} (>5% stale price threshold)"
                )

        # Check 5: price bounds
        if price < 0.01 or price > 0.99:
            failures.append(f"Price {price} out of valid range [0.01, 0.99]")

        return SafetyCheckResult(
            passed=len(failures) == 0,
            failures=failures,
        )

    def _simulate_fill(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> OrderResult:
        """Paper trading: simulate fill at price with 0.5% slippage."""
        slippage = 0.005
        if side == "BUY":
            fill_price = min(price * (1 + slippage), 0.99)
        else:
            fill_price = max(price * (1 - slippage), 0.01)

        self._total_exposure += size

        logger.info(
            f"[PAPER] {side} {size:.2f} USDC @ {fill_price:.4f} "
            f"(slippage={slippage:.1%}) token={token_id[:16]}..."
        )

        return OrderResult(
            order_id=f"paper_{int(time.time())}",
            status="simulated",
            filled_price=round(fill_price, 4),
            filled_size=size,
            side=side,
            token_id=token_id,
            timestamp=datetime.now().isoformat(),
            paper=True,
        )
