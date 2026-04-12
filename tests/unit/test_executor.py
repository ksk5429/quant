"""Tests for the execution layer — safety checks, paper fills, limits."""
from __future__ import annotations

import pytest
import asyncio
from src.execution.polymarket_executor import PolymarketExecutor
from src.execution.order_types import OrderResult


@pytest.fixture
def paper_executor():
    return PolymarketExecutor(
        paper_trading=True, max_position_usd=50, max_exposure_usd=300,
    )


class TestPaperMode:
    def test_simulates_fill(self, paper_executor):
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 0.65, 25.0, midpoint=0.65)
        )
        assert result.status == "simulated"
        assert result.paper == True
        assert result.filled_size == 25.0
        assert result.filled_price > 0

    def test_slippage_applied(self, paper_executor):
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 0.50, 10.0, midpoint=0.50)
        )
        # BUY slippage increases price
        assert result.filled_price > 0.50


class TestSafetyChecks:
    def test_rejects_over_max_position(self, paper_executor):
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 0.50, 100.0, midpoint=0.50)
        )
        assert result.status == "rejected"
        assert "exceeds max" in result.rejection_reason

    def test_rejects_over_max_exposure(self, paper_executor):
        # Fill up exposure
        for _ in range(6):
            asyncio.get_event_loop().run_until_complete(
                paper_executor.place_limit_order("tok_1", "BUY", 0.50, 50.0, midpoint=0.50)
            )
        # 6 * 50 = 300 = max. Next should reject.
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 0.50, 10.0, midpoint=0.50)
        )
        assert result.status == "rejected"
        assert "Exposure" in result.rejection_reason

    def test_rejects_drawdown_halt(self, paper_executor):
        paper_executor.set_drawdown_halt(True)
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 0.50, 10.0, midpoint=0.50)
        )
        assert result.status == "rejected"
        assert "Drawdown" in result.rejection_reason

    def test_rejects_stale_price(self, paper_executor):
        # Price 0.50 but midpoint 0.80 → 60% away
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 0.50, 10.0, midpoint=0.80)
        )
        assert result.status == "rejected"
        assert "stale" in result.rejection_reason.lower()

    def test_rejects_invalid_price(self, paper_executor):
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 1.50, 10.0)
        )
        assert result.status == "rejected"

    def test_accepts_valid_order(self, paper_executor):
        result = asyncio.get_event_loop().run_until_complete(
            paper_executor.place_limit_order("tok_1", "BUY", 0.50, 25.0, midpoint=0.50)
        )
        assert result.status == "simulated"


class TestLiveMode:
    def test_requires_private_key(self):
        with pytest.raises(ValueError, match="POLYMARKET_PRIVATE_KEY"):
            PolymarketExecutor(paper_trading=False, private_key="")
