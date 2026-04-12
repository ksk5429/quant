"""Tests for edge detection, Kelly sizing, and drawdown monitoring."""
from __future__ import annotations

import pytest
from src.risk.portfolio import (
    EdgeDetector, KellyPositionSizer, DrawdownMonitor, MarketSignal,
)


class TestEdgeDetection:
    def test_detects_edge_above_threshold(self):
        detector = EdgeDetector(min_edge=0.05, tx_cost=0.02)
        signals = [MarketSignal(
            market_id="m1", question="Q", market_price=0.50,
            swarm_probability=0.65, confidence=0.7, spread=0.10,
            disagreement_flag=False,
        )]
        result = detector.detect_edges(signals)
        assert len(result) == 1

    def test_rejects_small_edge(self):
        detector = EdgeDetector(min_edge=0.05, tx_cost=0.02)
        signals = [MarketSignal(
            market_id="m1", question="Q", market_price=0.50,
            swarm_probability=0.55, confidence=0.7, spread=0.10,
            disagreement_flag=False,
        )]
        result = detector.detect_edges(signals)
        assert len(result) == 0

    def test_rejects_low_confidence(self):
        detector = EdgeDetector(min_confidence=0.40)
        signals = [MarketSignal(
            market_id="m1", question="Q", market_price=0.30,
            swarm_probability=0.60, confidence=0.20, spread=0.10,
            disagreement_flag=False,
        )]
        result = detector.detect_edges(signals)
        assert len(result) == 0

    def test_rejects_high_spread(self):
        detector = EdgeDetector(max_spread=0.35)
        signals = [MarketSignal(
            market_id="m1", question="Q", market_price=0.30,
            swarm_probability=0.60, confidence=0.7, spread=0.50,
            disagreement_flag=True,
        )]
        result = detector.detect_edges(signals)
        assert len(result) == 0

    def test_rejects_extreme_price(self):
        detector = EdgeDetector()
        signals = [MarketSignal(
            market_id="m1", question="Q", market_price=0.99,
            swarm_probability=0.80, confidence=0.7, spread=0.10,
            disagreement_flag=False,
        )]
        result = detector.detect_edges(signals)
        assert len(result) == 0


class TestKellySizing:
    def test_quarter_kelly(self):
        sizer = KellyPositionSizer(kelly_fraction=0.25, bankroll_usd=1000)
        signal = MarketSignal(
            market_id="m1", question="Q", market_price=0.40,
            swarm_probability=0.60, confidence=0.7, spread=0.10,
            disagreement_flag=False,
        )
        pos = sizer.size_position(signal)
        assert pos is not None
        assert pos.position_size_usd > 0
        assert pos.side == "YES"

    def test_no_side_when_no_edge(self):
        sizer = KellyPositionSizer(kelly_fraction=0.25, bankroll_usd=1000)
        signal = MarketSignal(
            market_id="m1", question="Q", market_price=0.50,
            swarm_probability=0.50, confidence=0.7, spread=0.10,
            disagreement_flag=False,
        )
        pos = sizer.size_position(signal)
        assert pos is None

    def test_never_exceeds_max_position(self):
        sizer = KellyPositionSizer(
            kelly_fraction=0.25, max_position_pct=0.05, bankroll_usd=1000,
        )
        signal = MarketSignal(
            market_id="m1", question="Q", market_price=0.10,
            swarm_probability=0.90, confidence=0.9, spread=0.05,
            disagreement_flag=False,
        )
        pos = sizer.size_position(signal)
        assert pos is not None
        assert pos.position_size_usd <= 1000 * 0.05 + 0.01

    def test_buy_no_when_overpriced(self):
        sizer = KellyPositionSizer(kelly_fraction=0.25, bankroll_usd=1000)
        signal = MarketSignal(
            market_id="m1", question="Q", market_price=0.70,
            swarm_probability=0.40, confidence=0.7, spread=0.10,
            disagreement_flag=False,
        )
        pos = sizer.size_position(signal)
        assert pos is not None
        assert pos.side == "NO"


class TestDrawdownMonitor:
    def test_no_halt_when_profitable(self):
        dd = DrawdownMonitor(max_drawdown_pct=0.15)
        dd.record_pnl(10.0)
        dd.record_pnl(5.0)
        assert dd.check_halt(1000) == False

    def test_halts_at_threshold(self):
        dd = DrawdownMonitor(max_drawdown_pct=0.15)
        dd.record_pnl(50.0)   # peak at 50
        dd.record_pnl(-100.0)  # now at -50, drawdown = 100 from peak
        dd.record_pnl(-60.0)   # now at -110, drawdown = 160
        assert dd.check_halt(1000) == True
        assert dd.halted == True

    def test_stays_halted(self):
        dd = DrawdownMonitor(max_drawdown_pct=0.15)
        dd.record_pnl(-200.0)
        dd.check_halt(1000)
        assert dd.halted == True
        # Recovery doesn't auto-clear
        dd.record_pnl(500.0)
        assert dd.check_halt(1000) == True
