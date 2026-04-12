"""Tests for Kelly Criterion position sizing."""

import pytest

from src.risk.kelly import KellyCriterion, BetSide, TradeSignal


class TestKellySignals:
    def test_positive_edge_yes_bet(self):
        kelly = KellyCriterion(bankroll=1000, min_edge=0.05)
        signal = kelly.compute_signal(
            market_id="m1",
            market_question="Test?",
            our_probability=0.75,
            market_price=0.60,
            confidence=0.8,
        )

        assert signal.side == BetSide.YES
        assert signal.edge > 0
        assert signal.position_size_usd > 0
        assert signal.is_actionable

    def test_positive_edge_no_bet(self):
        kelly = KellyCriterion(bankroll=1000, min_edge=0.05)
        signal = kelly.compute_signal(
            market_id="m1",
            market_question="Test?",
            our_probability=0.30,
            market_price=0.60,
            confidence=0.8,
        )

        assert signal.side == BetSide.NO
        assert signal.edge > 0
        assert signal.is_actionable

    def test_no_edge_abstains(self):
        kelly = KellyCriterion(bankroll=1000, min_edge=0.05)
        signal = kelly.compute_signal(
            market_id="m1",
            market_question="Test?",
            our_probability=0.61,
            market_price=0.60,
            confidence=0.8,
        )

        assert signal.side == BetSide.ABSTAIN
        assert not signal.is_actionable
        assert signal.position_size_usd == 0

    def test_quarter_kelly_reduces_position(self):
        # Use high max_position_pct to avoid cap masking the Kelly difference
        full = KellyCriterion(bankroll=1000, kelly_fraction=1.0, max_position_pct=0.50, min_edge=0.05)
        quarter = KellyCriterion(bankroll=1000, kelly_fraction=0.25, max_position_pct=0.50, min_edge=0.05)

        s_full = full.compute_signal("m1", "T?", 0.80, 0.60, 0.9)
        s_quarter = quarter.compute_signal("m1", "T?", 0.80, 0.60, 0.9)

        assert s_quarter.position_size_usd < s_full.position_size_usd

    def test_max_position_cap(self):
        kelly = KellyCriterion(
            bankroll=10000, kelly_fraction=1.0,
            max_position_pct=0.05, min_edge=0.01,
        )
        signal = kelly.compute_signal("m1", "T?", 0.95, 0.50, 1.0)

        # Position should be capped at 5% of bankroll = $500
        assert signal.position_size_usd <= 500.0

    def test_confidence_scales_position(self):
        kelly = KellyCriterion(bankroll=1000, min_edge=0.05)
        s_high = kelly.compute_signal("m1", "T?", 0.80, 0.60, 0.9)

        kelly2 = KellyCriterion(bankroll=1000, min_edge=0.05)
        s_low = kelly2.compute_signal("m1", "T?", 0.80, 0.60, 0.3)

        assert s_high.position_size_usd > s_low.position_size_usd


class TestKellyRiskControls:
    def test_drawdown_stops_trading(self):
        kelly = KellyCriterion(bankroll=1000, max_drawdown_pct=0.15, min_edge=0.05)

        # Simulate 20% loss
        kelly.record_pnl(-200)  # bankroll = 800, peak = 1000, drawdown = 20%

        signal = kelly.compute_signal("m1", "T?", 0.80, 0.60, 0.9)
        assert signal.side == BetSide.ABSTAIN

    def test_pnl_tracking(self):
        kelly = KellyCriterion(bankroll=1000)
        kelly.record_pnl(100)
        assert kelly.bankroll == 1100

        kelly.record_pnl(-50)
        assert kelly.bankroll == 1050

    def test_total_return(self):
        kelly = KellyCriterion(bankroll=1000)
        kelly.record_pnl(200)
        assert kelly.total_return == pytest.approx(0.2)

    def test_current_drawdown(self):
        kelly = KellyCriterion(bankroll=1000)
        kelly.record_pnl(200)   # peak = 1200
        kelly.record_pnl(-300)  # bankroll = 900
        assert kelly.current_drawdown == pytest.approx(0.25)
