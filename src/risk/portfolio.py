"""Portfolio optimizer — converts calibrated probabilities into positions.

The pipeline that turns prediction accuracy into profit:
  Swarm probability → Calibration → Edge detection → Kelly sizing → Portfolio

Key principles (from the literature review):
1. Only bet where you have edge (edge = |calibrated_prob - market_price|)
2. Size positions by Kelly criterion (quarter-Kelly for safety)
3. Never bet on high-disagreement markets (spread > 0.30)
4. Diversify across uncorrelated markets
5. Circuit breakers: stop after drawdown exceeds threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class MarketSignal:
    """A single market with swarm prediction and market price."""
    market_id: str
    question: str
    market_price: float        # current Polymarket YES price
    swarm_probability: float   # calibrated swarm estimate
    confidence: float          # effective confidence (after disagreement penalty)
    spread: float              # Fish disagreement spread
    disagreement_flag: bool
    volume_usd: float = 0.0
    category: str = ""


@dataclass
class Position:
    """A recommended position in a prediction market."""
    market_id: str
    question: str
    side: str           # "YES" or "NO"
    edge: float         # absolute edge (our prob - market price)
    kelly_fraction: float  # optimal fraction of bankroll
    position_size_usd: float
    expected_value: float  # expected profit per dollar
    confidence: float
    reason: str = ""


@dataclass
class Portfolio:
    """Complete portfolio of positions with risk metrics."""
    positions: list[Position]
    total_exposure_usd: float
    total_expected_value: float
    n_markets: int
    n_skipped: int          # markets with no edge or too risky
    bankroll_usd: float
    exposure_pct: float     # total_exposure / bankroll
    sharpe_estimate: float  # estimated Sharpe from edge distribution


class EdgeDetector:
    """Identifies profitable edges between swarm predictions and market prices.

    An edge exists when the calibrated swarm probability differs from the
    market price by more than the minimum threshold (accounting for
    transaction costs and spread).
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        min_confidence: float = 0.40,
        max_spread: float = 0.35,
        tx_cost: float = 0.02,  # ~2% round-trip (Polymarket spread + fees)
    ) -> None:
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.max_spread = max_spread
        self.tx_cost = tx_cost

    def detect_edges(self, signals: list[MarketSignal]) -> list[MarketSignal]:
        """Filter signals to only those with tradeable edges.

        A market has an edge when:
        1. |swarm_prob - market_price| > min_edge + tx_cost
        2. confidence > min_confidence
        3. Fish spread < max_spread (low disagreement)
        4. Market has sufficient volume (implicit in signal selection)
        """
        tradeable = []
        skipped_reasons: dict[str, int] = {
            "no_edge": 0, "low_confidence": 0,
            "high_disagreement": 0, "price_extreme": 0,
        }

        for signal in signals:
            edge = abs(signal.swarm_probability - signal.market_price)

            # Skip markets with prices near 0 or 1 (already resolved or nearly so)
            if signal.market_price < 0.03 or signal.market_price > 0.97:
                skipped_reasons["price_extreme"] += 1
                continue

            # Skip if edge doesn't cover transaction costs
            if edge < self.min_edge + self.tx_cost:
                skipped_reasons["no_edge"] += 1
                continue

            # Skip if swarm confidence is too low
            if signal.confidence < self.min_confidence:
                skipped_reasons["low_confidence"] += 1
                continue

            # Skip if Fish disagree too much
            if signal.disagreement_flag or signal.spread > self.max_spread:
                skipped_reasons["high_disagreement"] += 1
                continue

            tradeable.append(signal)

        logger.info(
            f"Edge detection: {len(tradeable)}/{len(signals)} tradeable. "
            f"Skipped: {skipped_reasons}"
        )
        return tradeable


class KellyPositionSizer:
    """Converts edges into position sizes using fractional Kelly criterion.

    Full Kelly maximizes log-wealth growth but has ~25% drawdowns.
    Quarter-Kelly (fraction=0.25) reduces volatility by 4x while keeping
    75% of the growth rate. This is the standard for prediction market
    practitioners (PolySwarm, Thorp).
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
        max_total_exposure_pct: float = 0.30,
        bankroll_usd: float = 1000.0,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure_pct
        self.bankroll = bankroll_usd

    def size_position(self, signal: MarketSignal) -> Position | None:
        """Compute Kelly-optimal position size for a single market.

        For binary prediction markets:
        - If swarm_prob > market_price → BUY YES (pay market_price, win 1.0)
        - If swarm_prob < market_price → BUY NO (pay 1-market_price, win 1.0)

        Kelly formula for binary bets:
            f* = (p * b - q) / b
        where:
            p = estimated probability of winning
            q = 1 - p
            b = odds (payout / cost - 1)
        """
        p_est = signal.swarm_probability
        p_mkt = signal.market_price

        if p_est > p_mkt:
            # Buy YES: cost = p_mkt, win = 1.0
            side = "YES"
            p_win = p_est
            cost = p_mkt
        else:
            # Buy NO: cost = 1 - p_mkt, win = 1.0
            side = "NO"
            p_win = 1.0 - p_est
            cost = 1.0 - p_mkt

        if cost <= 0.01 or cost >= 0.99:
            return None

        # Odds: how much you win per dollar risked
        b = (1.0 - cost) / cost  # payout ratio
        q_lose = 1.0 - p_win

        # Kelly fraction
        kelly_raw = (p_win * b - q_lose) / b
        if kelly_raw <= 0:
            return None  # negative Kelly = no edge

        # Apply fractional Kelly and cap.
        # Note: we do NOT multiply by LLM confidence here. Kelly already
        # accounts for edge size (which reflects confidence implicitly).
        # Multiplying by uncalibrated LLM confidence double-deflates sizing.
        # Confidence filtering happens upstream in EdgeDetector.
        kelly_adj = kelly_raw * self.kelly_fraction

        # Cap at max position size
        kelly_adj = min(kelly_adj, self.max_position_pct)

        position_usd = kelly_adj * self.bankroll
        edge = abs(p_est - p_mkt)

        # Expected value per dollar
        ev_per_dollar = p_win * (1.0 / cost - 1.0) - q_lose

        return Position(
            market_id=signal.market_id,
            question=signal.question,
            side=side,
            edge=round(edge, 4),
            kelly_fraction=round(kelly_adj, 6),
            position_size_usd=round(position_usd, 2),
            expected_value=round(ev_per_dollar, 4),
            confidence=signal.confidence,
            reason=f"P_swarm={p_est:.3f} vs P_mkt={p_mkt:.3f}, edge={edge:.3f}",
        )

    def build_portfolio(self, signals: list[MarketSignal]) -> Portfolio:
        """Build a complete portfolio from a set of market signals.

        Steps:
        1. Size each position independently
        2. Sort by edge (highest first)
        3. Accumulate until total exposure hits limit
        4. Compute portfolio-level risk metrics
        """
        positions = []
        for signal in signals:
            pos = self.size_position(signal)
            if pos and pos.position_size_usd >= 1.0:  # minimum $1
                positions.append(pos)

        # Sort by edge (descending) — best opportunities first
        positions.sort(key=lambda p: -p.edge)

        # Enforce total exposure limit
        accepted = []
        total_exposure = 0.0
        max_exposure_usd = self.bankroll * self.max_total_exposure

        for pos in positions:
            if total_exposure + pos.position_size_usd > max_exposure_usd:
                # Scale down to fit remaining budget
                remaining = max_exposure_usd - total_exposure
                if remaining >= 1.0:
                    scale = remaining / pos.position_size_usd
                    pos.position_size_usd = round(remaining, 2)
                    pos.kelly_fraction = round(pos.kelly_fraction * scale, 6)
                    accepted.append(pos)
                    total_exposure += pos.position_size_usd
                break
            accepted.append(pos)
            total_exposure += pos.position_size_usd

        # Estimate Sharpe from edge distribution
        if accepted:
            edges = np.array([p.edge for p in accepted])
            # Rough Sharpe ≈ mean_edge / std_edge * sqrt(n_trades)
            sharpe_est = float(
                np.mean(edges) / (np.std(edges) + 1e-6) * math.sqrt(len(accepted))
            )
        else:
            sharpe_est = 0.0

        total_ev = sum(p.expected_value * p.position_size_usd for p in accepted)
        n_skipped = len(signals) - len(accepted)

        portfolio = Portfolio(
            positions=accepted,
            total_exposure_usd=round(total_exposure, 2),
            total_expected_value=round(total_ev, 2),
            n_markets=len(accepted),
            n_skipped=n_skipped,
            bankroll_usd=self.bankroll,
            exposure_pct=round(total_exposure / self.bankroll, 4),
            sharpe_estimate=round(sharpe_est, 2),
        )

        logger.info(
            f"Portfolio: {len(accepted)} positions, "
            f"${total_exposure:.0f} exposure ({portfolio.exposure_pct:.0%}), "
            f"EV=${total_ev:.2f}, est. Sharpe={sharpe_est:.2f}"
        )
        return portfolio

    def print_portfolio(self, portfolio: Portfolio) -> None:
        """Print a human-readable portfolio report."""
        print(f"\n{'='*72}")
        print(f"MIROFISH PORTFOLIO")
        print(f"{'='*72}")
        print(f"Bankroll:    ${portfolio.bankroll_usd:,.0f}")
        print(f"Exposure:    ${portfolio.total_exposure_usd:,.2f} ({portfolio.exposure_pct:.0%})")
        print(f"Positions:   {portfolio.n_markets}")
        print(f"Skipped:     {portfolio.n_skipped}")
        print(f"Expected EV: ${portfolio.total_expected_value:,.2f}")
        print(f"Est. Sharpe: {portfolio.sharpe_estimate:.2f}")

        if portfolio.positions:
            print(f"\n{'─'*72}")
            print(f"{'Side':<5} {'Edge':>6} {'Size':>8} {'EV/$':>7} {'Conf':>5}  Question")
            print(f"{'─'*72}")
            for p in portfolio.positions:
                print(
                    f"{p.side:<5} {p.edge:>5.1%} "
                    f"${p.position_size_usd:>7.2f} "
                    f"{p.expected_value:>+6.1%} "
                    f"{p.confidence:>4.0%}  "
                    f"{p.question[:45]}"
                )
        print(f"{'='*72}")


class DrawdownMonitor:
    """Circuit breaker that halts trading after excessive losses.

    Tracks cumulative P&L and stops the system when drawdown
    from peak exceeds the threshold. This prevents ruin.
    """

    def __init__(self, max_drawdown_pct: float = 0.15) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.pnl_history: list[float] = [0.0]  # start at 0
        self.peak: float = 0.0
        self.halted: bool = False
        self.halt_reason: str = ""

    def record_pnl(self, pnl: float) -> None:
        """Record a P&L event (positive = profit, negative = loss)."""
        cumulative = self.pnl_history[-1] + pnl
        self.pnl_history.append(cumulative)
        self.peak = max(self.peak, cumulative)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak (as fraction of initial bankroll)."""
        if self.peak <= 0:
            return abs(self.pnl_history[-1])
        return self.peak - self.pnl_history[-1]

    def check_halt(self, bankroll: float) -> bool:
        """Check if drawdown exceeds threshold. Returns True if halted."""
        if self.halted:
            return True

        dd_pct = self.current_drawdown / bankroll if bankroll > 0 else 0
        if dd_pct >= self.max_drawdown_pct:
            self.halted = True
            self.halt_reason = (
                f"Drawdown {dd_pct:.1%} exceeded limit {self.max_drawdown_pct:.0%}. "
                f"Peak: {self.peak:.2f}, Current: {self.pnl_history[-1]:.2f}"
            )
            logger.warning(f"CIRCUIT BREAKER: {self.halt_reason}")
            return True
        return False
