"""Kelly Criterion position sizing for prediction market trading.

The Kelly Criterion determines the optimal fraction of bankroll to wager
on a bet with positive expected value. We use Quarter-Kelly (conservative)
as recommended by PolySwarm (arXiv:2604.03888) to reduce variance.

Theory:
- Full Kelly:  f* = (bp - q) / b
  where b = odds, p = win probability, q = 1-p
- For prediction markets (binary, price = implied probability):
  f* = (p_our - p_market) / (1 - p_market)  [for YES bet]
  f* = (p_market - p_our) / p_market          [for NO bet]

References:
- Thorp, E.O. (1962). "Beat the Dealer" — original Kelly application
- Kelly, J.L. (1956). "A New Interpretation of Information Rate"
- PolySwarm: Quarter-Kelly for prediction markets
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loguru import logger


class BetSide(str, Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


@dataclass
class TradeSignal:
    """A recommended trade from the risk engine."""

    market_id: str
    market_question: str
    side: BetSide
    our_probability: float
    market_price: float
    edge: float          # our_probability - market_price (for YES)
    kelly_fraction: float  # Raw Kelly fraction
    position_fraction: float  # After applying Kelly multiplier + caps
    position_size_usd: float  # Dollar amount to risk
    expected_value: float  # Expected profit per dollar
    confidence: float

    @property
    def is_actionable(self) -> bool:
        return self.side != BetSide.ABSTAIN and self.position_size_usd > 0


class KellyCriterion:
    """Quarter-Kelly position sizing with risk controls.

    Usage:
        kelly = KellyCriterion(
            bankroll=1000,
            kelly_fraction=0.25,  # Quarter-Kelly
            max_position_pct=0.05,
            max_drawdown_pct=0.15,
        )

        signal = kelly.compute_signal(
            market_id="abc",
            market_question="Will X happen?",
            our_probability=0.72,
            market_price=0.60,
            confidence=0.85,
        )

        if signal.is_actionable:
            print(f"BET {signal.side}: ${signal.position_size_usd:.2f}")
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        min_edge: float = 0.05,
        paper_trading: bool = True,
    ) -> None:
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.min_edge = min_edge
        self.paper_trading = paper_trading

        self._initial_bankroll = bankroll
        self._peak_bankroll = bankroll
        self._trade_history: list[TradeSignal] = []

    def compute_signal(
        self,
        market_id: str,
        market_question: str,
        our_probability: float,
        market_price: float,
        confidence: float = 0.5,
    ) -> TradeSignal:
        """Compute a trade signal with Kelly-optimal position sizing.

        Args:
            our_probability: Swarm's calibrated probability of YES outcome
            market_price: Current YES price on the prediction market
            confidence: Swarm's confidence in its estimate (0-1)
        """
        # Determine side and edge
        yes_edge = our_probability - market_price
        no_edge = (1 - our_probability) - (1 - market_price)  # = market_price - our_probability

        if abs(yes_edge) < self.min_edge:
            return self._abstain_signal(market_id, market_question, our_probability, market_price, confidence)

        if yes_edge > 0:
            side = BetSide.YES
            edge = yes_edge
            # Kelly for YES: f* = (p - market_price) / (1 - market_price)
            raw_kelly = edge / (1 - market_price) if market_price < 1 else 0
        else:
            side = BetSide.NO
            edge = -yes_edge
            # Kelly for NO: f* = (market_price - p) / market_price
            raw_kelly = edge / market_price if market_price > 0 else 0

        # Apply fractional Kelly (Quarter-Kelly by default)
        adjusted_kelly = raw_kelly * self.kelly_fraction

        # Scale by confidence — less confident = smaller position
        adjusted_kelly *= confidence

        # Cap at max position percentage
        position_fraction = min(adjusted_kelly, self.max_position_pct)

        # Check drawdown limit
        if self._check_drawdown():
            logger.warning(f"Drawdown limit reached — ABSTAIN")
            return self._abstain_signal(market_id, market_question, our_probability, market_price, confidence)

        # Convert to dollar amount
        position_size = position_fraction * self.bankroll

        # Expected value per dollar risked
        if side == BetSide.YES:
            ev = our_probability * (1 / market_price - 1) - (1 - our_probability)
        else:
            ev = (1 - our_probability) * (1 / (1 - market_price) - 1) - our_probability

        signal = TradeSignal(
            market_id=market_id,
            market_question=market_question,
            side=side,
            our_probability=round(our_probability, 4),
            market_price=round(market_price, 4),
            edge=round(edge, 4),
            kelly_fraction=round(raw_kelly, 4),
            position_fraction=round(position_fraction, 4),
            position_size_usd=round(position_size, 2),
            expected_value=round(ev, 4),
            confidence=round(confidence, 4),
        )

        self._trade_history.append(signal)

        if signal.is_actionable:
            mode = "[PAPER]" if self.paper_trading else "[LIVE]"
            logger.info(
                f"{mode} Signal: {side.value.upper()} '{market_question[:40]}...' "
                f"${position_size:.2f} (edge={edge:.3f}, Kelly={raw_kelly:.3f})"
            )

        return signal

    def record_pnl(self, pnl: float) -> None:
        """Record profit/loss from a resolved trade."""
        self.bankroll += pnl
        self._peak_bankroll = max(self._peak_bankroll, self.bankroll)
        logger.info(f"PnL recorded: ${pnl:+.2f} → bankroll=${self.bankroll:.2f}")

    def _check_drawdown(self) -> bool:
        """Check if current drawdown exceeds the limit."""
        if self._peak_bankroll == 0:
            return False
        drawdown = (self._peak_bankroll - self.bankroll) / self._peak_bankroll
        return drawdown >= self.max_drawdown_pct

    def _abstain_signal(
        self, market_id: str, question: str, our_prob: float, mkt_price: float, conf: float
    ) -> TradeSignal:
        return TradeSignal(
            market_id=market_id,
            market_question=question,
            side=BetSide.ABSTAIN,
            our_probability=our_prob,
            market_price=mkt_price,
            edge=0.0,
            kelly_fraction=0.0,
            position_fraction=0.0,
            position_size_usd=0.0,
            expected_value=0.0,
            confidence=conf,
        )

    @property
    def current_drawdown(self) -> float:
        if self._peak_bankroll == 0:
            return 0.0
        return (self._peak_bankroll - self.bankroll) / self._peak_bankroll

    @property
    def total_return(self) -> float:
        if self._initial_bankroll == 0:
            return 0.0
        return (self.bankroll - self._initial_bankroll) / self._initial_bankroll
