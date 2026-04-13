"""AI bias detector — identifies where AI traders are systematically wrong.

Core insight (KSK, 2026-04-13): AI systems trained with RLHF exhibit
systematic hedging toward 0.5 on uncertain markets. If AI traders
dominate marginal price-setting, markets will be underconfident on
events the AI is uncertain about. The divergence between AI consensus
and crowd price is the signal.

Three detection patterns:
1. Hedging bias: AI clusters near 0.50 while crowd has moved to 0.30/0.70
2. Correlation blindness: AI treats related markets independently
3. Overconfidence on known events: AI is too confident on well-documented
   events (within training data) but underconfident on novel events

Usage:
    detector = AIBiasDetector()
    signals = detector.detect_hedging_bias(fish_predictions, market_prices)
    print(signals)  # markets where AI-crowd divergence creates opportunity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class HedgingBiasSignal:
    """Signal where AI hedging creates exploitable divergence from crowd."""
    market_id: str
    question: str
    ai_probability: float      # our swarm consensus
    market_price: float        # crowd price on Polymarket
    divergence: float          # |ai - crowd|
    ai_spread: float           # Fish disagreement spread
    direction: str             # 'ai_underconfident' or 'ai_overconfident'
    recommendation: str        # 'follow_crowd' or 'fade_crowd'
    confidence: float          # 0-1


class AIBiasDetector:
    """Detects where AI systematic biases create trading opportunities.

    The key insight: AI hedging is a FEATURE, not a bug — for us.
    When our Fish all cluster near 0.50 but the Polymarket crowd
    has moved to 0.30 or 0.70, the crowd likely has information
    the AI doesn't (insider knowledge, breaking news not in training
    data). We should FOLLOW the crowd in these cases, not our Fish.

    Conversely, when our Fish are confident (0.80+) but the market
    is at 0.60, the AI may be correct and the crowd is slow to
    update. We should FOLLOW our Fish in these cases.

    The meta-strategy: use AI bias detection to decide WHEN to
    trust our Fish vs WHEN to trust the crowd.
    """

    def __init__(
        self,
        hedging_threshold: float = 0.15,  # max distance from 0.50 to count as hedging
        divergence_threshold: float = 0.10,  # min AI-crowd divergence to signal
        confidence_threshold: float = 0.25,  # min distance from 0.50 for "confident"
    ) -> None:
        self.hedging_threshold = hedging_threshold
        self.divergence_threshold = divergence_threshold
        self.confidence_threshold = confidence_threshold

    def detect_hedging_bias(
        self,
        predictions: list[dict[str, Any]],
    ) -> list[HedgingBiasSignal]:
        """Detect markets where AI hedging creates opportunity.

        Each prediction dict should have:
        - market_id, question
        - ai_probability: swarm consensus
        - market_price: Polymarket YES price
        - spread: Fish disagreement spread

        Returns ranked list of hedging bias signals.
        """
        signals = []

        for pred in predictions:
            ai_prob = pred.get("ai_probability", 0.5)
            mkt_price = pred.get("market_price", 0.5)
            spread = pred.get("spread", 0.0)

            # AI distance from 0.50 (how opinionated is the AI?)
            ai_conviction = abs(ai_prob - 0.50)
            # Crowd distance from 0.50
            crowd_conviction = abs(mkt_price - 0.50)
            # Divergence between AI and crowd
            divergence = abs(ai_prob - mkt_price)

            if divergence < self.divergence_threshold:
                continue  # AI and crowd agree — no signal

            # Pattern 1: AI hedging, crowd moved
            # AI near 0.50 but crowd at 0.30 or 0.70
            if ai_conviction < self.hedging_threshold and crowd_conviction > self.confidence_threshold:
                signals.append(HedgingBiasSignal(
                    market_id=pred.get("market_id", ""),
                    question=pred.get("question", ""),
                    ai_probability=ai_prob,
                    market_price=mkt_price,
                    divergence=round(divergence, 4),
                    ai_spread=spread,
                    direction="ai_underconfident",
                    recommendation="follow_crowd",
                    confidence=round(min(divergence * 2, 1.0), 4),
                ))

            # Pattern 2: AI confident, crowd hasn't caught up
            # AI at 0.80 but crowd at 0.60 — AI may know something
            elif ai_conviction > self.confidence_threshold and crowd_conviction < ai_conviction:
                signals.append(HedgingBiasSignal(
                    market_id=pred.get("market_id", ""),
                    question=pred.get("question", ""),
                    ai_probability=ai_prob,
                    market_price=mkt_price,
                    divergence=round(divergence, 4),
                    ai_spread=spread,
                    direction="ai_overconfident",
                    recommendation="follow_ai" if spread < 0.20 else "caution",
                    confidence=round(min(ai_conviction, 1.0 - spread), 4),
                ))

        signals.sort(key=lambda s: -s.divergence)
        logger.info(f"AI bias detector: {len(signals)} signals from {len(predictions)} markets")
        return signals

    def classify_market_regime(
        self,
        ai_probability: float,
        market_price: float,
        fish_spread: float,
    ) -> dict[str, Any]:
        """Classify a single market into an AI bias regime.

        Returns a dict with regime classification and recommended action.
        """
        ai_conv = abs(ai_probability - 0.50)
        crowd_conv = abs(market_price - 0.50)
        divergence = abs(ai_probability - market_price)

        # Regime classification
        if ai_conv < self.hedging_threshold:
            if crowd_conv > self.confidence_threshold:
                regime = "ai_hedging_crowd_moved"
                action = "follow_crowd"
                reason = (
                    "AI is uncertain (hedging near 0.50) but the crowd has "
                    "moved. Crowd likely has information not in AI training data."
                )
            else:
                regime = "both_uncertain"
                action = "skip"
                reason = (
                    "Both AI and crowd are near 0.50. Genuinely uncertain market. "
                    "No edge for either strategy."
                )
        elif ai_conv > self.confidence_threshold:
            if divergence > self.divergence_threshold:
                if fish_spread < 0.15:
                    regime = "ai_confident_crowd_disagrees"
                    action = "follow_ai"
                    reason = (
                        "AI Fish agree strongly and diverge from crowd. "
                        "Low Fish spread suggests genuine signal, not noise."
                    )
                else:
                    regime = "ai_split_crowd_disagrees"
                    action = "caution"
                    reason = (
                        "AI average diverges from crowd but Fish disagree "
                        "among themselves. Mixed signal — reduce position size."
                    )
            else:
                regime = "ai_crowd_agree"
                action = "standard"
                reason = "AI and crowd agree. Use standard K-Fish pipeline."
        else:
            regime = "moderate_conviction"
            action = "standard"
            reason = "Moderate AI conviction, no strong bias signal detected."

        return {
            "regime": regime,
            "action": action,
            "reason": reason,
            "ai_probability": ai_probability,
            "market_price": market_price,
            "divergence": round(divergence, 4),
            "ai_conviction": round(ai_conv, 4),
            "crowd_conviction": round(crowd_conv, 4),
            "fish_spread": fish_spread,
        }

    def print_signals(self, signals: list[HedgingBiasSignal]) -> None:
        """Print detected bias signals."""
        if not signals:
            print("No AI bias signals detected.")
            return

        print(f"\n{'='*72}")
        print(f"AI BIAS SIGNALS — {len(signals)} markets with exploitable divergence")
        print(f"{'='*72}")
        print(f"{'Dir':>15} {'AI':>6} {'Crowd':>6} {'Gap':>6} {'Action':>13} Question")
        print(f"{'─'*72}")
        for s in signals:
            print(
                f"{s.direction:>15} {s.ai_probability:>5.0%} "
                f"{s.market_price:>5.0%} {s.divergence:>5.1%} "
                f"{s.recommendation:>13} {s.question[:30]}"
            )
        print(f"{'='*72}")
