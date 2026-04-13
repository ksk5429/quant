"""Cross-market arbitrage detection and hedged position construction.

Core insight (KSK, 2026-04-13): If AI traders dominate marginal
price-setting on prediction markets, their systematic biases
(hedging toward 0.5, failure to enforce logical constraints across
correlated markets) create exploitable mispricings.

Three detection methods:
1. Logical constraint violations: P(A) must >= P(A and B)
2. Subset violations: markets about the same event with different
   thresholds must have monotone prices
3. Complement violations: P(A) + P(not A) should sum to ~1.0

Two exploitation strategies:
1. Arbitrage: risk-free profit from logically inconsistent prices
2. Hedged pairs: correlated-market positions that profit from the
   spread, reducing net risk vs single-market bets

Reference: Saguillo et al. (2025) detected $40M in arbitrage on
Polymarket via semantic matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class MarketPair:
    """A pair of semantically related markets."""
    market_a_id: str
    market_b_id: str
    question_a: str
    question_b: str
    price_a: float  # YES price of market A
    price_b: float  # YES price of market B
    similarity: float  # semantic similarity [0, 1]
    relationship: str  # 'subset', 'complement', 'correlated', 'independent'


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage or mispricing between related markets."""
    pair: MarketPair
    type: str  # 'logical_violation', 'complement_violation', 'spread_mispricing'
    description: str
    expected_profit_pct: float  # estimated profit as % of capital deployed
    confidence: float  # 0-1, how confident we are this is real arbitrage

    @property
    def is_risk_free(self) -> bool:
        """True arbitrage (logical violation) vs statistical edge."""
        return self.type in ("logical_violation", "complement_violation")


@dataclass
class HedgedPosition:
    """A hedged pair trade across two correlated markets."""
    market_a_id: str
    market_b_id: str
    question_a: str
    question_b: str

    # Legs
    leg_a_side: str   # "YES" or "NO"
    leg_a_price: float
    leg_a_size: float
    leg_b_side: str
    leg_b_price: float
    leg_b_size: float

    # Risk metrics
    max_loss: float    # worst-case loss across all outcome scenarios
    max_gain: float    # best-case gain
    expected_pnl: float
    hedge_ratio: float  # correlation between legs (1.0 = perfect hedge)
    scenarios: list[dict]  # P&L under each possible outcome combination

    @property
    def risk_reward(self) -> float:
        if self.max_loss == 0:
            return float("inf")
        return self.max_gain / abs(self.max_loss)


class ArbitrageDetector:
    """Detects cross-market arbitrage opportunities.

    Uses semantic similarity to find related markets, then checks
    for logical constraint violations and spread mispricings.
    """

    def __init__(
        self,
        min_similarity: float = 0.65,
        min_profit_pct: float = 0.02,
    ) -> None:
        self.min_similarity = min_similarity
        self.min_profit_pct = min_profit_pct

    def find_pairs(
        self,
        markets: list[dict[str, Any]],
        similarity_matrix: np.ndarray | None = None,
    ) -> list[MarketPair]:
        """Find semantically related market pairs.

        Args:
            markets: list of dicts with 'id', 'question', 'yes_price'
            similarity_matrix: NxN cosine similarity matrix (from SemanticMatcher)
                If None, computes from questions using sentence-transformers.

        Returns:
            List of MarketPair objects above similarity threshold.
        """
        n = len(markets)
        if n < 2:
            return []

        if similarity_matrix is None:
            similarity_matrix = self._compute_similarity(
                [m["question"] for m in markets]
            )

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(similarity_matrix[i, j])
                if sim < self.min_similarity:
                    continue

                relationship = self._classify_relationship(
                    markets[i]["question"], markets[j]["question"], sim,
                )

                pairs.append(MarketPair(
                    market_a_id=markets[i]["id"],
                    market_b_id=markets[j]["id"],
                    question_a=markets[i]["question"],
                    question_b=markets[j]["question"],
                    price_a=markets[i]["yes_price"],
                    price_b=markets[j]["yes_price"],
                    similarity=round(sim, 4),
                    relationship=relationship,
                ))

        pairs.sort(key=lambda p: -p.similarity)
        logger.info(f"Found {len(pairs)} related market pairs (sim >= {self.min_similarity})")
        return pairs

    def detect_arbitrage(self, pairs: list[MarketPair]) -> list[ArbitrageOpportunity]:
        """Detect arbitrage opportunities from market pairs.

        Checks:
        1. Subset violations: if B is a subset of A, then P(B) <= P(A)
        2. Complement violations: if B = not A, then P(A) + P(B) ~= 1.0
        3. Spread mispricing: correlated markets with spreads wider than
           justified by their semantic relationship
        """
        opportunities = []

        for pair in pairs:
            # Check 1: Subset violation
            # If question B is a specific instance of question A,
            # then P(B) must be <= P(A).
            if pair.relationship == "subset":
                if pair.price_b > pair.price_a + 0.02:
                    profit = pair.price_b - pair.price_a
                    opportunities.append(ArbitrageOpportunity(
                        pair=pair,
                        type="logical_violation",
                        description=(
                            f"Subset violation: '{pair.question_b[:40]}' (P={pair.price_b:.2f}) "
                            f"is a subset of '{pair.question_a[:40]}' (P={pair.price_a:.2f}) "
                            f"but is priced higher. Buy NO on B, buy YES on A."
                        ),
                        expected_profit_pct=round(profit, 4),
                        confidence=pair.similarity,
                    ))

            # Check 2: Complement violation
            elif pair.relationship == "complement":
                sum_prices = pair.price_a + pair.price_b
                if abs(sum_prices - 1.0) > 0.05:
                    profit = abs(sum_prices - 1.0)
                    opportunities.append(ArbitrageOpportunity(
                        pair=pair,
                        type="complement_violation",
                        description=(
                            f"Complement violation: P(A) + P(B) = {sum_prices:.3f} != 1.0. "
                            f"A='{pair.question_a[:30]}' ({pair.price_a:.2f}), "
                            f"B='{pair.question_b[:30]}' ({pair.price_b:.2f})."
                        ),
                        expected_profit_pct=round(profit / 2, 4),
                        confidence=pair.similarity * 0.8,
                    ))

            # Check 3: Spread mispricing on correlated markets
            elif pair.relationship == "correlated" and pair.similarity > 0.75:
                spread = abs(pair.price_a - pair.price_b)
                expected_spread = (1.0 - pair.similarity) * 0.5
                if spread > expected_spread + 0.10:
                    excess = spread - expected_spread
                    opportunities.append(ArbitrageOpportunity(
                        pair=pair,
                        type="spread_mispricing",
                        description=(
                            f"Spread {spread:.2f} exceeds expected {expected_spread:.2f} "
                            f"for similarity {pair.similarity:.2f}. "
                            f"A={pair.price_a:.2f}, B={pair.price_b:.2f}."
                        ),
                        expected_profit_pct=round(excess / 2, 4),
                        confidence=pair.similarity * 0.6,
                    ))

        # Filter by minimum profit
        opportunities = [
            o for o in opportunities
            if o.expected_profit_pct >= self.min_profit_pct
        ]

        opportunities.sort(key=lambda o: -o.expected_profit_pct)
        logger.info(f"Detected {len(opportunities)} arbitrage opportunities")
        return opportunities

    def _classify_relationship(self, q_a: str, q_b: str, similarity: float) -> str:
        """Classify the relationship between two market questions."""
        a_lower = q_a.lower()
        b_lower = q_b.lower()

        # Check for complement (negation)
        negation_pairs = [
            ("will", "will not"), ("won't", "will"),
            ("before", "after"), ("yes", "no"),
        ]
        for pos, neg in negation_pairs:
            if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
                if similarity > 0.80:
                    return "complement"

        # Check for subset (one is a specific instance of the other)
        # e.g., "Will OpenAI release a model?" vs "Will GPT-6 be released?"
        if similarity > 0.85:
            # If one question is substantially longer (more specific), it's a subset
            len_ratio = len(q_b) / max(len(q_a), 1)
            if len_ratio > 1.3 or len_ratio < 0.7:
                return "subset"

        if similarity > 0.70:
            return "correlated"

        return "independent"

    def _compute_similarity(self, questions: list[str]) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        try:
            from src.semantic.news_extractor import SemanticMatcher
            matcher = SemanticMatcher()
            return matcher.compute_market_similarity(questions)
        except ImportError:
            # Fallback: simple word overlap Jaccard similarity
            n = len(questions)
            sim = np.zeros((n, n))
            for i in range(n):
                words_i = set(questions[i].lower().split())
                for j in range(i + 1, n):
                    words_j = set(questions[j].lower().split())
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    sim[i, j] = sim[j, i] = intersection / max(union, 1)
            np.fill_diagonal(sim, 1.0)
            return sim


class HedgedPositionBuilder:
    """Constructs hedged pair trades from correlated markets.

    Instead of Kelly-sizing single markets, constructs paired
    positions that profit from the SPREAD between correlated
    markets, reducing net risk.
    """

    def __init__(
        self,
        max_leg_size: float = 50.0,
        min_hedge_ratio: float = 0.5,
    ) -> None:
        self.max_leg_size = max_leg_size
        self.min_hedge_ratio = min_hedge_ratio

    def build_hedged_position(
        self,
        pair: MarketPair,
        our_prob_a: float,
        our_prob_b: float,
        total_size: float = 50.0,
    ) -> HedgedPosition | None:
        """Build a hedged position from a market pair.

        The hedge works by taking opposite-direction positions on
        correlated markets. Under the 4 possible outcome scenarios
        (A=yes/B=yes, A=yes/B=no, A=no/B=yes, A=no/B=no), at least
        one leg profits in most scenarios.

        Args:
            pair: the correlated market pair
            our_prob_a: our calibrated probability for market A
            our_prob_b: our calibrated probability for market B
            total_size: total capital to deploy across both legs
        """
        p_a = pair.price_a
        p_b = pair.price_b

        # Determine sides: buy the underpriced, sell the overpriced
        # relative to our estimates
        edge_a = our_prob_a - p_a  # positive = buy YES, negative = buy NO
        edge_b = our_prob_b - p_b

        leg_a_side = "YES" if edge_a > 0 else "NO"
        leg_b_side = "YES" if edge_b > 0 else "NO"

        # Size legs proportional to edge
        total_edge = abs(edge_a) + abs(edge_b)
        if total_edge < 0.02:
            return None  # no meaningful edge

        leg_a_size = min(total_size * abs(edge_a) / total_edge, self.max_leg_size)
        leg_b_size = min(total_size * abs(edge_b) / total_edge, self.max_leg_size)

        leg_a_price = p_a if leg_a_side == "YES" else 1.0 - p_a
        leg_b_price = p_b if leg_b_side == "YES" else 1.0 - p_b

        # Compute P&L under all 4 outcome scenarios
        scenarios = []
        for outcome_a in [1.0, 0.0]:
            for outcome_b in [1.0, 0.0]:
                pnl_a = self._leg_pnl(leg_a_side, leg_a_price, leg_a_size, outcome_a)
                pnl_b = self._leg_pnl(leg_b_side, leg_b_price, leg_b_size, outcome_b)
                total_pnl = pnl_a + pnl_b

                # Estimate scenario probability
                if pair.relationship == "subset":
                    # B is subset of A: if B=yes then A=yes
                    if outcome_b == 1.0 and outcome_a == 0.0:
                        prob = 0.0  # impossible
                    elif outcome_a == 1.0 and outcome_b == 1.0:
                        prob = our_prob_b
                    elif outcome_a == 1.0 and outcome_b == 0.0:
                        prob = our_prob_a - our_prob_b
                    else:
                        prob = 1.0 - our_prob_a
                else:
                    # Independent approximation
                    prob_a = our_prob_a if outcome_a == 1.0 else 1.0 - our_prob_a
                    prob_b = our_prob_b if outcome_b == 1.0 else 1.0 - our_prob_b
                    prob = prob_a * prob_b

                scenarios.append({
                    "outcome_a": outcome_a,
                    "outcome_b": outcome_b,
                    "pnl_a": round(pnl_a, 2),
                    "pnl_b": round(pnl_b, 2),
                    "total_pnl": round(total_pnl, 2),
                    "probability": round(max(0, prob), 4),
                })

        pnls = [s["total_pnl"] for s in scenarios if s["probability"] > 0.001]
        probs = [s["probability"] for s in scenarios if s["probability"] > 0.001]

        if not pnls:
            return None

        max_loss = min(pnls)
        max_gain = max(pnls)
        expected_pnl = sum(p * pnl for p, pnl in zip(probs, pnls))

        # Hedge ratio: what fraction of scenarios have both legs offsetting
        n_hedged = sum(
            1 for s in scenarios
            if s["probability"] > 0.001 and s["pnl_a"] * s["pnl_b"] < 0
        )
        hedge_ratio = n_hedged / max(len([s for s in scenarios if s["probability"] > 0.001]), 1)

        if hedge_ratio < self.min_hedge_ratio and expected_pnl <= 0:
            return None

        return HedgedPosition(
            market_a_id=pair.market_a_id,
            market_b_id=pair.market_b_id,
            question_a=pair.question_a,
            question_b=pair.question_b,
            leg_a_side=leg_a_side,
            leg_a_price=round(leg_a_price, 4),
            leg_a_size=round(leg_a_size, 2),
            leg_b_side=leg_b_side,
            leg_b_price=round(leg_b_price, 4),
            leg_b_size=round(leg_b_size, 2),
            max_loss=round(max_loss, 2),
            max_gain=round(max_gain, 2),
            expected_pnl=round(expected_pnl, 2),
            hedge_ratio=round(hedge_ratio, 4),
            scenarios=scenarios,
        )

    @staticmethod
    def _leg_pnl(side: str, price: float, size: float, outcome: float) -> float:
        """Calculate P&L for a single leg of a hedged position."""
        shares = size / max(price, 0.01)
        if side == "YES":
            payout = shares * outcome
        else:
            payout = shares * (1.0 - outcome)
        return payout - size

    def print_hedged_position(self, hp: HedgedPosition) -> None:
        """Print a hedged position with scenario analysis."""
        print(f"\n{'='*70}")
        print(f"HEDGED PAIR TRADE")
        print(f"{'='*70}")
        print(f"Leg A: {hp.leg_a_side} ${hp.leg_a_size:.2f} @ {hp.leg_a_price:.3f}")
        print(f"  {hp.question_a[:60]}")
        print(f"Leg B: {hp.leg_b_side} ${hp.leg_b_size:.2f} @ {hp.leg_b_price:.3f}")
        print(f"  {hp.question_b[:60]}")
        print(f"\nMax gain:     ${hp.max_gain:+.2f}")
        print(f"Max loss:     ${hp.max_loss:+.2f}")
        print(f"Expected P&L: ${hp.expected_pnl:+.2f}")
        print(f"Risk/reward:  {hp.risk_reward:.1f}x")
        print(f"Hedge ratio:  {hp.hedge_ratio:.0%}")

        print(f"\n{'Outcome A':>10} {'Outcome B':>10} {'P&L A':>8} {'P&L B':>8} {'Total':>8} {'Prob':>6}")
        print(f"{'─'*52}")
        for s in hp.scenarios:
            if s["probability"] < 0.001:
                continue
            oa = "YES" if s["outcome_a"] == 1.0 else "NO"
            ob = "YES" if s["outcome_b"] == 1.0 else "NO"
            print(
                f"{oa:>10} {ob:>10} "
                f"${s['pnl_a']:>+7.2f} ${s['pnl_b']:>+7.2f} "
                f"${s['total_pnl']:>+7.2f} {s['probability']:>5.1%}"
            )
        print(f"{'='*70}")
