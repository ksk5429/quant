"""AI Bias Detector v2 — deep model for exploiting LLM hedging patterns.

Core insight (KSK, 2026-04-13): AI systems trained with RLHF compress
their probability outputs toward 0.50 due to penalty for confident wrong
answers. This compression is detectable from the GAP between the Fish's
directional reasoning and its assigned probability.

Three regimes the detector distinguishes:
1. Genuine uncertainty — reasoning is vague, confidence low, spread low.
   Action: skip (no edge).
2. RLHF compression — reasoning is directional but probability is neutral.
   Action: decompress the probability and trade on the decompressed estimate.
3. Knowledge gap — Fish references training cutoff, can't verify current state.
   Action: follow the crowd price (they have post-cutoff information).

Five detection layers:
  Layer 1: Reasoning-probability coherence (text direction vs number)
  Layer 2: Bimodal spread detection (distribution shape, not just range)
  Layer 3: Knowledge cutoff classification (temporal markers in reasoning)
  Layer 4: Confidence-probability inconsistency (confident about 0.50 = paradox)
  Layer 5: Self-calibrating thresholds (learn from resolved markets)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FishAnalysisSignals:
    """Extracted signals from a single Fish's output."""
    persona: str
    probability: float
    confidence: float
    reasoning: str

    # Layer 1: reasoning direction
    directional_score: float = 0.0    # -1 (strong NO) to +1 (strong YES)
    reasoning_probability_gap: float = 0.0  # |implied_prob - stated_prob|

    # Layer 3: knowledge gap
    references_cutoff: bool = False
    temporal_uncertainty: bool = False

    # Layer 4: confidence paradox
    confidence_probability_inconsistency: float = 0.0


@dataclass
class MarketBiasProfile:
    """Complete bias analysis for a single market."""
    market_id: str
    question: str

    # Inputs
    swarm_probability: float   # raw swarm aggregate
    market_price: float        # Polymarket crowd price
    fish_predictions: list[FishAnalysisSignals]

    # Layer 1: Reasoning coherence
    mean_directional_score: float = 0.0
    mean_rp_gap: float = 0.0           # reasoning-probability gap
    compression_detected: bool = False

    # Layer 2: Distribution shape
    is_bimodal: bool = False
    n_directional_yes: int = 0
    n_directional_no: int = 0
    n_neutral: int = 0
    bimodal_strength: float = 0.0

    # Layer 3: Knowledge gap
    knowledge_gap_detected: bool = False
    n_cutoff_references: int = 0

    # Layer 4: Confidence paradox
    mean_confidence_inconsistency: float = 0.0
    paradox_detected: bool = False

    # Final classification
    regime: str = "unknown"           # genuine_uncertainty, rlhf_compression, knowledge_gap
    decompressed_probability: float = 0.5
    recommended_action: str = "skip"  # skip, decompress, follow_crowd, standard
    action_confidence: float = 0.0
    reasoning: str = ""


@dataclass
class BiasDetectorState:
    """Persistent state for self-calibration (Layer 5)."""
    regime_outcomes: dict[str, list[dict]] = field(default_factory=dict)

    def record_outcome(
        self, regime: str, action: str, our_prob: float,
        crowd_prob: float, decompressed_prob: float, outcome: float,
    ) -> None:
        if regime not in self.regime_outcomes:
            self.regime_outcomes[regime] = []
        self.regime_outcomes[regime].append({
            "our_prob": our_prob,
            "crowd_prob": crowd_prob,
            "decompressed_prob": decompressed_prob,
            "outcome": outcome,
            "brier_ours": (our_prob - outcome) ** 2,
            "brier_crowd": (crowd_prob - outcome) ** 2,
            "brier_decompressed": (decompressed_prob - outcome) ** 2,
            "timestamp": datetime.now().isoformat(),
        })

    def get_regime_accuracy(self, regime: str) -> dict[str, float] | None:
        entries = self.regime_outcomes.get(regime, [])
        if len(entries) < 10:
            return None
        return {
            "n": len(entries),
            "brier_ours": np.mean([e["brier_ours"] for e in entries]),
            "brier_crowd": np.mean([e["brier_crowd"] for e in entries]),
            "brier_decompressed": np.mean([e["brier_decompressed"] for e in entries]),
        }


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1: REASONING-PROBABILITY COHERENCE
# ═══════════════════════════════════════════════════════════════════════

# Directional language indicators
_YES_INDICATORS = [
    "likely", "probable", "strong evidence", "suggests yes",
    "expected to", "will likely", "high probability", "almost certain",
    "favors", "points toward", "indicates yes", "should happen",
    "on track", "momentum toward", "evidence supports",
    "historically", "precedent suggests", "base rate is high",
]

_NO_INDICATORS = [
    "unlikely", "improbable", "strong evidence against", "suggests no",
    "not expected", "will likely not", "low probability", "almost certainly not",
    "opposes", "points away", "indicates no", "should not happen",
    "off track", "momentum against", "evidence contradicts",
    "no precedent", "base rate is low", "historically rare",
]

_UNCERTAINTY_INDICATORS = [
    "uncertain", "unclear", "hard to say", "could go either way",
    "insufficient evidence", "genuinely ambiguous", "toss-up",
    "no strong evidence", "50/50", "coin flip", "too close to call",
    "lack of information", "cannot determine", "don't know",
]


def _score_reasoning_direction(reasoning: str, question: str = "") -> float:
    """Score the directional strength of reasoning text.

    Returns float in [-1, +1]:
    - Positive = reasoning argues for YES
    - Negative = reasoning argues for NO
    - Near zero = genuinely uncertain, mixed, OR contradictory

    IMPORTANT: strips the question text from reasoning before scoring
    to avoid false positives from question contamination.
    """
    text = reasoning.lower()

    # Strip question text to avoid contamination (reviewer fix)
    if question:
        q_lower = question.lower()
        text = text.replace(q_lower, "")

    # Use word boundary matching to avoid "not expected to fail" → NO
    # when it should be YES. We check for negation context.
    yes_count = 0
    no_count = 0
    uncertain_count = 0

    for indicator in _YES_INDICATORS:
        if indicator in text:
            # Check for preceding negation within 5 words
            idx = text.find(indicator)
            prefix = text[max(0, idx - 30):idx]
            if any(neg in prefix for neg in ["not ", "no ", "don't ", "doesn't ", "isn't ", "won't "]):
                no_count += 1  # negated YES = NO
            else:
                yes_count += 1

    for indicator in _NO_INDICATORS:
        if indicator in text:
            idx = text.find(indicator)
            prefix = text[max(0, idx - 30):idx]
            if any(neg in prefix for neg in ["not ", "no ", "don't ", "doesn't ", "isn't ", "won't "]):
                yes_count += 1  # negated NO = YES
            else:
                no_count += 1

    for indicator in _UNCERTAINTY_INDICATORS:
        if indicator in text:
            uncertain_count += 1

    total = yes_count + no_count + uncertain_count
    if total == 0:
        return 0.0

    # Directional score
    direction = (yes_count - no_count) / total

    # Dampen if uncertainty indicators are present
    if uncertain_count > 0:
        uncertainty_weight = uncertain_count / total
        direction *= (1.0 - uncertainty_weight * 0.7)

    return max(-1.0, min(1.0, direction))


def _implied_probability_from_direction(direction_score: float) -> float:
    """Convert a directional reasoning score to an implied probability.

    Maps [-1, +1] → [0.05, 0.95] via sigmoid-like transform.
    """
    # Softer sigmoid: direction ±0.5 maps to ~0.30/0.70
    implied = 0.5 + direction_score * 0.4
    return max(0.05, min(0.95, implied))


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2: BIMODAL SPREAD DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _detect_bimodal(probabilities: list[float], threshold: float = 0.10) -> dict:
    """Detect bimodal distribution in Fish probabilities.

    Bimodal = Fish split into two camps (some say YES likely, some say NO likely).
    This means the MEAN is misleading — the swarm is not uncertain,
    it's DISAGREEING ABOUT DIRECTION.

    Returns:
        dict with is_bimodal, n_yes, n_no, n_neutral, strength
    """
    probs = np.array(probabilities)
    n = len(probs)

    above = np.sum(probs > 0.5 + threshold)   # directional YES camp
    below = np.sum(probs < 0.5 - threshold)   # directional NO camp
    neutral = n - above - below               # near 0.50

    # Bimodal if both camps have 2+ Fish AND neutral is minority
    is_bimodal = above >= 2 and below >= 2 and neutral < n / 2

    # Strength: how separated are the two modes?
    if is_bimodal and above > 0 and below > 0:
        yes_mean = float(np.mean(probs[probs > 0.5 + threshold]))
        no_mean = float(np.mean(probs[probs < 0.5 - threshold]))
        strength = yes_mean - no_mean  # distance between modes
    else:
        strength = 0.0

    return {
        "is_bimodal": is_bimodal,
        "n_yes": int(above),
        "n_no": int(below),
        "n_neutral": int(neutral),
        "strength": round(strength, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# LAYER 3: KNOWLEDGE CUTOFF CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

_CUTOFF_INDICATORS = [
    "knowledge cutoff", "training data", "as of my", "cannot verify",
    "unable to confirm", "no information after", "beyond my knowledge",
    "i don't have access to", "not in my training", "post-cutoff",
    "i am uncertain about current", "cannot access recent",
    "my information may be outdated", "i lack current data",
]

_TEMPORAL_UNCERTAINTY = [
    "recently", "breaking news", "just announced", "latest reports",
    "current status", "as of today", "developing situation",
    "ongoing", "unfolding", "live updates",
]


def _detect_knowledge_gap(reasoning: str) -> dict:
    """Detect if the Fish is reasoning from a knowledge gap.

    Returns dict with references_cutoff, temporal_uncertainty, n_indicators.
    """
    text = reasoning.lower()
    cutoff_hits = sum(1 for ind in _CUTOFF_INDICATORS if ind in text)
    temporal_hits = sum(1 for ind in _TEMPORAL_UNCERTAINTY if ind in text)

    return {
        "references_cutoff": cutoff_hits > 0,
        "temporal_uncertainty": temporal_hits > 0,
        "n_cutoff_indicators": cutoff_hits,
        "n_temporal_indicators": temporal_hits,
    }


# ═══════════════════════════════════════════════════════════════════════
# LAYER 4: CONFIDENCE-PROBABILITY INCONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

def _confidence_probability_inconsistency(
    probability: float, confidence: float,
) -> float:
    """Measure the paradox of being confident about neutrality.

    High confidence + neutral probability = paradox.
    The Fish is saying "I'm very sure the answer is 50/50" which
    is either genuine (very rare events) or RLHF compression.

    Returns: inconsistency score [0, 1]. High = paradox detected.
    """
    neutrality = 1.0 - abs(probability - 0.5) * 2  # 1.0 at 0.50, 0.0 at 0/1
    inconsistency = confidence * neutrality

    # Only flag when confidence is meaningfully high
    if confidence < 0.4:
        return 0.0

    return max(0.0, min(1.0, inconsistency))


# ═══════════════════════════════════════════════════════════════════════
# HEDGING DECOMPRESSOR
# ═══════════════════════════════════════════════════════════════════════

def _decompress_probability(
    stated_probability: float,
    directional_score: float,
    confidence: float,
    is_bimodal: bool,
    compression_strength: float = 0.6,
) -> float:
    """Estimate the "pre-hedging" probability the Fish intended.

    When RLHF compression is detected, the stated probability (e.g., 0.52)
    is pulled toward 0.50 from where the reasoning points (e.g., 0.35).
    This function estimates the original, decompressed probability.

    Args:
        stated_probability: the Fish's output probability
        directional_score: reasoning direction [-1, +1]
        confidence: Fish's stated confidence
        is_bimodal: whether the swarm is split (bimodal)
        compression_strength: how much to decompress (0=none, 1=full)

    Returns:
        Decompressed probability estimate.
    """
    implied_prob = _implied_probability_from_direction(directional_score)

    # Blend stated and implied based on compression strength
    # Higher compression_strength = trust the reasoning direction more
    weight = compression_strength * confidence
    decompressed = stated_probability * (1 - weight) + implied_prob * weight

    # If bimodal: the swarm is DISAGREEING about direction.
    # This means higher uncertainty, not stronger signal.
    # REDUCE the distance from 0.50 to express wider uncertainty.
    if is_bimodal:
        distance_from_half = decompressed - 0.50
        decompressed = 0.50 + distance_from_half * 0.7  # shrink toward 0.50

    return max(0.05, min(0.95, decompressed))


# ═══════════════════════════════════════════════════════════════════════
# MAIN DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════

class AIBiasDetector:
    """Deep model for detecting and exploiting AI hedging bias.

    Five layers of detection:
    1. Reasoning-probability coherence: does the text contradict the number?
    2. Bimodal spread: is the swarm split, not uncertain?
    3. Knowledge cutoff: is the Fish reasoning from a gap?
    4. Confidence paradox: is the Fish confident about being neutral?
    5. Self-calibration: learn which regime classifications actually work.

    Usage:
        detector = AIBiasDetector()
        profile = detector.analyze_market(
            market_id="m1",
            question="Will X happen?",
            swarm_probability=0.52,
            market_price=0.70,
            fish_predictions=[...],  # FishPrediction objects
        )
        print(profile.regime)                # 'rlhf_compression'
        print(profile.decompressed_probability)  # 0.38
        print(profile.recommended_action)    # 'decompress'
    """

    def __init__(
        self,
        compression_strength: float = 0.6,
        rp_gap_threshold: float = 0.15,
        bimodal_threshold: float = 0.10,
        paradox_threshold: float = 0.50,
        divergence_threshold: float = 0.10,
    ) -> None:
        self.compression_strength = compression_strength
        self.rp_gap_threshold = rp_gap_threshold
        self.bimodal_threshold = bimodal_threshold
        self.paradox_threshold = paradox_threshold
        self.divergence_threshold = divergence_threshold
        self._state = BiasDetectorState()

    def analyze_market(
        self,
        market_id: str,
        question: str,
        swarm_probability: float,
        market_price: float,
        fish_predictions: list[Any],
    ) -> MarketBiasProfile:
        """Run all 5 detection layers on a single market.

        Args:
            market_id: market identifier
            question: market question text
            swarm_probability: raw swarm aggregate probability
            market_price: Polymarket YES price
            fish_predictions: list of FishPrediction objects (from llm_fish.py)

        Returns:
            MarketBiasProfile with regime classification and recommended action.
        """
        # Guard: empty Fish list
        if not fish_predictions:
            return MarketBiasProfile(
                market_id=market_id, question=question,
                swarm_probability=swarm_probability,
                market_price=market_price, fish_predictions=[],
                regime="no_data", recommended_action="skip",
                decompressed_probability=swarm_probability,
                reasoning="No Fish predictions available.",
            )

        # ── Extract per-Fish signals ──
        fish_signals = []
        for fp in fish_predictions:
            reasoning = getattr(fp, "reasoning", "") or ""
            probability = getattr(fp, "probability", 0.5)
            confidence = getattr(fp, "confidence", 0.5)
            persona = getattr(fp, "persona", "unknown")

            # Layer 1 (pass question for contamination stripping)
            dir_score = _score_reasoning_direction(reasoning, question)
            implied = _implied_probability_from_direction(dir_score)
            rp_gap = abs(implied - probability)

            # Layer 3
            kg = _detect_knowledge_gap(reasoning)

            # Layer 4
            ci = _confidence_probability_inconsistency(probability, confidence)

            fish_signals.append(FishAnalysisSignals(
                persona=persona,
                probability=probability,
                confidence=confidence,
                reasoning=reasoning,
                directional_score=dir_score,
                reasoning_probability_gap=round(rp_gap, 4),
                references_cutoff=kg["references_cutoff"],
                temporal_uncertainty=kg["temporal_uncertainty"],
                confidence_probability_inconsistency=round(ci, 4),
            ))

        # ── Aggregate across Fish ──
        probs = [fs.probability for fs in fish_signals]
        dir_scores = [fs.directional_score for fs in fish_signals]
        rp_gaps = [fs.reasoning_probability_gap for fs in fish_signals]
        ci_scores = [fs.confidence_probability_inconsistency for fs in fish_signals]
        n_cutoff = sum(1 for fs in fish_signals if fs.references_cutoff)

        mean_dir = float(np.mean(dir_scores)) if dir_scores else 0.0
        mean_rp_gap = float(np.mean(rp_gaps)) if rp_gaps else 0.0
        mean_ci = float(np.mean(ci_scores)) if ci_scores else 0.0

        # Layer 2: bimodal detection
        bimodal = _detect_bimodal(probs, self.bimodal_threshold)

        # ── Classify regime ──
        mean_confidence = float(np.mean([fs.confidence for fs in fish_signals]))
        # Compression requires both a gap AND sufficient confidence to decompress
        compression_detected = (
            mean_rp_gap > self.rp_gap_threshold
            and mean_confidence > 0.3  # no point decompressing if confidence is too low
        )
        knowledge_gap = n_cutoff >= len(fish_signals) * 0.3  # 30%+ Fish reference cutoff
        paradox = mean_ci > self.paradox_threshold

        if knowledge_gap:
            regime = "knowledge_gap"
            action = "follow_crowd"
            # Do NOT set decompressed = market_price (circular — market price
            # is the thing under investigation). Instead, blend swarm with crowd
            # giving crowd higher weight proportional to knowledge gap severity.
            crowd_weight = min(n_cutoff / len(fish_signals), 0.8)  # cap at 80% crowd
            decompressed = swarm_probability * (1 - crowd_weight) + market_price * crowd_weight
            action_confidence = min(n_cutoff / len(fish_signals), 1.0)
            reason = (
                f"{n_cutoff}/{len(fish_signals)} Fish reference knowledge cutoff. "
                f"Blending with crowd (weight={crowd_weight:.0%}): {decompressed:.3f}."
            )

        elif compression_detected or (paradox and bimodal["is_bimodal"]):
            regime = "rlhf_compression"
            # Decompress each Fish and re-aggregate
            decompressed_probs = []
            confs = []
            for fs in fish_signals:
                dp = _decompress_probability(
                    fs.probability, fs.directional_score, fs.confidence,
                    bimodal["is_bimodal"], self.compression_strength,
                )
                decompressed_probs.append(dp)
                confs.append(fs.confidence)

            # Confidence-weighted decompressed aggregate
            total_w = sum(confs)
            if total_w > 0:
                decompressed = sum(p * c for p, c in zip(decompressed_probs, confs)) / total_w
            else:
                decompressed = float(np.mean(decompressed_probs))
            decompressed = max(0.05, min(0.95, decompressed))

            action = "decompress"
            action_confidence = min(mean_rp_gap * 2, 1.0)
            reason = (
                f"RLHF compression detected: mean reasoning-probability gap = {mean_rp_gap:.2f}. "
                f"Fish reasoning points {'YES' if mean_dir > 0 else 'NO'} (direction={mean_dir:+.2f}) "
                f"but stated probability is {swarm_probability:.2f}. "
                f"Decompressed estimate: {decompressed:.3f}."
            )
            if bimodal["is_bimodal"]:
                reason += (
                    f" Bimodal split: {bimodal['n_yes']} Fish directional YES, "
                    f"{bimodal['n_no']} directional NO."
                )

        elif abs(swarm_probability - 0.50) < 0.08 and abs(market_price - 0.50) < 0.08:
            regime = "genuine_uncertainty"
            action = "skip"
            decompressed = swarm_probability
            action_confidence = 0.3
            reason = (
                "Both AI and crowd near 0.50. No directional reasoning detected. "
                "Genuinely uncertain market — no edge."
            )

        else:
            regime = "standard"
            action = "standard"
            decompressed = swarm_probability
            divergence = abs(swarm_probability - market_price)
            action_confidence = min(divergence * 3, 1.0)
            reason = (
                f"No bias pattern detected. Standard K-Fish pipeline applies. "
                f"Divergence from crowd: {divergence:.2f}."
            )

        profile = MarketBiasProfile(
            market_id=market_id,
            question=question,
            swarm_probability=swarm_probability,
            market_price=market_price,
            fish_predictions=fish_signals,
            mean_directional_score=round(mean_dir, 4),
            mean_rp_gap=round(mean_rp_gap, 4),
            compression_detected=compression_detected,
            is_bimodal=bimodal["is_bimodal"],
            n_directional_yes=bimodal["n_yes"],
            n_directional_no=bimodal["n_no"],
            n_neutral=bimodal["n_neutral"],
            bimodal_strength=bimodal["strength"],
            knowledge_gap_detected=knowledge_gap,
            n_cutoff_references=n_cutoff,
            mean_confidence_inconsistency=round(mean_ci, 4),
            paradox_detected=paradox,
            regime=regime,
            decompressed_probability=round(decompressed, 4),
            recommended_action=action,
            action_confidence=round(action_confidence, 4),
            reasoning=reason,
        )

        logger.info(
            f"Bias [{market_id}]: regime={regime} action={action} "
            f"swarm={swarm_probability:.3f} decompressed={decompressed:.3f} "
            f"crowd={market_price:.3f} rp_gap={mean_rp_gap:.3f}"
        )

        return profile

    def analyze_batch(
        self,
        markets: list[dict[str, Any]],
    ) -> list[MarketBiasProfile]:
        """Analyze a batch of markets.

        Each dict should have: market_id, question, swarm_probability,
        market_price, fish_predictions.
        """
        profiles = []
        for m in markets:
            profile = self.analyze_market(
                market_id=m["market_id"],
                question=m["question"],
                swarm_probability=m["swarm_probability"],
                market_price=m["market_price"],
                fish_predictions=m["fish_predictions"],
            )
            profiles.append(profile)
        return profiles

    def record_outcome(self, profile: MarketBiasProfile, outcome: float) -> None:
        """Record a resolved market for self-calibration (Layer 5)."""
        self._state.record_outcome(
            regime=profile.regime,
            action=profile.recommended_action,
            our_prob=profile.swarm_probability,
            crowd_prob=profile.market_price,
            decompressed_prob=profile.decompressed_probability,
            outcome=outcome,
        )

    def get_regime_performance(self) -> dict[str, Any]:
        """Get per-regime accuracy statistics.

        Returns which regimes/actions actually work based on resolved data.
        """
        report = {}
        for regime in ["genuine_uncertainty", "rlhf_compression", "knowledge_gap", "standard"]:
            stats = self._state.get_regime_accuracy(regime)
            if stats:
                best_source = min(
                    [("ours", stats["brier_ours"]),
                     ("crowd", stats["brier_crowd"]),
                     ("decompressed", stats["brier_decompressed"])],
                    key=lambda x: x[1],
                )
                report[regime] = {
                    **stats,
                    "best_source": best_source[0],
                    "best_brier": round(best_source[1], 4),
                }
        return report

    def print_profile(self, p: MarketBiasProfile) -> None:
        """Print a detailed bias analysis."""
        print(f"\n{'='*72}")
        print(f"AI BIAS ANALYSIS — {p.question[:55]}")
        print(f"{'='*72}")
        print(f"Regime:       {p.regime}")
        print(f"Action:       {p.recommended_action} (confidence={p.action_confidence:.0%})")
        print(f"\nProbabilities:")
        print(f"  Swarm raw:       {p.swarm_probability:.4f}")
        print(f"  Decompressed:    {p.decompressed_probability:.4f}")
        print(f"  Market (crowd):  {p.market_price:.4f}")

        print(f"\nLayer 1 — Reasoning Coherence:")
        print(f"  Direction score: {p.mean_directional_score:+.3f} ({'YES' if p.mean_directional_score > 0 else 'NO' if p.mean_directional_score < 0 else 'neutral'})")
        print(f"  R-P gap:         {p.mean_rp_gap:.3f} {'COMPRESSION' if p.compression_detected else 'normal'}")

        print(f"\nLayer 2 — Distribution Shape:")
        if p.is_bimodal:
            print(f"  BIMODAL: {p.n_directional_yes} YES / {p.n_directional_no} NO / {p.n_neutral} neutral (strength={p.bimodal_strength:.2f})")
        else:
            print(f"  Unimodal: {p.n_directional_yes} YES / {p.n_directional_no} NO / {p.n_neutral} neutral")

        print(f"\nLayer 3 — Knowledge Gap:")
        print(f"  Cutoff references: {p.n_cutoff_references}/{len(p.fish_predictions)} Fish {'GAP DETECTED' if p.knowledge_gap_detected else ''}")

        print(f"\nLayer 4 — Confidence Paradox:")
        print(f"  Mean inconsistency: {p.mean_confidence_inconsistency:.3f} {'PARADOX' if p.paradox_detected else ''}")

        print(f"\n{p.reasoning}")

        # Per-Fish breakdown
        print(f"\n{'Persona':<22} {'Prob':>5} {'Dir':>5} {'Gap':>5} {'CI':>5} {'Cutoff':>7}")
        print(f"{'─'*50}")
        for fs in sorted(p.fish_predictions, key=lambda x: -abs(x.directional_score)):
            cut = "CUT" if fs.references_cutoff else ""
            print(
                f"{fs.persona:<22} {fs.probability:>4.2f} "
                f"{fs.directional_score:>+4.2f} {fs.reasoning_probability_gap:>4.2f} "
                f"{fs.confidence_probability_inconsistency:>4.2f} {cut:>7}"
            )
        print(f"{'='*72}")

    def save_state(self, path: str = "data/bias_detector_state.json") -> None:
        """Persist self-calibration state to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "regime_outcomes": self._state.regime_outcomes,
        }, indent=2), encoding="utf-8")

    def load_state(self, path: str = "data/bias_detector_state.json") -> None:
        """Load persisted self-calibration state."""
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self._state.regime_outcomes = data.get("regime_outcomes", {})
            total = sum(len(v) for v in self._state.regime_outcomes.values())
            logger.info(f"Bias detector state loaded: {total} outcome records")
