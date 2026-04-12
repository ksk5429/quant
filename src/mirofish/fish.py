"""Fish v1 — DEPRECATED. Use llm_fish.py (v2) instead.

This module is retained for reference only. The v2 implementation in
llm_fish.py provides 9 orthogonal personas, zero-cost CLI execution,
extremized aggregation, and multi-round Delphi protocol.

The FishAnalysis model here has richer fields (cross_market_adjustment,
risk_factors) that may be ported to v2 in the future.

DEPRECATED: 2026-04-12. Do not import from this module for new code.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

class FishPersona(str, Enum):
    """Diverse persona types to ensure heterogeneous swarm analysis."""

    GEOPOLITICAL_ANALYST = "geopolitical_analyst"
    FINANCIAL_QUANT = "financial_quant"
    BAYESIAN_STATISTICIAN = "bayesian_statistician"
    INVESTIGATIVE_JOURNALIST = "investigative_journalist"
    CONTRARIAN_THINKER = "contrarian_thinker"
    DOMAIN_EXPERT = "domain_expert"
    CALIBRATION_SPECIALIST = "calibration_specialist"


PERSONA_SYSTEM_PROMPTS: dict[FishPersona, str] = {
    FishPersona.GEOPOLITICAL_ANALYST: (
        "You are a geopolitical analyst specializing in international relations, "
        "political risk, and macro trends. You assess prediction markets through "
        "the lens of power dynamics, alliances, sanctions, and institutional behavior. "
        "You think in scenarios and probability distributions, not point estimates."
    ),
    FishPersona.FINANCIAL_QUANT: (
        "You are a quantitative analyst with deep expertise in derivatives pricing, "
        "market microstructure, and statistical arbitrage. You evaluate prediction "
        "markets using implied probability, liquidity analysis, and historical base "
        "rates. You are precise with numbers and skeptical of narrative."
    ),
    FishPersona.BAYESIAN_STATISTICIAN: (
        "You are a Bayesian statistician. You start with base rates, update beliefs "
        "incrementally with evidence, and express uncertainty as probability distributions. "
        "You are explicit about priors, likelihoods, and posteriors. You flag when "
        "evidence is weak or when priors dominate."
    ),
    FishPersona.INVESTIGATIVE_JOURNALIST: (
        "You are an investigative journalist who digs beneath surface narratives. "
        "You look for hidden information, conflicts of interest, and signals that "
        "the consensus might be wrong. You value primary sources over secondary analysis. "
        "You are skeptical of official narratives."
    ),
    FishPersona.CONTRARIAN_THINKER: (
        "You are a contrarian thinker. Your job is to find reasons WHY the current "
        "market price might be wrong. You look for crowding, recency bias, anchoring, "
        "and neglected tail risks. If the market says 80%, you consider why it might "
        "be 60% or 95%. You are the devil's advocate of the swarm."
    ),
    FishPersona.DOMAIN_EXPERT: (
        "You are a domain expert who deeply understands the specific subject matter "
        "of the market you are analyzing. You bring specialized knowledge — whether "
        "scientific, technological, legal, or cultural — that general analysts miss. "
        "You identify technical details that shift probabilities."
    ),
    FishPersona.CALIBRATION_SPECIALIST: (
        "You are a calibration specialist trained in the art of precise probability "
        "estimation. You study your own track record, adjust for known biases "
        "(overconfidence, anchoring), and aim for well-calibrated probability "
        "judgments. You think about your Brier score and resolution."
    ),
}


# ---------------------------------------------------------------------------
# Analysis output
# ---------------------------------------------------------------------------

class FishAnalysis(BaseModel):
    """Structured output from a single Fish's analysis of a market."""

    fish_id: str
    persona: FishPersona
    market_id: str
    market_question: str

    # Core prediction
    probability: float = Field(ge=0.0, le=1.0, description="Predicted probability of YES outcome")
    confidence: float = Field(ge=0.0, le=1.0, description="Fish's confidence in its own estimate")

    # Reasoning chain (interpretable by humans)
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning chain for human review",
    )
    key_evidence: list[str] = Field(
        default_factory=list,
        description="Key evidence items that informed the prediction",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Identified risk factors or uncertainties",
    )

    # Cross-market signals (populated after MessageBus communication)
    related_markets: list[str] = Field(
        default_factory=list,
        description="IDs of markets this analysis is correlated with",
    )
    cross_market_adjustment: float = Field(
        default=0.0,
        description="Probability adjustment based on cross-market signals",
    )

    # Metadata
    timestamp: float = Field(default_factory=time.time)
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Fish agent
# ---------------------------------------------------------------------------

class Fish:
    """A single agent in the Mirofish swarm.

    Each Fish:
    1. Receives a market question + context (news, related markets)
    2. Applies its persona-specific analysis lens
    3. Produces a probability estimate with reasoning chain
    4. Shares insights via the MessageBus
    5. Optionally revises after receiving cross-market signals
    """

    def __init__(
        self,
        persona: FishPersona,
        llm_client: Any = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        fish_id: str | None = None,
    ) -> None:
        self.fish_id = fish_id or f"fish-{persona.value}-{uuid4().hex[:8]}"
        self.persona = persona
        self.system_prompt = PERSONA_SYSTEM_PROMPTS[persona]
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

        # Track historical performance for self-calibration
        self.history: list[FishAnalysis] = []
        self.brier_scores: list[float] = []

        logger.info(f"Fish spawned: {self.fish_id} ({persona.value})")

    async def analyze(
        self,
        market_id: str,
        market_question: str,
        market_description: str = "",
        current_price: float | None = None,
        news_context: list[str] | None = None,
        cross_market_signals: list[dict[str, Any]] | None = None,
    ) -> FishAnalysis:
        """Analyze a prediction market and produce a probability estimate.

        The Fish does NOT see the current market price during initial analysis
        (preserve independence, per PolySwarm methodology). Price is only used
        in the aggregation stage.
        """
        start_time = time.monotonic()

        # Build the analysis prompt
        prompt = self._build_prompt(
            market_question=market_question,
            market_description=market_description,
            news_context=news_context or [],
            cross_market_signals=cross_market_signals or [],
        )

        # Call LLM
        if self.llm_client is not None:
            response = await self._call_llm(prompt)
            analysis = self._parse_response(response, market_id, market_question)
        else:
            # Stub mode for testing — deterministic output
            analysis = self._stub_analysis(market_id, market_question)

        analysis.latency_ms = (time.monotonic() - start_time) * 1000
        analysis.model_used = self.model

        self.history.append(analysis)
        logger.info(
            f"[{self.fish_id}] Analyzed '{market_question[:50]}...' "
            f"→ P={analysis.probability:.3f} (conf={analysis.confidence:.2f})"
        )
        return analysis

    def _build_prompt(
        self,
        market_question: str,
        market_description: str,
        news_context: list[str],
        cross_market_signals: list[dict[str, Any]],
    ) -> str:
        """Build the analysis prompt for the LLM."""
        sections = [
            f"## Market Question\n{market_question}",
        ]

        if market_description:
            sections.append(f"## Market Description\n{market_description}")

        if news_context:
            news_block = "\n".join(f"- {item}" for item in news_context[:10])
            sections.append(f"## Recent News Context\n{news_block}")

        if cross_market_signals:
            signals_block = "\n".join(
                f"- Market '{s.get('question', '?')}': P={s.get('probability', '?')}"
                for s in cross_market_signals[:5]
            )
            sections.append(f"## Related Market Signals\n{signals_block}")

        sections.append(
            "## Your Task\n"
            "Analyze this prediction market and estimate the probability of the YES outcome.\n\n"
            "Respond in this exact format:\n"
            "PROBABILITY: <float between 0.0 and 1.0>\n"
            "CONFIDENCE: <float between 0.0 and 1.0>\n"
            "REASONING:\n"
            "1. <step 1>\n"
            "2. <step 2>\n"
            "...\n"
            "KEY_EVIDENCE:\n"
            "- <evidence 1>\n"
            "- <evidence 2>\n"
            "RISK_FACTORS:\n"
            "- <risk 1>\n"
            "- <risk 2>\n"
        )

        return "\n\n".join(sections)

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the LLM API. Supports Anthropic and OpenAI clients."""
        if hasattr(self.llm_client, "messages"):
            # Anthropic client
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "text": response.content[0].text,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        elif hasattr(self.llm_client, "chat"):
            # OpenAI client
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return {
                "text": response.choices[0].message.content,
                "tokens": response.usage.total_tokens if response.usage else 0,
            }
        else:
            raise ValueError(f"Unsupported LLM client type: {type(self.llm_client)}")

    def _parse_response(
        self, response: dict[str, Any], market_id: str, market_question: str
    ) -> FishAnalysis:
        """Parse structured LLM response into FishAnalysis."""
        text = response.get("text", "")
        tokens = response.get("tokens", 0)

        # Extract probability
        probability = 0.5  # default fallback
        confidence = 0.5
        reasoning_steps: list[str] = []
        key_evidence: list[str] = []
        risk_factors: list[str] = []

        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("PROBABILITY:"):
                try:
                    probability = float(line.split(":", 1)[1].strip())
                    probability = max(0.0, min(1.0, probability))
                except ValueError:
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                current_section = "reasoning"
            elif line.startswith("KEY_EVIDENCE:"):
                current_section = "evidence"
            elif line.startswith("RISK_FACTORS:"):
                current_section = "risk"
            elif line and current_section == "reasoning" and line[0].isdigit():
                reasoning_steps.append(line.lstrip("0123456789. "))
            elif line.startswith("- ") and current_section == "evidence":
                key_evidence.append(line[2:])
            elif line.startswith("- ") and current_section == "risk":
                risk_factors.append(line[2:])

        return FishAnalysis(
            fish_id=self.fish_id,
            persona=self.persona,
            market_id=market_id,
            market_question=market_question,
            probability=probability,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            key_evidence=key_evidence,
            risk_factors=risk_factors,
            tokens_used=tokens,
        )

    def _stub_analysis(self, market_id: str, market_question: str) -> FishAnalysis:
        """Generate a deterministic stub analysis for testing."""
        import hashlib

        # Deterministic but persona-varied probability
        seed = hashlib.sha256(
            f"{self.persona.value}:{market_question}".encode()
        ).hexdigest()
        base_prob = int(seed[:8], 16) / 0xFFFFFFFF  # 0.0 to 1.0

        return FishAnalysis(
            fish_id=self.fish_id,
            persona=self.persona,
            market_id=market_id,
            market_question=market_question,
            probability=round(base_prob, 4),
            confidence=round(0.5 + base_prob * 0.3, 4),
            reasoning_steps=[
                f"[STUB] {self.persona.value} analysis of '{market_question[:40]}...'",
                f"[STUB] Base probability from persona hash: {base_prob:.4f}",
            ],
            key_evidence=["[STUB] No live LLM — using deterministic hash"],
            risk_factors=["[STUB] This is a stub analysis, not a real prediction"],
        )

    def record_outcome(self, market_id: str, actual_outcome: float) -> float | None:
        """Record the actual outcome and compute Brier score for this Fish."""
        matching = [a for a in self.history if a.market_id == market_id]
        if not matching:
            return None

        latest = matching[-1]
        brier = (latest.probability - actual_outcome) ** 2
        self.brier_scores.append(brier)

        logger.info(
            f"[{self.fish_id}] Brier score for '{market_id}': {brier:.4f} "
            f"(predicted={latest.probability:.3f}, actual={actual_outcome})"
        )
        return brier

    @property
    def average_brier_score(self) -> float | None:
        """Running average Brier score across all resolved predictions."""
        if not self.brier_scores:
            return None
        return sum(self.brier_scores) / len(self.brier_scores)
