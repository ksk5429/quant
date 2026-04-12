"""LLM-powered Fish agent — zero-cost architecture (v2).

Four execution modes (all free):
1. CLI mode: calls `claude -p` (Claude Code CLI piped mode, Max subscription)
2. FILE mode: generates prompt files for manual Claude Code sessions
3. OLLAMA mode: calls local Ollama models (llama3, mistral, qwen, etc.)
4. GEMINI mode: calls Google Gemini free tier (15 RPM, 1500 RPD)

v2 upgrades over v1:
- 9 orthogonal personas designed for maximum structural diversity
- Extremized aggregation (Tetlock's recalibration: push away from 0.5)
- Disagreement-aware confidence (high spread → lower effective confidence)
- Improved prompt: forces step-by-step decomposition before probability
- Trimmed-mean aggregation option (drop highest/lowest, reduce outlier impact)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import sys

import numpy as np
from loguru import logger

from src.utils.cli import find_claude_binary as _find_claude_binary


# ═══════════════════════════════════════════════════════════════════════
# PERSONAS — designed for ORTHOGONAL reasoning, not just different knowledge
#
# Design principle from Schoenegger et al. (2024, Science Advances):
#   "Diversity of reasoning frameworks > capability of individual models"
#
# Each persona encodes a structurally different DECOMPOSITION STRATEGY.
# Two Fish given the same evidence should arrive at different probability
# estimates because they weight evidence differently, not because they
# hallucinate different facts.
# ═══════════════════════════════════════════════════════════════════════

PERSONA_PROMPTS: dict[str, str] = {

    # ── ANCHOR FISH: starts from reference class, adjusts minimally ───
    "base_rate_anchor": (
        "You are a reference-class forecaster. For every question, your FIRST "
        "step is to identify the most relevant reference class and compute its "
        "historical base rate. For example, how often do incumbent parties win "
        "reelection (about 55 percent), or how often does the top seed win the "
        "final (about 65 percent). "
        "You adjust from the base rate ONLY when specific evidence is strong "
        "enough to justify a shift. Your adjustments are small (rarely more "
        "than 15 percentage points from the base rate). You are the anchor "
        "of the swarm. When you are genuinely uncertain about the base rate, "
        "you say so and default toward 50%."
    ),

    # ── DECOMPOSER: breaks question into sub-probabilities ────────────
    "decomposer": (
        "You are a probability decomposer. You NEVER estimate a probability "
        "directly. Instead, you break the question into 2-4 independent "
        "sub-questions whose joint probability answers the original question. "
        "Example: 'Will X win the election?' → P(X wins primary) × P(X wins "
        "general | X wins primary). You estimate each sub-probability "
        "separately, then multiply (or apply the appropriate conditional "
        "logic). Show your decomposition explicitly. This method catches "
        "cases where a seemingly likely outcome depends on an unlikely "
        "prerequisite."
    ),

    # ── INSIDE VIEW: deep domain-specific evidence ────────────────────
    "inside_view": (
        "You are a domain specialist who focuses exclusively on the specific "
        "subject matter of each question. You look for the most informative "
        "single piece of evidence that most people would miss. For sports: "
        "injury reports, head-to-head records, venue effects. For politics: "
        "polling crosstabs, early voting data, redistricting. For economics: "
        "leading indicators, yield curve signals, central bank minutes. You "
        "weight this specific evidence heavily, even if it contradicts the "
        "general narrative. You are the swarm's specialist."
    ),

    # ── CONTRARIAN: systematically argues for the less popular outcome ─
    "contrarian": (
        "You are a professional contrarian. Before forming any judgment, you "
        "identify what the CONSENSUS view is likely to be on this question. "
        "Then you construct the strongest possible case for the opposite "
        "outcome. You look for: anchoring bias (people stuck on a salient "
        "number), availability bias (recent events overweighted), herding "
        "(people copying each other rather than thinking independently), "
        "and neglected tail scenarios. If the consensus says 80%, you explore "
        "why the true probability might be 55% or 95%. Your job is NOT to "
        "always disagree — it is to stress-test the consensus and report "
        "your honest posterior AFTER considering the contrarian case."
    ),

    # ── TEMPORAL: focuses on timing, deadlines, momentum ──────────────
    "temporal_analyst": (
        "You are a timing and momentum analyst. You focus on WHEN things "
        "happen, not just whether they happen. You ask: How much time "
        "remains before the deadline? What is the current trajectory? Is "
        "momentum accelerating or decelerating? What fraction of the "
        "necessary conditions are already met? You use analogies to similar "
        "situations with known timelines. For questions with explicit "
        "deadlines (by date X), you estimate the daily hazard rate and "
        "compute the cumulative probability over the remaining window. "
        "You are sensitive to the difference between 'will ever happen' "
        "and 'will happen by date X'."
    ),

    # ── INSTITUTIONAL: analyzes organizational incentives ─────────────
    "institutional_analyst": (
        "You are an institutional analyst who focuses on the incentives, "
        "constraints, and decision-making processes of the organizations "
        "and individuals involved. You ask: Who makes this decision? What "
        "are their incentives? What institutional barriers exist? What "
        "is the default action if no one intervenes? You understand that "
        "large organizations are slow, risk-averse, and path-dependent. "
        "You weight institutional inertia heavily — the status quo is "
        "usually the best predictor unless there is a forcing function "
        "for change. For political questions, you analyze the decision "
        "maker's coalition and career incentives."
    ),

    # ── PREMORTEM: imagines failure modes and surprise scenarios ───────
    "premortem": (
        "You are a premortem analyst. For the MOST LIKELY outcome, you "
        "imagine it is one year later and that outcome DID NOT occur. "
        "Then you work backward: what went wrong? What surprise derailed "
        "it? You generate 2-3 specific, plausible failure scenarios. You "
        "then estimate the combined probability of these failure modes. "
        "This technique (from Gary Klein's research) counteracts the "
        "planning fallacy and overconfidence. You are particularly "
        "valuable for questions where the obvious answer feels too easy."
    ),

    # ── CALIBRATOR: Tetlock superforecaster methodology ───────────────
    "calibrator": (
        "You are a superforecaster trained in Philip Tetlock's methodology. "
        "You follow a strict protocol: (1) Start with the outside view "
        "(base rate). (2) Identify 2-3 pieces of specific evidence. "
        "(3) Update incrementally from the base rate — each update should "
        "be small unless evidence is overwhelming. (4) Check for known "
        "biases: Are you anchored on a salient number? Are you "
        "overweighting recent events? Is your probability suspiciously "
        "round (0.50, 0.70, 0.90)? Adjust if so. (5) Report your final "
        "probability with appropriate precision (use increments of 0.05 "
        "for uncertain estimates, 0.02 for confident ones). You value "
        "calibration over resolution — being roughly right beats being "
        "precisely wrong."
    ),

    # ── BAYESIAN UPDATER: explicit prior + likelihood reasoning ───────
    "bayesian_updater": (
        "You are a Bayesian statistician. You reason in three explicit "
        "steps: (1) STATE YOUR PRIOR: What is your probability estimate "
        "before looking at any question-specific evidence? This should "
        "come from the reference class or your general knowledge. (2) "
        "IDENTIFY EVIDENCE: List 2-3 pieces of evidence relevant to this "
        "specific question. For each, estimate the likelihood ratio: how "
        "much more likely is this evidence under the YES outcome vs. the "
        "NO outcome? (3) UPDATE: Apply Bayes' rule by multiplying your "
        "prior odds by the likelihood ratios to get posterior odds, then "
        "convert back to probability. Show your work. Even approximate "
        "Bayesian reasoning produces better-calibrated estimates than "
        "intuitive judgment."
    ),
}

# Default swarm composition — 9 Fish for maximum diversity
DEFAULT_PERSONAS = list(PERSONA_PROMPTS.keys())


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FishPrediction:
    """A single Fish agent's prediction for one market."""
    persona: str
    probability: float
    confidence: float
    reasoning: str = ""
    model: str = ""
    tokens_used: int = 0
    elapsed_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING (v2)
# ═══════════════════════════════════════════════════════════════════════

def build_fish_prompt(
    question: str,
    description: str = "",
    outcomes: list[str] | None = None,
    persona: str = "calibrator",
) -> str:
    """Build a time-safe, price-withheld prompt that forces decomposition."""
    if outcomes is None:
        outcomes = ["Yes", "No"]

    system = PERSONA_PROMPTS.get(
        persona,
        f"You are a {persona.replace('_', ' ')} analyzing prediction markets.",
    )

    desc_block = ""
    if description:
        desc_block = f"\nCONTEXT:\n{description[:1200]}\n"

    # v2 prompt forces step-by-step BEFORE the probability.
    # This reduces anchoring and forces genuine reasoning.
    return f"""{system}

QUESTION: {question}
{desc_block}
OUTCOMES: {', '.join(outcomes)}
TARGET (estimate probability for this): "{outcomes[0]}"

INSTRUCTIONS:
Think step by step. First reason about the question using your specific
analytical framework. Then estimate the probability.

You MUST respond with ONLY this JSON (no other text, no markdown):
{{"steps": ["<step 1>", "<step 2>", "<step 3>"], "probability": <0.01-0.99>, "confidence": <0.1-1.0>, "reasoning": "<1-2 sentences summarizing your conclusion>"}}

CONSTRAINTS:
- You do NOT know the current market price. Form independent judgment.
- Do NOT default to 0.50 unless you genuinely have no information.
- Your confidence = how certain you are in YOUR estimate (not in the outcome).
- Avoid round numbers (0.50, 0.70, 0.80) unless truly warranted.
- Be precise: use 0.23 not 0.20, use 0.67 not 0.70."""


def parse_fish_response(text: str, persona: str = "") -> FishPrediction:
    """Parse LLM response into FishPrediction. Handles multiple formats."""
    # Try direct JSON
    try:
        data = json.loads(text.strip())
        return _validated_prediction(data, persona)
    except (json.JSONDecodeError, ValueError):
        pass

    # JSON in code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return _validated_prediction(data, persona)
        except (json.JSONDecodeError, ValueError):
            pass

    # Any JSON object with probability
    match = re.search(r'\{[^{}]*"probability"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return _validated_prediction(data, persona)
        except (json.JSONDecodeError, ValueError):
            pass

    # Regex fallback
    prob_match = re.search(r'"probability"\s*:\s*([\d.]+)', text)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    prob = float(prob_match.group(1)) if prob_match else 0.5
    conf = float(conf_match.group(1)) if conf_match else 0.3

    logger.warning(f"Fish [{persona}] fell back to regex parsing")
    return FishPrediction(
        persona=persona,
        probability=max(0.01, min(0.99, prob)),
        confidence=max(0.1, min(1.0, conf)),
        reasoning=text[:300],
    )


def _validated_prediction(data: dict, persona: str) -> FishPrediction:
    prob = float(data.get("probability", 0.5))
    conf = float(data.get("confidence", 0.5))
    reasoning = str(data.get("reasoning", ""))
    steps = data.get("steps", [])
    if steps and not reasoning:
        reasoning = "; ".join(str(s) for s in steps[:3])
    return FishPrediction(
        persona=persona,
        probability=max(0.01, min(0.99, prob)),
        confidence=max(0.1, min(1.0, conf)),
        reasoning=reasoning[:500],
    )


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATION ENGINE (v2)
#
# Three improvements over v1 naive weighted average:
# 1. Trimmed mean: drop the single most extreme Fish to reduce outliers
# 2. Extremization: Tetlock's finding that aggregated probabilities
#    should be pushed AWAY from 0.5 (crowds are underconfident after avg)
# 3. Disagreement penalty: high spread → reduce effective confidence
# ═══════════════════════════════════════════════════════════════════════

def aggregate_predictions(
    predictions: list[FishPrediction],
    extremize: float = 1.5,
    trim: bool = True,
    disagreement_penalty: bool = True,
) -> dict[str, Any]:
    """Advanced aggregation of Fish predictions.

    Args:
        predictions: list of FishPrediction from the swarm.
        extremize: Extremization factor (>1 pushes away from 0.5).
            1.0 = no extremization. 1.5 = moderate (Tetlock's recommendation).
            2.0 = strong. Set to 1.0 to disable.
        trim: If True, drop the most extreme Fish before averaging.
        disagreement_penalty: If True, reduce confidence when Fish disagree.

    Returns:
        Dict with raw_probability, fish_predictions, spread, etc.
    """
    if not predictions:
        return {
            "raw_probability": 0.5, "fish_predictions": [],
            "spread": 0.0, "mean_confidence": 0.0, "n_fish": 0,
            "extremized_probability": 0.5, "disagreement_flag": False,
            "swarm_healthy": False, "n_failed": 0,
        }

    # ── Health check: detect silent total failure ──
    n_failed = sum(
        1 for p in predictions
        if p.confidence <= 0.1 and ("error" in p.reasoning.lower() or "timeout" in p.reasoning.lower())
    )
    swarm_healthy = n_failed < len(predictions) * 0.5
    if not swarm_healthy:
        logger.error(
            f"SWARM FAILURE: {n_failed}/{len(predictions)} Fish failed. "
            f"Results are unreliable. Do NOT trade on this prediction."
        )

    probs = np.array([p.probability for p in predictions])
    confs = np.array([p.confidence for p in predictions])

    # ── Step 1: Trimmed weighted mean ──
    if trim and len(probs) >= 5:
        # Remove the single Fish furthest from the median
        median = np.median(probs)
        distances = np.abs(probs - median)
        keep_mask = np.ones(len(probs), dtype=bool)
        keep_mask[np.argmax(distances)] = False
        trim_probs = probs[keep_mask]
        trim_confs = confs[keep_mask]
    else:
        trim_probs = probs
        trim_confs = confs

    # Confidence-weighted average
    total_w = trim_confs.sum()
    if total_w > 1e-8:
        raw_prob = float(np.sum(trim_probs * trim_confs) / total_w)
    else:
        raw_prob = float(np.mean(trim_probs))
    raw_prob = max(0.01, min(0.99, raw_prob))

    # ── Step 2: Disagreement metrics (compute BEFORE extremization) ──
    spread = float(np.max(trim_probs) - np.min(trim_probs))
    std_dev = float(np.std(trim_probs))
    mean_conf = float(np.mean(trim_confs))

    # ── Step 3: Asymmetric extremization (Tetlock recalibration) ──
    # Key insight: extremize LESS when Fish disagree (high spread).
    # Extremization assumes the swarm is directionally correct but
    # underconfident. When Fish disagree, the direction itself is
    # uncertain — extremizing amplifies the wrong direction for half
    # the Fish. Decay extremization as disagreement increases.
    if extremize != 1.0 and 0.01 < raw_prob < 0.99:
        # Decay: at spread=0 use full extremize, at spread>=0.5 use 1.0 (none)
        effective_extremize = max(1.0, extremize * max(0.0, 1.0 - spread / 0.5))
        log_odds = math.log(raw_prob / (1 - raw_prob))
        scaled_log_odds = log_odds * effective_extremize
        ext_prob = 1.0 / (1.0 + math.exp(-scaled_log_odds))
        ext_prob = max(0.01, min(0.99, ext_prob))
    else:
        ext_prob = raw_prob

    # Flag high disagreement (spread > 0.30 or std > 0.15)
    disagreement_flag = spread > 0.30 or std_dev > 0.15

    # Penalize confidence if Fish disagree heavily
    if disagreement_penalty and disagreement_flag:
        # Scale confidence down proportional to disagreement
        penalty = max(0.3, 1.0 - std_dev)  # floor at 0.3
        effective_confidence = mean_conf * penalty
    else:
        effective_confidence = mean_conf

    return {
        "raw_probability": round(raw_prob, 4),
        "extremized_probability": round(ext_prob, 4),
        "fish_predictions": predictions,
        "spread": round(spread, 4),
        "std_dev": round(std_dev, 4),
        "mean_confidence": round(mean_conf, 4),
        "effective_confidence": round(effective_confidence, 4),
        "n_fish": len(predictions),
        "n_trimmed": len(predictions) - len(trim_probs),
        "disagreement_flag": disagreement_flag,
        "extremization_factor": extremize,
        "swarm_healthy": swarm_healthy,
        "n_failed": n_failed,
    }


# ═══════════════════════════════════════════════════════════════════════
# BACKENDS
# ═══════════════════════════════════════════════════════════════════════

# ─── Claude CLI (free via Max subscription) ───────────────────────────

class CLIFish:
    """Fish via `claude -p`. Free, automated, uses Max subscription."""

    CLAUDE_BIN = _find_claude_binary()

    def __init__(self, persona: str, model: str = "haiku", claude_bin: str = "") -> None:
        self.persona = persona
        self.model = model
        self.claude_bin = claude_bin or self.CLAUDE_BIN

    async def predict(
        self, question: str, description: str = "", outcomes: list[str] | None = None,
    ) -> FishPrediction:
        prompt = build_fish_prompt(question, description, outcomes, self.persona)
        t0 = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                self.claude_bin, "-p",
                "--model", self.model,
                "--output-format", "text",
                "--no-session-persistence",
                "--tools", "",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "CLAUDECODE": ""},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")), timeout=120.0,
            )
            raw_text = stdout.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")[:200]
                logger.error(f"CLI Fish [{self.persona}] exit {proc.returncode}: {err}")
                return FishPrediction(persona=self.persona, probability=0.5, confidence=0.1,
                                      reasoning=f"CLI error: {err}")

            pred = parse_fish_response(raw_text, self.persona)
            pred.model = f"cli-{self.model}"
            pred.elapsed_s = round(time.monotonic() - t0, 1)
            return pred

        except asyncio.TimeoutError:
            logger.error(f"CLI Fish [{self.persona}] timeout 120s")
            return FishPrediction(persona=self.persona, probability=0.5, confidence=0.1,
                                  reasoning="CLI timeout", elapsed_s=120.0)
        except Exception as e:
            logger.error(f"CLI Fish [{self.persona}] error: {e}")
            return FishPrediction(persona=self.persona, probability=0.5, confidence=0.1,
                                  reasoning=f"CLI error: {e}")


# ─── Ollama (free, local GPU) ─────────────────────────────────────────

class OllamaFish:
    """Fish via local Ollama. Free, requires local model downloaded."""

    def __init__(self, persona: str, model: str = "llama3.3",
                 base_url: str = "http://localhost:11434") -> None:
        self.persona = persona
        self.model = model
        self.base_url = base_url

    async def predict(self, question: str, description: str = "",
                      outcomes: list[str] | None = None) -> FishPrediction:
        import httpx
        prompt = build_fish_prompt(question, description, outcomes, self.persona)
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{self.base_url}/api/generate", json={
                    "model": self.model, "prompt": prompt,
                    "stream": False, "options": {"temperature": 0.7},
                })
                r.raise_for_status()
                raw = r.json().get("response", "")
            pred = parse_fish_response(raw, self.persona)
            pred.model = f"ollama-{self.model}"
            pred.elapsed_s = round(time.monotonic() - t0, 1)
            return pred
        except Exception as e:
            logger.error(f"Ollama Fish [{self.persona}] error: {e}")
            return FishPrediction(persona=self.persona, probability=0.5,
                                  confidence=0.1, reasoning=f"Ollama error: {e}")


# ─── Gemini free tier (15 RPM, 1500 RPD) ──────────────────────────────

class GeminiFish:
    """Fish via Google Gemini free tier."""

    def __init__(self, persona: str, api_key: str = "",
                 model: str = "gemini-2.0-flash") -> None:
        self.persona = persona
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model

    async def predict(self, question: str, description: str = "",
                      outcomes: list[str] | None = None) -> FishPrediction:
        import httpx
        prompt = build_fish_prompt(question, description, outcomes, self.persona)
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512},
                })
                r.raise_for_status()
                raw = (r.json().get("candidates", [{}])[0]
                       .get("content", {}).get("parts", [{}])[0].get("text", ""))
            pred = parse_fish_response(raw, self.persona)
            pred.model = f"gemini-{self.model}"
            pred.elapsed_s = round(time.monotonic() - t0, 1)
            return pred
        except Exception as e:
            logger.error(f"Gemini Fish [{self.persona}] error: {e}")
            return FishPrediction(persona=self.persona, probability=0.5,
                                  confidence=0.1, reasoning=f"Gemini error: {e}")


# ─── File-based batch (manual, zero dependencies) ────────────────────

class FileBatchGenerator:
    """Generates prompt files for manual Claude Code sessions."""

    def __init__(self, output_dir: str | Path = "shared_state/fish_tasks") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_batch(self, markets: list[dict[str, Any]],
                       personas: list[str] | None = None) -> list[Path]:
        if personas is None:
            personas = DEFAULT_PERSONAS
        generated = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for mkt in markets:
            mid = mkt.get("id", "unknown")
            for persona in personas:
                prompt = build_fish_prompt(
                    mkt.get("question", ""), mkt.get("description", ""),
                    mkt.get("outcomes", ["Yes", "No"]), persona)
                fp = self.output_dir / f"{ts}_{mid}_{persona}.txt"
                fp.write_text(prompt, encoding="utf-8")
                generated.append(fp)
        logger.info(f"Generated {len(generated)} prompts in {self.output_dir}")
        return generated

    def collect_results(self, analyses_dir: str | Path = "shared_state/analyses"
                        ) -> dict[str, list[FishPrediction]]:
        path = Path(analyses_dir)
        if not path.exists():
            return {}
        results: dict[str, list[FishPrediction]] = {}
        for fp in sorted(path.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                mid = data.get("market_id", fp.stem)
                pred = FishPrediction(
                    persona=data.get("persona", "unknown"),
                    probability=max(0.01, min(0.99, float(data.get("probability", 0.5)))),
                    confidence=max(0.1, min(1.0, float(data.get("confidence", 0.5)))),
                    reasoning=data.get("reasoning", ""),
                    model=data.get("model", "claude-code"),
                )
                results.setdefault(mid, []).append(pred)
            except Exception as e:
                logger.warning(f"Parse error {fp.name}: {e}")
        return results


# ═══════════════════════════════════════════════════════════════════════
# FISH SWARM (v2)
# ═══════════════════════════════════════════════════════════════════════

class FishSwarm:
    """Unified swarm supporting all four free backends.

    v2 improvements:
    - 9 orthogonal personas by default
    - Extremized aggregation (factor=1.5)
    - Trimmed mean (drop most extreme Fish)
    - Disagreement-aware confidence penalty
    - Per-market timing and token tracking
    """

    def __init__(
        self,
        mode: Literal["cli", "ollama", "gemini", "file"] = "cli",
        personas: list[str] | None = None,
        model: str = "",
        api_key: str = "",
        max_concurrent: int = 3,
        extremize: float = 1.5,
        trim: bool = True,
        ollama_url: str = "http://localhost:11434",
        output_dir: str = "shared_state/fish_tasks",
        claude_bin: str = "",
    ) -> None:
        self.mode = mode
        self.personas = personas or DEFAULT_PERSONAS
        self.max_concurrent = max_concurrent
        self.extremize = extremize
        self.trim = trim
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._total_calls = 0
        self._total_elapsed = 0.0

        fish_cls: type
        fish_kwargs: dict[str, Any] = {}

        if mode == "cli":
            self.model = model or "haiku"
            fish_cls = CLIFish
            fish_kwargs = {"model": self.model, "claude_bin": claude_bin}
        elif mode == "ollama":
            self.model = model or "llama3.3"
            fish_cls = OllamaFish
            fish_kwargs = {"model": self.model, "base_url": ollama_url}
        elif mode == "gemini":
            self.model = model or "gemini-2.0-flash"
            fish_cls = GeminiFish
            fish_kwargs = {"api_key": api_key, "model": self.model}
        elif mode == "file":
            self.model = "manual"
            self._batch_gen = FileBatchGenerator(output_dir=output_dir)
            self._fish: list = []
            logger.info(f"FishSwarm: mode=file, personas={len(self.personas)}")
            return
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self._fish = [fish_cls(persona=p, **fish_kwargs) for p in self.personas]
        logger.info(
            f"FishSwarm: mode={mode}, model={self.model}, "
            f"fish={len(self._fish)}, concurrent={max_concurrent}, "
            f"extremize={extremize}, trim={trim}"
        )

    async def predict(
        self,
        question: str,
        description: str = "",
        outcomes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run full swarm on a single market. Returns aggregated result."""
        if self.mode == "file":
            raise RuntimeError("Use generate_batch() + collect_results() in file mode")

        t0 = time.monotonic()

        async def run_one(fish) -> FishPrediction:
            async with self._semaphore:
                return await fish.predict(question, description, outcomes)

        tasks = [run_one(f) for f in self._fish]
        predictions = await asyncio.gather(*tasks)
        self._total_calls += len(predictions)

        result = aggregate_predictions(
            predictions, extremize=self.extremize, trim=self.trim,
        )

        elapsed = time.monotonic() - t0
        self._total_elapsed += elapsed
        result["total_calls"] = self._total_calls
        result["wall_time_s"] = round(elapsed, 1)

        return result

    def generate_batch(self, markets: list[dict[str, Any]]) -> list[Path]:
        gen = getattr(self, "_batch_gen", FileBatchGenerator())
        return gen.generate_batch(markets, self.personas)

    def collect_results(self, d: str = "shared_state/analyses") -> dict[str, list[FishPrediction]]:
        gen = getattr(self, "_batch_gen", FileBatchGenerator())
        return gen.collect_results(d)
