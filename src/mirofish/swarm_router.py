"""SwarmRouter — adaptive swarm configuration based on market characteristics.

Inspired by kyegomez/swarms SwarmRouter pattern:
Instead of running the same 9-Fish swarm on every market, route each market
to the optimal swarm configuration based on:
- Market category (politics, sports, crypto, geopolitics)
- Time horizon (hours, days, weeks, months)
- Expected difficulty (clear favorite vs. coin flip)
- Volume/liquidity (high-volume markets deserve more Fish)

This directly improves Sharpe by:
- Spending more compute on markets where we have edge
- Reducing compute waste on easy markets (3 Fish is enough for 95%+ favorites)
- Using specialized persona sets for different categories

Also implements multi-model competition (NoFx pattern):
Run Fish across Claude, Gemini, and Ollama for true model diversity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger


@dataclass
class SwarmConfig:
    """Configuration for a specific swarm invocation."""
    personas: list[str]
    model: str
    mode: str  # "cli", "ollama", "gemini"
    max_rounds: int
    max_concurrent: int
    extremize: float
    use_researcher: bool
    researcher_model: str


# ── Category-specific persona sets ────────────────────────────────────

POLITICS_PERSONAS = [
    "base_rate_anchor",
    "decomposer",
    "inside_view",
    "institutional_analyst",
    "contrarian",
    "temporal_analyst",
    "premortem",
    "calibrator",
    "bayesian_updater",
]

SPORTS_PERSONAS = [
    "base_rate_anchor",
    "inside_view",     # matchup data, injuries
    "decomposer",
    "contrarian",
    "calibrator",
]

CRYPTO_PERSONAS = [
    "base_rate_anchor",
    "temporal_analyst",  # momentum, deadlines
    "contrarian",
    "inside_view",       # on-chain data, whale movements
    "premortem",
    "calibrator",
    "bayesian_updater",
]

GEOPOLITICS_PERSONAS = [
    "base_rate_anchor",
    "institutional_analyst",  # state actor incentives
    "decomposer",
    "inside_view",
    "contrarian",
    "premortem",
    "temporal_analyst",
    "calibrator",
    "bayesian_updater",
]

FAST_PERSONAS = [
    "base_rate_anchor",
    "calibrator",
    "contrarian",
]


def classify_market(
    question: str,
    description: str = "",
    volume_usd: float = 0,
    outcomes: list[str] | None = None,
) -> dict[str, Any]:
    """Classify a market by category, difficulty, and time horizon.

    Uses keyword heuristics. Fast, no LLM calls needed.
    """
    q = question.lower()
    d = description.lower()
    text = f"{q} {d}"

    # Category — score-based to handle multi-topic markets correctly.
    # Each keyword match adds 1 point to its category. Highest score wins.
    # This avoids priority-order collisions (e.g., "Ukraine Bitcoin" scores
    # in both crypto and geopolitics; the one with more matches wins).
    scores: dict[str, int] = {
        "politics": 0, "crypto": 0, "sports": 0,
        "geopolitics": 0, "economics": 0,
    }

    politics_kw = ["election", "president", "governor", "senator", "congress",
                    "vote", "poll", "democrat", "republican", "mayor",
                    "cabinet", "nominate", "impeach", "partisan"]
    crypto_kw = ["bitcoin", "btc", "eth", "crypto", "token", "solana",
                  "defi", "nft", "blockchain", "altcoin"]
    sports_kw = ["finals", "championship", "super bowl", "nba", "nfl",
                  "mlb", "premier league", "la liga", "champions league",
                  "world cup", "mvp", "vs.", "match winner"]
    geopolitics_kw = ["ceasefire", "invasion", "war", "military", "sanctions",
                       "nato", "regime", "supreme leader", "nuclear", "iran",
                       "ukraine", "russia", "troops", "strike", "diplomacy"]
    economics_kw = ["fed ", "interest rate", "inflation", "gdp",
                     "unemployment", "recession", "tariff", "central bank",
                     "monetary policy", "basis points"]

    for kw in politics_kw:
        if kw in text:
            scores["politics"] += 1
    for kw in crypto_kw:
        if kw in text:
            scores["crypto"] += 1
    for kw in sports_kw:
        if kw in text:
            scores["sports"] += 1
    for kw in geopolitics_kw:
        if kw in text:
            scores["geopolitics"] += 1
    for kw in economics_kw:
        if kw in text:
            scores["economics"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        category = "general"
    else:
        category = max(scores, key=scores.get)

    # Difficulty: head-to-head is harder; deadline questions medium;
    # volume also factors in (high-volume markets attract more
    # informed traders, making them harder to beat)
    if "vs" in q or "vs." in q:
        difficulty = "hard"
    elif volume_usd >= 10_000_000:
        difficulty = "hard"
    elif volume_usd < 50_000:
        difficulty = "easy"
    else:
        difficulty = "medium"

    # Volume tier
    if volume_usd >= 1_000_000:
        volume_tier = "high"
    elif volume_usd >= 100_000:
        volume_tier = "medium"
    else:
        volume_tier = "low"

    return {
        "category": category,
        "difficulty": difficulty,
        "volume_tier": volume_tier,
    }


def route_swarm(
    question: str,
    description: str = "",
    volume_usd: float = 0,
    outcomes: list[str] | None = None,
    available_modes: list[str] | None = None,
) -> SwarmConfig:
    """Route a market to the optimal swarm configuration.

    Returns a SwarmConfig tailored to the market's characteristics.
    """
    if available_modes is None:
        available_modes = ["cli"]

    classification = classify_market(question, description, volume_usd, outcomes)
    cat = classification["category"]
    difficulty = classification["difficulty"]
    vol_tier = classification["volume_tier"]

    # Select personas by category
    if cat == "politics":
        personas = POLITICS_PERSONAS
    elif cat == "sports":
        personas = SPORTS_PERSONAS
    elif cat == "crypto":
        personas = CRYPTO_PERSONAS
    elif cat in ("geopolitics",):
        personas = GEOPOLITICS_PERSONAS
    elif cat == "economics":
        personas = GEOPOLITICS_PERSONAS  # similar structure
    else:
        personas = POLITICS_PERSONAS  # full set for unknown

    # Adjust Fish count by difficulty and volume
    if difficulty == "hard" or vol_tier == "high":
        # Full swarm + researcher for important/hard markets
        use_researcher = True
        max_rounds = 3
    elif difficulty == "easy" or vol_tier == "low":
        # Minimal swarm for easy/low-volume markets
        personas = FAST_PERSONAS
        use_researcher = False
        max_rounds = 1
    else:
        use_researcher = True
        max_rounds = 2

    # Select model and mode
    mode = available_modes[0]
    model = "haiku" if mode == "cli" else "llama3.3"
    researcher_model = "sonnet" if mode == "cli" else model

    # Extremization: higher for categories where we have more data
    extremize_map = {
        "politics": 1.8,    # most data, extremize more
        "sports": 1.3,      # moderate data
        "crypto": 1.5,
        "geopolitics": 1.5,
        "economics": 1.5,
        "general": 1.2,     # least data, extremize less
    }
    extremize = extremize_map.get(cat, 1.5)

    config = SwarmConfig(
        personas=personas,
        model=model,
        mode=mode,
        max_rounds=max_rounds,
        max_concurrent=3,
        extremize=extremize,
        use_researcher=use_researcher,
        researcher_model=researcher_model,
    )

    logger.info(
        f"SwarmRouter: [{cat}/{difficulty}/{vol_tier}] "
        f"→ {len(personas)} Fish, {max_rounds} rounds, "
        f"researcher={'ON' if use_researcher else 'OFF'}, "
        f"extremize={extremize}"
    )
    return config


# ── Multi-model competition (NoFx pattern) ────────────────────────────

@dataclass
class ModelCompetition:
    """Track per-model accuracy for multi-model competition.

    When multiple backends are available (CLI + Gemini + Ollama),
    run Fish across all of them. Track which model produces the
    best-calibrated predictions. Dynamically shift weight toward
    the best-performing model.
    """
    model_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, model: str, brier: float) -> None:
        """Record a Brier score for a model."""
        if model not in self.model_stats:
            self.model_stats[model] = {
                "total_brier": 0.0,
                "n_predictions": 0,
                "consecutive_misses": 0,
                "safety_mode": False,
            }
        stats = self.model_stats[model]
        stats["total_brier"] += brier
        stats["n_predictions"] += 1

        # NoFx consecutive failure detection
        if brier > 0.25:  # wrong direction
            stats["consecutive_misses"] += 1
        else:
            stats["consecutive_misses"] = 0

        # Safety mode: activate after 3 consecutive misses
        if stats["consecutive_misses"] >= 3:
            stats["safety_mode"] = True
            logger.warning(
                f"Model [{model}] entered SAFETY MODE "
                f"({stats['consecutive_misses']} consecutive misses)"
            )
        elif stats["consecutive_misses"] == 0 and stats["safety_mode"]:
            stats["safety_mode"] = False
            logger.info(f"Model [{model}] exited safety mode")

    def get_weight(self, model: str) -> float:
        """Get aggregation weight for a model based on track record."""
        if model not in self.model_stats:
            return 1.0

        stats = self.model_stats[model]
        if stats["safety_mode"]:
            return 0.25  # heavily downweight in safety mode

        if stats["n_predictions"] < 5:
            return 1.0  # not enough data to judge

        avg_brier = stats["total_brier"] / stats["n_predictions"]
        # Weight inversely proportional to Brier (lower Brier = higher weight)
        # Normalize: Brier 0.05 → weight 1.9, Brier 0.25 → weight 1.0
        return max(0.3, 2.0 - avg_brier * 4)

    def summary(self) -> dict[str, Any]:
        result = {}
        for model, stats in self.model_stats.items():
            n = stats["n_predictions"]
            avg = stats["total_brier"] / n if n > 0 else 0
            result[model] = {
                "avg_brier": round(avg, 4),
                "n_predictions": n,
                "consecutive_misses": stats["consecutive_misses"],
                "safety_mode": stats["safety_mode"],
                "weight": round(self.get_weight(model), 3),
            }
        return result
