"""Researcher Fish — context gathering before swarm prediction.

Inspired by MiroFish (666ghj) knowledge graph approach:
Instead of Fish analyzing cold, the Researcher first gathers
relevant context (news, facts, historical precedent) and distributes
it to all Fish. Better context → better predictions → higher profit.

The Researcher runs ONCE per market, then its output is injected into
every Fish prompt as "RESEARCH BRIEFING". This is the single highest-ROI
improvement: a 5-minute research step that improves all 9 Fish predictions.

Usage:
    researcher = ResearcherFish(model="sonnet")
    context = await researcher.research(question, description)
    # context is then injected into each Fish prompt
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.utils.cli import find_claude_binary as _find_claude_binary


@dataclass
class ResearchContext:
    """Structured context gathered by the Researcher Fish."""
    question: str
    base_rate: str          # reference class and base rate estimate
    key_facts: list[str]    # 3-5 most relevant facts
    recent_developments: str  # what has changed recently
    resolution_criteria: str  # how exactly this market resolves
    time_remaining: str     # deadline analysis
    contrarian_case: str    # strongest case for the less likely outcome
    raw_response: str = ""
    elapsed_s: float = 0.0

    def to_briefing(self) -> str:
        """Format as a research briefing to inject into Fish prompts."""
        facts_block = "\n".join(f"  - {f}" for f in self.key_facts[:5])
        return f"""RESEARCH BRIEFING (gathered independently, treat as background):
Base rate: {self.base_rate}
Key facts:
{facts_block}
Recent developments: {self.recent_developments}
Resolution criteria: {self.resolution_criteria}
Time remaining: {self.time_remaining}
Strongest contrarian case: {self.contrarian_case}"""


RESEARCHER_PROMPT = """You are a research analyst preparing a briefing for a team of forecasters.
Your job is to gather the most relevant factual context for a prediction market question.
You do NOT make predictions yourself. You provide FACTS that help others predict.

QUESTION: {question}

CONTEXT:
{description}

Analyze this question and provide a structured research briefing.
Respond with ONLY this JSON (no other text):
{{
  "base_rate": "<What is the reference class? How often does this type of event occur historically? Be specific with numbers.>",
  "key_facts": ["<fact 1>", "<fact 2>", "<fact 3>", "<fact 4>", "<fact 5>"],
  "recent_developments": "<What has changed recently that is relevant? Any breaking news or shifts?>",
  "resolution_criteria": "<How EXACTLY does this market resolve? What is the precise condition?>",
  "time_remaining": "<How much time remains before resolution? Is the deadline tight or distant?>",
  "contrarian_case": "<What is the strongest argument for the LESS LIKELY outcome?>"
}}

Rules:
- Focus on VERIFIABLE FACTS, not opinions.
- Include specific numbers, dates, and sources where possible.
- For the base rate, identify the most relevant reference class and cite the frequency.
- Be honest about what you do not know. Say "uncertain" rather than guessing."""


class ResearcherFish:
    """Gathers context before the swarm runs. Runs once per market."""

    CLAUDE_BIN = _find_claude_binary()

    def __init__(self, model: str = "sonnet", claude_bin: str = "") -> None:
        self.model = model
        self.claude_bin = claude_bin or self.CLAUDE_BIN

    async def research(
        self, question: str, description: str = "",
    ) -> ResearchContext:
        """Gather research context for a market question.

        Uses a stronger model (sonnet by default) since this runs once
        and the quality of context affects all Fish predictions.
        """
        prompt = RESEARCHER_PROMPT.format(
            question=question,
            description=description[:1500],
        )
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
                proc.communicate(input=prompt.encode("utf-8")), timeout=180.0,
            )
            raw = stdout.decode("utf-8", errors="replace").strip()
            elapsed = time.monotonic() - t0

            parsed = self._parse_response(raw)
            parsed.question = question
            parsed.raw_response = raw
            parsed.elapsed_s = round(elapsed, 1)

            logger.info(
                f"Researcher: gathered context in {elapsed:.1f}s "
                f"({len(parsed.key_facts)} facts)"
            )
            return parsed

        except Exception as e:
            logger.error(f"Researcher error: {e}")
            return ResearchContext(
                question=question,
                base_rate="Unknown",
                key_facts=["Research failed"],
                recent_developments="Unknown",
                resolution_criteria="See market description",
                time_remaining="Unknown",
                contrarian_case="Unknown",
                elapsed_s=time.monotonic() - t0,
            )

    def _parse_response(self, text: str) -> ResearchContext:
        """Parse researcher LLM response."""
        # Try JSON extraction
        for pattern in [
            r'```(?:json)?\s*(\{.*?\})\s*```',
            r'(\{[^{}]*"base_rate"[^{}]*\})',
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return ResearchContext(
                        question="",
                        base_rate=str(data.get("base_rate", "Unknown")),
                        key_facts=[str(f) for f in data.get("key_facts", [])],
                        recent_developments=str(data.get("recent_developments", "")),
                        resolution_criteria=str(data.get("resolution_criteria", "")),
                        time_remaining=str(data.get("time_remaining", "")),
                        contrarian_case=str(data.get("contrarian_case", "")),
                    )
                except (json.JSONDecodeError, ValueError):
                    continue

        # Try direct parse
        try:
            data = json.loads(text.strip())
            return ResearchContext(
                question="",
                base_rate=str(data.get("base_rate", "Unknown")),
                key_facts=[str(f) for f in data.get("key_facts", [])],
                recent_developments=str(data.get("recent_developments", "")),
                resolution_criteria=str(data.get("resolution_criteria", "")),
                time_remaining=str(data.get("time_remaining", "")),
                contrarian_case=str(data.get("contrarian_case", "")),
            )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        return ResearchContext(
            question="",
            base_rate="Unknown",
            key_facts=[text[:200]],
            recent_developments="",
            resolution_criteria="",
            time_remaining="",
            contrarian_case="",
        )
