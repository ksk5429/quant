"""GOD Node — the event injection and swarm orchestration layer.

The GOD Node is a privileged agent that:
1. Receives real-world events from the user or external feeds
2. Analyzes which markets are affected by the event
3. Broadcasts event signals to relevant Fish agents
4. Triggers swarm-wide re-evaluation when probability shifts are significant
5. Acts as the "third party" observer that connects the swarm to reality

Named after the concept of an omniscient observer in information theory.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.mirofish.message_bus import MessageBus, Message, MessageType


class EventImpact(BaseModel):
    """Analysis of how a real-world event impacts prediction markets."""

    event_description: str
    affected_market_ids: list[str] = Field(default_factory=list)
    impact_direction: dict[str, str] = Field(
        default_factory=dict,
        description="market_id → 'increase' | 'decrease' | 'uncertain'",
    )
    impact_magnitude: dict[str, float] = Field(
        default_factory=dict,
        description="market_id → estimated probability shift magnitude (0-1)",
    )
    reasoning: str = ""
    urgency: str = "normal"  # "critical" | "high" | "normal" | "low"
    timestamp: float = Field(default_factory=time.time)


class GodNode:
    """The omniscient observer that connects real-world events to the swarm.

    Usage:
        god = GodNode(message_bus=bus, llm_client=client)

        # Inject a real-world event
        impact = await god.inject_event(
            "Supreme Court rules against crypto regulation bill",
            market_ids=["market-123", "market-456"]
        )

        # Trigger full swarm re-evaluation
        await god.trigger_reanalysis(market_ids=["market-123"])
    """

    def __init__(
        self,
        message_bus: MessageBus,
        llm_client: Any = None,
        model: str = "claude-opus-4-6",
        reanalysis_threshold: float = 0.15,
        propagation_mode: str = "broadcast",
    ) -> None:
        self.bus = message_bus
        self.llm_client = llm_client
        self.model = model
        self.reanalysis_threshold = reanalysis_threshold
        self.propagation_mode = propagation_mode
        self.node_id = "god-node"

        self.event_history: list[EventImpact] = []
        self._known_markets: dict[str, dict[str, Any]] = {}

        logger.info("GOD Node initialized")

    def register_market(self, market_id: str, metadata: dict[str, Any]) -> None:
        """Register a market so GOD can assess event impact on it."""
        self._known_markets[market_id] = metadata
        logger.debug(f"GOD: Registered market {market_id}")

    async def inject_event(
        self,
        event_description: str,
        market_ids: list[str] | None = None,
        urgency: str = "normal",
    ) -> EventImpact:
        """Inject a real-world event into the swarm.

        Steps:
        1. Analyze which markets are affected (using LLM if available)
        2. Estimate impact direction and magnitude per market
        3. Broadcast EVENT_INJECTION message to affected Fish
        4. If impact exceeds threshold, trigger REANALYSIS_REQUEST
        """
        logger.info(f"GOD: Event injection — '{event_description[:80]}...'")

        # Determine affected markets
        if market_ids is None:
            affected = await self._identify_affected_markets(event_description)
        else:
            affected = market_ids

        # Analyze impact
        impact = await self._analyze_impact(event_description, affected, urgency)
        self.event_history.append(impact)

        # Broadcast to swarm
        message = Message(
            msg_type=MessageType.EVENT_INJECTION,
            sender_id=self.node_id,
            payload={
                "event": event_description,
                "impact": impact.model_dump(),
            },
            target_ids=affected if self.propagation_mode == "targeted" else None,
        )
        delivered = await self.bus.publish(message)
        logger.info(f"GOD: Event broadcast to {delivered} Fish agents")

        # Trigger re-analysis for significantly impacted markets
        high_impact_markets = [
            mid for mid, mag in impact.impact_magnitude.items()
            if mag >= self.reanalysis_threshold
        ]
        if high_impact_markets:
            await self.trigger_reanalysis(high_impact_markets)

        return impact

    async def trigger_reanalysis(self, market_ids: list[str]) -> int:
        """Request specific Fish to re-analyze their markets."""
        message = Message(
            msg_type=MessageType.REANALYSIS_REQUEST,
            sender_id=self.node_id,
            payload={"market_ids": market_ids, "reason": "significant_event_impact"},
            target_ids=market_ids,
        )
        delivered = await self.bus.publish(message)
        logger.info(f"GOD: Reanalysis requested for {len(market_ids)} markets → {delivered} Fish")
        return delivered

    async def _identify_affected_markets(self, event_description: str) -> list[str]:
        """Use LLM to determine which registered markets are affected by an event."""
        if not self._known_markets:
            return []

        if self.llm_client is None:
            # Stub: return all known markets
            return list(self._known_markets.keys())

        # Build prompt listing known markets
        market_list = "\n".join(
            f"- {mid}: {meta.get('question', 'Unknown')}"
            for mid, meta in self._known_markets.items()
        )

        prompt = (
            f"Given this real-world event:\n\n"
            f'"{event_description}"\n\n'
            f"Which of these prediction markets would be affected? "
            f"List only the market IDs that are directly or indirectly impacted.\n\n"
            f"Markets:\n{market_list}\n\n"
            f"Respond with one market ID per line, nothing else."
        )

        response = await self._call_llm(prompt)
        affected_ids = []
        for line in response.split("\n"):
            line = line.strip().lstrip("- ")
            if line in self._known_markets:
                affected_ids.append(line)

        return affected_ids or list(self._known_markets.keys())

    async def _analyze_impact(
        self,
        event_description: str,
        affected_market_ids: list[str],
        urgency: str,
    ) -> EventImpact:
        """Analyze impact direction and magnitude for each affected market."""
        if self.llm_client is None:
            # Stub analysis
            return EventImpact(
                event_description=event_description,
                affected_market_ids=affected_market_ids,
                impact_direction={mid: "uncertain" for mid in affected_market_ids},
                impact_magnitude={mid: 0.1 for mid in affected_market_ids},
                reasoning="[STUB] No LLM available — default impact assessment",
                urgency=urgency,
            )

        # Build detailed analysis prompt
        market_details = "\n".join(
            f"- {mid}: {self._known_markets.get(mid, {}).get('question', 'Unknown')}"
            for mid in affected_market_ids
        )

        prompt = (
            f"Analyze the impact of this event on each prediction market.\n\n"
            f'Event: "{event_description}"\n\n'
            f"Markets:\n{market_details}\n\n"
            f"For each market, provide:\n"
            f"MARKET_ID: <id>\n"
            f"DIRECTION: increase | decrease | uncertain\n"
            f"MAGNITUDE: <float 0.0 to 1.0, how much the probability should shift>\n"
            f"REASONING: <one sentence>\n\n"
            f"Be precise. A magnitude of 0.05 = minor shift, 0.20 = major shift."
        )

        response = await self._call_llm(prompt)
        return self._parse_impact_response(
            response, event_description, affected_market_ids, urgency
        )

    def _parse_impact_response(
        self,
        response: str,
        event_description: str,
        affected_market_ids: list[str],
        urgency: str,
    ) -> EventImpact:
        """Parse the LLM's impact analysis response."""
        directions: dict[str, str] = {}
        magnitudes: dict[str, float] = {}
        reasoning_parts: list[str] = []

        current_market = None
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("MARKET_ID:"):
                current_market = line.split(":", 1)[1].strip()
            elif line.startswith("DIRECTION:") and current_market:
                directions[current_market] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("MAGNITUDE:") and current_market:
                try:
                    magnitudes[current_market] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    magnitudes[current_market] = 0.1
            elif line.startswith("REASONING:") and current_market:
                reasoning_parts.append(f"{current_market}: {line.split(':', 1)[1].strip()}")

        return EventImpact(
            event_description=event_description,
            affected_market_ids=affected_market_ids,
            impact_direction=directions,
            impact_magnitude=magnitudes,
            reasoning=" | ".join(reasoning_parts),
            urgency=urgency,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for GOD-level analysis."""
        if hasattr(self.llm_client, "messages"):
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        elif hasattr(self.llm_client, "chat"):
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        return ""
