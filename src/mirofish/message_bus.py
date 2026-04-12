"""MessageBus — inter-agent communication system for the Mirofish swarm.

Fish communicate through the MessageBus to share cross-market insights,
correlation signals, and event-triggered updates. The bus supports:
- Broadcast: GOD node sends events to all Fish
- Targeted: Fish sends insight to specific related Fish
- Topic-based: Fish subscribe to market categories
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from loguru import logger


class MessageType(str, Enum):
    """Types of messages that flow through the bus."""

    # Fish → Fish
    CROSS_MARKET_SIGNAL = "cross_market_signal"
    CORRELATION_ALERT = "correlation_alert"
    PROBABILITY_UPDATE = "probability_update"

    # GOD → Fish
    EVENT_INJECTION = "event_injection"
    REANALYSIS_REQUEST = "reanalysis_request"

    # System
    FISH_SPAWNED = "fish_spawned"
    FISH_REMOVED = "fish_removed"
    SWARM_CONSENSUS = "swarm_consensus"


@dataclass
class Message:
    """A single message on the bus."""

    msg_type: MessageType
    sender_id: str
    payload: dict[str, Any]
    target_ids: list[str] | None = None  # None = broadcast
    topic: str | None = None
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: f"msg-{int(time.time()*1000)}")


# Type alias for message handlers
MessageHandler = Callable[[Message], Awaitable[None]]


class MessageBus:
    """Async message bus for swarm communication.

    Supports three routing modes:
    1. Broadcast (target_ids=None): delivered to all subscribers
    2. Targeted (target_ids=[...]):  delivered only to named Fish
    3. Topic-based (topic="politics"): delivered to topic subscribers

    The bus is ordered and guarantees delivery within a single swarm.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, MessageHandler] = {}
        self._topic_subscribers: dict[str, set[str]] = defaultdict(set)
        self._message_log: list[Message] = []
        self._max_log_size = 10_000

    def subscribe(self, subscriber_id: str, handler: MessageHandler) -> None:
        """Register a Fish (or GOD node) to receive messages."""
        self._subscribers[subscriber_id] = handler
        logger.debug(f"MessageBus: {subscriber_id} subscribed")

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber."""
        self._subscribers.pop(subscriber_id, None)
        for topic_subs in self._topic_subscribers.values():
            topic_subs.discard(subscriber_id)

    def subscribe_topic(self, subscriber_id: str, topic: str) -> None:
        """Subscribe a Fish to a specific topic (e.g., 'politics', 'crypto')."""
        self._topic_subscribers[topic].add(subscriber_id)

    async def publish(self, message: Message) -> int:
        """Publish a message to the bus. Returns number of recipients."""
        self._log_message(message)

        recipients = self._resolve_recipients(message)
        if not recipients:
            logger.warning(f"MessageBus: No recipients for {message.msg_type} from {message.sender_id}")
            return 0

        # Deliver concurrently to all recipients
        tasks = []
        for recipient_id in recipients:
            if recipient_id == message.sender_id:
                continue  # Don't deliver to sender
            handler = self._subscribers.get(recipient_id)
            if handler:
                tasks.append(self._safe_deliver(handler, message, recipient_id))

        if tasks:
            await asyncio.gather(*tasks)

        delivered = len(tasks)
        logger.debug(
            f"MessageBus: {message.msg_type} from {message.sender_id} "
            f"→ {delivered} recipients"
        )
        return delivered

    def _resolve_recipients(self, message: Message) -> set[str]:
        """Determine which subscribers should receive this message."""
        if message.target_ids is not None:
            # Targeted delivery
            return set(message.target_ids) & set(self._subscribers.keys())

        if message.topic is not None:
            # Topic-based delivery
            return self._topic_subscribers.get(message.topic, set()) & set(self._subscribers.keys())

        # Broadcast
        return set(self._subscribers.keys())

    async def _safe_deliver(
        self, handler: MessageHandler, message: Message, recipient_id: str
    ) -> None:
        """Deliver a message, catching and logging any handler errors."""
        try:
            await handler(message)
        except Exception as e:
            logger.error(
                f"MessageBus: Handler error for {recipient_id} "
                f"on {message.msg_type}: {e}"
            )

    def _log_message(self, message: Message) -> None:
        """Append message to the log, trimming if over capacity."""
        self._message_log.append(message)
        if len(self._message_log) > self._max_log_size:
            self._message_log = self._message_log[-self._max_log_size // 2:]

    @property
    def message_count(self) -> int:
        return len(self._message_log)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def get_messages_for(
        self,
        subscriber_id: str,
        msg_type: MessageType | None = None,
        since: float | None = None,
    ) -> list[Message]:
        """Retrieve historical messages relevant to a subscriber."""
        results = []
        for msg in self._message_log:
            if msg_type and msg.msg_type != msg_type:
                continue
            if since and msg.timestamp < since:
                continue
            recipients = self._resolve_recipients(msg)
            if subscriber_id in recipients:
                results.append(msg)
        return results
