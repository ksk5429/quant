"""Polymarket API client — Gamma API for metadata, CLOB API for trading.

Two APIs:
- Gamma API (https://gamma-api.polymarket.com): Market metadata, no auth required
- CLOB API (https://clob.polymarket.com): Trading, HMAC-SHA256 auth required

Data hierarchy: Series → Events → Markets
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel, Field


class PolymarketMarket(BaseModel):
    """A single Polymarket market (binary outcome)."""

    id: str
    question: str
    description: str = ""
    category: str = ""
    end_date: str = ""

    # Pricing
    yes_price: float = 0.5
    no_price: float = 0.5
    spread: float = 0.0

    # Volume & Liquidity
    volume: float = 0.0
    liquidity: float = 0.0
    volume_24h: float = 0.0

    # Metadata
    slug: str = ""
    active: bool = True
    closed: bool = False
    resolved: bool = False
    outcome: str | None = None

    # Token IDs for CLOB trading
    yes_token_id: str = ""
    no_token_id: str = ""

    # Tags for topic-based filtering
    tags: list[str] = Field(default_factory=list)


class PolymarketEvent(BaseModel):
    """An event containing one or more markets."""

    id: str
    title: str
    description: str = ""
    slug: str = ""
    markets: list[PolymarketMarket] = Field(default_factory=list)


class GammaClient:
    """Client for the Polymarket Gamma API (market metadata, no auth)."""

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or self.BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PolymarketMarket]:
        """Fetch markets from Gamma API with filters."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }

        try:
            response = await self._client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Gamma API error: {e}")
            return []

        markets = []
        for item in data if isinstance(data, list) else []:
            try:
                market = self._parse_market(item)
                markets.append(market)
            except Exception as e:
                logger.warning(f"Failed to parse market: {e}")
                continue

        logger.info(f"Gamma: Fetched {len(markets)} markets")
        return markets

    async def get_events(
        self,
        active: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PolymarketEvent]:
        """Fetch events (each containing one or more markets)."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
        }

        try:
            response = await self._client.get("/events", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Gamma API events error: {e}")
            return []

        events = []
        for item in data if isinstance(data, list) else []:
            try:
                event = self._parse_event(item)
                events.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
                continue

        return events

    async def get_market_by_slug(self, slug: str) -> PolymarketMarket | None:
        """Fetch a specific market by its slug."""
        try:
            response = await self._client.get(f"/markets", params={"slug": slug})
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                return self._parse_market(data[0])
        except httpx.HTTPError as e:
            logger.error(f"Gamma API slug lookup error: {e}")
        return None

    async def search_markets(
        self,
        query: str,
        min_volume: float = 0,
        min_liquidity: float = 0,
        limit: int = 50,
    ) -> list[PolymarketMarket]:
        """Search for markets matching a query, with volume/liquidity filters."""
        all_markets = await self.get_markets(limit=200)

        # Filter by query (simple substring match on question + description)
        query_lower = query.lower()
        filtered = [
            m for m in all_markets
            if query_lower in m.question.lower()
            or query_lower in m.description.lower()
        ]

        # Apply volume/liquidity filters
        filtered = [
            m for m in filtered
            if m.volume >= min_volume and m.liquidity >= min_liquidity
        ]

        return filtered[:limit]

    def _parse_market(self, raw: dict[str, Any]) -> PolymarketMarket:
        """Parse raw API response into PolymarketMarket."""
        # Extract token IDs from outcomes
        tokens = raw.get("clobTokenIds", "").split(",") if raw.get("clobTokenIds") else []
        yes_token = tokens[0].strip() if len(tokens) > 0 else ""
        no_token = tokens[1].strip() if len(tokens) > 1 else ""

        # Extract prices
        outcomes_prices = raw.get("outcomePrices", "").split(",") if raw.get("outcomePrices") else []
        yes_price = float(outcomes_prices[0]) if len(outcomes_prices) > 0 else 0.5
        no_price = float(outcomes_prices[1]) if len(outcomes_prices) > 1 else 0.5

        return PolymarketMarket(
            id=str(raw.get("id", "")),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            end_date=raw.get("endDate", ""),
            yes_price=yes_price,
            no_price=no_price,
            spread=abs(yes_price - no_price),
            volume=float(raw.get("volume", 0)),
            liquidity=float(raw.get("liquidity", 0)),
            volume_24h=float(raw.get("volume24hr", 0)),
            slug=raw.get("slug", ""),
            active=raw.get("active", True),
            closed=raw.get("closed", False),
            resolved=raw.get("resolved", False),
            outcome=raw.get("outcome"),
            yes_token_id=yes_token,
            no_token_id=no_token,
            tags=raw.get("tags", []) or [],
        )

    def _parse_event(self, raw: dict[str, Any]) -> PolymarketEvent:
        """Parse raw API response into PolymarketEvent."""
        markets = []
        for m in raw.get("markets", []):
            try:
                markets.append(self._parse_market(m))
            except Exception:
                continue

        return PolymarketEvent(
            id=str(raw.get("id", "")),
            title=raw.get("title", ""),
            description=raw.get("description", ""),
            slug=raw.get("slug", ""),
            markets=markets,
        )


class CLOBClient:
    """Client for the Polymarket CLOB API (trading, auth required).

    NOTE: Trading is disabled by default (paper_trading=True in config).
    This client provides read-only market data and order book access.
    Actual order placement requires proper authentication and is gated
    behind the risk management module.
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or self.BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_price(self, token_id: str, side: str = "BUY") -> float | None:
        """Get best bid or ask price for a token."""
        try:
            response = await self._client.get(
                "/price", params={"token_id": token_id, "side": side}
            )
            response.raise_for_status()
            data = response.json()
            return float(data.get("price", 0))
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning(f"CLOB price error for {token_id}: {e}")
            return None

    async def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token."""
        try:
            response = await self._client.get(
                "/midpoint", params={"token_id": token_id}
            )
            response.raise_for_status()
            data = response.json()
            return float(data.get("mid", 0))
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning(f"CLOB midpoint error for {token_id}: {e}")
            return None

    async def get_order_book(self, token_id: str) -> dict[str, Any]:
        """Get full order book for a token."""
        try:
            response = await self._client.get(
                "/book", params={"token_id": token_id}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"CLOB order book error for {token_id}: {e}")
            return {"bids": [], "asks": []}

    async def get_spread(self, token_id: str) -> dict[str, float | None]:
        """Get bid-ask spread for a token."""
        try:
            response = await self._client.get(
                "/spread", params={"token_id": token_id}
            )
            response.raise_for_status()
            data = response.json()
            return {
                "bid": float(data.get("bid", 0)) if data.get("bid") else None,
                "ask": float(data.get("ask", 0)) if data.get("ask") else None,
                "spread": float(data.get("spread", 0)) if data.get("spread") else None,
            }
        except httpx.HTTPError as e:
            logger.warning(f"CLOB spread error for {token_id}: {e}")
            return {"bid": None, "ask": None, "spread": None}
