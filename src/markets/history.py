"""Historical resolved market scraper for Polymarket Gamma API.

Fetches all resolved markets with their outcomes, volumes, and metadata.
Stores results as a local JSON corpus for retrodiction evaluation and
calibration pipeline training.

Usage:
    python -m src.markets.history           # scrape all resolved markets
    python -m src.markets.history --limit 50  # scrape 50 for testing
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


@dataclass
class ResolvedMarket:
    """A resolved Polymarket market with ground truth outcome."""

    id: str
    question: str
    description: str
    category: str
    slug: str

    # Resolution
    outcomes: list[str]
    outcome_prices: list[float]  # [1.0, 0.0] means first outcome won
    winning_outcome: str
    winning_index: int  # 0 or 1

    # Timing
    created_at: str
    end_date: str
    closed_time: str

    # Volume and liquidity
    volume_usd: float
    volume_clob: float

    # Metadata
    resolution_source: str = ""
    tags: list[str] = field(default_factory=list)
    event_title: str = ""
    event_id: str = ""

    @property
    def ground_truth(self) -> float:
        """Binary ground truth: 1.0 if first outcome won, 0.0 otherwise.

        Raises ValueError if outcome_prices is empty or ambiguous,
        rather than silently returning 0.5.
        """
        if not self.outcome_prices or len(self.outcome_prices) < 2:
            raise ValueError(f"Market {self.id} has no outcome prices")
        # outcome_prices[0] is 1.0 if first outcome won, 0.0 otherwise
        return float(self.outcome_prices[0])


class HistoricalMarketScraper:
    """Scrapes resolved markets from Polymarket Gamma API.

    Paginates through all closed+resolved markets, parses outcome data,
    and stores the corpus locally as JSON for offline evaluation.
    """

    BASE_URL = "https://gamma-api.polymarket.com"
    PAGE_SIZE = 100  # max per request
    RATE_LIMIT_DELAY = 0.35  # seconds between requests (stay under 300/10s)

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or "data/resolved_markets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def scrape_resolved_markets(
        self,
        max_markets: int | None = None,
        min_volume_usd: float = 1000.0,
    ) -> list[ResolvedMarket]:
        """Scrape all resolved markets from Gamma API.

        Args:
            max_markets: Stop after this many markets (None = all).
            min_volume_usd: Skip markets below this volume threshold.

        Returns:
            List of ResolvedMarket objects with ground truth outcomes.
        """
        all_markets: list[ResolvedMarket] = []
        offset = 0
        empty_pages = 0

        logger.info(
            f"Starting scrape: max={max_markets or 'all'}, "
            f"min_volume=${min_volume_usd:.0f}"
        )

        while True:
            params = {
                "closed": "true",
                "limit": self.PAGE_SIZE,
                "offset": offset,
                "order": "volumeNum",
                "ascending": "false",
            }

            # Retry with exponential backoff (3 attempts)
            data = None
            for attempt in range(3):
                try:
                    response = await self._client.get("/markets", params=params)
                    response.raise_for_status()
                    data = response.json()
                    break
                except httpx.HTTPError as e:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"API error at offset {offset} (attempt {attempt + 1}/3): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    if attempt < 2:
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"API failed after 3 attempts at offset {offset}")

            if data is None:
                # Skip this page, continue to next
                offset += self.PAGE_SIZE
                continue

            if not data or not isinstance(data, list):
                empty_pages += 1
                if empty_pages >= 3:
                    logger.info("Three consecutive empty pages, stopping.")
                    break
                offset += self.PAGE_SIZE
                continue

            empty_pages = 0
            page_count = 0

            for raw in data:
                market = self._parse_resolved(raw)
                if market is None:
                    continue
                if market.volume_usd < min_volume_usd:
                    continue
                all_markets.append(market)
                page_count += 1

            logger.info(
                f"Page {offset // self.PAGE_SIZE + 1}: "
                f"{page_count} resolved markets "
                f"(total: {len(all_markets)})"
            )

            offset += self.PAGE_SIZE

            if max_markets and len(all_markets) >= max_markets:
                all_markets = all_markets[:max_markets]
                break

            if len(data) < self.PAGE_SIZE:
                logger.info("Last page reached.")
                break

            await asyncio.sleep(self.RATE_LIMIT_DELAY)

        logger.info(f"Scrape complete: {len(all_markets)} resolved markets")
        return all_markets

    def _parse_resolved(self, raw: dict[str, Any]) -> ResolvedMarket | None:
        """Parse a raw API response into a ResolvedMarket.

        Returns None if the market is not fully resolved or has invalid data.
        """
        # Must be closed
        if not raw.get("closed", False):
            return None

        # Parse outcome prices to determine resolution
        prices_str = raw.get("outcomePrices", "")
        outcomes_str = raw.get("outcomes", "")

        try:
            if isinstance(prices_str, str):
                prices = json.loads(prices_str)
            else:
                prices = prices_str
            prices = [float(p) for p in prices]
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

        try:
            if isinstance(outcomes_str, str):
                outcomes = json.loads(outcomes_str)
            else:
                outcomes = outcomes_str
        except (json.JSONDecodeError, TypeError):
            outcomes = ["Yes", "No"]

        # Must have exactly 2 outcomes with a clear resolution
        if len(prices) < 2:
            return None

        # Reject ambiguous resolutions (both prices near 0.5)
        if max(prices) < 0.95:
            return None

        # Reject markets that are closed but not actually resolved
        uma_status = raw.get("umaResolutionStatus", "")
        if uma_status and uma_status not in ("resolved", ""):
            return None

        # Determine winner: the outcome with the higher price wins
        if prices[0] > prices[1]:
            winning_index = 0
        elif prices[1] > prices[0]:
            winning_index = 1
        else:
            # Equal prices = ambiguous resolution, skip
            return None

        winning_outcome = outcomes[winning_index] if winning_index < len(outcomes) else "Unknown"

        # Extract event info
        events = raw.get("events", [])
        event_title = events[0].get("title", "") if events else ""
        event_id = str(events[0].get("id", "")) if events else ""

        volume_usd = 0.0
        try:
            volume_usd = float(raw.get("volumeNum", 0) or raw.get("volume", 0) or 0)
        except (ValueError, TypeError):
            pass

        volume_clob = 0.0
        try:
            volume_clob = float(raw.get("volumeClob", 0) or 0)
        except (ValueError, TypeError):
            pass

        return ResolvedMarket(
            id=str(raw.get("id", "")),
            question=raw.get("question", ""),
            description=raw.get("description", "")[:2000],  # cap description length
            category=raw.get("category", ""),
            slug=raw.get("slug", ""),
            outcomes=outcomes,
            outcome_prices=prices,
            winning_outcome=winning_outcome,
            winning_index=winning_index,
            created_at=raw.get("createdAt", ""),
            end_date=raw.get("endDate", ""),
            closed_time=raw.get("closedTime", ""),
            volume_usd=volume_usd,
            volume_clob=volume_clob,
            resolution_source=raw.get("resolutionSource", ""),
            tags=raw.get("tags", []) or [],
            event_title=event_title,
            event_id=event_id,
        )

    def save_corpus(self, markets: list[ResolvedMarket], filename: str = "resolved_corpus.json") -> Path:
        """Save resolved markets corpus to JSON file."""
        output_path = self.output_dir / filename
        data = [asdict(m) for m in markets]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(markets)} markets to {output_path}")
        return output_path

    @staticmethod
    def load_corpus(filepath: str | Path) -> list[ResolvedMarket]:
        """Load resolved markets corpus from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        markets = []
        for item in data:
            try:
                markets.append(ResolvedMarket(**item))
            except TypeError as e:
                logger.warning(f"Failed to load market: {e}")
                continue

        logger.info(f"Loaded {len(markets)} markets from {filepath}")
        return markets

    def corpus_stats(self, markets: list[ResolvedMarket]) -> dict[str, Any]:
        """Compute summary statistics for the corpus."""
        if not markets:
            return {"count": 0}

        volumes = [m.volume_usd for m in markets]
        categories = {}
        for m in markets:
            cat = m.category or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1

        yes_wins = sum(1 for m in markets if m.winning_index == 0)
        base_rate = yes_wins / len(markets)

        return {
            "count": len(markets),
            "total_volume_usd": sum(volumes),
            "mean_volume_usd": sum(volumes) / len(volumes),
            "median_volume_usd": sorted(volumes)[len(volumes) // 2],
            "max_volume_usd": max(volumes),
            "categories": dict(sorted(categories.items(), key=lambda x: -x[1])),
            "base_rate_yes": round(base_rate, 4),
            "date_range": {
                "earliest": min(m.created_at for m in markets if m.created_at),
                "latest": max(m.closed_time for m in markets if m.closed_time),
            },
        }


async def main() -> None:
    """CLI entry point for scraping resolved markets."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape resolved Polymarket markets")
    parser.add_argument("--limit", type=int, default=None, help="Max markets to scrape")
    parser.add_argument("--min-volume", type=float, default=1000.0, help="Min volume USD")
    parser.add_argument("--output-dir", type=str, default="data/resolved_markets")
    args = parser.parse_args()

    scraper = HistoricalMarketScraper(output_dir=args.output_dir)

    try:
        markets = await scraper.scrape_resolved_markets(
            max_markets=args.limit,
            min_volume_usd=args.min_volume,
        )

        if markets:
            path = scraper.save_corpus(markets)
            stats = scraper.corpus_stats(markets)
            print(f"\nCorpus saved to: {path}")
            print(f"Total markets: {stats['count']}")
            print(f"Total volume: ${stats['total_volume_usd']:,.0f}")
            print(f"Base rate (first outcome wins): {stats['base_rate_yes']:.2%}")
            print(f"\nCategories:")
            for cat, count in list(stats["categories"].items())[:10]:
                print(f"  {cat}: {count}")
        else:
            print("No resolved markets found.")

    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
