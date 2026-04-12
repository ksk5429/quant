"""Live market scanner — discovers and ranks active Polymarket markets.

Scans the Polymarket Gamma API for active markets, classifies them,
estimates our edge potential, and returns a ranked list of markets
worth analyzing with the full swarm.

This is the entry point for the live trading pipeline:
  Scanner → Engine v4 → Portfolio → (future: Order Execution)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from src.mirofish.swarm_router import classify_market


@dataclass
class ActiveMarket:
    """A live Polymarket market candidate for analysis."""
    id: str
    question: str
    description: str
    category: str
    difficulty: str
    yes_price: float
    no_price: float
    volume_usd: float
    liquidity_usd: float
    end_date: str
    slug: str
    outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])
    tags: list[str] = field(default_factory=list)
    score: float = 0.0  # ranking score (higher = more attractive)


class MarketScanner:
    """Scans Polymarket for active markets worth analyzing.

    Filters by volume, liquidity, and price range (avoid resolved markets).
    Ranks by a composite score favoring high-volume markets with prices
    away from 0 or 1 (more room for edge).
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        min_volume_usd: float = 50_000,
        min_liquidity_usd: float = 10_000,
        price_range: tuple[float, float] = (0.05, 0.95),
        max_markets: int = 50,
    ) -> None:
        self.min_volume = min_volume_usd
        self.min_liquidity = min_liquidity_usd
        self.price_range = price_range
        self.max_markets = max_markets

    async def scan(self) -> list[ActiveMarket]:
        """Scan Polymarket for tradeable active markets.

        Returns ranked list sorted by composite attractiveness score.
        """
        raw_markets = await self._fetch_active_markets()
        filtered = self._filter_markets(raw_markets)
        ranked = self._rank_markets(filtered)
        return ranked[:self.max_markets]

    async def _fetch_active_markets(self) -> list[dict]:
        """Fetch active markets from Gamma API with pagination."""
        all_markets = []
        offset = 0
        page_size = 100

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                try:
                    r = await client.get(
                        f"{self.BASE_URL}/markets",
                        params={
                            "active": "true",
                            "closed": "false",
                            "limit": page_size,
                            "offset": offset,
                            "order": "volumeNum",
                            "ascending": "false",
                        },
                    )
                    r.raise_for_status()
                    data = r.json()
                except httpx.HTTPError as e:
                    logger.error(f"Gamma API error at offset {offset}: {e}")
                    break

                if not data or not isinstance(data, list):
                    break

                all_markets.extend(data)
                offset += page_size

                if len(data) < page_size:
                    break

                # Stop after enough pages (don't scrape everything)
                if len(all_markets) >= 500:
                    break

                await asyncio.sleep(0.3)

        logger.info(f"Scanner: fetched {len(all_markets)} active markets")
        return all_markets

    def _filter_markets(self, raw_markets: list[dict]) -> list[ActiveMarket]:
        """Filter and parse raw markets into ActiveMarket objects."""
        filtered = []

        for raw in raw_markets:
            try:
                volume = float(raw.get("volumeNum", 0) or 0)
                liquidity = float(raw.get("liquidity", 0) or 0)

                if volume < self.min_volume:
                    continue
                if liquidity < self.min_liquidity:
                    continue

                # Parse prices
                prices_str = raw.get("outcomePrices", "")
                if isinstance(prices_str, str):
                    prices = json.loads(prices_str) if prices_str else []
                else:
                    prices = prices_str or []
                prices = [float(p) for p in prices]

                if len(prices) < 2:
                    continue

                yes_price = prices[0]
                no_price = prices[1]

                # Skip markets near resolution (price near 0 or 1)
                if yes_price < self.price_range[0] or yes_price > self.price_range[1]:
                    continue

                # Parse outcomes
                outcomes_str = raw.get("outcomes", "")
                if isinstance(outcomes_str, str):
                    outcomes = json.loads(outcomes_str) if outcomes_str else ["Yes", "No"]
                else:
                    outcomes = outcomes_str or ["Yes", "No"]

                question = raw.get("question", "")
                description = raw.get("description", "")[:2000]

                classification = classify_market(question, description, volume)

                market = ActiveMarket(
                    id=str(raw.get("id", "")),
                    question=question,
                    description=description,
                    category=classification["category"],
                    difficulty=classification["difficulty"],
                    yes_price=yes_price,
                    no_price=no_price,
                    volume_usd=volume,
                    liquidity_usd=liquidity,
                    end_date=raw.get("endDate", ""),
                    slug=raw.get("slug", ""),
                    outcomes=outcomes,
                    tags=raw.get("tags", []) or [],
                )
                filtered.append(market)

            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        logger.info(f"Scanner: {len(filtered)} markets pass filters")
        return filtered

    def _rank_markets(self, markets: list[ActiveMarket]) -> list[ActiveMarket]:
        """Rank markets by composite attractiveness score.

        Score factors:
        - Volume (log-scaled, higher = more liquid = better execution)
        - Price distance from 0.5 (closer to 0.5 = more uncertain = more edge)
        - Liquidity (higher = lower slippage)
        """
        import math

        for m in markets:
            vol_score = math.log10(max(m.volume_usd, 1))  # 5 = $100K, 7 = $10M
            price_uncertainty = 1.0 - abs(m.yes_price - 0.5) * 2  # 1.0 at 0.5, 0.0 at 0/1
            liq_score = math.log10(max(m.liquidity_usd, 1)) / 7  # normalize

            m.score = round(vol_score * 0.4 + price_uncertainty * 0.4 + liq_score * 0.2, 4)

        markets.sort(key=lambda m: -m.score)
        return markets

    def save_scan(self, markets: list[ActiveMarket], path: str = "data/scans") -> Path:
        """Save scan results to JSON."""
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out_dir / f"scan_{ts}.json"

        data = [
            {
                "id": m.id,
                "question": m.question,
                "category": m.category,
                "difficulty": m.difficulty,
                "yes_price": m.yes_price,
                "volume_usd": m.volume_usd,
                "liquidity_usd": m.liquidity_usd,
                "score": m.score,
                "slug": m.slug,
                "outcomes": m.outcomes,
            }
            for m in markets
        ]
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Scan saved: {filepath} ({len(markets)} markets)")
        return filepath

    def print_scan(self, markets: list[ActiveMarket], top_n: int = 20) -> None:
        """Print scan results."""
        print(f"\n{'='*80}")
        print(f"MIROFISH MARKET SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*80}")
        print(f"Active markets scanned: {len(markets)}")
        print(f"\n{'Rank':>4} {'Score':>5} {'Cat':>10} {'Price':>6} {'Vol($)':>12} {'Question'}")
        print(f"{'─'*80}")
        for i, m in enumerate(markets[:top_n], 1):
            print(
                f"{i:>4} {m.score:>5.2f} {m.category:>10} "
                f"{m.yes_price:>5.0%} {m.volume_usd:>11,.0f}  "
                f"{m.question[:42]}"
            )
        print(f"{'='*80}")


async def main():
    """CLI entry point for market scanning."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan Polymarket for active markets")
    parser.add_argument("--min-volume", type=float, default=50000)
    parser.add_argument("--min-liquidity", type=float, default=10000)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    scanner = MarketScanner(
        min_volume_usd=args.min_volume,
        min_liquidity_usd=args.min_liquidity,
    )
    markets = await scanner.scan()
    scanner.print_scan(markets, top_n=args.top)
    scanner.save_scan(markets)


if __name__ == "__main__":
    asyncio.run(main())
