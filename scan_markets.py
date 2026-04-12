"""Step 1: Scan Polymarket and write markets to shared_state/market_data/.

This script fetches active markets from Polymarket (no API key needed),
filters by volume/liquidity, and writes them as JSON files that Fish
agents can read from their VS Code windows.

Usage:
    python scan_markets.py                    # Top 10 markets, volume > $50k
    python scan_markets.py --limit 20         # Top 20 markets
    python scan_markets.py --min-volume 10000 # Lower volume threshold
    python scan_markets.py --category crypto  # Filter by category keyword
    python scan_markets.py --query "bitcoin"  # Search by keyword
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

GAMMA_URL = "https://gamma-api.polymarket.com"
SHARED_STATE = Path(__file__).parent / "shared_state"
MARKET_DATA_DIR = SHARED_STATE / "market_data"


def fetch_markets(limit: int = 100, min_volume: float = 50_000) -> list[dict]:
    """Fetch active markets from Polymarket Gamma API."""
    response = httpx.get(
        f"{GAMMA_URL}/markets",
        params={"limit": 200, "active": "true", "closed": "false"},
        timeout=20,
    )
    response.raise_for_status()
    raw_markets = response.json()

    markets = []
    for m in raw_markets:
        volume = float(m.get("volume", 0))
        if volume < min_volume:
            continue

        # Parse prices (API returns JSON-encoded string)
        raw_prices = m.get("outcomePrices", "")
        try:
            if isinstance(raw_prices, str) and raw_prices.startswith("["):
                prices = json.loads(raw_prices)
            elif isinstance(raw_prices, list):
                prices = raw_prices
            else:
                prices = raw_prices.split(",") if raw_prices else []
            yes_price = float(prices[0]) if prices else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
        except (json.JSONDecodeError, ValueError, IndexError):
            yes_price, no_price = 0.5, 0.5

        markets.append({
            "id": str(m.get("id", "")),
            "question": m.get("question", ""),
            "description": (m.get("description", "") or "")[:1000],
            "category": m.get("category", ""),
            "yes_price": round(yes_price, 4),
            "no_price": round(no_price, 4),
            "volume": volume,
            "volume_24h": float(m.get("volume24hr", 0)),
            "liquidity": float(m.get("liquidity", 0)),
            "slug": m.get("slug", ""),
            "end_date": m.get("endDate", ""),
            "tags": m.get("tags", []) or [],
            "fetched_at": datetime.now().isoformat(),
        })

    markets.sort(key=lambda x: x["volume"], reverse=True)
    return markets[:limit]


def write_markets(markets: list[dict]) -> list[Path]:
    """Write markets to shared_state/market_data/ as individual JSON files."""
    MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old market files
    for old in MARKET_DATA_DIR.glob("market_*.json"):
        old.unlink()

    paths = []
    for m in markets:
        short_id = m["id"][:12] if len(m["id"]) > 12 else m["id"]
        path = MARKET_DATA_DIR / f"market_{short_id}.json"
        path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(path)

    return paths


def display_markets(markets: list[dict]) -> None:
    """Display fetched markets in a rich table."""
    table = Table(title=f"Polymarket: {len(markets)} Active Markets")
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", max_width=55)
    table.add_column("YES", justify="right", style="green")
    table.add_column("Volume", justify="right", style="cyan")
    table.add_column("Category", style="dim", max_width=12)
    table.add_column("File", style="dim", max_width=20)

    for i, m in enumerate(markets, 1):
        short_id = m["id"][:12]
        table.add_row(
            str(i),
            m["question"][:55],
            f"{m['yes_price']:.3f}",
            f"${m['volume']:,.0f}",
            m["category"][:12],
            f"market_{short_id}.json",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Scan Polymarket for active markets")
    parser.add_argument("--limit", type=int, default=10, help="Max markets to fetch")
    parser.add_argument("--min-volume", type=float, default=50_000, help="Min volume USD")
    parser.add_argument("--category", type=str, default="", help="Filter by category keyword")
    parser.add_argument("--query", type=str, default="", help="Search by keyword in question")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]MIROFISH MARKET SCANNER[/]\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan",
    ))

    try:
        markets = fetch_markets(limit=args.limit * 3, min_volume=args.min_volume)

        # Apply keyword filters
        if args.category:
            markets = [m for m in markets if args.category.lower() in m["category"].lower()]
        if args.query:
            q = args.query.lower()
            markets = [m for m in markets if q in m["question"].lower() or q in m["description"].lower()]

        markets = markets[:args.limit]

        if not markets:
            console.print("[yellow]No markets found. Try lower --min-volume or different --query.[/]")
            return

        # Write to shared_state
        paths = write_markets(markets)
        display_markets(markets)

        console.print(f"\n[green]Wrote {len(paths)} market files to shared_state/market_data/[/]")
        console.print("[dim]Next: Open Fish folders in VS Code and analyze these markets.[/]")
        console.print("[dim]  code Swarm_Intelligence/fish_geopolitical[/]")
        console.print("[dim]  code Swarm_Intelligence/fish_quant[/]")
        console.print("[dim]  code Swarm_Intelligence/fish_contrarian[/]")

    except httpx.HTTPError as e:
        console.print(f"[red]Network error: {e}[/]")
    except KeyboardInterrupt:
        console.print("[yellow]Cancelled.[/]")


if __name__ == "__main__":
    main()
