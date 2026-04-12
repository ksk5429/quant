"""Mirofish Live Demo — Full pipeline from Polymarket to Claude to Trade Signal.

This demo:
1. Fetches real active markets from Polymarket Gamma API
2. Analyzes each market using Claude-powered Fish agents (real LLM calls)
3. Aggregates predictions using Bayesian confidence-weighted fusion
4. Calibrates probabilities (if enough history exists)
5. Computes trade signals using Quarter-Kelly criterion
6. Visualizes everything with Plotly

Usage:
    # With API key in environment
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo_live.py

    # Or with config/local.yaml (created by setup.py)
    python demo_live.py

    # Paper mode (no API key — uses stubs)
    python demo_live.py --stub
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix Windows encoding for Korean locale
os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.mirofish.fish import Fish, FishPersona, FishAnalysis
from src.mirofish.swarm import Swarm, SwarmPrediction
from src.mirofish.message_bus import MessageBus
from src.risk.kelly import KellyCriterion, TradeSignal
from src.utils.config import load_config

console = Console()


# ---------------------------------------------------------------------------
# Step 1: Fetch real markets from Polymarket
# ---------------------------------------------------------------------------

async def fetch_markets(
    limit: int = 10,
    min_volume: float = 50_000,
) -> list[dict]:
    """Fetch active markets from Polymarket Gamma API."""
    console.print("\n[bold cyan]Step 1: Fetching markets from Polymarket...[/]")

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "limit": 100,
                "active": "true",
                "closed": "false",
            },
        )
        response.raise_for_status()
        all_markets = response.json()

    # Filter by volume and parse
    markets = []
    for m in all_markets:
        volume = float(m.get("volume", 0))
        if volume < min_volume:
            continue

        # Parse prices — API returns JSON string like '["0.535","0.465"]'
        raw_prices = m.get("outcomePrices", "")
        try:
            if isinstance(raw_prices, str):
                prices = json.loads(raw_prices) if raw_prices.startswith("[") else raw_prices.split(",")
            elif isinstance(raw_prices, list):
                prices = raw_prices
            else:
                prices = []
            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 0.5
        except (json.JSONDecodeError, ValueError, IndexError):
            yes_price, no_price = 0.5, 0.5

        markets.append({
            "id": str(m.get("id", "")),
            "question": m.get("question", ""),
            "description": m.get("description", "")[:500],
            "category": m.get("category", ""),
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": volume,
            "slug": m.get("slug", ""),
        })

    # Sort by volume descending, take top N
    markets.sort(key=lambda x: x["volume"], reverse=True)
    markets = markets[:limit]

    # Display
    table = Table(title=f"Top {len(markets)} Active Markets (vol > ${min_volume:,.0f})")
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", max_width=55)
    table.add_column("YES Price", justify="right", style="green")
    table.add_column("Volume", justify="right", style="cyan")

    for i, m in enumerate(markets, 1):
        table.add_row(
            str(i),
            m["question"][:55],
            f"${m['yes_price']:.3f}",
            f"${m['volume']:,.0f}",
        )

    console.print(table)
    return markets


# ---------------------------------------------------------------------------
# Step 2: Analyze with Claude-powered Fish swarm
# ---------------------------------------------------------------------------

async def analyze_with_swarm(
    markets: list[dict],
    llm_client=None,
    num_fish: int = 5,
    model: str = "claude-haiku-4-5-20251001",
) -> list[SwarmPrediction]:
    """Run the full swarm analysis on each market."""
    console.print("\n[bold cyan]Step 2: Swarm analysis with Claude Fish agents...[/]")

    swarm = Swarm(
        num_fish=num_fish,
        llm_client=llm_client,
        model=model,
        max_concurrent=3,
    )

    mode = "[bold green]LIVE (Claude API)[/]" if llm_client else "[bold yellow]STUB MODE[/]"
    console.print(f"  Mode: {mode}")
    console.print(f"  Fish: {num_fish} agents, Model: {model}")
    console.print(f"  Personas: {', '.join(f.persona.value for f in swarm.fish)}")

    predictions = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, market in enumerate(markets):
            task = progress.add_task(
                f"Analyzing [{i+1}/{len(markets)}]: {market['question'][:45]}...",
                total=None,
            )

            prediction = await swarm.analyze_market(
                market_id=market["id"],
                market_question=market["question"],
                market_description=market.get("description", ""),
                current_price=market["yes_price"],
            )
            predictions.append(prediction)

            progress.update(task, completed=True)
            progress.remove_task(task)

    return predictions


# ---------------------------------------------------------------------------
# Step 3: Generate trade signals
# ---------------------------------------------------------------------------

def generate_signals(
    predictions: list[SwarmPrediction],
    markets: list[dict],
    bankroll: float = 1000,
) -> list[TradeSignal]:
    """Apply Kelly criterion to generate trade signals."""
    console.print("\n[bold cyan]Step 3: Generating trade signals (Quarter-Kelly)...[/]")

    kelly = KellyCriterion(
        bankroll=bankroll,
        kelly_fraction=0.25,
        max_position_pct=0.05,
        min_edge=0.05,
        paper_trading=True,
    )

    signals = []
    for pred, market in zip(predictions, markets):
        signal = kelly.compute_signal(
            market_id=pred.market_id,
            market_question=pred.market_question,
            our_probability=pred.probability,
            market_price=market["yes_price"],
            confidence=pred.confidence,
        )
        signals.append(signal)

    return signals


# ---------------------------------------------------------------------------
# Step 4: Display results
# ---------------------------------------------------------------------------

def display_results(
    predictions: list[SwarmPrediction],
    signals: list[TradeSignal],
    markets: list[dict],
):
    """Rich table display of all results."""
    console.print("\n[bold cyan]Step 4: Results[/]")

    # Main results table
    table = Table(title="Mirofish Swarm Predictions vs Market")
    table.add_column("#", style="dim", width=3)
    table.add_column("Market", max_width=40)
    table.add_column("Market P", justify="right", style="yellow")
    table.add_column("Swarm P", justify="right", style="green")
    table.add_column("Edge", justify="right")
    table.add_column("Conf", justify="right", style="cyan")
    table.add_column("Spread", justify="right", style="dim")
    table.add_column("Signal", justify="center")
    table.add_column("Size", justify="right")

    for i, (pred, signal, market) in enumerate(zip(predictions, signals, markets), 1):
        edge = pred.edge or 0
        edge_style = "green" if edge > 0.05 else ("red" if edge < -0.05 else "dim")
        edge_str = f"[{edge_style}]{edge:+.3f}[/]"

        if signal.side.value == "yes":
            sig_str = "[bold green]BUY YES[/]"
        elif signal.side.value == "no":
            sig_str = "[bold red]BUY NO[/]"
        else:
            sig_str = "[dim]PASS[/]"

        size_str = f"${signal.position_size_usd:.0f}" if signal.is_actionable else "-"

        table.add_row(
            str(i),
            pred.market_question[:40],
            f"{market['yes_price']:.3f}",
            f"{pred.probability:.3f}",
            edge_str,
            f"{pred.confidence:.2f}",
            f"{pred.spread:.3f}",
            sig_str,
            size_str,
        )

    console.print(table)

    # Summary stats
    actionable = [s for s in signals if s.is_actionable]
    total_exposure = sum(s.position_size_usd for s in actionable)
    avg_edge = sum(abs(s.edge) for s in actionable) / len(actionable) if actionable else 0

    console.print(Panel(
        f"Markets analyzed: {len(predictions)}\n"
        f"Actionable signals: {len(actionable)}\n"
        f"Total exposure: ${total_exposure:.2f}\n"
        f"Avg edge: {avg_edge:.3f}\n"
        f"Mode: [bold yellow]PAPER TRADING[/]",
        title="Summary",
        border_style="green",
    ))

    # Detailed breakdown for top signal
    if actionable:
        best = max(actionable, key=lambda s: abs(s.edge))
        best_pred = next(p for p in predictions if p.market_id == best.market_id)

        console.print(f"\n[bold]Best Signal: {best.side.value.upper()} '{best.market_question[:50]}...'[/]")
        console.print(f"  Our P: {best.our_probability:.3f}  |  Market: {best.market_price:.3f}  |  Edge: {best.edge:+.3f}")
        console.print(f"  Kelly: {best.kelly_fraction:.3f}  |  Position: ${best.position_size_usd:.2f}  |  EV: {best.expected_value:+.4f}")

        # Show individual Fish analyses
        if best_pred.fish_analyses:
            console.print("\n  [dim]Individual Fish predictions:[/]")
            for fa in best_pred.fish_analyses:
                bar_len = int(fa.probability * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                console.print(
                    f"    {fa.persona.value:<25} P={fa.probability:.3f} "
                    f"[{'green' if fa.probability > 0.5 else 'red'}]{bar}[/] "
                    f"(conf={fa.confidence:.2f})"
                )


# ---------------------------------------------------------------------------
# Step 5: Save results to shared_state (for Fish agents to read)
# ---------------------------------------------------------------------------

def save_to_shared_state(
    predictions: list[SwarmPrediction],
    signals: list[TradeSignal],
    markets: list[dict],
):
    """Write results to shared_state/ for Fish agents in Swarm_Intelligence/."""
    shared = Path("shared_state")

    # Write market data for Fish agents
    market_data_dir = shared / "market_data"
    market_data_dir.mkdir(parents=True, exist_ok=True)

    for market in markets:
        path = market_data_dir / f"market_{market['id'][:12]}.json"
        path.write_text(json.dumps(market, indent=2), encoding="utf-8")

    # Write consensus predictions
    consensus_dir = shared / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for pred, signal in zip(predictions, signals):
        result = {
            "market_id": pred.market_id,
            "market_question": pred.market_question,
            "swarm_probability": pred.probability,
            "swarm_confidence": pred.confidence,
            "market_price": pred.market_price,
            "edge": pred.edge,
            "spread": pred.spread,
            "signal_side": signal.side.value,
            "signal_size_usd": signal.position_size_usd,
            "fish_count": len(pred.fish_analyses),
            "fish_predictions": {
                fa.persona.value: {"probability": fa.probability, "confidence": fa.confidence}
                for fa in pred.fish_analyses
            },
            "timestamp": timestamp,
        }
        path = consensus_dir / f"consensus_{pred.market_id[:12]}_{timestamp}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    console.print(f"\n[dim]Results saved to shared_state/ ({len(markets)} markets)[/]")


# ---------------------------------------------------------------------------
# Step 6: Generate visualizations
# ---------------------------------------------------------------------------

def generate_visualizations(predictions: list[SwarmPrediction], markets: list[dict]):
    """Generate and save Plotly visualizations."""
    from src.visualization.plots import (
        plot_swarm_prediction,
        plot_edge_distribution,
    )

    viz_dir = Path("data/processed")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Edge distribution
    edges = [p.edge for p in predictions if p.edge is not None]
    if edges:
        fig = plot_edge_distribution(edges)
        path = viz_dir / "edge_distribution.html"
        fig.write_html(str(path))
        console.print(f"  Edge distribution: {path}")

    # Best prediction detail
    actionable = [p for p in predictions if p.edge and abs(p.edge) > 0.05]
    if actionable:
        best = max(actionable, key=lambda p: abs(p.edge))
        fig = plot_swarm_prediction(
            fish_probabilities=[fa.probability for fa in best.fish_analyses],
            fish_personas=[fa.persona.value for fa in best.fish_analyses],
            fish_confidences=[fa.confidence for fa in best.fish_analyses],
            swarm_probability=best.probability,
            market_price=best.market_price,
            market_question=best.market_question,
        )
        path = viz_dir / "best_signal_swarm.html"
        fig.write_html(str(path))
        console.print(f"  Best signal swarm: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Mirofish Live Demo")
    parser.add_argument("--stub", action="store_true", help="Run without API key (stub mode)")
    parser.add_argument("--markets", type=int, default=5, help="Number of markets to analyze")
    parser.add_argument("--fish", type=int, default=5, help="Number of Fish agents")
    parser.add_argument("--min-volume", type=float, default=50_000, help="Min market volume USD")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001", help="Claude model")
    parser.add_argument("--bankroll", type=float, default=1000, help="Paper trading bankroll")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]MIROFISH PREDICTION ENGINE[/]\n"
        "Swarm Intelligence for Prediction Markets\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan",
    ))

    # Initialize Claude client
    llm_client = None
    if not args.stub:
        config = load_config()
        api_key = config.api_keys.anthropic or os.environ.get("ANTHROPIC_API_KEY", "")

        if api_key:
            try:
                import anthropic
                llm_client = anthropic.Anthropic(api_key=api_key)
                console.print("[green]Claude API: Connected[/]")
            except Exception as e:
                console.print(f"[yellow]Claude API: Failed ({e}) — falling back to stub mode[/]")
        else:
            console.print("[yellow]No Anthropic API key — running in stub mode[/]")
            console.print("[dim]  Set ANTHROPIC_API_KEY or run: python setup.py[/]")

    # Pipeline
    try:
        # Step 1: Fetch markets
        markets = await fetch_markets(limit=args.markets, min_volume=args.min_volume)
        if not markets:
            console.print("[red]No markets found matching filters. Try lowering --min-volume.[/]")
            return

        # Step 2: Swarm analysis
        predictions = await analyze_with_swarm(
            markets=markets,
            llm_client=llm_client,
            num_fish=args.fish,
            model=args.model,
        )

        # Step 3: Trade signals
        signals = generate_signals(predictions, markets, bankroll=args.bankroll)

        # Step 4: Display
        display_results(predictions, signals, markets)

        # Step 5: Save to shared state
        save_to_shared_state(predictions, signals, markets)

        # Step 6: Visualizations
        console.print("\n[bold cyan]Step 5: Generating visualizations...[/]")
        generate_visualizations(predictions, markets)

    except httpx.HTTPError as e:
        console.print(f"[red]Network error: {e}[/]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")


if __name__ == "__main__":
    asyncio.run(main())
