"""Step 3: Aggregate Fish analyses into trade signals.

Reads all JSON analyses from shared_state/analyses/, applies Bayesian
confidence-weighted aggregation, and produces calibrated trade signals.

This script requires NO API key — all computation is local math.

Usage:
    python aggregate.py                       # Aggregate all analyses
    python aggregate.py --market btc          # Filter by market keyword
    python aggregate.py --min-edge 0.10       # Higher edge threshold
    python aggregate.py --bankroll 5000       # Custom bankroll
    python aggregate.py --export signals.json # Export signals to file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SHARED_STATE = Path(__file__).parent / "shared_state"
ANALYSES_DIR = SHARED_STATE / "analyses"
MARKET_DATA_DIR = SHARED_STATE / "market_data"
CONSENSUS_DIR = SHARED_STATE / "consensus"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_market_data() -> dict[str, dict]:
    """Load all market data from shared_state/market_data/."""
    markets = {}
    if not MARKET_DATA_DIR.exists():
        return markets

    for path in MARKET_DATA_DIR.glob("market_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            market_id = data.get("id", path.stem)
            markets[market_id] = data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            console.print(f"[yellow]Warning: Could not read {path.name}: {e}[/]")
    return markets


def load_analyses() -> dict[str, list[dict]]:
    """Load all Fish analyses from shared_state/analyses/, grouped by market_id."""
    analyses_by_market: dict[str, list[dict]] = defaultdict(list)

    if not ANALYSES_DIR.exists():
        return analyses_by_market

    # Search recursively (supports round_1/, round_2/ subdirectories)
    for path in sorted(ANALYSES_DIR.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            market_id = data.get("market_id", "")
            if not market_id:
                continue
            data["_source_file"] = str(path.relative_to(ANALYSES_DIR))
            analyses_by_market[market_id].append(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            console.print(f"[yellow]Warning: Could not read {path.name}: {e}[/]")

    return dict(analyses_by_market)


# ---------------------------------------------------------------------------
# Bayesian aggregation
# ---------------------------------------------------------------------------

def bayesian_aggregate(analyses: list[dict]) -> dict:
    """Aggregate multiple Fish analyses using confidence-weighted Bayesian fusion.

    Formula: P_swarm = sum(w_i * P_i) / sum(w_i)
    where w_i = confidence_i * (1 + accuracy_bonus_i)

    Returns a consensus dict with probability, confidence, and metadata.
    """
    if not analyses:
        return {"probability": 0.5, "confidence": 0.0, "fish_count": 0}

    probabilities = []
    confidences = []
    fish_details = []

    for a in analyses:
        p = a.get("probability")
        c = a.get("confidence", 0.5)

        # Validate
        if p is None or not isinstance(p, (int, float)):
            continue
        p = max(0.0, min(1.0, float(p)))
        c = max(0.0, min(1.0, float(c)))

        probabilities.append(p)
        confidences.append(c)
        fish_details.append({
            "fish_name": a.get("fish_name", "unknown"),
            "probability": round(p, 4),
            "confidence": round(c, 4),
            "reasoning_summary": (a.get("reasoning_steps", [""])[0])[:100] if a.get("reasoning_steps") else "",
            "source_file": a.get("_source_file", ""),
        })

    if not probabilities:
        return {"probability": 0.5, "confidence": 0.0, "fish_count": 0}

    probs = np.array(probabilities)
    confs = np.array(confidences)

    # Confidence-weighted aggregation
    weights = confs.copy()
    total_weight = np.sum(weights)
    if total_weight == 0:
        agg_prob = float(np.mean(probs))
    else:
        agg_prob = float(np.sum(weights * probs) / total_weight)

    agg_prob = round(float(np.clip(agg_prob, 0.001, 0.999)), 4)

    # Ensemble metrics
    spread = float(np.max(probs) - np.min(probs))
    std_dev = float(np.std(probs))
    agreement_bonus = max(0, 1.0 - spread * 2)
    agg_confidence = float(np.mean(confs)) * (0.7 + 0.3 * agreement_bonus)
    agg_confidence = round(float(np.clip(agg_confidence, 0.0, 1.0)), 4)

    return {
        "probability": agg_prob,
        "confidence": agg_confidence,
        "spread": round(spread, 4),
        "std_dev": round(std_dev, 4),
        "fish_count": len(probabilities),
        "fish_details": fish_details,
        "aggregation_method": "bayesian_confidence_weighted",
    }


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def compute_signal(
    market_id: str,
    market_question: str,
    our_probability: float,
    market_price: float,
    confidence: float,
    bankroll: float = 1000,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.05,
    min_edge: float = 0.05,
) -> dict:
    """Compute a trade signal using Quarter-Kelly criterion."""
    yes_edge = our_probability - market_price

    if abs(yes_edge) < min_edge:
        return {
            "market_id": market_id,
            "market_question": market_question,
            "side": "PASS",
            "our_probability": round(our_probability, 4),
            "market_price": round(market_price, 4),
            "edge": 0.0,
            "position_usd": 0.0,
            "kelly_raw": 0.0,
            "expected_value": 0.0,
            "confidence": round(confidence, 4),
        }

    if yes_edge > 0:
        side = "BUY YES"
        edge = yes_edge
        raw_kelly = edge / (1 - market_price) if market_price < 1 else 0
        ev = our_probability * (1 / market_price - 1) - (1 - our_probability)
    else:
        side = "BUY NO"
        edge = -yes_edge
        raw_kelly = edge / market_price if market_price > 0 else 0
        ev = (1 - our_probability) * (1 / (1 - market_price) - 1) - our_probability

    adjusted = raw_kelly * kelly_fraction * confidence
    position_frac = min(adjusted, max_position_pct)
    position_usd = round(position_frac * bankroll, 2)

    return {
        "market_id": market_id,
        "market_question": market_question,
        "side": side,
        "our_probability": round(our_probability, 4),
        "market_price": round(market_price, 4),
        "edge": round(edge, 4),
        "position_usd": position_usd,
        "kelly_raw": round(raw_kelly, 4),
        "expected_value": round(ev, 4),
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Display & export
# ---------------------------------------------------------------------------

def display_results(
    consensuses: list[dict],
    signals: list[dict],
) -> None:
    """Display aggregation results in rich tables."""

    # Fish contributions table
    console.print("\n[bold cyan]Fish Contributions[/]")
    for c in consensuses:
        if not c.get("fish_details"):
            continue
        q = c.get("market_question", "?")[:50]
        console.print(f"\n  [bold]{q}...[/]")
        for fd in c["fish_details"]:
            p = fd["probability"]
            bar_len = int(p * 25)
            bar = "#" * bar_len + "." * (25 - bar_len)
            style = "green" if p > 0.5 else "red"
            console.print(
                f"    {fd['fish_name']:<28} P={p:.3f} [{style}]{bar}[/] "
                f"(conf={fd['confidence']:.2f})"
            )
        console.print(
            f"    {'SWARM CONSENSUS':<28} P={c['probability']:.3f} "
            f"[bold yellow]{'=' * 25}[/] "
            f"(conf={c['confidence']:.2f}, spread={c['spread']:.3f})"
        )

    # Trade signals table
    console.print("\n[bold cyan]Trade Signals[/]")
    table = Table()
    table.add_column("#", style="dim", width=3)
    table.add_column("Market", max_width=40)
    table.add_column("Mkt P", justify="right", style="yellow")
    table.add_column("Our P", justify="right", style="green")
    table.add_column("Edge", justify="right")
    table.add_column("Conf", justify="right", style="cyan")
    table.add_column("Signal", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("EV", justify="right")

    for i, s in enumerate(signals, 1):
        edge = s["edge"]
        edge_style = "green" if edge > 0.05 else "dim"

        if s["side"] == "BUY YES":
            sig_str = "[bold green]BUY YES[/]"
        elif s["side"] == "BUY NO":
            sig_str = "[bold red]BUY NO[/]"
        else:
            sig_str = "[dim]PASS[/]"

        size_str = f"${s['position_usd']:.0f}" if s["position_usd"] > 0 else "-"
        ev_str = f"{s['expected_value']:+.3f}" if s["position_usd"] > 0 else "-"

        table.add_row(
            str(i),
            s["market_question"][:40],
            f"{s['market_price']:.3f}",
            f"{s['our_probability']:.3f}",
            f"[{edge_style}]{edge:+.3f}[/]",
            f"{s['confidence']:.2f}",
            sig_str,
            size_str,
            ev_str,
        )

    console.print(table)

    # Summary
    actionable = [s for s in signals if s["side"] != "PASS"]
    total_exposure = sum(s["position_usd"] for s in actionable)
    avg_edge = np.mean([s["edge"] for s in actionable]) if actionable else 0

    console.print(Panel(
        f"Markets with analyses: {len(signals)}\n"
        f"Actionable signals: {len(actionable)}\n"
        f"Total exposure: ${total_exposure:.2f}\n"
        f"Average edge: {avg_edge:.3f}\n"
        f"Mode: [bold yellow]PAPER TRADING[/]",
        title="Aggregation Summary",
        border_style="green",
    ))


def save_consensus(consensuses: list[dict], signals: list[dict]) -> None:
    """Save consensus and signals to shared_state/consensus/."""
    CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual consensus files
    for c, s in zip(consensuses, signals):
        mid = c.get("market_id", "unknown")[:12]
        result = {**c, **s, "timestamp": timestamp}
        path = CONSENSUS_DIR / f"consensus_{mid}_{timestamp}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save combined signals file
    combined_path = CONSENSUS_DIR / f"all_signals_{timestamp}.json"
    combined = {
        "timestamp": timestamp,
        "generated_at": datetime.now().isoformat(),
        "total_markets": len(signals),
        "actionable": len([s for s in signals if s["side"] != "PASS"]),
        "signals": signals,
    }
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[green]Saved to shared_state/consensus/[/]")
    console.print(f"  Combined: {combined_path.name}")


def export_signals(signals: list[dict], export_path: str) -> None:
    """Export signals to a standalone JSON file."""
    path = Path(export_path)
    output = {
        "generated_at": datetime.now().isoformat(),
        "signals": signals,
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Exported to {path}[/]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate Fish analyses into trade signals")
    parser.add_argument("--market", type=str, default="", help="Filter by market keyword")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Min edge threshold")
    parser.add_argument("--bankroll", type=float, default=1000, help="Bankroll for position sizing")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (0.25 = quarter)")
    parser.add_argument("--export", type=str, default="", help="Export signals to file")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]MIROFISH AGGREGATION ENGINE[/]\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Bankroll: ${args.bankroll:,.0f} | Kelly: {args.kelly}x | Min edge: {args.min_edge}",
        border_style="cyan",
    ))

    # Load data
    console.print("\n[bold]Loading data...[/]")
    markets = load_market_data()
    analyses_by_market = load_analyses()

    console.print(f"  Markets loaded: {len(markets)}")
    console.print(f"  Markets with analyses: {len(analyses_by_market)}")

    total_analyses = sum(len(v) for v in analyses_by_market.values())
    console.print(f"  Total Fish analyses: {total_analyses}")

    if total_analyses == 0:
        console.print("\n[yellow]No Fish analyses found in shared_state/analyses/[/]")
        console.print("[dim]Run Fish agents first. See docs/USER_MANUAL.md for instructions.[/]")
        console.print("[dim]")
        console.print("[dim]Quick start:[/]")
        console.print("[dim]  1. python scan_markets.py          # Fetch markets[/]")
        console.print("[dim]  2. Open Fish folders in VS Code     # Analyze with Claude Max[/]")
        console.print("[dim]  3. python aggregate.py              # You are here[/]")
        return

    # Filter by keyword
    if args.market:
        q = args.market.lower()
        filtered = {}
        for mid, analyses in analyses_by_market.items():
            market_data = markets.get(mid, {})
            question = market_data.get("question", "") or (analyses[0].get("market_question", "") if analyses else "")
            if q in question.lower():
                filtered[mid] = analyses
        analyses_by_market = filtered
        console.print(f"  After filter '{args.market}': {len(analyses_by_market)} markets")

    if not analyses_by_market:
        console.print("[yellow]No matching analyses found.[/]")
        return

    # Aggregate each market
    console.print("\n[bold]Aggregating...[/]")
    consensuses = []
    signals = []

    for market_id, analyses in analyses_by_market.items():
        market_data = markets.get(market_id, {})
        market_question = market_data.get("question", "") or (analyses[0].get("market_question", "Unknown") if analyses else "Unknown")
        market_price = market_data.get("yes_price", 0.5)

        # Bayesian aggregation
        consensus = bayesian_aggregate(analyses)
        consensus["market_id"] = market_id
        consensus["market_question"] = market_question
        consensus["market_price"] = market_price
        consensuses.append(consensus)

        # Trade signal
        signal = compute_signal(
            market_id=market_id,
            market_question=market_question,
            our_probability=consensus["probability"],
            market_price=market_price,
            confidence=consensus["confidence"],
            bankroll=args.bankroll,
            kelly_fraction=args.kelly,
            min_edge=args.min_edge,
        )
        signals.append(signal)

    # Sort by absolute edge descending
    paired = sorted(zip(consensuses, signals), key=lambda x: abs(x[1]["edge"]), reverse=True)
    consensuses = [p[0] for p in paired]
    signals = [p[1] for p in paired]

    # Display
    display_results(consensuses, signals)

    # Save
    save_consensus(consensuses, signals)

    # Export
    if args.export:
        export_signals(signals, args.export)

    # Generate visualization
    try:
        from src.visualization.plots import plot_swarm_prediction, plot_edge_distribution

        viz_dir = Path("data/processed")
        viz_dir.mkdir(parents=True, exist_ok=True)

        edges = [s["edge"] for s in signals if s["side"] != "PASS"]
        if edges:
            fig = plot_edge_distribution(edges)
            fig.write_html(str(viz_dir / "edge_distribution.html"))
            console.print(f"  Viz: data/processed/edge_distribution.html")

        # Best signal detail
        best_consensus = next((c for c in consensuses if c.get("fish_details")), None)
        if best_consensus and best_consensus.get("fish_details"):
            fd = best_consensus["fish_details"]
            fig = plot_swarm_prediction(
                fish_probabilities=[f["probability"] for f in fd],
                fish_personas=[f["fish_name"] for f in fd],
                fish_confidences=[f["confidence"] for f in fd],
                swarm_probability=best_consensus["probability"],
                market_price=best_consensus.get("market_price"),
                market_question=best_consensus.get("market_question", ""),
            )
            fig.write_html(str(viz_dir / "best_signal_swarm.html"))
            console.print(f"  Viz: data/processed/best_signal_swarm.html")
    except ImportError:
        pass  # Visualization optional


if __name__ == "__main__":
    main()
