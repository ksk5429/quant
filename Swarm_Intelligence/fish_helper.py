"""Fish Helper — Read markets and write structured analyses.

Run from any Fish folder to list available markets and write your analysis.

Usage (from inside a Fish folder, e.g. fish_geopolitical/):
    python ../fish_helper.py list                    # List markets to analyze
    python ../fish_helper.py read market_553882      # Read a specific market
    python ../fish_helper.py write                   # Interactive analysis writer
    python ../fish_helper.py status                  # See what's been analyzed
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

# Resolve paths relative to this script (in Swarm_Intelligence/)
SCRIPT_DIR = Path(__file__).parent
SHARED_STATE = SCRIPT_DIR.parent / "shared_state"
MARKET_DATA = SHARED_STATE / "market_data"
ANALYSES_DIR = SHARED_STATE / "analyses"
EVENTS_DIR = SHARED_STATE / "events"


def cmd_list(args):
    """List all markets available for analysis."""
    if not MARKET_DATA.exists():
        print("No markets found. Run: python scan_markets.py")
        return

    markets = []
    for path in sorted(MARKET_DATA.glob("market_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        markets.append(data)

    if not markets:
        print("No markets found. Run: python scan_markets.py")
        return

    print(f"\n{'#':<4} {'Question':<55} {'YES':>6} {'Volume':>12} {'File'}")
    print("-" * 95)
    for i, m in enumerate(markets, 1):
        q = m["question"][:55]
        print(f"{i:<4} {q:<55} {m['yes_price']:>6.3f} ${m['volume']:>10,.0f} {Path(MARKET_DATA / f'market_{m[\"id\"][:12]}.json').name}")

    # Show analysis status
    print(f"\n--- Analysis Status ---")
    existing = set()
    if ANALYSES_DIR.exists():
        for a in ANALYSES_DIR.rglob("*.json"):
            try:
                data = json.loads(a.read_text(encoding="utf-8"))
                mid = data.get("market_id", "")
                fname = data.get("fish_name", "")
                existing.add(f"{fname}:{mid}")
            except Exception:
                pass

    cwd_name = Path.cwd().name
    fish_name = cwd_name if cwd_name.startswith("fish_") else "unknown_fish"

    analyzed = 0
    pending = 0
    for m in markets:
        key = f"{fish_name}:{m['id']}"
        if key in existing:
            analyzed += 1
        else:
            pending += 1

    print(f"  Your Fish ({fish_name}): {analyzed} analyzed, {pending} pending")
    print(f"  Total analyses in shared_state: {len(existing)}")


def cmd_read(args):
    """Read a specific market's details."""
    # Find matching market
    for path in MARKET_DATA.glob("market_*.json"):
        if args.market_id in path.stem or args.market_id in path.read_text(encoding="utf-8"):
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"\n=== MARKET: {data['question']} ===")
            print(f"ID:          {data['id']}")
            print(f"Category:    {data.get('category', 'N/A')}")
            print(f"YES Price:   {data['yes_price']:.4f}")
            print(f"NO Price:    {data['no_price']:.4f}")
            print(f"Volume:      ${data['volume']:,.0f}")
            print(f"End Date:    {data.get('end_date', 'N/A')}")
            if data.get("description"):
                print(f"\nDescription:\n{data['description'][:500]}")

            # Show existing analyses
            print(f"\n--- Existing Analyses ---")
            if ANALYSES_DIR.exists():
                for a_path in ANALYSES_DIR.rglob("*.json"):
                    try:
                        a_data = json.loads(a_path.read_text(encoding="utf-8"))
                        if a_data.get("market_id") == data["id"]:
                            print(f"  {a_data.get('fish_name', '?')}: P={a_data.get('probability', '?'):.3f} (conf={a_data.get('confidence', '?'):.2f})")
                    except Exception:
                        pass

            # Show events
            if EVENTS_DIR.exists():
                events = sorted(EVENTS_DIR.glob("*.json"))
                if events:
                    print(f"\n--- Recent Events ---")
                    for e_path in events[-3:]:
                        e_data = json.loads(e_path.read_text(encoding="utf-8"))
                        print(f"  [{e_data.get('timestamp', '?')}] {e_data.get('event', '?')[:80]}")
            return

    print(f"Market '{args.market_id}' not found. Run: python ../fish_helper.py list")


def cmd_write(args):
    """Interactive analysis writer."""
    cwd_name = Path.cwd().name
    fish_name = cwd_name if cwd_name.startswith("fish_") else input("Fish name (e.g. fish_quant): ").strip()

    # List markets
    cmd_list(argparse.Namespace())

    market_id = input("\nEnter market ID (or # from list): ").strip()

    # Resolve # to market ID
    if market_id.isdigit():
        idx = int(market_id) - 1
        paths = sorted(MARKET_DATA.glob("market_*.json"))
        if 0 <= idx < len(paths):
            data = json.loads(paths[idx].read_text(encoding="utf-8"))
            market_id = data["id"]
            market_question = data["question"]
        else:
            print("Invalid number.")
            return
    else:
        # Try to find by partial ID
        market_question = ""
        for path in MARKET_DATA.glob("market_*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if market_id in data["id"]:
                market_id = data["id"]
                market_question = data["question"]
                break

    print(f"\nAnalyzing: {market_question}")
    print(f"Fish: {fish_name}")
    print()

    # Collect analysis
    prob_str = input("Your probability estimate (0.0 to 1.0): ").strip()
    probability = float(prob_str)

    conf_str = input("Your confidence (0.0 to 1.0): ").strip()
    confidence = float(conf_str)

    print("\nReasoning steps (enter each step, empty line to finish):")
    reasoning = []
    while True:
        step = input(f"  Step {len(reasoning)+1}: ").strip()
        if not step:
            break
        reasoning.append(step)

    print("\nKey evidence (enter each item, empty line to finish):")
    evidence = []
    while True:
        item = input("  - ").strip()
        if not item:
            break
        evidence.append(item)

    print("\nRisk factors (enter each item, empty line to finish):")
    risks = []
    while True:
        item = input("  - ").strip()
        if not item:
            break
        risks.append(item)

    # Build analysis JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis = {
        "fish_name": fish_name,
        "market_id": market_id,
        "market_question": market_question,
        "probability": round(probability, 4),
        "confidence": round(confidence, 4),
        "reasoning_steps": reasoning,
        "key_evidence": evidence,
        "risk_factors": risks,
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSES_DIR / f"{fish_name}_{market_id[:12]}_{timestamp}.json"
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAnalysis saved to: {out_path}")
    print(f"  P={probability:.3f}, Conf={confidence:.2f}, Steps={len(reasoning)}")
    print(f"\nNext: Run 'python aggregate.py' from the QUANT root to generate signals.")


def cmd_status(args):
    """Show overall swarm analysis status."""
    markets = {}
    if MARKET_DATA.exists():
        for p in MARKET_DATA.glob("market_*.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            markets[d["id"]] = d

    analyses = {}
    if ANALYSES_DIR.exists():
        for p in ANALYSES_DIR.rglob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                mid = d.get("market_id", "")
                fname = d.get("fish_name", "unknown")
                if mid not in analyses:
                    analyses[mid] = []
                analyses[mid].append(fname)
            except Exception:
                pass

    print(f"\n=== SWARM STATUS ===")
    print(f"Markets loaded: {len(markets)}")
    print(f"Markets analyzed: {len(analyses)}")
    print(f"Total analyses: {sum(len(v) for v in analyses.values())}")

    if markets:
        print(f"\n{'Market':<50} {'Fish Analyses':>15}")
        print("-" * 67)
        for mid, mdata in markets.items():
            q = mdata["question"][:50]
            fish_list = analyses.get(mid, [])
            count = len(fish_list)
            fish_str = ", ".join(f.replace("fish_", "") for f in fish_list) if fish_list else "[none]"
            print(f"{q:<50} {count:>3} ({fish_str})")


def main():
    parser = argparse.ArgumentParser(description="Fish Helper — read markets, write analyses")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available markets")

    read_p = sub.add_parser("read", help="Read a specific market")
    read_p.add_argument("market_id", help="Market ID or partial match")

    sub.add_parser("write", help="Interactive analysis writer")
    sub.add_parser("status", help="Show swarm analysis status")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "read":
        cmd_read(args)
    elif args.command == "write":
        cmd_write(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
