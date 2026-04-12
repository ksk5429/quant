# Max-Plan Swarm Workflow

> Complete step-by-step workflow using Claude Max (no API credits needed).

## The 3-Step Pipeline

```
Step 1: python scan_markets.py        # Fetch real Polymarket data (free)
Step 2: Open Fish in VS Code          # Analyze with Claude Max (unlimited)
Step 3: python aggregate.py           # Aggregate into trade signals (free)
```

**Total API cost: $0.00** — everything runs on your Claude Max subscription.

---

## Step 1: Scan Markets

From the QUANT root:

```bash
python scan_markets.py --limit 10 --min-volume 50000
```

This writes JSON files to `shared_state/market_data/`. Each file contains:
- Market question
- Current YES/NO prices
- Volume and liquidity
- Description

---

## Step 2: Fish Analysis (Claude Max)

### Option A: Claude Code in VS Code (recommended)

Open 3-7 Fish folders in separate VS Code windows:

```bash
code Swarm_Intelligence/fish_geopolitical
code Swarm_Intelligence/fish_quant
code Swarm_Intelligence/fish_contrarian
```

In each Fish window, prompt Claude Code:

```
Read the markets in ../../shared_state/market_data/ and analyze them.
For each market, write your analysis as a JSON file to 
../../shared_state/analyses/ following the format in CLAUDE.md.

Focus on the markets where you have the strongest analytical edge
given your persona.
```

Claude Code (running on your Max plan) will:
1. Read the market data files
2. Apply the persona-specific analysis from CLAUDE.md
3. Write structured JSON analyses to shared_state/analyses/

### Option B: Interactive Fish Helper

From any Fish folder:

```bash
cd Swarm_Intelligence/fish_quant
python ../fish_helper.py list          # See available markets
python ../fish_helper.py read 561251   # Read a specific market
python ../fish_helper.py write         # Interactive analysis writer
python ../fish_helper.py status        # See what's been analyzed
```

### Option C: Direct JSON Writing

Write analysis files directly (by hand or via Claude in any window):

```bash
# Template — copy, edit, save
cat > shared_state/analyses/fish_quant_561251_20260412.json << 'EOF'
{
  "fish_name": "fish_quant",
  "market_id": "561251",
  "market_question": "Will LeBron James win the 2028 US Presidential Election",
  "probability": 0.003,
  "confidence": 0.90,
  "reasoning_steps": [
    "Base rate: 2/46 non-politicians elected = 4.3%",
    "No FEC filing, no campaign staff, no declared interest",
    "Market at 0.005 may already overvalue this"
  ],
  "key_evidence": [
    "Zero political infrastructure",
    "Focus on business empire, not politics"
  ],
  "risk_factors": [
    "Celebrity-to-president path exists (Trump precedent)",
    "Thin market susceptible to whale manipulation"
  ],
  "timestamp": "2026-04-12T15:30:00Z"
}
EOF
```

---

## Step 3: Aggregate

From the QUANT root:

```bash
python aggregate.py                     # Standard aggregation
python aggregate.py --bankroll 5000     # Custom bankroll
python aggregate.py --min-edge 0.10     # Higher edge threshold
python aggregate.py --export signals.json  # Export to file
```

This reads all analyses from `shared_state/analyses/`, applies:
1. Bayesian confidence-weighted aggregation
2. Quarter-Kelly position sizing
3. Generates trade signals

Output:
- Rich table showing all Fish contributions and consensus
- Trade signals (BUY YES / BUY NO / PASS)
- Files saved to `shared_state/consensus/`
- Plotly visualizations to `data/processed/`

---

## Tips for Best Results

1. **Use at least 3 Fish** — Geopolitical + Quant + Contrarian is the minimum effective set
2. **The Contrarian matters most** — if all Fish agree, the system might be overconfident
3. **Don't show Fish the market price** — let Claude analyze independently
4. **Run the Calibrator last** — it reads all other Fish analyses and checks for bias
5. **Diversity > quantity** — 3 diverse Fish beats 7 similar Fish
6. **Save your analyses** — they become training data for future calibration
