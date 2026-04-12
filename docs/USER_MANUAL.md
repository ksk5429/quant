# User Manual

> Step-by-step guide for operating the K-Fish prediction engine.

## Table of Contents
1. [Getting Started](#1-getting-started)
2. [Running the Demo](#2-running-the-demo)
3. [Operating the Swarm](#3-operating-the-swarm)
4. [Interpreting Results](#4-interpreting-results)
5. [Common Workflows](#5-common-workflows)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Getting Started

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12 |
| RAM | 4 GB | 8 GB |
| Internet | Required | Stable connection |
| Anthropic API key | Required for live mode | $5+ credits |
| VS Code | Optional | Recommended (for Swarm Intelligence) |

### Installation

```bash
# Clone the repository
git clone https://github.com/ksk5429/quant.git
cd quant

# Install dependencies
pip install -e ".[dev]"

# Setup API keys (interactive)
python setup.py

# Verify installation
pytest tests/ -v
```

### API Key Setup

**Option A: Interactive setup (recommended)**
```bash
python setup.py
```

**Option B: Environment variable**
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

**Option C: Config file**
Edit `config/local.yaml`:
```yaml
api_keys:
  anthropic: "sk-ant-api03-your-key-here"
```

---

## 2. Running the Demo

### Stub Mode (No API key needed)

```bash
python demo_live.py --stub --markets 5 --fish 5
```

This fetches **real markets from Polymarket** but uses deterministic hash-based analysis (no LLM calls). Good for testing the pipeline.

### Live Mode (Requires API key)

```bash
# Default: 5 markets, 5 Fish, Haiku model (~$0.001 per Fish analysis)
python demo_live.py

# More markets, more Fish
python demo_live.py --markets 10 --fish 7

# Stronger model (more expensive but better analysis)
python demo_live.py --model claude-sonnet-4-6

# Custom bankroll
python demo_live.py --bankroll 5000
```

### Cost Estimates

| Model | Cost per Fish Analysis | 5 Fish x 5 Markets | 7 Fish x 10 Markets |
|-------|----------------------|---------------------|----------------------|
| claude-haiku-4-5 | ~$0.001 | ~$0.025 | ~$0.07 |
| claude-sonnet-4-6 | ~$0.01 | ~$0.25 | ~$0.70 |
| claude-opus-4-6 | ~$0.05 | ~$1.25 | ~$3.50 |

---

## 3. Operating the Swarm

### Method A: Programmatic Swarm (Single Terminal)

All Fish run as async coroutines in one Python process:

```bash
python demo_live.py --markets 5 --fish 7
```

**Pros:** Simple, fast, automated aggregation.
**Cons:** All Fish use the same model and temperature.

### Method B: VS Code Swarm (Multiple Windows)

Each Fish is a separate Claude Code instance:

**Step 1:** Open the master intelligence
```bash
code f:/TREE_OF_THOUGHT/QUANT
```

**Step 2:** Open Fish agents in separate VS Code windows
```bash
code f:/TREE_OF_THOUGHT/QUANT/Swarm_Intelligence/fish_geopolitical
code f:/TREE_OF_THOUGHT/QUANT/Swarm_Intelligence/fish_quant
code f:/TREE_OF_THOUGHT/QUANT/Swarm_Intelligence/fish_contrarian
```

**Step 3:** In the master window, write a market to analyze
```bash
# Master creates market_data file
echo '{"id":"test1","question":"Will BTC exceed 100k by Dec 2026?","description":"...","yes_price":0.45}' > shared_state/market_data/market_test1.json
```

**Step 4:** In each Fish window, prompt Claude Code:
```
Read the market in shared_state/market_data/market_test1.json
and analyze it. Write your analysis to shared_state/analyses/
following the protocol in CLAUDE.md.
```

**Step 5:** In the master window, read all analyses:
```
Read all files in shared_state/analyses/ and aggregate them
using Bayesian confidence-weighted fusion.
```

**Pros:** Each Fish has full Claude Code capabilities, different models possible, human can intervene per Fish.
**Cons:** Requires multiple VS Code windows, manual coordination.

---

## 4. Interpreting Results

### The Results Table

```
# | Market    | Market P | Swarm P | Edge   | Conf | Spread | Signal  | Size
1 | Will BTC  |   0.450  |  0.620  | +0.170 | 0.72 |  0.250 | BUY YES |  $42
2 | Will ETH  |   0.300  |  0.280  | -0.020 | 0.55 |  0.400 | PASS    |   -
3 | Fed rate  |   0.700  |  0.550  | -0.150 | 0.68 |  0.300 | BUY NO  |  $38
```

**Column meanings:**

| Column | Meaning | Good Value |
|--------|---------|------------|
| Market P | Current Polymarket price (implied probability) | N/A |
| Swarm P | Our ensemble prediction | Differs from Market P |
| Edge | Swarm P - Market P (for YES) | > +0.05 or < -0.05 |
| Conf | Ensemble confidence | > 0.60 |
| Spread | Max - Min across Fish | < 0.30 (agreement) |
| Signal | BUY YES / BUY NO / PASS | Actionable when edge > 5% |
| Size | Dollar amount to risk | Within risk limits |

### What the Signals Mean

| Signal | Meaning | Action |
|--------|---------|--------|
| **BUY YES** | We think the event is MORE likely than the market says | Buy YES shares |
| **BUY NO** | We think the event is LESS likely than the market says | Buy NO shares |
| **PASS** | Edge too small or confidence too low | Do nothing |

### Warning Signs

- **High spread (> 0.40):** Fish strongly disagree. The prediction is unreliable. Wait for more information.
- **Low confidence (< 0.40):** The ensemble isn't sure. Reduce position size or skip.
- **All Fish on one side:** Possible groupthink. The contrarian Fish should disagree sometimes. If it doesn't, the system may be overconfident.
- **Edge > 0.30:** Suspiciously large. Either the market is very thin or our analysis has a blind spot. Double-check.

---

## 5. Common Workflows

### Workflow 1: Daily Market Scan

```bash
# Fetch top markets, analyze with stub, identify candidates
python demo_live.py --stub --markets 20 --min-volume 100000

# Re-analyze top candidates with live LLM
python demo_live.py --markets 5 --fish 7 --model claude-sonnet-4-6
```

### Workflow 2: Event-Driven Re-Analysis

When a major news event happens:

```python
# In Python or via the master Claude Code window:
from src.mirofish.swarm import Swarm
import anthropic

swarm = Swarm(num_fish=5, llm_client=anthropic.Anthropic())

# Inject the event
await swarm.inject_event(
    "Supreme Court blocks crypto regulation bill",
    urgency="high"
)

# Re-analyze affected markets
prediction = await swarm.analyze_market(
    market_id="crypto-reg-001",
    market_question="Will the crypto regulation bill pass by 2026?",
)
```

### Workflow 3: Performance Review

After markets resolve:

```python
# Record actual outcomes
scores = swarm.record_outcome("market-123", actual_outcome=1.0)
print(f"Ensemble Brier: {scores['ensemble']:.4f}")

# Check individual Fish performance
summary = swarm.performance_summary
print(f"Best Fish: {summary['best_fish']}")
print(f"Worst Fish: {summary['worst_fish']}")
```

---

## 6. Troubleshooting

### "Credit balance too low"
Your Anthropic API account needs credits.
- Go to https://console.anthropic.com/settings/billing
- Add at least $5
- Or use `--stub` mode for testing

### "No markets found"
The volume filter may be too high.
- Try: `--min-volume 10000` (lower threshold)
- Or: `--min-volume 0` (all markets)

### Unicode errors on Windows
```bash
set PYTHONUTF8=1
python demo_live.py
```

### Import errors
```bash
pip install -e ".[dev]"
```

### Tests failing
```bash
# Run with verbose output
pytest tests/ -v --tb=long

# Run specific test file
pytest tests/unit/test_fish.py -v
```
