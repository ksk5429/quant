# Swarm Protocol

> How Fish agents communicate, coordinate, and collectively produce intelligence.

## Overview

K-Fish uses a **hybrid Strategy A + C architecture**: a tiered pipeline (Trading Floor) with adversarial debate rounds (Debate Tournament) at the analysis tier.

```
TIER 1: RESEARCH         Gather data, news, context
         |
TIER 2: ANALYSIS          Produce probability estimates
         |                  + Adversarial debate rounds
         |
TIER 3: CALIBRATION       Meta-analysis, bias detection, trade signals
```

---

## The Seven Fish Personas

### Tier 1: Research Fish

These Fish gather raw information. They don't produce final probabilities — they produce research briefs for Tier 2.

| Fish | Role | Writes To |
|------|------|-----------|
| **Investigative Journalist** | Finds primary sources, hidden info, narrative analysis | `shared_state/research/` |
| **Domain Expert** | Provides specialized knowledge (legal, scientific, technical) | `shared_state/research/` |

### Tier 2: Analysis Fish

These Fish read Tier 1 research and produce probability estimates. They **do not see market prices**.

| Fish | Role | Analytical Lens |
|------|------|-----------------|
| **Geopolitical Analyst** | Power dynamics, institutional behavior | Scenario-based |
| **Financial Quant** | Base rates, statistics, order books | Mathematical |
| **Bayesian Statistician** | Prior updating, evidence weighting | Probabilistic |
| **Contrarian Thinker** | Why consensus might be wrong | Adversarial |

### Tier 3: Meta-Analysis Fish

| Fish | Role | Special Power |
|------|------|---------------|
| **Calibration Specialist** | Reads ALL other Fish, checks for ensemble bias | Can override |

---

## Communication Protocol

### File-Based Communication

All communication happens through JSON files in `shared_state/`:

```
shared_state/
+-- market_data/                   Master -> Fish (what to analyze)
|   +-- market_{id}.json
+-- events/                        Master -> Fish (real-world events)
|   +-- {YYYYMMDD_HHMMSS}_event.json
+-- research/                      Tier 1 -> Tier 2
|   +-- {fish_name}_{market_id}_{timestamp}.json
+-- analyses/                      Tier 2 -> Tier 3 -> Master
|   +-- round_1/                   Independent analysis
|   +-- round_2/                   Post-debate revision
|   +-- {fish_name}_{market_id}_{timestamp}.json
+-- debates/                       Paired adversarial debates
|   +-- {fish_A}_vs_{fish_B}_{market_id}_{timestamp}.json
+-- signals/                       Fish -> Fish (direct messages)
|   +-- {from}_to_{to}_{timestamp}.json
+-- consensus/                     Master -> public (aggregated result)
    +-- consensus_{market_id}_{timestamp}.json
```

### File Naming Convention

```
{fish_name}_{market_id}_{YYYYMMDD_HHMMSS}.json
```

- Timestamp-keyed filenames prevent collision (no file locking needed)
- Each Fish writes to its own namespace — no overwriting others' files
- Any human can read the JSON to inspect reasoning

### JSON Analysis Format

Every Fish analysis must contain:

```json
{
  "fish_name": "fish_quant",
  "market_id": "abc123",
  "market_question": "Will Bitcoin exceed $100k by Dec 2026?",
  "probability": 0.62,
  "confidence": 0.75,
  "reasoning_steps": [
    "Step 1: Historical base rate of BTC crossing major thresholds...",
    "Step 2: Current momentum and halving cycle analysis...",
    "Step 3: Macro environment (interest rates, liquidity)..."
  ],
  "key_evidence": [
    "BTC has crossed previous ATH within 18 months of halving (3/3 times)",
    "Current price $87k, needs 15% gain, within historical post-halving range"
  ],
  "risk_factors": [
    "Regulatory crackdown could suppress price",
    "Global recession could reduce risk appetite"
  ],
  "timestamp": "2026-04-12T15:30:00Z"
}
```

---

## The Pipeline (Step by Step)

### Round 0: Market Assignment

The Master (GOD Node) writes market data to `shared_state/market_data/`:

```json
{
  "id": "btc-100k-2026",
  "question": "Will Bitcoin exceed $100k by December 2026?",
  "description": "Resolves YES if BTC/USD exceeds $100,000...",
  "category": "crypto",
  "yes_price": 0.45,
  "volume": 5000000,
  "end_date": "2026-12-31"
}
```

### Round 1: Independent Analysis

Each Tier 2 Fish analyzes the market **independently, without seeing other Fish or market price**.

```
Fish_Geopolitical --> shared_state/analyses/round_1/fish_geopolitical_btc100k_*.json
Fish_Quant        --> shared_state/analyses/round_1/fish_quant_btc100k_*.json
Fish_Bayesian     --> shared_state/analyses/round_1/fish_bayesian_btc100k_*.json
Fish_Contrarian   --> shared_state/analyses/round_1/fish_contrarian_btc100k_*.json
```

**Rule:** No Fish reads another Fish's Round 1 output until all have submitted.

### Round 2: Adversarial Debate

Fish are paired for structured debate:

| Pairing | Purpose |
|---------|---------|
| Geopolitical vs. Contrarian | Challenge geopolitical assumptions |
| Quant vs. Bayesian | Challenge statistical reasoning |

**Debate format:**
1. Fish A reads Fish B's Round 1 analysis
2. Fish A writes a response challenging Fish B's weakest points
3. Fish B reads Fish A's challenge and responds
4. Both write revised probability estimates

Debate files go to `shared_state/debates/`:

```json
{
  "debate_id": "geopolitical_vs_contrarian_btc100k",
  "market_id": "btc-100k-2026",
  "fish_a": "fish_geopolitical",
  "fish_b": "fish_contrarian",
  "fish_a_round1_probability": 0.65,
  "fish_b_round1_probability": 0.40,
  "fish_a_challenge": "The contrarian ignores the halving cycle...",
  "fish_b_response": "The geopolitical analyst overweights momentum...",
  "fish_a_revised_probability": 0.60,
  "fish_b_revised_probability": 0.45
}
```

### Round 3: Final Estimates

After the debate, all Fish write **final** probability estimates to `shared_state/analyses/round_2/`.

These estimates incorporate:
- Their original analysis (Round 1)
- Insights from the debate (Round 2)
- Any Tier 1 research they hadn't seen before

### Round 4: Aggregation

The Master reads all Round 2 analyses and applies:
1. Bayesian confidence-weighted aggregation
2. Isotonic calibration (if enough history)
3. Quarter-Kelly position sizing
4. Trade signal generation

Results written to `shared_state/consensus/`.

---

## Event Injection Protocol

When a real-world event occurs:

### Step 1: Human Detection
You (the human operator) notice a significant event:
> "Supreme Court blocks crypto regulation bill"

### Step 2: GOD Node Broadcast
Write to `shared_state/events/`:

```json
{
  "event": "Supreme Court blocks crypto regulation bill in 5-4 decision",
  "timestamp": "2026-04-12T16:00:00Z",
  "urgency": "high",
  "affected_categories": ["crypto", "regulation", "politics"],
  "source": "Reuters"
}
```

### Step 3: Fish Re-Analysis
Each Fish reads the event and writes a **new** analysis incorporating the event.

### Step 4: Re-Aggregation
Master re-aggregates with the new analyses.

---

## Adding a New Fish

1. Create folder: `Swarm_Intelligence/fish_newname/`
2. Create `CLAUDE.md` with:
   - Persona description
   - Communication protocol (read/write paths)
   - Analytical rules
3. Open in VS Code with Claude Code
4. Start analyzing

**Template:**
```markdown
# Fish: [Name]

## Identity
You are a **[role]** Fish agent in the K-Fish swarm.

## Persona
- [Key analytical trait 1]
- [Key analytical trait 2]
- [Key analytical trait 3]

## Communication Protocol
- Read: `../../shared_state/market_data/`, `../../shared_state/events/`
- Write: `../../shared_state/analyses/fish_[name]_{market_id}_{timestamp}.json`
```

---

## Performance Tracking

### Per-Fish Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Brier Score | mean((P_predicted - outcome)^2) | < 0.20 |
| Calibration | ECE across 10 bins | < 0.05 |
| Hit Rate | % of directional predictions correct | > 55% |
| Confidence Accuracy | correlation(confidence, correctness) | > 0.3 |

### Swarm Metrics

| Metric | What It Measures |
|--------|-----------------|
| Ensemble Brier | Overall prediction quality |
| Diversity Score | spread across Fish (higher = more diverse, usually better) |
| Debate Impact | How much Round 2 changes from Round 1 (should be moderate) |
| Contrarian Value | How often the contrarian was right when others were wrong |
