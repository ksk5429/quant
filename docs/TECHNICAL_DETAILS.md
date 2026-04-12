# Technical Details

> Complete technical documentation of the Mirofish prediction engine.

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Layer](#data-layer)
3. [Swarm Layer](#swarm-layer)
4. [Prediction Layer](#prediction-layer)
5. [Network Layer](#network-layer)
6. [Risk Layer](#risk-layer)
7. [Visualization Layer](#visualization-layer)
8. [Configuration System](#configuration-system)
9. [Testing](#testing)

---

## System Overview

Mirofish is a **six-layer prediction pipeline**:

```
Layer 1: DATA         Polymarket Gamma/CLOB API -> market questions, prices, volume
Layer 2: SWARM        N Fish agents analyze independently (Claude API)
Layer 3: PREDICTION   Bayesian aggregation + isotonic calibration
Layer 4: NETWORK      Market correlation graph (semantic + price)
Layer 5: RISK         Quarter-Kelly position sizing + drawdown protection
Layer 6: VISUALIZATION Plotly dashboards, heatmaps, network graphs
```

### Data Flow

```
Polymarket API ──> Market Data ──> Fish Agent 1 ──┐
                                   Fish Agent 2 ──┤
                                   Fish Agent 3 ──┤──> Bayesian ──> Calibration ──> Kelly ──> Signal
                                   Fish Agent 4 ──┤    Aggregation   (isotonic)    Criterion
                                   Fish Agent 5 ──┘
```

Key design principle: **Fish never see the market price during analysis.** This prevents anchoring bias and preserves the independence required for ensemble accuracy. The market price is only incorporated at the aggregation stage as a Bayesian prior.

---

## Data Layer

### Polymarket Gamma API (Market Metadata)

**Base URL:** `https://gamma-api.polymarket.com`
**Authentication:** None required for reads
**Rate Limit:** ~60 requests/minute

**Endpoints used:**

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `GET /markets` | List active markets | `?limit=100&active=true` |
| `GET /events` | List events (contain multiple markets) | `?limit=50` |

**Response parsing:**
- `outcomePrices` is a JSON-encoded string: `'["0.535","0.465"]'`
- `clobTokenIds` are comma-separated: `"token_yes,token_no"`
- `volume` is cumulative USD volume (not 24h)

**Implementation:** [src/markets/polymarket.py](../src/markets/polymarket.py)

The `GammaClient` class handles:
- Async HTTP via `httpx.AsyncClient`
- JSON parsing of Polymarket's non-standard response formats
- Filtering by volume, liquidity, and active status
- Market search by keyword

### Polymarket CLOB API (Trading)

**Base URL:** `https://clob.polymarket.com`
**Authentication:** HMAC-SHA256 (L2 level)
**Chain:** Polygon (chain_id=137)

**Read-only endpoints (no auth):**

| Endpoint | Purpose |
|----------|---------|
| `GET /price?token_id=X&side=BUY` | Best bid/ask |
| `GET /midpoint?token_id=X` | Midpoint price |
| `GET /book?token_id=X` | Full order book |
| `GET /spread?token_id=X` | Bid-ask spread |

**Trading endpoints (L2 auth required):**

| Endpoint | Purpose |
|----------|---------|
| `POST /order` | Place limit order |
| `DELETE /order` | Cancel order |

Trading is **gated behind `paper_trading: true`** in configuration. The system never places real orders unless explicitly configured and approved by the human operator.

---

## Swarm Layer

### Fish Agent (`src/mirofish/fish.py`)

Each Fish is an independent LLM-powered analyst with:

**Persona:** A system prompt defining the Fish's analytical lens (7 types available). See [Swarm Protocol](SWARM_PROTOCOL.md) for persona definitions.

**Input:** Market question + description + optional news context. **Never the current price.**

**Output:** A `FishAnalysis` object containing:
```python
FishAnalysis:
    fish_id: str              # Unique identifier
    persona: FishPersona      # Which analytical lens
    market_id: str            # Which market
    probability: float        # P(YES), range [0, 1]
    confidence: float         # Self-assessed confidence [0, 1]
    reasoning_steps: list     # Step-by-step chain for human review
    key_evidence: list        # Evidence items
    risk_factors: list        # Identified uncertainties
    tokens_used: int          # LLM token consumption
    latency_ms: float         # Response time
```

**LLM Integration:**
- Supports Anthropic (Claude) and OpenAI clients
- Structured output parsing: `PROBABILITY:`, `CONFIDENCE:`, `REASONING:`, etc.
- Async execution via `asyncio.to_thread` (non-blocking)
- Deterministic stub mode for testing (SHA-256 hash of persona + question)

**Performance tracking:**
- Each Fish maintains a history of all analyses
- Brier scores computed per Fish when outcomes are recorded
- Running average Brier score used for aggregation weight adjustment

### Swarm Orchestrator (`src/mirofish/swarm.py`)

The Swarm manages N Fish and coordinates the analysis pipeline:

1. **Spawn:** Creates Fish with diverse personas (cycles through all 7 types)
2. **Concurrent analysis:** `asyncio.gather` with semaphore rate limiting
3. **Aggregation:** Bayesian confidence-weighted fusion (see [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md))
4. **History:** Stores all `SwarmPrediction` objects for performance tracking

**Concurrency control:**
```python
semaphore = asyncio.Semaphore(max_concurrent)  # Default: 5

async def bounded_analyze(fish):
    async with semaphore:
        return await fish.analyze(...)

results = await asyncio.gather(*[bounded_analyze(f) for f in self.fish])
```

This prevents API rate limit violations while maximizing throughput.

### GOD Node (`src/mirofish/god_node.py`)

The GOD (Global Omniscient Dispatcher) Node handles real-world event injection:

1. **Receives** an event description from the human operator
2. **Identifies** which markets are affected (LLM-based or manual)
3. **Analyzes** impact direction and magnitude per market
4. **Broadcasts** via MessageBus to all affected Fish
5. **Triggers re-analysis** if impact exceeds threshold (default: 15% probability shift)

**Event propagation modes:**
- `broadcast`: All Fish receive the event
- `targeted`: Only Fish assigned to affected markets
- `cascade`: Event propagates through the correlation graph

### MessageBus (`src/mirofish/message_bus.py`)

Three routing modes for inter-agent communication:

| Mode | How It Works | When to Use |
|------|-------------|-------------|
| **Broadcast** | `target_ids=None` → all subscribers | GOD node events |
| **Targeted** | `target_ids=["fish_1", "fish_2"]` → named Fish only | Cross-market signals |
| **Topic-based** | `topic="politics"` → subscribers of that topic | Category-specific alerts |

Messages are delivered concurrently via `asyncio.gather` with error isolation — one failed handler doesn't block others.

---

## Prediction Layer

### Bayesian Aggregation

See [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md) for full derivation.

**Algorithm:**
```
weights[i] = confidence[i] * (1 + accuracy_bonus[i])
P_swarm = sum(weights[i] * P[i]) / sum(weights)
```

Where `accuracy_bonus` = `max(0, 1 - 2 * brier_score)`, rewarding Fish with lower (better) Brier scores.

### Probability Calibration (`src/prediction/calibration.py`)

Three methods available:

| Method | When | How |
|--------|------|-----|
| **Isotonic Regression** | >= 100 resolved predictions | Non-parametric monotonic function |
| **Platt Scaling** | < 100 predictions | Sigmoid fit to log-odds |
| **Temperature Scaling** | Quick baseline | Single scalar parameter |

**Evaluation metrics:**
- **Brier Score:** `mean((predicted - actual)^2)` — lower is better. Target: 0.18 (human superforecaster level)
- **ECE:** Expected Calibration Error — mean absolute gap between predicted and observed frequency
- **MCE:** Maximum Calibration Error — worst-case bin
- **Log-Loss:** Logarithmic scoring rule

---

## Network Layer

### Market Correlation Graph (`src/network/market_graph.py`)

A NetworkX weighted graph where:
- **Nodes** = prediction markets
- **Edges** = correlations between markets
- **Edge types:** semantic (embedding similarity), price (historical correlation), negation (arbitrage pair), causal

**Edge construction methods:**

1. **Semantic edges:** Cosine similarity of sentence-transformer embeddings. Markets about related topics get linked.
2. **Price correlation edges:** Pearson correlation of historical price series. Markets that move together get linked.
3. **Negation pair detection:** Heuristic matching of questions like "Will X?" and "Will X NOT?" — price deviations from sum=1.0 are arbitrage opportunities.

**Analysis capabilities:**
- **Centrality scoring:** Betweenness, eigenvector, PageRank, degree — identifies "systemically important" markets
- **Community detection:** Louvain modularity — groups correlated markets into clusters
- **Divergence matrix:** Jensen-Shannon divergence between all market pairs
- **Adjacency export:** NumPy matrix for GNN input

---

## Risk Layer

### Kelly Criterion (`src/risk/kelly.py`)

See [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md) for full derivation.

**Position sizing pipeline:**

```
1. Compute edge:     edge = |P_our - P_market|
2. Raw Kelly:        f* = edge / (1 - P_market)     [YES bet]
                     f* = edge / P_market            [NO bet]
3. Fractional Kelly: f_adj = f* × 0.25              [quarter-Kelly]
4. Confidence scale: f_adj = f_adj × confidence
5. Position cap:     f_adj = min(f_adj, 0.05)        [max 5% per position]
6. Dollar amount:    position = f_adj × bankroll
```

**Risk controls:**
- **Max position:** 5% of bankroll per market (configurable)
- **Drawdown stop:** 15% peak-to-trough stops all trading
- **Paper trading:** Default mode — no real money until human approves
- **Min edge:** 5% minimum edge required to generate a signal

---

## Visualization Layer

Six plot types in `src/visualization/plots.py`:

| Plot | Purpose | Library |
|------|---------|---------|
| Correlation Heatmap | Market-market similarity matrix | Plotly Heatmap |
| Market Network | Interactive graph of market relationships | Plotly Scatter + NetworkX |
| Swarm Prediction | Individual Fish vs. ensemble vs. market price | Plotly Bar + HLine |
| Calibration Diagram | Reliability plot (predicted vs. observed frequency) | Plotly Scatter + Histogram |
| PnL Curve | Portfolio performance with drawdown shading | Plotly Scatter + Fill |
| Edge Distribution | Histogram of prediction edges across markets | Plotly Histogram |

All plots use `plotly_dark` theme and export to interactive HTML files.

---

## Configuration System

### File hierarchy (highest priority wins):

```
1. Environment variables (ANTHROPIC_API_KEY, etc.)
2. config/local.yaml (gitignored, per-user)
3. config/default.yaml (committed, shared defaults)
```

**Deep merge:** local.yaml overrides specific keys in default.yaml without replacing the entire section.

### Key configuration sections:

| Section | Controls |
|---------|----------|
| `api_keys` | Anthropic, OpenAI, Polymarket, news API |
| `swarm` | Number of Fish, personas, model, temperature, concurrency |
| `god_node` | Model, propagation mode, reanalysis threshold |
| `markets.polymarket` | API URLs, scan interval, volume/liquidity filters |
| `semantic` | Embedding model, similarity threshold, RAG config |
| `network` | Divergence metric, edge threshold, centrality metric |
| `prediction` | Aggregation method, calibration method, Brier target |
| `risk` | Kelly fraction, max position, max drawdown, bankroll, paper_trading |
| `visualization` | Dashboard port, theme, refresh interval |
| `logging` | Level, file path, rotation, retention |

---

## Testing

**Framework:** pytest + pytest-asyncio
**Coverage target:** 80%
**Current:** 45 tests, all passing

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_fish.py` | 13 | Fish creation, persona diversity, stub analysis, Brier scoring |
| `test_swarm.py` | 12 | Swarm creation, prediction pipeline, aggregation methods, outcome recording |
| `test_calibration.py` | 12 | All 3 calibration methods, metrics, batch calibration |
| `test_kelly.py` | 8 | Position sizing, edge detection, drawdown stops, PnL tracking |

Run: `pytest tests/ -v --tb=short`
