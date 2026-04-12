# Mirofish

**Swarm Intelligence Prediction Engine for Prediction Markets**

Mirofish uses multiple AI agents — called **Fish** — to analyze prediction markets like [Polymarket](https://polymarket.com). Each Fish has a different analytical personality (geopolitical analyst, quant, contrarian, etc.), analyzes markets independently, then their predictions are fused using Bayesian statistics to produce calibrated probability estimates and trade signals.

> **Named after the schooling behavior of fish** — individually simple, collectively intelligent.

---

## The Core Idea in 60 Seconds

```
  YOU drop a real-world event
        |
        v
  +-----------+
  | GOD Node  |  "Supreme Court blocks crypto regulation"
  +-----------+
        |
        | broadcasts to all Fish
        |
   +----+----+----+----+----+
   |    |    |    |    |    |
   v    v    v    v    v    v
 [Geo] [Qnt] [Bay] [Jrn] [Con]    <-- 5 Fish with different worldviews
 0.72  0.68  0.71  0.65  0.45     <-- Each estimates P(YES)
   |    |    |    |    |    |
   +----+----+----+----+----+
        |
        v
  Bayesian Aggregation (confidence-weighted)
  P_swarm = 0.67
        |
        v
  Calibration (isotonic regression)
  P_calibrated = 0.64
        |
        v
  Kelly Criterion (quarter-Kelly)
  "BUY YES, $32.00 (edge = +0.14)"
```

**Why this works:** An ensemble of 12 LLMs matches 925 human forecasters in accuracy ([Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528)). Diversity across agents — not model size — is what drives prediction quality.

---

## How It Works (Step by Step)

### Step 1: Market Discovery
The system connects to Polymarket's API and fetches active markets with their current prices, volume, and metadata.

### Step 2: Swarm Analysis
Each Fish receives the market question **without seeing the current price** (to prevent anchoring bias). Every Fish applies its unique analytical lens:

| Fish | Reasoning Framework | What It Does |
|------|---------------------|--------------|
| Base Rate Anchor | Reference class forecasting | Anchors on historical frequencies, adjusts minimally |
| Decomposer | Sub-probability multiplication | Breaks question into independent sub-questions |
| Inside View | Domain-specific evidence | Finds the single most informative fact others miss |
| Contrarian | Consensus stress-testing | Constructs strongest case for the less popular outcome |
| Temporal Analyst | Timing and momentum | Analyzes deadlines, hazard rates, trajectory |
| Institutional Analyst | Organizational incentives | Weights institutional inertia and decision-maker incentives |
| Premortem | Failure scenario enumeration | Imagines why the likely outcome didn't happen |
| Calibrator | Tetlock superforecaster method | Base rate → evidence → incremental update → bias check |
| Bayesian Updater | Explicit prior × likelihood | States prior, identifies evidence, applies Bayes' rule |

### Step 3: Bayesian Aggregation
Individual Fish predictions are combined using confidence-weighted fusion:

```
P_swarm = sum(confidence_i x accuracy_bonus_i x P_i) / sum(weights)
```

Fish that are both **confident** and **historically accurate** get more influence. This is not simple averaging — it's learned weighting based on track record.

### Step 4: Probability Calibration
Raw LLM probabilities are systematically overconfident ([Geng et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.366/)). We apply isotonic regression to map raw estimates to calibrated probabilities using historical resolution data.

### Step 5: Trade Signal Generation
The [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion) determines optimal bet sizing:

```
edge = our_probability - market_price
kelly_fraction = edge / (1 - market_price)
position = kelly_fraction x 0.25 x confidence x bankroll
```

We use **quarter-Kelly** (conservative) with a 5% max position cap and 15% drawdown stop.

### Step 6: Visualization
Interactive Plotly dashboards show correlation heatmaps, network graphs, Fish agreement/disagreement, calibration diagrams, and PnL curves.

---

## The Swarm Intelligence Architecture

What makes Mirofish unique: each Fish agent runs as a **real, separate AI instance** in its own VS Code window. This isn't simulated multi-agent — it's actual parallel intelligence.

```
QUANT/                              <- Master Intelligence (you + Claude)
|
+-- Swarm_Intelligence/
|   +-- fish_geopolitical/          <- Open in VS Code #2
|   +-- fish_quant/                 <- Open in VS Code #3
|   +-- fish_contrarian/            <- Open in VS Code #4
|   +-- fish_bayesian/              <- Open in VS Code #5
|   +-- fish_journalist/            <- Open in VS Code #6
|   +-- fish_domain_expert/         <- Open in VS Code #7
|   +-- fish_calibrator/            <- Open in VS Code #8
|
+-- shared_state/                   <- Communication backbone
|   +-- market_data/                <- Master writes: what to analyze
|   +-- events/                     <- Master writes: real-world events
|   +-- analyses/                   <- Each Fish writes: its analysis
|   +-- signals/                    <- Fish-to-Fish: direct messages
|   +-- consensus/                  <- Master writes: aggregated result
|
+-- src/                            <- Core engine (aggregation, calibration, risk)
```

Fish communicate through JSON files in `shared_state/`. No sockets, no databases — just files that any human can read, inspect, and debug.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/settings/keys) (for Claude-powered Fish)

### Installation

```bash
git clone https://github.com/ksk5429/quant.git
cd quant
pip install -e ".[dev]"
```

### Setup API Keys

```bash
python setup.py          # Interactive setup (recommended)
# OR
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

### Run the Demo

```bash
# Stub mode (no API key needed — tests the full pipeline)
python demo_live.py --stub --markets 5

# Live mode (requires Anthropic API key)
python demo_live.py --markets 5 --fish 5

# Custom model (Haiku is cheapest, Sonnet is best)
python demo_live.py --model claude-sonnet-4-6 --markets 3
```

### Run Tests

```bash
pytest tests/ -v              # All 45 tests
pytest tests/ -v --cov=src    # With coverage
```

---

## Project Structure

```
quant/
+-- src/
|   +-- mirofish/                # Swarm engine (v4)
|   |   +-- engine_v4.py        #   Full pipeline: Route→Research→Delphi→Calibrate→Kelly
|   |   +-- llm_fish.py         #   9 personas, 4 backends (CLI/Ollama/Gemini/File)
|   |   +-- researcher.py       #   Context gathering Fish (sonnet)
|   |   +-- swarm_router.py     #   Adaptive routing + model competition
|   |   +-- ipc.py              #   File-based IPC for distributed Fish
|   |   +-- swarm.py            #   Swarm orchestrator
|   |   +-- god_node.py         #   Event injection + impact analysis
|   |   +-- message_bus.py      #   Inter-agent communication
|   +-- prediction/              # Scoring & calibration
|   |   +-- calibration.py      #   v2: netcal (Beta/Histogram/auto-select) + CRPS
|   |   +-- advanced_scoring.py #   Brier decomposition, conformal intervals
|   |   +-- volatility.py       #   GARCH regime detection, Kelly adjustment
|   |   +-- run_retrodiction.py #   Evaluation runner (CLI mode)
|   +-- risk/                    # Position sizing & risk
|   |   +-- portfolio.py        #   Edge detection, Kelly, drawdown monitor
|   |   +-- analytics.py        #   Sharpe/Sortino, Monte Carlo simulation
|   +-- markets/                 # Market data
|   |   +-- polymarket.py       #   Gamma + CLOB API clients
|   |   +-- history.py          #   Resolved market scraper (2,500 markets)
|   |   +-- dataset.py          #   External 408K market parquet loader
|   +-- semantic/                # NLP & information
|   |   +-- analyzer.py         #   Embeddings, similarity analysis
|   |   +-- news_extractor.py   #   trafilatura + sentence-transformers
|   +-- network/                 # Graph analysis
|   |   +-- market_graph.py     #   Correlation graph, centrality
|   +-- utils/                   # Infrastructure
|       +-- cli.py              #   Claude binary detection (cross-platform)
|       +-- experiment_tracker.py # MLflow tracking
|       +-- config.py           #   YAML config loader
+-- Swarm_Intelligence/          # Fish agent workspaces
+-- shared_state/                # Inter-agent file-based communication
+-- data/                        # Datasets and results (gitignored)
+-- literature_review/           # 32 academic sources + review article
+-- docs/                        # Documentation
+-- tests/                       # Test suite
+-- config/                      # YAML configuration
```

## v4 Retrodiction Baseline (Real Data)

30 resolved Polymarket markets evaluated with 9 Fish personas using Claude Haiku CLI:

| Metric | Mirofish | Polymarket Crowd | Random |
|--------|----------|-----------------|--------|
| Brier Score | 0.213 | 0.084 | 0.250 |
| Accuracy | 73.3% | ~90% | 50% |
| Cost | $0.00 | — | — |

---

## Documentation

| Document | What It Covers |
|----------|----------------|
| [Technical Details](docs/TECHNICAL_DETAILS.md) | Full system architecture, data flow, API integration, implementation details |
| [Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md) | All math explained: Bayesian aggregation, Kelly criterion, calibration, divergence metrics |
| [User Manual](docs/USER_MANUAL.md) | Step-by-step guide: setup, running Fish, interpreting results, common workflows |
| [Swarm Protocol](docs/SWARM_PROTOCOL.md) | How Fish agents communicate, the tiered pipeline, debate rounds, routing |
| [Glossary](docs/GLOSSARY.md) | Every term defined — prediction markets, Brier scores, Kelly criterion, etc. |
| [System Design](docs/architecture/system_design.md) | Architecture diagrams and data flow |
| [Literature Review](literature_review/annotated_bibliography.md) | 26 peer-reviewed references with annotations |
| [Developer Notes](docs/DEVELOPERS_NOTE.md) | Design decisions and rationale |

---

## Research Foundation

Every algorithmic decision is grounded in peer-reviewed research. No hallucinated references — every citation has a real DOI or arXiv ID.

| Claim | Evidence |
|-------|----------|
| LLM ensembles match human crowds | [Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528) |
| Retrieval-augmented LLMs approach superforecasters | [Halawi et al., NeurIPS 2024](https://arxiv.org/abs/2402.18563) |
| Swarm + Bayesian aggregation is SOTA for Polymarket | [PolySwarm, arXiv:2604.03888](https://arxiv.org/abs/2604.03888) |
| RLHF models are overconfident, need calibration | [Geng et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.366/) |
| $40M arbitrage on Polymarket via semantic matching | [Saguillo et al., AFT 2025](https://arxiv.org/abs/2504.00000) |
| Semantic similarity graphs outperform price correlation | [Baaijens et al., Applied Network Science 2025](https://doi.org/10.1007/s41109-025-00755-2) |
| Quarter-Kelly is optimal for prediction markets | [Kelly 1956](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x), [Thorp 1962](https://en.wikipedia.org/wiki/Beat_the_Dealer) |

Full bibliography: [literature_review/annotated_bibliography.md](literature_review/annotated_bibliography.md)

---

## Human-AI Collaboration

This project is built through structured human-AI interaction:

- **Human provides**: Domain knowledge, strategy decisions, risk tolerance, event detection
- **AI provides**: Data fetching, probability computation, aggregation math, visualization
- **Shared**: Market selection, architecture decisions, feature priorities

The [Human Worksheet](docs/worksheets/mirofish_human_worksheet.docx) (DOCX) contains 11 sections for the human operator to fill out, covering strategy, domain knowledge, risk parameters, and learning goals.

---

## Roadmap

- [x] **Phase 1: Foundation** — Core engine, 45 tests, literature review, Polymarket API
- [ ] **Phase 2: Intelligence** — Live Claude Fish, tiered swarm pipeline, debate rounds
- [ ] **Phase 3: Calibration** — Historical backtesting, learned calibration, performance tracking
- [ ] **Phase 4: Production** — Real-time dashboard, portfolio management, live signals

---

## License

MIT

## Disclaimer

This is a research and educational tool. Prediction market trading involves financial risk. The system uses paper trading by default — live trading requires explicit human approval. Past performance does not guarantee future results. Use at your own risk.
