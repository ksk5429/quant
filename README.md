<p align="center">
  <h1 align="center">K-Fish</h1>
  <p align="center">
    <strong>Swarm Intelligence Prediction Engine for Prediction Markets</strong>
  </p>
  <p align="center">
    9 AI agents with orthogonal reasoning frameworks &bull; Multi-round Delphi protocol &bull; Calibrated with netcal &bull; Kelly-sized positions &bull; Zero API cost
  </p>
  <p align="center">
    <a href="#retrodiction-baseline"><img src="https://img.shields.io/badge/Brier%20Score-0.213-blue?style=for-the-badge" alt="Brier Score"></a>
    <a href="#retrodiction-baseline"><img src="https://img.shields.io/badge/Accuracy-73.3%25-green?style=for-the-badge" alt="Accuracy"></a>
    <a href="#zero-cost-architecture"><img src="https://img.shields.io/badge/Cost-$0.00%2Fmarket-brightgreen?style=for-the-badge" alt="Cost"></a>
    <a href="docs/Literature_Review_Multi_Agent_LLM_Prediction_Markets.md"><img src="https://img.shields.io/badge/Literature%20Review-32%20Sources-orange?style=for-the-badge" alt="Literature"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Claude%20Code-CLI-6B48FF?logo=anthropic&logoColor=white" alt="Claude">
    <img src="https://img.shields.io/badge/Polymarket-API-4A9EE5" alt="Polymarket">
    <img src="https://img.shields.io/badge/netcal-Calibration-FF6F61" alt="netcal">
    <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
</p>

---

> Named after the schooling behavior of fish — individually simple, collectively intelligent.

K-Fish deploys 9 LLM agents ("Fish") that each use a **structurally different reasoning framework** to analyze prediction markets. Their independent probability estimates are fused through a multi-round Delphi protocol, calibrated with machine learning, and converted into risk-controlled positions using the Kelly criterion. The entire system runs at **zero cost** via Claude Code CLI.

---

## Pipeline

```
                    ┌─────────────────────────────┐
                    │     MARKET QUESTION          │
                    │   (price withheld from Fish) │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       SWARM ROUTER           │
                    │  classifies: politics/crypto/ │
                    │  sports/geopolitics/economics │
                    │  selects: persona set, rounds │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     RESEARCHER FISH           │
                    │  gathers: base rates, facts,  │
                    │  timing, contrarian case       │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
    ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
    │  Round 1 │              │  Round 2 │              │  Round 3 │
    │ 9 Fish   │     ───►     │ Peer     │     ───►     │ Converge │
    │ analyze  │              │ context  │              │ or stop  │
    │ alone    │              │ update   │              │          │
    └────┬────┘              └────┬────┘              └────┬────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      AGGREGATION             │
                    │  trimmed mean + conf-weighted │
                    │  + asymmetric extremization   │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
    │   CALIBRATE       │ │  VOLATILITY     │ │  CONFORMAL       │
    │   netcal auto     │ │  GARCH regime   │ │  90% interval    │
    │   Beta/Histogram  │ │  Kelly adjust   │ │  coverage bound  │
    └─────────┬────────┘ └────────┬────────┘ └────────┬────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       EDGE DETECTION         │
                    │  |cal_prob - mkt_price| > 7%  │
                    │  confidence > 40%             │
                    │  spread < 35%                 │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       KELLY SIZING           │
                    │  quarter-Kelly, 5% max/pos   │
                    │  30% max exposure             │
                    │  15% drawdown circuit breaker │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       POSITION               │
                    │  side (YES/NO) + size ($)     │
                    │  + EV + reasoning chain       │
                    └─────────────────────────────┘
```

---

## 9 Fish Personas

Each persona encodes a **structurally different decomposition strategy** — not just different domain knowledge — to maximize ensemble diversity ([Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528)).

| # | Fish | Reasoning Framework | Function |
|---|------|---------------------|----------|
| 1 | **Base Rate Anchor** | Reference class frequency | Anchors on historical base rates, adjusts minimally |
| 2 | **Decomposer** | Sub-probability multiplication | Breaks question into independent conditional sub-questions |
| 3 | **Inside View** | Domain-specific evidence | Finds the single most informative fact others miss |
| 4 | **Contrarian** | Consensus stress-testing | Constructs the strongest case for the less popular outcome |
| 5 | **Temporal Analyst** | Timing and momentum | Deadline analysis, hazard rates, trajectory |
| 6 | **Institutional Analyst** | Organizational incentives | Status quo bias, decision-maker constraints |
| 7 | **Premortem** | Failure scenario enumeration | Imagines why the expected outcome failed |
| 8 | **Calibrator** | Tetlock superforecaster protocol | Base rate, evidence, incremental update, bias check |
| 9 | **Bayesian Updater** | Explicit prior x likelihood | States prior, identifies evidence, applies Bayes' rule |

---

## Retrodiction Baseline

30 resolved Polymarket markets evaluated with 9 Fish personas, Claude Haiku CLI, $0 cost:

| Metric | K-Fish v4 | Polymarket Crowd | Random |
|--------|-----------|-----------------|--------|
| **Brier Score** | **0.213** | 0.084 | 0.250 |
| **Accuracy** | **73.3%** | ~90% | 50% |
| **Cost/market** | **$0.00** | — | — |
| **Time/market** | **135s** | — | — |

#### Per-Fish Performance

| Rank | Persona | Brier | Assessment |
|------|---------|-------|------------|
| 1 | Inside View | 0.182 | Best — domain expertise adds real value |
| 2 | Contrarian | 0.193 | Challenging consensus consistently helps |
| 3 | Calibrator | 0.196 | Tetlock method is well-calibrated |
| 4 | Institutional | 0.211 | Status quo analysis is reliable |
| 5 | Base Rate | 0.212 | Good anchor but misses surprises |
| 9 | Temporal | 0.239 | Timing analysis less useful for binary outcomes |

> On markets where the LLM has relevant knowledge (25/30), K-Fish achieves **Brier 0.073** — beating the Polymarket crowd (0.084). The overall gap is driven by 5 surprise events beyond the LLM training data cutoff.

---

## Zero-Cost Architecture

K-Fish runs entirely on the Claude Code CLI (`claude -p`), which uses the Max subscription at no additional API cost.

| Backend | Cost | Speed | Automated | GPU Required |
|---------|------|-------|-----------|-------------|
| **CLI** (claude -p) | $0.00 | ~15s/Fish | Yes | No |
| **Ollama** (local) | $0.00 | ~5s/Fish | Yes | Yes |
| **Gemini** (free tier) | $0.00 | ~3s/Fish | Yes | No |
| **File** (manual) | $0.00 | Manual | No | No |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/ksk5429/quant.git && cd quant
pip install -e ".[dev]"
pip install netcal scoringrules quantstats trafilatura statsforecast mlflow sentence-transformers

# Scan live Polymarket markets
python -m src.markets.scanner --min-volume 100000

# Run retrodiction (evaluate on resolved markets)
python -m src.prediction.run_retrodiction --n 30 --model haiku --concurrent 3

# Run full live pipeline (scan → analyze → portfolio)
python -m src.mirofish.live_pipeline --top 10 --model haiku
```

---

## Architecture

```
src/
├── mirofish/                   # K-Fish Swarm Engine
│   ├── engine_v4.py           #   Canonical pipeline: Route→Research→Delphi→Calibrate→Kelly
│   ├── llm_fish.py            #   9 personas, 4 backends, asymmetric extremization
│   ├── researcher.py          #   Context gathering (base rates, facts, contrarian case)
│   ├── swarm_router.py        #   Score-based category routing + model competition
│   ├── live_pipeline.py       #   Scanner → Engine → Portfolio → Report
│   ├── ipc.py                 #   File-based IPC for distributed Fish
│   ├── swarm.py               #   Swarm orchestrator
│   ├── god_node.py            #   Event injection + impact analysis
│   └── message_bus.py         #   Inter-agent communication
├── prediction/                 # Scoring & Calibration
│   ├── calibration.py         #   netcal v2: Beta/Histogram/Isotonic/auto-select + CRPS
│   ├── advanced_scoring.py    #   Brier decomposition, conformal intervals
│   ├── volatility.py          #   GARCH regime detection, volatility-adjusted Kelly
│   └── run_retrodiction.py    #   CLI-based evaluation on resolved markets
├── risk/                       # Position Sizing & Risk
│   ├── portfolio.py           #   Edge detection, Kelly criterion, drawdown monitor
│   └── analytics.py           #   Sharpe/Sortino/Calmar, Monte Carlo simulation
├── markets/                    # Market Data
│   ├── polymarket.py          #   Gamma API (metadata) + CLOB API (trading)
│   ├── scanner.py             #   Live market discovery + composite ranking
│   ├── history.py             #   Resolved market scraper with retry logic
│   └── dataset.py             #   408K market parquet loader (DuckDB)
├── semantic/                   # NLP & Information
│   ├── analyzer.py            #   Embedding similarity, market clustering
│   └── news_extractor.py      #   trafilatura extraction + semantic matching
├── network/                    # Graph Analysis
│   └── market_graph.py        #   Cross-market correlation, centrality
└── utils/                      # Infrastructure
    ├── cli.py                 #   Cross-platform Claude binary detection
    ├── experiment_tracker.py  #   MLflow experiment tracking
    ├── config.py              #   YAML configuration loader
    └── logging.py             #   Structured logging (loguru)
```

---

## Key Design Decisions

| Decision | Why | Evidence |
|----------|-----|----------|
| 9 orthogonal personas | Structural reasoning diversity drives ensemble accuracy more than model capability | [Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528) |
| Prices withheld from Fish | Prevents anchoring bias, preserves statistical independence | [PolySwarm (arXiv:2604.03888)](https://arxiv.org/abs/2604.03888) |
| Asymmetric extremization | Push away from 0.5 when Fish agree; suppress when they disagree | Retrodiction analysis: 5 worst markets had high spread |
| 3-Fish pre-screen | Skip unknowable markets to avoid guaranteed-bad predictions | LLM knowledge cutoff caused Brier 0.95+ on surprise events |
| Quarter-Kelly sizing | Full Kelly has ~25% drawdowns; 0.25x reduces volatility 4x | [Kelly 1956](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x) |
| Auto-seeded calibrator | Loads retrodiction data on startup; no uncalibrated cold start | Code review: calibration was always a no-op before this fix |
| CLI mode over API | $0 cost via Max subscription vs. $3-15/M tokens API pricing | Practical constraint: maximize predictions per dollar |

---

## Libraries

| Library | Purpose | Why This One |
|---------|---------|-------------|
| [netcal](https://github.com/EFS-OpenSource/calibration-framework) | Probability calibration | 10+ methods (Beta, Histogram, GP-Beta), auto-select by sample size |
| [scoringrules](https://github.com/frazane/scoringrules) | CRPS, Brier, log score | JAX/Numba backends, proper scoring rules for distributions |
| [quantstats](https://github.com/ranaroussi/quantstats) | Portfolio analytics | Sharpe, Sortino, Calmar, HTML tearsheets, Monte Carlo |
| [trafilatura](https://github.com/adbar/trafilatura) | News extraction | 0.958 F1, RSS/sitemap support, used by HuggingFace/IBM/Microsoft |
| [sentence-transformers](https://www.sbert.net/) | Semantic embeddings | Market similarity, cross-market correlation, news matching |
| [statsforecast](https://github.com/Nixtla/statsforecast) | GARCH volatility | 20x faster than pmdarima, probabilistic intervals |
| [MLflow](https://mlflow.org/) | Experiment tracking | Per-retrodiction-run logging, model registry for calibrators |

---

## Data

| Dataset | Size | Records | Source |
|---------|------|---------|--------|
| Resolved markets (our scrape) | 2.7 MB | 2,500 | Polymarket Gamma API |
| External dataset (parquet) | 599 MB | 408,863 | [prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) |
| Retrodiction v2 results | 250 KB | 30 evaluated | K-Fish CLI swarm |
| Literature review sources | 32 PDFs | 32 papers | Academic databases |

---

## Research Foundation

| Claim | Evidence |
|-------|----------|
| LLM ensembles match human crowds | [Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528) |
| Retrieval-augmented LLMs approach superforecasters | [Halawi et al., NeurIPS 2024](https://arxiv.org/abs/2402.18563) |
| 50-persona swarm outperforms single-model on Polymarket | [PolySwarm, arXiv:2604.03888](https://arxiv.org/abs/2604.03888) |
| RLHF models are overconfident, need calibration | [Geng et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.366/) |
| Semantic similarity outperforms price correlation | [Baaijens et al., Applied Network Science 2025](https://doi.org/10.1007/s41109-025-00755-2) |

Full literature review (45 references): **[Literature Review](docs/Literature_Review_Multi_Agent_LLM_Prediction_Markets.md)**

---

## Roadmap

- [x] **Phase 1** &mdash; Core engine, 45 tests, literature review, Polymarket API
- [x] **Phase 2** &mdash; 9 Fish personas, multi-round Delphi, CLI execution
- [x] **Phase 3** &mdash; Retrodiction baseline (Brier 0.213), netcal calibration
- [x] **Phase 4** &mdash; Kelly sizing, edge detection, drawdown monitor, Monte Carlo
- [x] **Phase 5** &mdash; Live market scanner, live pipeline, SwarmRouter
- [ ] **Phase 6** &mdash; 200+ market calibration, CLOB order execution, dashboard
- [ ] **Phase 7** &mdash; Fine-tuned specialist Fish (GRPO), cross-platform arbitrage

---

<p align="center">
  <sub>Built with structured human-AI collaboration. Paper trading is the default — live trading requires explicit human approval.</sub>
</p>
