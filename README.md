# K-Fish

**Swarm Intelligence Prediction Engine for Prediction Markets**

9 AI agents with orthogonal reasoning frameworks analyze prediction markets independently, then their estimates are fused through multi-round Delphi protocol, calibrated with netcal, and converted into Kelly-sized positions. Zero API cost — runs entirely on Claude Code CLI.

> Named after the schooling behavior of fish — individually simple, collectively intelligent.

---

## How It Works

```
Market Question (price withheld from agents)
    |
    v
SwarmRouter ─── classifies: politics / crypto / sports / geopolitics
    |            selects: persona set, round count, extremization factor
    v
Researcher Fish (sonnet) ─── gathers: base rates, key facts, contrarian case
    |
    v
Multi-Round Delphi Protocol
    Round 1: 9 Fish analyze independently
    Round 2: Fish see anonymized peer median, update or hold
    Round 3: repeat until std_dev < 0.03 or max rounds
    |
    v
Aggregation ─── trimmed mean + confidence-weighted + asymmetric extremization
    |                (extremize less when Fish disagree)
    v
Calibration ─── netcal auto-select (Beta / Histogram / Isotonic / Temperature)
    |                seeded from retrodiction results on startup
    v
Edge Detection ─── |calibrated_prob - market_price| > 5% + 2% tx cost
    |                    filter: confidence > 40%, spread < 35%
    v
Kelly Sizing ─── quarter-Kelly, 5% max per position, volatility-adjusted
    |                 drawdown circuit breaker at 15%
    v
Position ─── side (YES/NO), size ($), expected value, reasoning chain
```

---

## 9 Fish Personas

Each persona encodes a structurally different reasoning framework — not just different domain knowledge — to maximize ensemble diversity.

| Fish | Method | Purpose |
|------|--------|---------|
| Base Rate Anchor | Reference class frequency | Prevents drift from statistical reality |
| Decomposer | Sub-probability multiplication | Catches unlikely prerequisites |
| Inside View | Domain-specific evidence | The single best fact others miss |
| Contrarian | Consensus stress-testing | Finds where the crowd is wrong |
| Temporal Analyst | Timing and momentum | Deadline analysis, hazard rates |
| Institutional Analyst | Organizational incentives | Status quo and decision-maker constraints |
| Premortem | Failure scenario enumeration | Counteracts planning fallacy |
| Calibrator | Tetlock superforecaster protocol | Explicit bias checks and incremental updating |
| Bayesian Updater | Prior x likelihood | Transparent, auditable reasoning |

---

## Retrodiction Baseline

30 resolved Polymarket markets, 9 Fish, Claude Haiku CLI, $0 cost:

| Metric | K-Fish v4 | Polymarket Crowd | Random Baseline |
|--------|------------|-----------------|-----------------|
| Brier Score | 0.213 | 0.084 | 0.250 |
| Accuracy | 73.3% | ~90% | 50% |
| Cost per market | $0.00 | — | — |
| Time per market | 135s | — | — |

Best-performing Fish: Inside View (0.182), Contrarian (0.193), Calibrator (0.196).

The gap to the Polymarket crowd is driven by 5 surprise events beyond the LLM knowledge cutoff. On the remaining 25 markets, K-Fish achieves Brier 0.073 (beats crowd).

---

## Quick Start

### Prerequisites
- Python 3.11+
- Claude Code CLI (Max subscription for zero-cost operation)

### Installation

```bash
git clone https://github.com/ksk5429/quant.git
cd quant
pip install -e ".[dev]"
pip install netcal scoringrules quantstats trafilatura statsforecast mlflow sentence-transformers
```

### Zero-Cost CLI Mode

No API key needed. Uses your Claude Code Max subscription:

```bash
# Run retrodiction on 30 resolved markets
python -m src.prediction.run_retrodiction --n 30 --model haiku --concurrent 3

# Scrape resolved markets for calibration corpus
python -m src.markets.history --limit 2500 --min-volume 5000
```

### API Mode (Optional)

For API-based execution, configure keys:

```bash
cp config/default.yaml config/local.yaml
# Edit config/local.yaml with your API keys
```

---

## Architecture

```
src/
├── mirofish/                # Swarm Engine (K-Fish core)
│   ├── engine_v4.py         # Full pipeline (canonical)
│   ├── llm_fish.py          # 9 personas, 4 backends (CLI/Ollama/Gemini/File)
│   ├── researcher.py        # Context gathering (base rates, facts, contrarian case)
│   ├── swarm_router.py      # Adaptive routing + model competition + safety mode
│   ├── ipc.py               # File-based IPC for distributed Fish
│   ├── swarm.py             # Swarm orchestration
│   ├── god_node.py          # Event injection
│   └── message_bus.py       # Inter-agent communication
├── prediction/               # Scoring and Calibration
│   ├── calibration.py       # netcal v2: Beta/Histogram/Isotonic/auto-select + CRPS
│   ├── advanced_scoring.py  # Brier decomposition, conformal intervals
│   ├── volatility.py        # GARCH regime detection, Kelly adjustment
│   └── run_retrodiction.py  # CLI-based evaluation on resolved markets
├── risk/                     # Position Sizing
│   ├── portfolio.py         # Edge detection, Kelly sizing, drawdown monitor
│   └── analytics.py         # Sharpe/Sortino/Calmar, Monte Carlo simulation
├── markets/                  # Market Data
│   ├── polymarket.py        # Gamma API (metadata) + CLOB API (trading)
│   ├── history.py           # Resolved market scraper with retry logic
│   └── dataset.py           # 408K market parquet loader (DuckDB)
├── semantic/                 # NLP
│   ├── analyzer.py          # Embedding similarity, market clustering
│   └── news_extractor.py    # trafilatura article extraction + semantic matching
├── network/                  # Graph Analysis
│   └── market_graph.py      # Cross-market correlation, centrality
└── utils/
    ├── cli.py               # Cross-platform Claude binary detection
    ├── experiment_tracker.py # MLflow experiment tracking
    ├── config.py            # YAML configuration loader
    └── logging.py           # Structured logging (loguru)
```

---

## Key Design Decisions

| Decision | Rationale | Evidence |
|----------|-----------|----------|
| 9 orthogonal personas over model diversity | Structural reasoning diversity drives ensemble accuracy | Schoenegger et al., Science Advances 2024 |
| Market prices withheld during Fish analysis | Prevents anchoring, preserves independence | PolySwarm methodology (arXiv:2604.03888) |
| Asymmetric extremization | Extremize less when Fish disagree (high spread = uncertain direction) | Tetlock recalibration + retrodiction analysis |
| 3-Fish pre-screen | Skip unknowable markets to avoid guaranteed-bad predictions | 5 catastrophic misses in baseline caused by LLM knowledge cutoff |
| Quarter-Kelly sizing | Full Kelly has ~25% drawdowns; 0.25x reduces volatility 4x | Kelly 1956, Thorp 1962, PolySwarm |
| Calibrator auto-seeded from retrodiction | Eliminates uncalibrated cold start on every restart | Code review finding: calibration was always a no-op |
| CLI mode (claude -p) over API calls | $0 cost via Max subscription vs $3-15/M tokens API pricing | Practical constraint: maximize runs per dollar |

---

## Data Assets

| Dataset | Size | Records | Source |
|---------|------|---------|--------|
| Resolved markets (our scrape) | 2.7 MB | 2,500 markets | Polymarket Gamma API |
| External dataset (parquet) | 599 MB | 408,863 markets | Jon-Becker/prediction-market-analysis |
| Retrodiction results | 250 KB | 30 evaluated markets | CLI Fish swarm |

---

## Libraries

| Library | Version | Used For |
|---------|---------|----------|
| netcal | 1.3.6 | Beta calibration, histogram binning, ECE/MCE metrics |
| scoringrules | 0.9.0 | CRPS scoring for distributional evaluation |
| quantstats | 0.0.81 | Portfolio tearsheets, Sharpe/Sortino analysis |
| trafilatura | 2.0.0 | News article extraction (0.958 F1) |
| sentence-transformers | 5.4.0 | Semantic embeddings for market similarity |
| statsforecast | 2.0.3 | GARCH/ARCH volatility modeling |
| mlflow | 3.11.1 | Experiment tracking and model registry |
| mapie | 1.3.0 | Conformal prediction intervals |

---

## Research Foundation

| Claim | Evidence |
|-------|----------|
| LLM ensembles match human crowds | Schoenegger et al., Science Advances 2024 |
| Retrieval-augmented LLMs approach superforecasters | Halawi et al., NeurIPS 2024 |
| Swarm + Bayesian aggregation is SOTA for Polymarket | PolySwarm, arXiv:2604.03888 |
| RLHF models are overconfident, need calibration | Geng et al., NAACL 2024 |
| Semantic similarity outperforms price correlation | Baaijens et al., Applied Network Science 2025 |
| Quarter-Kelly is optimal for prediction markets | Kelly 1956, Thorp 1962 |

Full literature review: [docs/Literature_Review_Multi_Agent_LLM_Prediction_Markets.docx](docs/Literature_Review_Multi_Agent_LLM_Prediction_Markets.docx)

---

## Roadmap

- [x] Phase 1: Foundation — core engine, tests, literature review, Polymarket API
- [x] Phase 2: Swarm Intelligence — 9 Fish personas, multi-round Delphi, CLI execution
- [x] Phase 3: Calibration — retrodiction baseline (Brier 0.213), netcal integration
- [x] Phase 4: Risk — Kelly sizing, edge detection, drawdown monitor, Monte Carlo
- [ ] Phase 5: Live Pipeline — real-time market scanning, position tracking, P&L
- [ ] Phase 6: Scale — 200+ market calibration, CLOB order execution, dashboard

---

## License

MIT

## Disclaimer

Research and educational tool. Prediction market trading involves financial risk. Paper trading is the default — live trading requires explicit human approval. Past performance does not guarantee future results.
