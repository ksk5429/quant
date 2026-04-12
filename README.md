<h1 align="center">K-Fish</h1>

<p align="center">
<strong>Swarm Intelligence Prediction Engine for Prediction Markets</strong>
</p>

<p align="center">
<a href="#retrodiction-baseline"><img src="https://img.shields.io/badge/Brier_Score-0.213-2196F3?style=for-the-badge" alt="Brier"></a>
<a href="#retrodiction-baseline"><img src="https://img.shields.io/badge/Accuracy-73.3%25-4CAF50?style=for-the-badge" alt="Accuracy"></a>
<a href="#zero-cost-architecture"><img src="https://img.shields.io/badge/Cost-$0.00/market-00C853?style=for-the-badge" alt="Cost"></a>
<a href="#9-fish-personas"><img src="https://img.shields.io/badge/Fish_Agents-9_Personas-7C4DFF?style=for-the-badge" alt="Fish"></a>
<a href="docs/Literature_Review_Multi_Agent_LLM_Prediction_Markets.md"><img src="https://img.shields.io/badge/Literature-32_Sources-FF6D00?style=for-the-badge" alt="Lit"></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Claude_Code-CLI-6B48FF?logo=anthropic&logoColor=white" alt="Claude">
<img src="https://img.shields.io/badge/Polymarket-Gamma_API-4A9EE5" alt="Polymarket">
<img src="https://img.shields.io/badge/netcal-Calibration-FF6F61" alt="netcal">
<img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
<img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<br/>

<blockquote>
Named after the schooling behavior of fish — individually simple, collectively intelligent.
</blockquote>

K-Fish deploys **9 LLM agents** ("Fish") that each use a structurally different reasoning framework to analyze prediction markets. Their independent probability estimates are fused through a **multi-round Delphi protocol**, calibrated with machine learning, and converted into risk-controlled positions using the **Kelly criterion**. The entire system runs at **zero cost** via Claude Code CLI.

---

## Pipeline

> Market → Route → Research → Delphi (multi-round) → Aggregate → Calibrate → Edge Detect → Kelly Size → Position

```mermaid
flowchart TD
    A["`**MARKET QUESTION**
    price withheld from all Fish agents`"]

    B["`**SWARM ROUTER**
    classifies category
    selects personas, rounds, extremization`"]

    C["`**RESEARCHER FISH**
    base rates · key facts
    timing analysis · contrarian case`"]

    subgraph DELPHI [" MULTI-ROUND DELPHI PROTOCOL "]
        direction TB
        R1["`**Round 1** — Independent`"]
        R2["`**Round 2** — Peer Context`"]
        RN["`**Round N** — Converge`"]
        R1 -- "anonymized estimates" --> R2 -- "update or hold" --> RN
    end

    D["`**AGGREGATION**
    trimmed mean · confidence-weighted
    asymmetric extremization`"]

    E1["`**CALIBRATE**
    netcal auto-select
    Beta · Histogram · Isotonic`"]

    E2["`**VOLATILITY**
    GARCH regime detection
    Kelly adjustment factor`"]

    E3["`**CONFORMAL**
    90% coverage interval
    prediction bounds`"]

    F["`**EDGE DETECTION**
    |calibrated − market_price| > 7%
    confidence > 40% · spread < 35%`"]

    G["`**KELLY SIZING**
    quarter-Kelly · 5% max/position
    30% max exposure · 15% drawdown stop`"]

    H["`**POSITION**
    side YES/NO · size $
    expected value · reasoning chain`"]

    A --> B --> C --> DELPHI --> D
    D --> E1 & E2 & E3
    E1 & E2 & E3 --> F
    F --> G --> H
```

### 9 Fish in the Delphi Swarm

```mermaid
graph LR
    subgraph SWARM [" 9 Fish Agents — each with orthogonal reasoning "]
        direction LR
        F1(("🎯<br/>Anchor<br/>0.72"))
        F2(("🔀<br/>Decomp<br/>0.68"))
        F3(("🔍<br/>Inside<br/>0.71"))
        F4(("⚡<br/>Contra<br/>0.45"))
        F5(("⏱️<br/>Temporal<br/>0.63"))
        F6(("🏛️<br/>Institut<br/>0.67"))
        F7(("💀<br/>Premort<br/>0.58"))
        F8(("📐<br/>Calibr<br/>0.70"))
        F9(("📊<br/>Bayes<br/>0.69"))
    end
```

---

## 9 Fish Personas

Each persona encodes a **structurally different decomposition strategy** to maximize ensemble diversity ([Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528)).

| # | Fish | Reasoning Framework | Function |
|:-:|------|---------------------|----------|
| 1 | **Base Rate Anchor** | Reference class frequency | Anchors on historical base rates, adjusts minimally |
| 2 | **Decomposer** | Sub-probability multiplication | Breaks question into independent conditional sub-questions |
| 3 | **Inside View** | Domain-specific evidence | Finds the single most informative fact others miss |
| 4 | **Contrarian** | Consensus stress-testing | Constructs the strongest case for the less popular outcome |
| 5 | **Temporal Analyst** | Timing and momentum | Deadline analysis, hazard rates, trajectory |
| 6 | **Institutional Analyst** | Organizational incentives | Status quo bias, decision-maker constraints |
| 7 | **Premortem** | Failure scenario enumeration | Imagines why the expected outcome failed |
| 8 | **Calibrator** | Tetlock superforecaster protocol | Base rate → evidence → incremental update → bias check |
| 9 | **Bayesian Updater** | Explicit prior x likelihood | States prior, identifies evidence, applies Bayes' rule |

---

## Retrodiction Baseline

30 resolved Polymarket markets · 9 Fish · Claude Haiku CLI · **$0 cost**

| Metric | K-Fish v4 | Polymarket Crowd | Random |
|:------:|:---------:|:----------------:|:------:|
| **Brier Score** | **0.213** | 0.084 | 0.250 |
| **Accuracy** | **73.3%** | ~90% | 50% |
| **Cost/market** | **$0.00** | — | — |
| **Time/market** | **135s** | — | — |

> [!NOTE]
> On markets where the LLM has relevant knowledge (25/30), K-Fish achieves **Brier 0.073** — beating the Polymarket crowd (0.084). The overall gap is driven by 5 surprise events beyond the LLM training data cutoff.

<details>
<summary><strong>Per-Fish Performance Rankings</strong> (click to expand)</summary>

| Rank | Persona | Brier | Assessment |
|:----:|---------|:-----:|------------|
| 1 | Inside View | 0.182 | Best — domain expertise adds real value |
| 2 | Contrarian | 0.193 | Challenging consensus consistently helps |
| 3 | Calibrator | 0.196 | Tetlock method is well-calibrated |
| 4 | Institutional | 0.211 | Status quo analysis is reliable |
| 5 | Base Rate | 0.212 | Good anchor but misses surprises |
| 6 | Decomposer | 0.224 | Conditional decomposition adds moderate value |
| 7 | Bayesian | 0.229 | Explicit prior/likelihood reasoning |
| 8 | Premortem | 0.235 | Failure scenarios less useful for binary outcomes |
| 9 | Temporal | 0.239 | Timing analysis least effective persona |

</details>

---

## Zero-Cost Architecture

K-Fish runs entirely on the Claude Code CLI (`claude -p`), which uses the Max subscription at **no additional API cost**.

| Backend | Cost | Speed | Automated | GPU |
|:-------:|:----:|:-----:|:---------:|:---:|
| **CLI** (`claude -p`) | $0.00 | ~15s/Fish | Yes | No |
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

<details>
<summary><strong>Project Structure</strong> (click to expand)</summary>

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

</details>

### Module Dependency Graph

```mermaid
graph TD
    subgraph ENGINE ["Engine Layer"]
        E4["engine_v4"]
        LP["live_pipeline"]
        SR["swarm_router"]
    end

    subgraph SWARM ["Swarm Layer"]
        LF["llm_fish"]
        RS["researcher"]
        IPC["ipc"]
    end

    subgraph PREDICTION ["Prediction Layer"]
        CAL["calibration"]
        ADV["advanced_scoring"]
        VOL["volatility"]
        RET["run_retrodiction"]
    end

    subgraph RISK ["Risk Layer"]
        PF["portfolio"]
        AN["analytics"]
    end

    subgraph MARKET ["Market Layer"]
        PM["polymarket"]
        SC["scanner"]
        HI["history"]
        DS["dataset"]
    end

    LP --> E4 --> LF & RS & SR & CAL & PF & VOL & ADV
    RET --> LF & HI
    SC --> PM & SR
    LF --> CAL
    E4 --> PF
    PF --> AN

    style ENGINE fill:#1a1a2e,stroke:#58a6ff
    style SWARM fill:#0a1929,stroke:#1f6feb
    style PREDICTION fill:#0a2910,stroke:#3fb950
    style RISK fill:#2a0a0a,stroke:#f85149
    style MARKET fill:#2d1f00,stroke:#d29922
```

---

## Key Design Decisions

> [!IMPORTANT]
> Every decision is grounded in peer-reviewed evidence or empirical retrodiction results.

| Decision | Why | Evidence |
|----------|-----|----------|
| 9 orthogonal personas | Structural reasoning diversity drives accuracy | [Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528) |
| Prices withheld from Fish | Prevents anchoring, preserves independence | [PolySwarm, arXiv:2604.03888](https://arxiv.org/abs/2604.03888) |
| Asymmetric extremization | Suppress when Fish disagree (high spread) | Retrodiction: 5 worst markets had high spread |
| 3-Fish pre-screen | Skip unknowable markets | LLM cutoff caused Brier 0.95+ on surprises |
| Quarter-Kelly | Full Kelly has ~25% drawdowns | [Kelly 1956](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x) |
| Auto-seeded calibrator | No uncalibrated cold start | Code review: calibration was always a no-op |
| CLI over API | $0 vs $3-15/M tokens | Maximize predictions per dollar |

---

## Libraries

<details>
<summary><strong>Full Library Inventory</strong> (click to expand)</summary>

| Library | Purpose | Why This One |
|---------|---------|-------------|
| [netcal](https://github.com/EFS-OpenSource/calibration-framework) | Probability calibration | 10+ methods, auto-select by sample size |
| [scoringrules](https://github.com/frazane/scoringrules) | CRPS, Brier, log score | JAX/Numba backends |
| [quantstats](https://github.com/ranaroussi/quantstats) | Portfolio analytics | Sharpe, Sortino, Calmar, Monte Carlo |
| [trafilatura](https://github.com/adbar/trafilatura) | News extraction | 0.958 F1, used by HuggingFace/IBM |
| [sentence-transformers](https://www.sbert.net/) | Semantic embeddings | Market similarity, news matching |
| [statsforecast](https://github.com/Nixtla/statsforecast) | GARCH volatility | 20x faster than pmdarima |
| [MLflow](https://mlflow.org/) | Experiment tracking | Model registry for calibrators |
| [mapie](https://github.com/scikit-learn-contrib/MAPIE) | Conformal prediction | Coverage-guaranteed intervals |

</details>

---

## Data Assets

| Dataset | Size | Records | Source |
|---------|:----:|:-------:|--------|
| Resolved markets | 2.7 MB | 2,500 | Polymarket Gamma API |
| External dataset | 599 MB | 408,863 | [prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) |
| Retrodiction results | 250 KB | 30 evaluated | K-Fish CLI swarm |
| Literature review | 32 PDFs | 32 papers | Academic databases |

---

## Research Foundation

> [!TIP]
> Full literature review with 45 references: **[Literature Review](docs/Literature_Review_Multi_Agent_LLM_Prediction_Markets.md)**

| Claim | Evidence |
|-------|----------|
| LLM ensembles match human crowds | [Schoenegger et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adp1528) |
| Retrieval-augmented LLMs approach superforecasters | [Halawi et al., NeurIPS 2024](https://arxiv.org/abs/2402.18563) |
| 50-persona swarm outperforms single-model on Polymarket | [PolySwarm, arXiv:2604.03888](https://arxiv.org/abs/2604.03888) |
| RLHF models are overconfident, need calibration | [Geng et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.366/) |
| Semantic similarity outperforms price correlation | [Baaijens et al., Applied Network Science 2025](https://doi.org/10.1007/s41109-025-00755-2) |

---

## Roadmap

```mermaid
graph TD
    P1["✅ <b>Phase 1</b> — Foundation<br/>Core engine · 45 tests · Literature review · Polymarket API"]
    P2["✅ <b>Phase 2</b> — Swarm Intelligence<br/>9 Fish personas · Multi-round Delphi · CLI execution"]
    P3["✅ <b>Phase 3</b> — Calibration<br/>Retrodiction baseline (Brier 0.213) · netcal integration"]
    P4["✅ <b>Phase 4</b> — Risk Management<br/>Kelly sizing · Edge detection · Drawdown monitor · Monte Carlo"]
    P5["✅ <b>Phase 5</b> — Live Pipeline<br/>Market scanner · Live pipeline · SwarmRouter"]
    P6["🔄 <b>Phase 6</b> — Scale<br/>200+ market calibration · CLOB order execution · Dashboard"]
    P7["⬜ <b>Phase 7</b> — Optimization<br/>Fine-tuned specialist Fish (GRPO) · Cross-platform arbitrage"]

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7

    style P1 fill:#0a2910,stroke:#3fb950,color:#3fb950
    style P2 fill:#0a2910,stroke:#3fb950,color:#3fb950
    style P3 fill:#0a2910,stroke:#3fb950,color:#3fb950
    style P4 fill:#0a2910,stroke:#3fb950,color:#3fb950
    style P5 fill:#0a2910,stroke:#3fb950,color:#3fb950
    style P6 fill:#1a1a2e,stroke:#58a6ff,color:#58a6ff
    style P7 fill:#1a1a2e,stroke:#8b949e,color:#8b949e
```

---

<p align="center">
<sub>Built with structured human-AI collaboration · Paper trading is the default · Live trading requires explicit human approval</sub>
</p>
