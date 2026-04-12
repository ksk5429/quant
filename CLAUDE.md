# CLAUDE.md — Mirofish Prediction Engine

## Project Identity

**Mirofish** is a swarm intelligence prediction engine for prediction markets. It uses multiple LLM agents ("Fish") organized in a swarm to analyze markets, detect cross-market correlations, and produce calibrated probability estimates.

**Human Lead:** Kyeong Sun Kim (KSK)
**AI System:** Claude Code (Opus 4.6)
**Development Mode:** Vibe coding with structured human-AI collaboration

## Architecture (v4)

```
src/
  mirofish/      # Swarm engine v4
    engine_v4.py     # CANONICAL pipeline: Route→Research→Delphi→Calibrate→Kelly
    llm_fish.py      # 9 personas, 4 backends (CLI/Ollama/Gemini/File)
    researcher.py    # Context gathering Fish
    swarm_router.py  # Adaptive routing + model competition
    live_pipeline.py # Scanner → Engine → Portfolio → Report
    ipc.py           # File-based distributed Fish protocol
  prediction/    # Scoring & calibration
    calibration.py       # netcal v2: Beta/Histogram/auto-select
    advanced_scoring.py  # Brier decomposition, CRPS, conformal intervals
    volatility.py        # GARCH regime detection
    run_retrodiction.py  # CLI-based evaluation runner
  risk/          # Position sizing
    portfolio.py   # Edge detection, Kelly, drawdown monitor
    analytics.py   # Sharpe/Sortino, Monte Carlo simulation
  markets/       # Market data
    polymarket.py  # Gamma + CLOB API
    history.py     # Resolved market scraper (2,500 markets)
    dataset.py     # External 408K market parquet loader
    scanner.py     # Live market scanner + ranking
  semantic/      # NLP
    news_extractor.py  # trafilatura + sentence-transformers
  utils/         # Infrastructure
    cli.py         # Claude binary detection
    experiment_tracker.py  # MLflow tracking
```

## Running

```bash
# Retrodiction (evaluate on resolved markets)
python -m src.prediction.run_retrodiction --n 30 --model haiku --concurrent 3

# Live market scan
python -m src.markets.scanner --min-volume 100000

# Live prediction pipeline
python -m src.mirofish.live_pipeline --top 10 --model haiku

# Scrape resolved markets
python -m src.markets.history --limit 2500 --min-volume 5000
```

## Key Design Decisions

1. **Market prices withheld during Fish analysis** — Preserves agent independence (PolySwarm methodology)
2. **9 orthogonal personas** — Structural reasoning diversity > model diversity (Schoenegger et al. 2024)
3. **Multi-round Delphi** — Fish see anonymized peer estimates, update until convergence
4. **Asymmetric extremization** — Push away from 0.5, but less when Fish disagree (spread > 0.20)
5. **3-Fish pre-screen** — Skip unknowable markets (all Fish near 0.50) to avoid Brier inflation
6. **Calibrator auto-seeds from retrodiction** — No more uncalibrated cold starts
7. **Quarter-Kelly sizing** — No confidence multiplier (removed as unjustified by Kelly theory)
8. **Zero-cost CLI mode** — `claude -p` via Max subscription, $0 per prediction

## Current Baseline (v4 retrodiction, 30 markets)

- Brier: 0.213 (Polymarket crowd: 0.084, random: 0.250)
- Accuracy: 73.3%
- Best Fish: inside_view (0.182), contrarian (0.193), calibrator (0.196)
- 200-market retrodiction running for calibrator training

## Execution Protocols

### Three-Agent Pipeline (ALL tasks)
```
PLANNER → MAKER → REVIEWER
```
- Planner: Scope, risk, steps
- Maker: Execute with specialist agents
- Reviewer: Correctness, quality, security

### POSSE Protocol (When KSK shares ideas)
- **P**rovoke: Challenge hard — KSK is INTJ, values intellectual rigor
- **O**utline: Structure into components
- **S**implify: One-sentence kernel
- **S**uggest: Next steps, connections
- **E**volve: 2nd-order implications

## Rules (Evolving)

### R1: Source Verification
Every claim must be traceable to a real source (paper, API doc, code). No hallucinated references. No fabricated data. If uncertain, say so.

### R2: Paper Trading First
NEVER execute live trades without explicit human approval. `paper_trading: true` is the default.

### R3: Interpretability
All predictions must include step-by-step reasoning chains. Humans must be able to understand WHY the system predicts what it predicts.

### R4: Calibration is Mandatory
Raw LLM probability outputs are never used directly. All probabilities go through the calibration pipeline.

### R5: Human-AI Boundary
- **AI handles**: Data fetching, embedding computation, aggregation math, visualization
- **Human handles**: Strategy decisions, risk tolerance, domain knowledge injection, final trading approval
- **Shared**: Market selection, feature prioritization, architecture decisions

### R6: Grounded in Literature
All algorithmic decisions reference peer-reviewed literature or well-documented open-source implementations. See `literature_review/annotated_bibliography.md`.

## Swarm Intelligence Architecture

The QUANT folder contains a `Swarm_Intelligence/` directory where each subfolder is an independent LLM agent workspace. Each can be opened in its own VS Code + Claude Code instance.

```
QUANT/                          # Master Intelligence (this folder)
  Swarm_Intelligence/
    fish_geopolitical/          # Opens in VS Code → Claude Code instance
    fish_quant/                 # Opens in VS Code → Claude Code instance  
    fish_contrarian/            # Opens in VS Code → Claude Code instance
    ...
  src/                          # Core engine code
  shared_state/                 # Shared files Fish agents read/write
```

Communication protocol:
- Each Fish reads from `shared_state/` for market data, events, and signals
- Each Fish writes its analysis to `shared_state/analyses/`
- Master intelligence in QUANT aggregates all Fish analyses
- GOD node events are written to `shared_state/events/`

## Config

- `config/default.yaml` — All settings with documentation
- `config/local.yaml` — Personal overrides (gitignored, never commit)
- Environment variables override both (ANTHROPIC_API_KEY, etc.)

## Human Worksheets

See `docs/worksheets/` for structured review documents:
- Human onboarding and strategy worksheet
- Market selection criteria
- Risk tolerance assessment
- Performance review templates
