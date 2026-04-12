# CLAUDE.md — Mirofish Prediction Engine

## Project Identity

**Mirofish** is a swarm intelligence prediction engine for prediction markets. It uses multiple LLM agents ("Fish") organized in a swarm to analyze markets, detect cross-market correlations, and produce calibrated probability estimates.

**Human Lead:** Kyeong Sun Kim (KSK)
**AI System:** Claude Code (Opus 4.6)
**Development Mode:** Vibe coding with structured human-AI collaboration

## Architecture

```
src/
  mirofish/      # Swarm engine: Fish agents, GOD node, MessageBus
  semantic/      # Embeddings, similarity, LLM semantic analysis
  markets/       # Polymarket Gamma/CLOB API clients
  prediction/    # Bayesian aggregation, calibration, backtesting
  network/       # Market correlation graph, divergence analysis
  visualization/ # Plotly dashboards, heatmaps, network plots
  risk/          # Kelly criterion, position sizing, drawdown limits
  utils/         # Config loader, logging
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/unit/ -v --cov=src
```

## Key Design Decisions

1. **Market prices withheld during Fish analysis** — Preserves agent independence (PolySwarm methodology)
2. **Bayesian confidence-weighted aggregation** — Not naive averaging (ref: Schoenegger et al. 2024)
3. **Quarter-Kelly position sizing** — Conservative risk management (ref: Thorp, PolySwarm)
4. **Isotonic regression calibration** — LLMs are overconfident (ref: Geng et al. NAACL 2024)
5. **Semantic similarity graphs** — Topic similarity > price correlation (ref: Baaijens et al. 2025)

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
