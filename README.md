# Mirofish: Swarm Intelligence Prediction Engine

> LLM-powered swarm intelligence for prediction market analysis and quant trading.

## What Is This?

Mirofish is a **semantic prediction engine** that uses multiple LLM agents ("Fish") organized in a swarm to analyze prediction markets (Polymarket, Kalshi, etc.). Each Fish handles one market, communicates with neighboring Fish through a correlation graph, and collectively produces calibrated probability estimates that outperform individual forecasters.

### Core Concept

```
Real-World Event (GOD Node)
        |
        v
  [Fish-1] <---> [Fish-2] <---> [Fish-3]
     |               |               |
  Market A        Market B        Market C
     |               |               |
     +-------+-------+-------+-------+
             |
     Bayesian Aggregation
             |
     Calibrated Probability (0, 1)
             |
     Quarter-Kelly Position Sizing
             |
         TRADE SIGNAL
```

### Key Features

- **Swarm Intelligence**: N diverse LLM personas, each specializing in one market
- **GOD Node**: Inject real-world events to trigger swarm-wide re-evaluation
- **Semantic Analysis**: LLM-based understanding of market descriptions, news, and cross-market relationships
- **Market Correlation Graph**: NetworkX-based graph detecting related markets via KL/JS divergence
- **Bayesian Aggregation**: Confidence-weighted probability fusion (not naive averaging)
- **Probability Calibration**: Isotonic regression + Brier score tracking
- **Risk Management**: Quarter-Kelly criterion position sizing
- **Visualization Dashboard**: Heatmaps, network graphs, probability distributions

## Architecture

```
src/
  mirofish/      # Swarm engine (Fish agents, GOD node, message bus)
  semantic/      # LLM semantic analysis, embeddings, correlations
  markets/       # Polymarket API client, data feeds, market graph
  prediction/    # Bayesian aggregation, calibration, backtesting
  network/       # Market correlation graph, divergence analysis
  visualization/ # Dashboard, heatmaps, network plots
  risk/          # Kelly criterion, position sizing, drawdown limits
  utils/         # Config, logging, rate limiting
```

See [docs/architecture/system_design.md](docs/architecture/system_design.md) for full architecture documentation.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/quant.git
cd quant
pip install -e ".[dev]"

# 2. Set API keys
cp config/default.yaml config/local.yaml
# Edit config/local.yaml with your API keys

# 3. Run the dashboard
python -m src.visualization.dashboard

# 4. Analyze a market
python -m src.mirofish.swarm --market "Will X happen by 2026?"

# 5. Run backtester
python -m src.prediction.backtester --start 2025-01-01 --end 2026-01-01
```

## Research Foundation

This project is built on peer-reviewed research. See [literature_review/](literature_review/) for:
- 26 annotated references spanning LLM forecasting, swarm intelligence, calibration, and market microstructure
- Architecture directly informed by PolySwarm (arXiv:2604.03888) and Wisdom of Silicon Crowd (Science Advances, 2024)

## Human-AI Development

This project is developed through structured human-AI collaboration:
- **Human Review Worksheets**: [docs/worksheets/](docs/worksheets/) contains DOCX workbooks for human review
- **Developer Notes**: [docs/DEVELOPERS_NOTE.md](docs/DEVELOPERS_NOTE.md) tracks design decisions
- **Changelog**: [docs/changelog/](docs/changelog/) tracks all changes with rationale
- **Rules**: [.claude/rules/](.claude/rules/) evolve based on human feedback

## Status

**Phase 1: Foundation** (current) -- Directory structure, core engine, literature review
**Phase 2: Intelligence** -- Swarm agents, semantic analysis, market data integration
**Phase 3: Calibration** -- Probability calibration, backtesting, risk management
**Phase 4: Production** -- Dashboard, live trading, monitoring

## License

MIT

## Disclaimer

This is a research and educational tool. Trading on prediction markets involves financial risk. Past performance does not guarantee future results. Use at your own risk.
