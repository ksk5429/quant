# Literature Review: Mirofish Prediction Engine

> 26 annotated references spanning LLM forecasting, swarm intelligence, probability calibration, market microstructure, and network analysis.

## How to Use This Folder

- **[annotated_bibliography.md](annotated_bibliography.md)** — Full annotated bibliography with all 26 references
- **[summaries/](summaries/)** — Individual paper summaries (one file per paper)
- **[papers/](papers/)** — Downloaded PDFs (when available)

## Reference Categories

| Category | Count | Key Insight |
|----------|-------|-------------|
| LLM Forecasting | 6 | Ensembles of 12+ LLMs match human crowd accuracy |
| Financial NLP / Semantic Analysis | 4 | LLM sentiment scores predict stock returns at ~90% hit rate |
| Multi-Agent / Swarm Intelligence | 4 | PolySwarm: 50-persona swarm with Bayesian aggregation is SOTA |
| Probability Calibration | 4 | RLHF models are overconfident; isotonic regression fixes it |
| Information Aggregation Theory | 4 | Prediction markets are the best information aggregation mechanism |
| Network / Graph Analysis | 2 | GNN-based correlation forecasting outperforms during market stress |
| Polymarket Empirical | 2 | $40M arbitrage extracted from Polymarket in one year |

## Critical Design Implications

1. **Ensemble > single model** — Diversity across models, not model size, drives accuracy
2. **Retrieval is mandatory** — Raw LLM knowledge cutoffs kill forecasting
3. **RL fine-tuning on resolved questions** — Specialized models beat frontier models
4. **Calibration is not optional** — RLHF models are systematically overconfident
5. **Market age matters** — Thin early-stage markets are noisy; weight by liquidity
6. **Two alpha sources** — Intra-market mispricing + inter-market semantic arbitrage
7. **Social-choice aggregation** — Bayesian confidence-weighted beats naive voting
8. **Semantic similarity graphs** — Use topic similarity, not just price correlation
