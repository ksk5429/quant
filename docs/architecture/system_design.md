# System Design: K-Fish Prediction Engine

## Overview

K-Fish is a multi-agent LLM system for prediction market analysis. The system uses swarm intelligence — multiple diverse AI agents independently analyzing markets, then aggregating their predictions through Bayesian consensus.

## Core Insight

> "An ensemble of 12 LLMs achieved forecasting accuracy statistically indistinguishable from 925 human forecasters." — Schoenegger et al., Science Advances 2024

Diversity across agents, not model size, drives prediction accuracy.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  ┌──────────┐  ┌──────────┐  ┌─��────────────────┐     │
│  │ Gamma API│  │ CLOB API │  │ News/RAG (ChromaDB│)    │
│  │ (metadata│  │ (prices) │  │ Tavily, Serper)   │     │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘     │
│       │              │                 │                │
└───────┼──────────────┼─────────────────┼────────────────┘
        │              │                 │
┌───────┼──────────────┼─────────────────┼────────────────┐
│       ▼              │                 ▼                │
│  ┌─────────────────────────────────────────────┐       │
│  │              SWARM LAYER                     │       │
│  │                                               │       │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │       │
│  │  │Fish1│ │Fish2│ │Fish3│ │Fish4│ │Fish5│  │       │
│  │  │Geopo│ │Quant│ │Bayes│ │Journ│ │Contr│  │       │
│  │  └──┬──┘ └──┬──┘ └──┬─���┘ └──┬──┘ └──┬──┘  │       │
│  │     │       │       │       │       │      │       │
│  │     └───────┴───────┴───────┴───────┘      │       │
│  │              MessageBus                      │       │
│  └──────────────────┬──────────────────────────┘       │
│                     │                                   │
│  ┌──────────────────▼──────────────────────────┐       │
│  │            GOD NODE                          │       │
│  │  Real-world event injection + routing        │       │
│  └──────────────────┬──────────────────────────┘       │
│                     │                INTELLIGENCE LAYER │
└─────────────────────┼──────────────────────────────────┘
                      │
┌─────────────────────┼──────────────────────────────────┐
│                     ▼              DECISION LAYER       │
│  ┌──────────────────────────────┐                      │
│  │  Bayesian Aggregation        │                      │
│  │  p_swarm = Σ(w_i·p_i)/Σ(w_i)│                      │
│  └──────────────┬───��───────────┘                      │
│                 │                                       │
│  ┌──────────────▼───────────────┐                      │
│  │  Probability Calibration     │                      │
│  │  Isotonic Regression         │                      │
│  └──────────────┬───────────────┘                      │
│                 │                                       │
│  ┌──────────────▼───────────────┐                      │
│  │  Kelly Criterion Sizing      │                      │
│  │  f* = edge / (1 - price)     │                      │
│  │  × 0.25 (quarter-Kelly)     │                      │
│  └──────────────┬───────────────┘                      │
│                 │                                       │
│  ┌──────────────▼───────────────┐                      │
│  │  TRADE SIGNAL                │                      │
│  │  BUY YES / BUY NO / ABSTAIN │                      │
│  └──────────────────────────────┘                      │
└────────────���────────────────────────────���──────────────┘
```

## Data Flow (Step by Step)

### Step 1: Market Discovery
- Gamma API scanned every 30s for active markets
- Filtered by: volume > $10k, liquidity > $5k, not resolved
- Markets embedded using sentence-transformers
- Correlation graph built from semantic similarity

### Step 2: Swarm Analysis
- Each Fish receives: market question, description, news context
- **Market price is WITHHELD** during Fish analysis (independence)
- Fish applies persona-specific reasoning lens
- Outputs: probability, confidence, reasoning chain, risk factors

### Step 3: Cross-Market Communication
- Fish share insights via MessageBus
- Signals: correlation alerts, probability updates, event reactions
- Topic-based routing (politics, crypto, economics, etc.)

### Step 4: Bayesian Aggregation
```
p_swarm = Σ(confidence_i × accuracy_bonus_i × p_i) / Σ(weights)
```
- Weights = confidence × (1 + historical accuracy bonus)
- Accuracy bonus from Fish's running Brier score
- This ensures reliable + confident Fish dominate

### Step 5: Calibration
- Raw swarm probability → isotonic regression → calibrated probability
- Requires 100+ resolved predictions before calibrating
- Evaluated via Brier score, ECE, log-loss

### Step 6: Position Sizing (Kelly Criterion)
```
edge = |p_our - p_market|
f* = edge / (1 - p_market)    # for YES bets
position = f* × 0.25 × confidence × bankroll
position = min(position, 0.05 × bankroll)  # cap at 5%
```

### Step 7: Risk Gate
- Drawdown > 15% → stop all trading
- Paper trading by default (no real money unless human approves)
- All signals logged for review

## Network Analysis Module

Markets form a graph:
- **Nodes** = prediction markets
- **Edges** = correlations (semantic similarity, price correlation, causal)
- **Edge types**: semantic, price, negation (arbitrage pair), causal

Analysis capabilities:
- Community detection (Louvain modularity)
- Centrality scoring (betweenness, eigenvector, PageRank)
- Negation pair detection (intra-market arbitrage)
- KL/JS divergence for cross-market mispricing

## Key References

| Component | Primary Reference |
|-----------|------------------|
| Swarm architecture | PolySwarm (arXiv:2604.03888) |
| Ensemble accuracy | Schoenegger et al. (Science Advances, 2024) |
| RAG pipeline | Halawi et al. (NeurIPS 2024) |
| Calibration | Geng et al. (NAACL 2024) |
| Kelly sizing | Thorp (1962), Kelly (1956) |
| Semantic graphs | Baaijens et al. (Applied Network Science, 2025) |
| Polymarket arbitrage | Saguillo et al. (AFT 2025) |
| Market microstructure | Tsang & Yang (2026) |
