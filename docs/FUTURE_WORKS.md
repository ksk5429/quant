# Future Works & Discussion Topics

> A structured agenda for team meetings. Each section contains a research question, current status, proposed approaches, and discussion prompts for the group.

---

## How to Use This Document

This document is designed for **team meetings and brainstorming sessions**. Each topic follows the same format:

1. **The Question** — What problem are we trying to solve?
2. **Current Status** — What exists today in Mirofish?
3. **Proposed Approaches** — Concrete options to discuss
4. **Discussion Prompts** — Questions to drive the meeting conversation
5. **Reading Material** — Papers and resources to review beforehand

Schedule one topic per meeting (30-60 minutes). Assign the reading material before the meeting.

---

## Topic 1: Model Specialization vs. Model Diversity

### The Question
Should we fine-tune specialized models per Fish persona, or keep using diverse prompts on general-purpose models?

### Current Status
All Fish use the same base model (Claude) with different system prompts. The diversity comes from persona prompts, not from model architecture.

### Proposed Approaches

**A) Prompt-only diversity (current)**
- Same model, different system prompts
- Cheapest, simplest
- Limited by the model's ability to role-play different analytical styles

**B) Multi-model diversity**
- Fish 1: Claude Sonnet (strong at reasoning)
- Fish 2: GPT-4o (strong at world knowledge)
- Fish 3: Gemini (strong at multimodal)
- Fish 4: Llama 3 local (free, private)
- Real architectural diversity, not just prompt diversity

**C) Fine-tuned specialists**
- Train each Fish on domain-specific data (financial news, political analysis, etc.)
- Highest accuracy ceiling but expensive to train
- Turtel et al. (2025) showed a 14B fine-tuned model beat GPT-4o on Polymarket

### Discussion Prompts
1. Is prompt diversity sufficient, or does true model diversity produce measurably better ensembles?
2. What's the cost/benefit of running 3 different API providers vs. 7 Claude instances?
3. Should we invest in fine-tuning once we have 1,000+ resolved predictions to train on?
4. How do we evaluate whether adding a new model actually improves the ensemble?

### Reading Material
- Schoenegger et al. (2024): "Wisdom of the Silicon Crowd" — diversity drives accuracy
- Turtel et al. (2025): "Outcome-based RL to Predict the Future" — fine-tuned models beat general

---

## Topic 2: Real-Time Event Detection

### The Question
How should the system detect market-moving events in real time, rather than relying on the human operator to inject them?

### Current Status
Events are injected manually by the human operator via the GOD Node. No automated event detection.

### Proposed Approaches

**A) News API polling**
- Poll Tavily/Serper/NewsAPI every 5 minutes for breaking news
- LLM classifies: does this affect any of our tracked markets?
- Cost: ~$0.10/hour for API + LLM classification

**B) Social media monitoring**
- Track X/Twitter for keywords related to tracked markets
- Sentiment spikes trigger Fish re-analysis
- Challenge: noise, false positives, API cost

**C) Polymarket WebSocket**
- Subscribe to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Detect sudden price movements (>5% in 5 minutes)
- Price movement implies someone knows something — trigger re-analysis
- Cheapest, most direct signal

**D) Hybrid: WebSocket price alerts + periodic news scan**
- WebSocket detects the price move first
- News scan identifies what caused it
- Fish analyze with the news context

### Discussion Prompts
1. What latency do we need? Minutes matter for latency arbitrage; hours are fine for fundamental analysis.
2. How do we avoid false positives from price noise vs. real information?
3. Should the system run 24/7 or only during human-active hours?
4. What's the budget for real-time data feeds?

### Reading Material
- PolySwarm (arXiv:2604.03888): 5-second scan loop with Gamma API
- Polymarket WebSocket docs: https://docs.polymarket.com/developers/CLOB/websocket/wss-overview

---

## Topic 3: Cross-Market Arbitrage Engine

### The Question
Can we systematically detect and exploit mispricings between related Polymarket contracts?

### Current Status
`src/network/market_graph.py` has negation pair detection and divergence computation. Not yet connected to automated trading.

### Proposed Approaches

**A) Negation pair arbitrage (simplest)**
- Find markets where YES + NO prices don't sum to $1.00
- If sum < $0.98: buy both sides for guaranteed profit
- Saguillo et al. (2025) found this generated millions on Polymarket

**B) Semantic combinatorial arbitrage**
- Markets A: "Will X win the election?" (0.60)
- Market B: "Will X win the primary?" (0.50)
- If X can't win the election without winning the primary, then P(A) <= P(B) should hold
- Violations are exploitable
- Requires LLM to understand logical relationships between markets

**C) Correlation-based mean reversion**
- Build correlation graph from historical price series
- When correlated markets diverge beyond 2 standard deviations, bet on reversion
- Classic statistical arbitrage applied to prediction markets

### Discussion Prompts
1. How much capital should we allocate to arbitrage vs. fundamental prediction?
2. What's the minimum edge that's worth executing after transaction costs?
3. How do we handle the risk that "arbitrage" is actually correct information we don't have?
4. Should the arbitrage engine operate independently from the Fish swarm?

### Reading Material
- Saguillo et al. (2025): "Empirical Arbitrage Analysis on Polymarket" — $40M extracted
- Tsang & Yang (2026): "Anatomy of Polymarket" — market microstructure

---

## Topic 4: Calibration & Learning Over Time

### The Question
How does the system get better over time? What's the learning loop?

### Current Status
Brier scores are tracked per Fish. Isotonic calibration is available but requires 100+ resolved predictions.

### Proposed Approaches

**A) Historical resolution database**
- Scrape all resolved Polymarket markets (tens of thousands)
- Build a labeled dataset: (market_question, resolution_date, outcome)
- Use this for: base rate estimation, calibration training, backtesting

**B) Online calibration updating**
- After each market resolves, update the calibration model incrementally
- Use a sliding window of the last 500 resolved predictions
- The calibrator gets better every day

**C) Fish performance-based pruning**
- If a Fish's Brier score is consistently above 0.30 (bad), reduce its weight to near zero
- If a Fish is consistently the best, clone its persona with slight variations
- Natural selection applied to AI agents

**D) Reinforcement learning on resolved predictions**
- Fine-tune a model using RLVR on resolved Polymarket questions
- Turtel et al. (2025) achieved ROI improvements of ~20% with this approach
- Requires 10,000+ resolved examples for stable training

### Discussion Prompts
1. How long until we have enough data for meaningful calibration? (100+ resolutions)
2. Should we weight recent performance more than historical? (recency weighting)
3. Is Fish pruning too aggressive? Could we lose valuable contrarian signal?
4. What's the ROI on fine-tuning vs. better prompting?

### Reading Material
- Kapoor & Gruver (2024): "LLMs Must Be Taught Uncertainty" — 1,000 examples for calibration
- Jenane et al. (2026): "Entropy to Calibrated Uncertainty" — intrinsic calibration

---

## Topic 5: Graph Neural Networks for Market Correlation

### The Question
Can GNNs learn market correlations that simple cosine similarity misses?

### Current Status
Market graph is built from embedding cosine similarity and price correlation. No learned graph representations yet.

### Proposed Approaches

**A) Graph Attention Network (GAT)**
- Learn which market connections matter most
- Attention weights reveal WHY markets are correlated
- Interpretable: we can see which edges the model considers important

**B) Temporal Graph Neural Network (TGNN)**
- Correlations change over time — static graphs miss this
- TGNN captures evolving market relationships
- Fanshawe et al. (2026) showed TGNN outperforms during market stress

**C) Hypergraph approach**
- Some correlations are multi-way (3+ markets are jointly correlated)
- Standard graphs only capture pairwise relationships
- Hypergraphs model group dependencies

### Discussion Prompts
1. Do we have enough data to train a GNN? (How many markets x time steps?)
2. Is the interpretability of attention weights valuable for our use case?
3. How often should the graph be rebuilt? (Every hour? Every day?)
4. Should the graph influence Fish assignments or just the aggregation?

### Reading Material
- Baaijens et al. (2025): Cosine similarity > price correlation for financial GNNs
- Fanshawe et al. (2026): Temporal-Heterogeneous GNN for correlation forecasting

---

## Topic 6: Expanding Beyond Polymarket

### The Question
Should we support multiple prediction market platforms?

### Current Status
Polymarket only (Gamma API + CLOB API). Architecture is not yet platform-agnostic.

### Proposed Approaches

**A) PMXT unified SDK**
- `pmxt` (pmxt.dev) provides a "CCXT for prediction markets"
- Supports: Polymarket, Kalshi, Limitless, Probable, Baozi
- Unified API: `fetch_markets()`, `create_order()`, etc.
- 622 stars, 70 releases, actively maintained

**B) Cross-platform arbitrage**
- Same event priced differently on Polymarket vs. Kalshi
- Buy on the cheaper platform, sell on the more expensive
- Requires accounts on both platforms

**C) Platform-specific strategies**
- Polymarket: crypto-heavy, blockchain-native, more volatile
- Kalshi: US-regulated, more conservative, institutional
- Different Fish personas for different platforms

### Discussion Prompts
1. Which platform has the most inefficiency (biggest alpha opportunity)?
2. Is cross-platform arbitrage practical given different settlement mechanisms?
3. Should each platform have its own Fish swarm or share one?
4. What's the regulatory risk of operating across platforms?

### Reading Material
- PMXT docs: https://pmxt.dev
- Kalshi docs: https://kalshi.com/docs

---

## Topic 7: Autonomous vs. Human-in-the-Loop

### The Question
How much autonomy should the system have? Where is the human essential?

### Current Status
Fully human-in-the-loop. Paper trading only. Human triggers every analysis and approves every signal.

### Proposed Approaches

**A) Fully manual (current)**
- Human triggers analysis, reviews signals, approves trades
- Safest, but doesn't scale

**B) Semi-autonomous**
- System scans markets automatically every hour
- Generates signals automatically
- Human approves/rejects signals before execution
- Alert system (email, Slack) for high-edge opportunities

**C) Autonomous with guardrails**
- System trades automatically within strict limits
- Max $10/day total exposure
- Instant stop if drawdown exceeds 10%
- Human reviews performance weekly
- All trades logged for audit

**D) Fully autonomous (distant future)**
- 24/7 operation with no human intervention
- Self-calibrating, self-healing
- Requires months of proven performance in (C) first

### Discussion Prompts
1. What's the minimum track record before we trust autonomous trading?
2. What's the worst-case scenario at each autonomy level?
3. Should the human retain veto power even in autonomous mode?
4. How do we handle the system making a trade we disagree with?

---

## Topic 8: Team Roles & Collaboration

### The Question
How should a team of humans work together with the AI system?

### Current Status
Single human operator (KSK) + Claude Code AI.

### Proposed Roles

| Role | Responsibility | Time Commitment |
|------|---------------|-----------------|
| **Strategist** | Decides which markets to target, risk parameters | 2 hrs/week |
| **Domain Specialist** | Provides expertise in specific categories (politics, crypto, etc.) | 3 hrs/week |
| **Quant Developer** | Maintains code, adds features, runs backtests | 5 hrs/week |
| **Risk Manager** | Reviews signals, approves/blocks trades, monitors drawdown | 1 hr/day |
| **AI Operator** | Runs Fish swarms, injects events, manages Claude Code sessions | 2 hrs/day |

### Discussion Prompts
1. Who makes the final call on trade execution?
2. How do we handle disagreements between team members?
3. Should domain specialists run their own Fish agent?
4. What's the minimum viable team size?
5. How do we split profits (and losses)?

---

## Topic 9: Ethical & Legal Considerations

### The Question
What are the ethical and legal boundaries of AI-powered prediction market trading?

### Discussion Prompts
1. Is it ethical to use AI to gain an advantage over human traders?
2. Are there regulatory requirements for algorithmic trading on Polymarket?
3. How do we handle markets about sensitive topics (elections, conflicts, health)?
4. Should we self-impose restrictions on which markets to trade?
5. What's our responsibility if our predictions influence market prices?
6. How do we handle information asymmetry — when our AI knows something the market doesn't?
7. What's the tax treatment of prediction market profits in our jurisdiction?

### Reading Material
- Polymarket Terms of Service
- CFTC guidance on prediction markets (Kalshi vs. CFTC ruling)
- Wolfers & Zitzewitz (2004): "Prediction Markets" — economic theory of information aggregation

---

## Topic 10: Long-Term Vision

### The Question
Where is Mirofish in 1 year? 3 years? 5 years?

### Discussion Prompts
1. **1 year:** A profitable, semi-autonomous system with proven calibration on 500+ markets?
2. **3 years:** A platform other teams can use? An open-source project with contributors?
3. **5 years:** A full-stack prediction intelligence company? Integration with traditional finance?
4. Should we publish our methodology? (Academic paper, blog posts, open-source)
5. What's the competitive moat? (Data, calibration, speed, or something else?)
6. How do we measure success? (ROI? Brier score? Number of accurate predictions?)
7. What would make us stop? (Consistent losses, regulatory changes, ethical concerns)

---

## Topic 11: Tiered Compute Architecture (KSK's Key Insight)

### The Insight
> "First pass of fish is expensive. Sequential and depth. Events or price spikes will be a separate flow that works on a subset of data. Think opus vs sonnet vs haiku. You need expensive first pass to understand and generate the graph. But after that you may not need as many fish running. Only diff based."

This is the single most important architectural evolution for Mirofish. It transforms the system from "run everything every time" to "build understanding once, update incrementally."

### Proposed Architecture: Full Pass + Differential Updates

```
FULL PASS (expensive, infrequent — weekly or when entering new markets)
  8 Fish x Opus 4.6 x 12 markets = 96 analyses
  Builds: market graph, cross-market correlations, base knowledge
  Cost: ~2 hours human time, $0 on Max plan
  Output: shared_state/knowledge_graph.json (the "brain")

DIFFERENTIAL UPDATE (cheap, frequent — daily or on events)
  Triggered by: price spike >5%, breaking news, market resolution
  Runs: 2-3 Fish (Researcher + most relevant specialist) x affected markets only
  Uses: Sonnet or Haiku model (if API), or quick Claude session (if Max)
  Cost: ~15 minutes, hits only changed data
  Output: patches to knowledge_graph.json

EVENT-TRIGGERED REANALYSIS (medium, on-demand)
  Triggered by: GOD node event injection (human drops real-world event)
  Runs: 3-5 Fish on affected market cluster only
  Reads: existing knowledge graph + new event
  Output: updated probabilities for affected markets only
```

### Why This Works (Computer Science Analogy)
- **Full Pass** = full build (`make clean && make all`)
- **Differential** = incremental build (`make` — only recompile changed files)
- **Event-triggered** = hot reload (patch running system without full restart)

The knowledge graph is the compiled artifact. Once built, you don't rebuild from scratch — you patch it.

### Implementation Plan
1. Define `shared_state/knowledge_graph.json` schema (markets, correlations, Fish estimates, timestamps)
2. Full pass writes to knowledge graph
3. `diff_update.py` script: takes event/price change, identifies affected markets, runs minimal Fish
4. `event_trigger.py` script: human describes event, system identifies affected graph nodes, runs targeted reanalysis

### Model Routing (KSK's opus/sonnet/haiku insight)
| Task | Model | Cost | When |
|------|-------|------|------|
| Full Pass analysis | Opus 4.6 (Max plan) | $0 | Weekly, new markets |
| Differential update | Sonnet 4.6 (API) | ~$0.01/market | Daily price checks |
| Quick screening | Haiku 4.5 (API) | ~$0.001/market | Hourly market scan |
| Event reanalysis | Opus 4.6 (Max plan) | $0 | On breaking news |

### Discussion Prompts
1. How often should the full pass run? Weekly? Only when entering a new market category?
2. What triggers a differential update? Price change >5%? Volume spike >2x?
3. Should the knowledge graph store Fish reasoning chains, or just probabilities?
4. How do we handle graph staleness? Auto-expire after 7 days?
5. Can we pre-compute which Fish are most relevant per market category?

---

## Meeting Schedule Template

| Week | Topic | Prep Reading | Duration |
|------|-------|-------------|----------|
| 1 | Topic 8: Team Roles | This document | 30 min |
| 2 | Topic 7: Autonomy Levels | User Manual | 45 min |
| 3 | Topic 1: Model Diversity | Schoenegger 2024 | 60 min |
| 4 | Topic 3: Arbitrage Engine | Saguillo 2025 | 60 min |
| 5 | Topic 2: Event Detection | PolySwarm paper | 45 min |
| 6 | Topic 4: Calibration & Learning | Kapoor 2024 | 60 min |
| 7 | Topic 5: GNNs | Baaijens 2025 | 60 min |
| 8 | Topic 6: Multi-Platform | PMXT docs | 45 min |
| 9 | Topic 9: Ethics & Legal | Polymarket ToS | 45 min |
| 10 | Topic 10: Long-Term Vision | All above | 60 min |
