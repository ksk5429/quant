# Mathematical Foundations

> Every equation used in Mirofish, explained step by step.

## Table of Contents
1. [Probability Basics](#1-probability-basics)
2. [Bayesian Confidence-Weighted Aggregation](#2-bayesian-confidence-weighted-aggregation)
3. [Kelly Criterion](#3-kelly-criterion)
4. [Probability Calibration](#4-probability-calibration)
5. [Scoring Rules](#5-scoring-rules)
6. [Information-Theoretic Divergence](#6-information-theoretic-divergence)
7. [Semantic Similarity](#7-semantic-similarity)
8. [Graph Metrics](#8-graph-metrics)

---

## 1. Probability Basics

### Prediction Market Prices as Probabilities

A prediction market price reflects the **implied probability** of an event occurring.

If a market trades at **$0.65** for YES:
- The market consensus is a **65% chance** the event happens
- A YES share costs $0.65 and pays $1.00 if the event occurs
- A NO share costs $0.35 and pays $1.00 if the event does not occur

**Profit from a correct YES bet:**
```
Profit = (1 / price) - 1 = (1 / 0.65) - 1 = 0.538  (53.8% return)
```

**Expected value of a bet:**
```
EV = P_true × (1/price - 1) - (1 - P_true) × 1
```

If our true probability estimate is 0.75 and market price is 0.65:
```
EV = 0.75 × (1/0.65 - 1) - 0.25 × 1
   = 0.75 × 0.538 - 0.25
   = 0.404 - 0.25
   = +0.154  (positive = profitable bet)
```

---

## 2. Bayesian Confidence-Weighted Aggregation

### Why Not Simple Averaging?

If 5 Fish predict: [0.70, 0.65, 0.80, 0.50, 0.72], simple average = 0.674.

But what if Fish 3 (0.80) has confidence 0.95 and a perfect track record, while Fish 4 (0.50) has confidence 0.30 and a terrible track record? Simple averaging treats them equally. That's wrong.

### The Aggregation Formula

```
P_swarm = SUM(w_i * P_i) / SUM(w_i)
```

where the weight of Fish i is:

```
w_i = confidence_i × (1 + accuracy_bonus_i)
```

and the accuracy bonus rewards Fish with good historical Brier scores:

```
accuracy_bonus_i = max(0, 1 - 2 × brier_score_i)
```

**Brier score range:** 0.0 (perfect) to 1.0 (maximally wrong)
- Brier = 0.0 → accuracy_bonus = 1.0 → weight doubles
- Brier = 0.25 → accuracy_bonus = 0.5 → weight × 1.5
- Brier = 0.50 → accuracy_bonus = 0.0 → no bonus (just confidence)

### Worked Example

| Fish | P_i | Confidence | Brier | Accuracy Bonus | Weight | Weighted P |
|------|-----|------------|-------|----------------|--------|------------|
| Geopolitical | 0.70 | 0.80 | 0.20 | 0.60 | 1.28 | 0.896 |
| Quant | 0.65 | 0.90 | 0.15 | 0.70 | 1.53 | 0.995 |
| Contrarian | 0.50 | 0.60 | 0.35 | 0.30 | 0.78 | 0.390 |
| Bayesian | 0.80 | 0.85 | 0.18 | 0.64 | 1.39 | 1.115 |
| Journalist | 0.72 | 0.70 | 0.25 | 0.50 | 1.05 | 0.756 |

```
Sum of weights = 1.28 + 1.53 + 0.78 + 1.39 + 1.05 = 6.03
Sum of weighted P = 0.896 + 0.995 + 0.390 + 1.115 + 0.756 = 4.152

P_swarm = 4.152 / 6.03 = 0.689
```

Notice: the contrarian (P=0.50, low confidence, poor track record) has minimal influence. The quant and Bayesian (high confidence, good track records) dominate.

### Ensemble Confidence

```
spread = max(P_i) - min(P_i)
agreement_bonus = max(0, 1 - 2 × spread)
confidence_swarm = mean(confidence_i) × (0.7 + 0.3 × agreement_bonus)
```

When Fish agree (low spread), confidence is boosted. When they disagree (high spread), confidence is penalized.

---

## 3. Kelly Criterion

### Origin

John Larry Kelly Jr. (1956) showed that **maximizing the expected log of wealth** produces the optimal long-run growth rate. Edward Thorp (1962) applied this to beat casinos and then financial markets.

### The Formula

For a binary bet with probability p of winning and odds b:

```
Full Kelly:  f* = (b × p - q) / b
```

where q = 1 - p.

For prediction markets where price = implied probability:

**YES bet (our P > market price):**
```
edge = P_our - P_market
f* = edge / (1 - P_market)
```

**NO bet (our P < market price):**
```
edge = P_market - P_our
f* = edge / P_market
```

### Why Quarter-Kelly?

Full Kelly maximizes growth rate but has wild variance. If your probability estimate is even slightly wrong, full Kelly can produce catastrophic drawdowns.

| Kelly Fraction | Growth Rate | Drawdown Risk | Recommended For |
|---------------|-------------|---------------|-----------------|
| Full (1.0) | Maximum | Very High | Never in practice |
| Half (0.5) | 75% of max | Moderate | Aggressive traders |
| **Quarter (0.25)** | **~56% of max** | **Low** | **Our default** |
| Tenth (0.1) | 19% of max | Very Low | Ultra-conservative |

Quarter-Kelly gives ~56% of the maximum growth rate with ~6% of the variance. Every serious Kelly practitioner (Thorp, Poundstone, Ziemba) recommends fractional Kelly.

### Worked Example

```
Our probability:  P_our = 0.72
Market price:     P_market = 0.60
Confidence:       0.85
Bankroll:         $1,000

edge = 0.72 - 0.60 = 0.12
raw_kelly = 0.12 / (1 - 0.60) = 0.30  (30% of bankroll)
quarter_kelly = 0.30 × 0.25 = 0.075
confidence_adjusted = 0.075 × 0.85 = 0.0638
position_capped = min(0.0638, 0.05) = 0.05  (5% cap)
position_dollars = 0.05 × $1,000 = $50.00

Signal: BUY YES, $50.00
```

### Expected Value

```
EV_per_dollar = P_our × (1/P_market - 1) - (1 - P_our)
              = 0.72 × (1/0.60 - 1) - 0.28
              = 0.72 × 0.667 - 0.28
              = 0.480 - 0.28
              = +0.200  (20 cents profit per dollar risked)
```

---

## 4. Probability Calibration

### The Problem

LLMs are **systematically overconfident** (Geng et al., NAACL 2024). When an LLM says "I'm 90% sure," the true probability is often closer to 75%. This miscalibration can destroy trading performance.

### Isotonic Regression

A non-parametric method that fits a **monotonically increasing** function from predicted probabilities to observed frequencies.

**How it works:**
1. Collect historical predictions and actual outcomes
2. Sort by predicted probability
3. Fit a step function that is non-decreasing and minimizes squared error

**Advantage:** No assumption about the shape of the calibration curve. If the miscalibration is S-shaped, linear, or arbitrary, isotonic regression will find it.

**Requirement:** Needs 100+ resolved predictions for reliable calibration.

### Platt Scaling

Fits a logistic (sigmoid) function to the log-odds:

```
P_calibrated = 1 / (1 + exp(-(a × log(P/(1-P)) + b)))
```

Parameters a and b are learned from historical data via logistic regression.

**Advantage:** Works with fewer samples (<100).
**Disadvantage:** Assumes sigmoid-shaped miscalibration.

### Temperature Scaling

The simplest method. Divides log-odds by a single scalar:

```
P_calibrated = sigmoid(log(P/(1-P)) / T)
```

If T > 1: predictions move toward 0.5 (reduces overconfidence).
If T < 1: predictions move toward 0 or 1 (increases confidence).

---

## 5. Scoring Rules

### Brier Score

```
BS = (1/N) × SUM((P_predicted - outcome)^2)
```

Where outcome is 0 (NO) or 1 (YES).

| Score | Meaning |
|-------|---------|
| 0.00 | Perfect prediction |
| 0.12-0.18 | Prediction market consensus level |
| 0.18-0.22 | Human superforecaster level |
| 0.25 | Random guessing at 0.5 |
| 1.00 | Maximally wrong |

**Our target: 0.18** (match human superforecasters).

### Expected Calibration Error (ECE)

Divide predictions into B bins by predicted probability. For each bin:

```
ECE = SUM(|bin_i| / N × |accuracy_i - confidence_i|)
```

A perfectly calibrated predictor has ECE = 0: in the bin of "70% predictions," exactly 70% of events occur.

### Log-Loss

```
LL = -(1/N) × SUM(y × log(P) + (1-y) × log(1-P))
```

Penalizes confident wrong predictions exponentially. A prediction of 0.99 for an event that doesn't happen contributes ~4.6 to log-loss.

---

## 6. Information-Theoretic Divergence

### Kullback-Leibler (KL) Divergence

Measures how different one probability distribution is from another:

```
KL(P || Q) = SUM(P_i × log(P_i / Q_i))
```

**Not symmetric:** KL(P||Q) != KL(Q||P). Undefined if Q_i = 0 where P_i > 0.

### Jensen-Shannon (JS) Divergence

A symmetric, bounded version of KL divergence:

```
M = (P + Q) / 2
JS(P || Q) = (KL(P || M) + KL(Q || M)) / 2
```

**Range:** 0 (identical) to 1 (maximally different). Used in Mirofish for cross-market mispricing detection.

### Application to Prediction Markets

For two related markets with implied probabilities P_1 = [p, 1-p] and P_2 = [q, 1-q]:

```
JS divergence tells us how "different" the two markets' beliefs are.
```

High JS divergence between semantically related markets suggests one or both are mispriced — an arbitrage opportunity.

---

## 7. Semantic Similarity

### Cosine Similarity

For two embedding vectors A and B:

```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

**Range:** -1 (opposite) to +1 (identical). In Mirofish, we use sentence-transformer embeddings (384 dimensions) to compute market-question similarity.

**Example:**
- "Will Bitcoin exceed $100k?" and "Will Ethereum exceed $10k?" → high similarity (~0.8)
- "Will Bitcoin exceed $100k?" and "Will the Democrats win in 2028?" → low similarity (~0.2)

### How We Use It

1. Embed all market questions using `all-MiniLM-L6-v2`
2. Compute pairwise cosine similarity matrix
3. Markets with similarity >= 0.7 get linked in the correlation graph
4. Linked markets share cross-market signals between Fish agents

---

## 8. Graph Metrics

### Betweenness Centrality

How often a node lies on the shortest path between other nodes:

```
C_B(v) = SUM(sigma_st(v) / sigma_st)
```

High betweenness = the market is a "bridge" connecting different clusters. Events affecting high-betweenness markets ripple widely.

### Eigenvector Centrality

A node's importance is proportional to the importance of its neighbors:

```
x_i = (1/lambda) × SUM(A_ij × x_j)
```

High eigenvector centrality = the market is connected to other important markets. Useful for identifying systemically significant markets.

### PageRank

Google's algorithm applied to markets:

```
PR(v) = (1-d)/N + d × SUM(PR(u) / out_degree(u))
```

Where d = 0.85 (damping factor). PageRank identifies markets that are both well-connected and connected to well-connected markets.

### Louvain Community Detection

Maximizes modularity Q to partition the graph into communities:

```
Q = (1/2m) × SUM((A_ij - k_i × k_j / 2m) × delta(c_i, c_j))
```

Markets in the same community share information more readily than markets across communities.
