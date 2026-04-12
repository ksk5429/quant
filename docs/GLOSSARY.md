# Glossary

> Every term used in Mirofish, defined for newcomers.

---

## A

**Aggregation** — Combining multiple probability estimates into a single consensus prediction. Mirofish uses Bayesian confidence-weighted aggregation, not simple averaging.

**Arbitrage** — Exploiting price discrepancies between related markets for risk-free profit. On Polymarket, two types exist: intra-market (YES + NO prices don't sum to $1) and inter-market (semantically related markets are mispriced relative to each other).

## B

**Bayesian** — A statistical approach that updates beliefs using Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E). Start with a prior, update with evidence, get a posterior.

**Brier Score** — A scoring rule that measures prediction accuracy. Formula: mean((predicted - actual)^2). Range: 0 (perfect) to 1 (maximally wrong). Human superforecasters score ~0.18.

## C

**Calibration** — The property that a predictor's stated probabilities match actual frequencies. If you say "70% likely" for 100 events, roughly 70 should occur. LLMs are systematically overconfident and need post-hoc calibration.

**CLOB** — Central Limit Order Book. Polymarket's trading engine where buy and sell orders are matched. Accessed via the CLOB API.

**Confidence** — A Fish's self-assessed certainty in its own prediction. Range: 0 (no confidence) to 1 (fully confident). Used as a weight in aggregation.

**Contrarian** — A Fish persona that deliberately looks for reasons the consensus might be wrong. The most valuable Fish for detecting groupthink and bias.

## D

**Drawdown** — The peak-to-trough decline in portfolio value. If your bankroll goes from $1,200 to $900, the drawdown is 25%. Mirofish stops trading at 15% drawdown.

**Divergence** — A measure of difference between probability distributions. Jensen-Shannon divergence is used to detect cross-market mispricing.

## E

**ECE (Expected Calibration Error)** — The average gap between predicted probabilities and observed frequencies across bins. Lower is better.

**Edge** — The difference between our predicted probability and the market price. Edge > 0.05 = potentially profitable trade.

**Ensemble** — A collection of models (Fish) whose combined prediction is more accurate than any individual model. The core principle behind Mirofish.

## F

**Fish** — An individual LLM agent in the Mirofish swarm. Each Fish has a unique persona (geopolitical analyst, quant, contrarian, etc.) and produces independent probability estimates.

## G

**Gamma API** — Polymarket's market metadata API. No authentication required for reads. Returns market questions, descriptions, prices, volumes.

**GOD Node** — Global Omniscient Dispatcher. The master intelligence that injects real-world events into the swarm, triggers re-analysis, and coordinates Fish communication.

## I

**Implied Probability** — The probability embedded in a market price. A YES price of $0.65 implies a 65% chance of the event occurring.

**Isotonic Regression** — A non-parametric calibration method that fits a monotonically increasing function from predicted probabilities to observed frequencies. The default calibration method in Mirofish.

## J

**Jensen-Shannon Divergence** — A symmetric measure of difference between two probability distributions. Range: 0 (identical) to 1 (maximally different). Used for cross-market mispricing detection.

## K

**Kelly Criterion** — A formula for optimal bet sizing that maximizes long-run wealth growth. f* = edge / (1 - price) for YES bets. Mirofish uses quarter-Kelly (0.25x) for safety.

## L

**LLM** — Large Language Model. AI models like Claude or GPT that understand and generate text. In Mirofish, LLMs power the Fish agents for semantic market analysis.

## M

**MessageBus** — The internal communication system for Fish agents. Supports broadcast (all Fish), targeted (specific Fish), and topic-based routing.

**Mirofish** — The complete swarm intelligence prediction engine. Named after the schooling behavior of fish.

## N

**Negation Pair** — Two markets that ask opposite questions ("Will X?" and "Will X NOT?"). Their prices should sum to $1.00. Deviations are arbitrage opportunities.

## P

**Paper Trading** — Simulated trading without real money. Mirofish defaults to paper trading mode. Live trading requires explicit human approval.

**Persona** — The analytical personality assigned to a Fish agent. Seven personas exist: geopolitical analyst, financial quant, Bayesian statistician, investigative journalist, contrarian thinker, domain expert, calibration specialist.

**Platt Scaling** — A calibration method that fits a sigmoid function to log-odds. Works well with small sample sizes (<100 predictions).

**Polymarket** — The largest prediction market platform. Trades on Polygon blockchain (chain_id=137). Markets are binary (YES/NO) with prices between $0 and $1.

**Prediction Market** — A market where participants trade contracts whose payoff depends on the outcome of a future event. Prices reflect the crowd's consensus probability.

## Q

**Quarter-Kelly** — Using 25% of the full Kelly Criterion bet size. Gives ~56% of maximum growth rate with ~6% of the variance. The recommended position sizing approach.

## R

**RAG (Retrieval-Augmented Generation)** — Enhancing LLM predictions by first retrieving relevant information (news articles, data) and including it in the prompt. Critical for prediction accuracy.

## S

**Semantic Similarity** — How similar two texts are in meaning (not just words). Computed using embedding vectors and cosine similarity. Markets about related topics have high semantic similarity.

**Spread** — In the swarm context: the difference between the highest and lowest Fish predictions. High spread = disagreement = lower confidence.

**Swarm Intelligence** — The collective behavior of decentralized, self-organized agents. In Mirofish, multiple Fish agents with diverse perspectives produce predictions that are more accurate than any single agent.

## T

**Temperature Scaling** — The simplest calibration method. Divides log-odds by a single scalar parameter to adjust confidence.

**Token** — A unit of text processing for LLMs. One token is roughly 4 characters. API costs are based on tokens consumed.

## V

**Vibe Coding** — A development approach where the human provides high-level direction and the AI writes the code. The approach used to build Mirofish.
