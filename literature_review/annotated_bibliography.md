# Annotated Bibliography

> Last updated: 2026-04-12
> Total references: 26

---

## 1. LLM Forecasting on Prediction Markets

### [1] Schoenegger et al. (2024) — "Wisdom of the Silicon Crowd"
- **Citation:** Schoenegger, P., Tuminauskaite, I., Park, P.S., Bastos, R.V.S., & Tetlock, P.E. (2024). "Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy." *Science Advances*, 10(45), eadp1528.
- **DOI:** 10.1126/sciadv.adp1528
- **Key Finding:** An ensemble of 12 LLMs achieved forecasting accuracy statistically indistinguishable from 925 human forecasters in a 3-month tournament. LLM predictions improved 17-28% when exposed to human median forecasts; simple averaging of human and LLM outputs yielded best results.
- **Relevance:** Core justification for ensemble architecture. Diversity across models beats any single model.
- **Design Implication:** Our swarm should use diverse personas AND diverse models, not just one LLM with different prompts.

### [2] Halawi et al. (2024) — "Approaching Human-Level Forecasting"
- **Citation:** Halawi, D., Zhang, F., Yueh-Han, C., & Steinhardt, J. (2024). "Approaching Human-Level Forecasting with Language Models." *NeurIPS 2024*. arXiv:2402.18563.
- **Key Finding:** A retrieval-augmented LM pipeline (search → forecast → aggregate) nears human crowd performance on competitive forecasting platforms. Introduces ForecastBench.
- **Relevance:** Blueprint for production LLM forecaster. RAG architecture is directly implementable.
- **Design Implication:** Retrieval-augment-aggregate is the proven pipeline. Our Fish agents must have news/RAG access.

### [3] Schoenegger & Park (2023) — "LLM Prediction Capabilities (Metaculus)"
- **Citation:** Schoenegger, P. & Park, P.S. (2023). "Large Language Model Prediction Capabilities: Evidence from a Real-World Forecasting Tournament." arXiv:2310.13014.
- **Key Finding:** GPT-4 in a 3-month Metaculus tournament was heavily outperformed by human crowds. Zero-shot LLM forecasting does not beat a 50% baseline.
- **Relevance:** Documents the performance floor. Naive prompting is insufficient.
- **Design Implication:** Never rely on zero-shot LLM prediction. Must add retrieval, ensembling, and calibration.

### [4] Turtel et al. (2025) — "Outcome-based RL for Polymarket"
- **Citation:** Turtel, B., Franklin, D., & Skotheim, K. (2025). "Outcome-based Reinforcement Learning to Predict the Future." *Transactions on Machine Learning Research*. arXiv:2505.17989.
- **Key Finding:** A 14B model fine-tuned with RLVR on 110k Polymarket questions matched or surpassed frontier models (o1). A 7-run ensemble earned $52 profit vs. $39 for o1 in Polymarket simulation.
- **Relevance:** Validates Polymarket-specific RL training. Smaller specialized models beat general-purpose frontier models.
- **Design Implication:** Future Phase: fine-tune a specialized model on resolved Polymarket questions using RLVR.

### [5] Alur et al. (2025) — "AIA Forecaster"
- **Citation:** Alur, R., Stadie, B.C., et al. (2025). "AIA Forecaster: Technical Report." arXiv:2511.07678.
- **Key Finding:** AIA Forecaster achieved performance statistically indistinguishable from human superforecasters on ForecastBench. Individual runs are unstable; ensembling over many runs is "absolutely essential."
- **Relevance:** Validates ensemble-over-runs pattern. Introduces MarketLiquid benchmark (1,610 questions).
- **Design Implication:** Run each Fish multiple times and aggregate, not just once. Instability in individual runs is expected.

### [6] Pratt et al. (2024) — "Can LLMs Use Forecasting Strategies?"
- **Citation:** Pratt, S., Blumberg, S., et al. (2024). "Can Language Models Use Forecasting Strategies?" arXiv:2406.04446.
- **Key Finding:** LLMs prompted with superforecasting strategies did not outperform simple baselines on GleanGen. Complex strategies did not improve over naive prediction.
- **Relevance:** Prompt engineering alone is insufficient for forecasting.
- **Design Implication:** Don't over-invest in prompt complexity. Invest in retrieval, ensembling, and calibration instead.

---

## 2. Semantic Analysis and Financial Prediction

### [7] Lopez-Lira & Tang (2023/2025) — "ChatGPT Stock Price Movements"
- **Citation:** Lopez-Lira, A. & Tang, Y. (2025). "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models." arXiv:2304.07619 (v6).
- **Key Finding:** GPT-4 sentiment scores on news headlines achieve ~90% hit rate for initial market reactions. Long-short strategy earned ~700% cumulative return (Oct 2021 - May 2024). Returns decline as LLM adoption rises.
- **Relevance:** Foundational result linking LLM semantics to financial alpha.
- **Design Implication:** Use LLM sentiment as a feature. But expect alpha decay as adoption increases.

### [8] Kirtac & Germano (2024) — "Sentiment Trading with LLMs"
- **Citation:** Kirtac, K. & Germano, G. (2024). "Sentiment Trading with Large Language Models." *Finance Research Letters*, 62. arXiv:2412.19245.
- **Key Finding:** OPT model outperforms BERT and FinBERT on 965K financial news articles, predicting stock returns with 74.4% accuracy. Sharpe ratio of 3.05.
- **Relevance:** Benchmarks LLM architectures for financial sentiment. Sharpe 3.05 is a strong target.
- **Design Implication:** LLM-based sentiment is strictly superior to traditional NLP (BERT, FinBERT) for financial prediction.

### [9] Nie et al. (2024) — "LLMs for Financial Applications (Survey)"
- **Citation:** Nie, Y., Kong, Y., et al. (2024). "A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges." arXiv:2406.11903.
- **Key Finding:** Domain-specific fine-tuning (FinBERT, FinGPT, BloombergGPT) outperforms general models. LLMs achieve higher Sharpe ratios than lexicon-based methods.
- **Relevance:** Comprehensive field map. Establishes SOTA per financial prediction subtask.
- **Design Implication:** Consider domain-specific fine-tuning for production deployment.

### [10] Wang & Wei (2025) — "Event-Aware LLM Sentiment Factors"
- **Citation:** Wang, Y. & Wei, Q. (2025). "Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets." arXiv:2508.07408.
- **Key Finding:** LLM multi-label event tagging on 100K+ tweets constructs interpretable alpha factors with IC > 0.05. Specific event categories are highly predictive.
- **Relevance:** Shows LLMs can extract typed semantic signals beyond raw sentiment.
- **Design Implication:** Tag news by event category, not just sentiment. Category-specific signals are stronger.

---

## 3. Multi-Agent / Swarm Intelligence

### [11] Barot & Borkhatariya (2026) — "PolySwarm"
- **Citation:** Barot, R.M. & Borkhatariya, A.S. (2026). "PolySwarm: A Multi-Agent LLM Framework for Prediction Market Trading and Latency Arbitrage." arXiv:2604.03888.
- **Key Finding:** 50-persona LLM swarm on Polymarket with Bayesian aggregation. Two-stage process: confidence-weighted swarm consensus, then combine with market-implied probability. Quarter-Kelly position sizing.
- **Relevance:** THE most directly relevant paper. Purpose-built for Polymarket.
- **Design Implication:** Our architecture directly follows this: diverse personas → withheld market price → Bayesian aggregation → Kelly sizing.

### [12] Jimenez-Romero et al. (2025) — "LLM MAS for Swarm Intelligence"
- **Citation:** Jimenez-Romero, C., Yegenoglu, A., & Blum, C. (2025). "Multi-Agent Systems Powered by LLMs: Applications in Swarm Intelligence." *Frontiers in AI*, 8:1593017.
- **Key Finding:** LLM agents in swarm simulations demonstrate decentralized collective intelligence. Structured prompts outperform autonomous prompts.
- **Relevance:** Theoretical grounding for swarm-based LLM architectures.
- **Design Implication:** Use structured persona prompts, not freeform. Structure drives swarm quality.

### [13] Zhao et al. (2024) — "Electoral LLM Collective Decision-Making"
- **Citation:** Zhao, X., Wang, K., & Peng, W. (2024). "An Electoral Approach to Diversify LLM-based Multi-Agent Collective Decision-Making." *EMNLP 2024*.
- **Key Finding:** GEDI electoral method using social choice theory significantly improves collective decision quality over majority vote or dictatorial aggregation.
- **Relevance:** Directly informs aggregation design.
- **Design Implication:** Consider GEDI/social-choice aggregation as alternative to simple weighted average.

### [14] Rosenberg (2023) — "Conversational Swarm Intelligence"
- **Citation:** Rosenberg, L. (2023). "Conversational Swarm Intelligence: A Pilot Study." arXiv:2309.03220.
- **Key Finding:** CSI using LLM agents as information-propagation conduits between small groups produced 30% more contributions (p<0.05) than standard chat. Consensus emerges faster.
- **Relevance:** Validates LLM-mediated swarm information propagation.
- **Design Implication:** Fish should propagate insights to neighbors before final aggregation, not just aggregate independently.

---

## 4. Probability Calibration

### [15] Geng et al. (2024) — "Survey: LLM Calibration (NAACL)"
- **Citation:** Geng, J., et al. (2024). "A Survey of Confidence Estimation and Calibration in Large Language Models." *NAACL 2024*.
- **Key Finding:** RLHF models are systematically overconfident. Temperature scaling and Platt scaling are strongest post-hoc methods. Fine-tuning on correctness labels significantly improves calibration.
- **Relevance:** Essential reference for calibration module.
- **Design Implication:** Must apply post-hoc calibration. Never trust raw LLM probability outputs.

### [16] Kapoor & Gruver (2024) — "LLMs Must Be Taught Uncertainty"
- **Citation:** Kapoor, S. & Gruver, N. (2024). "Large Language Models Must Be Taught to Know What They Don't Know." arXiv:2406.08391.
- **Key Finding:** Fine-tuning on ~1,000 graded examples with LoRA produces well-calibrated uncertainty estimates that generalize out-of-distribution.
- **Relevance:** Practical calibration recipe: labeled dataset → LoRA fine-tune → calibrated model.
- **Design Implication:** Collect resolved Polymarket questions as calibration training data. Fine-tune after 1,000+ samples.

### [17] Wang et al. (2024) — "Calibrating Verbalized Probabilities"
- **Citation:** Wang, C., et al. (2024). "Calibrating Verbalized Probabilities for Large Language Models." arXiv:2410.06707.
- **Key Finding:** Verbalized confidence scores can be post-hoc calibrated via Platt scaling or isotonic regression, significantly reducing ECE.
- **Relevance:** Directly applicable — our Fish output verbalized probabilities.
- **Design Implication:** Pipeline: Fish outputs probability → isotonic regression calibration → final output.

### [18] Jenane et al. (2026) — "Entropy to Calibrated Uncertainty"
- **Citation:** Jenane, A., et al. (2026). "From Entropy to Calibrated Uncertainty: Training Language Models to Reason About Uncertainty." arXiv:2603.06317.
- **Key Finding:** Three-stage pipeline (entropy scoring → Platt scaling → GRPO RL) produces intrinsically calibrated LLMs at inference time without sampling overhead.
- **Relevance:** Most advanced calibration result. Target architecture for production.
- **Design Implication:** Long-term: train an intrinsically calibrated model. Short-term: use isotonic regression.

---

## 5. Information Aggregation Theory

### [19] Wolfers & Zitzewitz (2004) — "Prediction Markets"
- **Citation:** Wolfers, J. & Zitzewitz, E. (2004). "Prediction Markets." *Journal of Economic Perspectives*, 18(2), 107-126.
- **Key Finding:** Prediction markets provide incentives for information seeking, truthful revelation, and efficient aggregation. Prices are generally superior forecasters vs. polls and expert panels.
- **Relevance:** Theoretical foundation. Polymarket prices contain genuine information signal.

### [20] Hayek (1945) — "The Use of Knowledge in Society"
- **Citation:** Hayek, F.A. (1945). "The Use of Knowledge in Society." *American Economic Review*, 35(4), 519-530.
- **Key Finding:** Prices aggregate dispersed local knowledge that no central planner can possess.
- **Relevance:** Intellectual foundation for prediction market theory.

### [21] Palan et al. (2019) — "Aggregation Mechanisms for Crowd Predictions"
- **Citation:** Palan, S., Huber, J., & Senninger, L. (2019). "Aggregation Mechanisms for Crowd Predictions." *Experimental Economics*.
- **Key Finding:** Continuous double auctions outperform arithmetic mean, geometric mean, and censored estimators for aggregating dispersed private information.
- **Relevance:** Justifies treating market prices as strong prior; use incentive-weighted aggregation.

### [22] Atanasov et al. (2020) — "Prediction Markets vs. Prediction Polls"
- **Citation:** Atanasov, P., et al. "Distilling the Wisdom of Crowds: Prediction Markets vs. Prediction Polls." *Management Science*.
- **Key Finding:** Markets and polls are closely matched; team polls outperform when information is complementary. Extremizing algorithms improve polls to near-market accuracy.
- **Relevance:** When swarm members hold complementary knowledge, team-poll aggregation with extremizing may outperform raw market prices.

---

## 6. Network / Graph Analysis

### [23] Fanshawe et al. (2026) — "Forecasting Equity Correlations with THGNN"
- **Citation:** Fanshawe, J., Masih, R., & Cameron, A. (2026). "Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network." arXiv:2601.04602.
- **Key Finding:** Temporal-Heterogeneous GNN combining Transformer with edge-aware graph attention reduces 10-day correlation forecasting error. Outperforms especially during market stress.
- **Relevance:** GNN-based correlation forecasting applicable to Polymarket contract dependencies.

### [24] Baaijens et al. (2025) — "Graph Learning on Financial Networks"
- **Citation:** Baaijens, T., Güven, C., & Nápoles, G. (2025). "Graph Learning on Financial Networks from Pairwise Similarity." *Applied Network Science*.
- **Key Finding:** Cosine-similarity-based Graph Attention Network (CS_GAT) outperforms Pearson-correlation-based GNNs for return prediction.
- **Relevance:** Use semantic similarity, not just price correlation, to build market graphs.
- **Design Implication:** Our market graph should be built from topic/semantic similarity between market descriptions.

---

## 7. Polymarket Empirical

### [25] Tsang & Yang (2026) — "Anatomy of Polymarket"
- **Citation:** Tsang, K.P. & Yang, Z. (2026). "The Anatomy of Polymarket: Evidence from the 2024 Presidential Election." arXiv:2603.03136.
- **Key Finding:** Market quality matures over contract life: early thin markets have wide arbitrage. Whale traders drove large price movements. Market is vulnerable to manipulation early, efficient late.
- **Relevance:** Most rigorous Polymarket microstructure study.
- **Design Implication:** Weight predictions by market age and liquidity. Less trust in thin/early-stage markets.

### [26] Saguillo et al. (2025) — "Empirical Arbitrage on Polymarket"
- **Citation:** Saguillo, O., et al. (2025). "Empirical Arbitrage Analysis on Polymarket." *AFT 2025*.
- **Key Finding:** $40M arbitrage extracted from 86M on-chain transactions (April 2024 - April 2025). Two types: Market Rebalancing Arbitrage (intra-market) and Combinatorial Arbitrage (inter-market). LLMs enable semantic matching.
- **Relevance:** Quantifies exploitable inefficiency. LLM semantic matching is the enabling technology.
- **Design Implication:** Implement both intra-market (sum-to-1) and inter-market (semantic) arbitrage detection.
