<h1 align="center">K-Fish</h1>

<p align="center">
<strong>Swarm-intelligence prediction-market forecasting + calibration stack.</strong><br>
<em>This repository is an overview. The live work moved to three dedicated repos.</em>
</p>

<p align="center">
  <a href="https://github.com/ksk5429/polymarket-oracle-risk"><img src="https://img.shields.io/badge/PyPI-polymarket--oracle--risk-0066cc?style=for-the-badge" alt="PyPI"></a>
  <a href="https://ksk5429.github.io/quant-notes/"><img src="https://img.shields.io/badge/Docs-quant--notes-4338ca?style=for-the-badge" alt="Docs"></a>
  <a href="#what-happened-to-this-repo"><img src="https://img.shields.io/badge/Status-index_only-6b7280?style=for-the-badge" alt="Status"></a>
</p>

---

## Where everything lives now

| Repo | Visibility | Purpose |
|---|---|---|
| **[ksk5429/kfish](https://github.com/ksk5429/kfish)** | 🔒 private | Production uv-workspace monorepo: 9-agent swarm, calibration spine, Korean Telegram bot |
| **[ksk5429/polymarket-oracle-risk](https://github.com/ksk5429/polymarket-oracle-risk)** | 🌍 public, MIT | Bayesian risk analyzer for Polymarket (UMA OO) resolutions — shippable to PyPI |
| **[ksk5429/quant-notes](https://github.com/ksk5429/quant-notes)** | 🌍 public | Development notes, architecture decisions, review log — MkDocs Material site |
| **[ksk5429.github.io/quant-notes](https://ksk5429.github.io/quant-notes/)** | 🌍 live site | Published docs site (render target of the above) |

## What is K-Fish?

A swarm-intelligence forecasting stack built in the spirit of Tetlock's
Good Judgment Project but run by 9 orthogonal LLM personas (contrarian,
inside-view, outside-view, pre-mortem, devil's-advocate, quant,
geopolitical, macro, red-team). Multi-round Delphi aggregation, full
post-hoc calibration (isotonic + Venn-Abers + Mondrian conformal +
empirical-Bayes shrinkage ensemble), and a DuckDB ASOF-join warehouse
that ties every forecast to the market price at the moment it was made.

Baseline: Brier **0.2026** on a 200-market Polymarket retrodiction.

Full story + architecture at **[ksk5429.github.io/quant-notes](https://ksk5429.github.io/quant-notes/)**.

## Three headlines

- **Public analyzer is publishable.** `polymarket-oracle-risk` on PyPI
  scores UMA-resolved Polymarket markets on manipulation risk. NumPyro
  Bayesian logistic, Zelenskyy-suit regression test, refusal gate.
- **Calibration is exact to the legacy baseline.** The new `kfish-core`
  reproduces the legacy engine's Brier to 4 decimal places on three
  archived retrodiction runs — first green on the ADR-0002 seven-night
  cutover gate.
- **Bot is paper-only, pending legal review.** Korean-first Telegram bot
  on Hyperliquid with per-user Fernet-encrypted agent wallets. Live
  mainnet blocked until Korean counsel reviews VASP + gambling-law
  classification (see [quant-notes/reviews/open-questions](https://ksk5429.github.io/quant-notes/reviews/open-questions/)).

## Canonical documents

Two planning documents drive the whole stack. Both live in the docs site:

- [90-Day Build Guide](https://ksk5429.github.io/quant-notes/blueprints/90-day-build-guide/)
- [Three-Repo GitHub Blueprint](https://ksk5429.github.io/quant-notes/blueprints/three-repo-blueprint/)

## How to engage

- **Read the docs site** (no login): <https://ksk5429.github.io/quant-notes/>
- **Suggest a correction or ask a question**: open an issue or
  [Discussion](https://github.com/ksk5429/quant-notes/discussions) on
  the quant-notes repo.
- **Install the public analyzer**: `pip install polymarket-oracle-risk`
  (once the first release is cut).
- **Trading alpha / proprietary logic**: lives in the private `kfish`
  repo. Not publicly available.

## What happened to this repo

This repository was the original scratch-space before the work split into
three focused repos. It's kept as a **public index** (this README) so
inbound links keep working. The code here is historical; current
development is on the three repos above.

If you're looking at this repo via an inbound link from a paper,
a Google result, or a shared message, you probably want one of:

- **The paper/research story** → <https://ksk5429.github.io/quant-notes/architecture/overview/>
- **The PyPI package** → <https://github.com/ksk5429/polymarket-oracle-risk>
- **The development log** → <https://ksk5429.github.io/quant-notes/progress/build-log/>

## Contact

- GitHub: [@ksk5429](https://github.com/ksk5429)
- Email: ksk54299@gmail.com
- LinkedIn: [linkedin.com/in/ksk5429](https://www.linkedin.com/in/ksk5429)
- Questions & reviews: [quant-notes discussions](https://github.com/ksk5429/quant-notes/discussions)

## License

This repository's historical code stays under its original MIT license.
Each of the three new repos carries its own license — see each repo.

---

<sub>Last updated: 2026-04-19. If this README is stale, the truth lives at
<https://ksk5429.github.io/quant-notes/progress/build-log/>.</sub>
