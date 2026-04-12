# Developer's Note

## Session 2026-04-12: Project Genesis

### What Was Built
K-Fish prediction engine — complete foundation from zero to 45 passing tests.

### Architecture Decisions

1. **Why 7 Fish, not 50?** PolySwarm uses 50 personas, but API costs scale linearly. 7 diverse personas capture ~90% of the diversity benefit at 14% of the cost. Scale up when profitable.

2. **Why withheld market prices during Fish analysis?** If Fish see the current market price, they anchor to it. Independence is the whole point of ensemble prediction. Prices are combined ONLY at the aggregation stage.

3. **Why Bayesian weighted, not simple average?** A Fish that is both confident AND historically accurate should have more influence. Simple averaging treats the contrarian equal to the calibrator, even when one has a 0.15 Brier and the other has 0.35.

4. **Why Quarter-Kelly?** Full Kelly maximizes long-run growth rate but has wild variance. Quarter-Kelly gives ~75% of the growth rate with ~25% of the variance. Thorp and every serious Kelly practitioner recommends fractional Kelly.

5. **Why isotonic regression for calibration?** It's non-parametric (no assumptions about the calibration curve shape), works well with 100+ samples, and is available in scikit-learn. Platt scaling assumes a sigmoid shape which may not hold for LLM outputs.

6. **Why file-based communication for the swarm?** Each Fish runs in its own VS Code + Claude Code instance. Files in `shared_state/` are the simplest, most debuggable communication protocol. No sockets, no databases, no complexity. Read JSON, write JSON.

### Key References Used
- PolySwarm (arXiv:2604.03888) — primary architecture reference
- Schoenegger et al. (Science Advances, 2024) — ensemble justification
- Saguillo et al. (AFT 2025) — $40M arbitrage proves opportunity exists
- Geng et al. (NAACL 2024) — calibration is mandatory

### What's Next
See the human worksheet at `docs/worksheets/K-Fish_human_worksheet.docx`.
The human needs to fill out Sections 1-8 before the AI can optimize the system.
