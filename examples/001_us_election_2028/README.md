# Exercise 001: 2028 US Election Markets

**Date:** 2026-04-12
**Topic:** 2028 US Presidential Election — Democratic nomination + general election
**Markets:** 15 (all Polymarket)
**Fish:** 3 (Geopolitical Analyst, Financial Quant, Contrarian Thinker)
**Result:** 0 actionable signals (all PASS)

## Files

| File | Description |
|------|-------------|
| [exercise_001_report.docx](exercise_001_report.docx) | Full analysis report with findings, Fish performance, critique |
| [consensus.json](consensus.json) | Raw aggregation output (machine-readable) |
| [generate_report.py](generate_report.py) | Script that generated the DOCX report |

## Summary

All 15 markets were ultra-low-probability tail candidates (0.5-0.9%). The swarm correctly identified that edges were too small for profitable trading. Key findings:
1. MrBeast is constitutionally ineligible (age 30 in 2028, must be 35)
2. Phil Murphy and Tim Walz are underpriced relative to celebrities
3. Cross-market anomaly: Walz general vs. nomination pricing inconsistency
4. Market prices all longshots in a lazy 0.55-0.95% band regardless of viability

## Lessons Learned

- Need higher-probability markets (20-80%) for meaningful edge detection
- Need news/RAG context for Fish agents
- Debate rounds between Fish would sharpen estimates
- Price-blinding was not consistently enforced
