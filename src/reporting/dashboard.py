"""Track record dashboard — generates comprehensive performance reports.

Reads all data from the SQLite database and produces a Markdown report
with: summary, calibration, per-category, per-Fish, position log, risk.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.db.manager import DatabaseManager
from src.prediction.calibration import compute_brier, compute_ece


class TrackRecordDashboard:
    """Generates comprehensive track record reports from DB."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def generate_report(self, output_path: str = "reports/track_record.md") -> Path:
        """Generate full Markdown track record report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            f"# K-Fish Track Record\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            self._section_summary(),
            self._section_calibration(),
            self._section_categories(),
            self._section_fish(),
            self._section_positions(),
            self._section_risk(),
        ]

        content = "\n---\n\n".join(s for s in sections if s)
        out.write_text(content, encoding="utf-8")
        logger.info(f"Track record report: {out}")
        return out

    def _section_summary(self) -> str:
        record = self.db.get_track_record()
        bankroll = float(self.db.get_system_state("bankroll") or "1000")

        lines = [
            "## Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total predictions | {record.n_predictions} |",
            f"| Total positions | {record.n_positions} |",
            f"| Closed positions | {record.n_closed} |",
            f"| Win rate | {record.win_rate:.1%} |",
            f"| Total P&L | ${record.total_pnl_usd:+.2f} |",
            f"| Avg P&L/trade | ${record.avg_pnl_usd:+.2f} |",
            f"| Best trade | ${record.best_trade_usd:+.2f} |",
            f"| Worst trade | ${record.worst_trade_usd:+.2f} |",
            f"| Avg Brier | {record.avg_brier:.4f} |",
            f"| Resolutions | {record.n_resolutions} |",
            f"| Bankroll | ${bankroll:,.2f} |",
        ]
        return "\n".join(lines)

    def _section_calibration(self) -> str:
        preds, outs = self.db.get_calibration_data(limit=5000)
        if len(preds) < 5:
            return "## Calibration\n\nInsufficient data (need 5+ resolved predictions)."

        p = np.array(preds)
        o = np.array(outs)
        brier = compute_brier(preds, outs)
        ece = compute_ece(preds, outs)

        lines = [
            "## Calibration\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Brier Score | {brier:.4f} |",
            f"| ECE | {ece:.4f} |",
            f"| Sample Count | {len(preds)} |",
            f"\n### Reliability Diagram\n",
            f"| Bin | N | Predicted | Actual | Error |",
            f"|-----|---|-----------|--------|-------|",
        ]

        edges = np.linspace(0, 1, 11)
        for i in range(10):
            mask = (p >= edges[i]) & (p < edges[i + 1])
            count = int(np.sum(mask))
            if count > 0:
                pred_avg = float(np.mean(p[mask]))
                act_avg = float(np.mean(o[mask]))
                err = abs(pred_avg - act_avg)
                lines.append(
                    f"| {edges[i]:.1f}-{edges[i+1]:.1f} | {count} | "
                    f"{pred_avg:.3f} | {act_avg:.3f} | {err:.3f} |"
                )

        return "\n".join(lines)

    def _section_categories(self) -> str:
        rows = self.db.conn.execute("""
            SELECT p.category,
                   COUNT(*) as n,
                   AVG((p.calibrated_probability - r.outcome) * (p.calibrated_probability - r.outcome)) as brier,
                   AVG(CASE WHEN (p.calibrated_probability >= 0.5 AND r.outcome = 1)
                            OR (p.calibrated_probability < 0.5 AND r.outcome = 0) THEN 1.0 ELSE 0.0 END) as accuracy
            FROM predictions p
            JOIN resolutions r ON p.market_id = r.market_id
            GROUP BY p.category
            ORDER BY brier ASC
        """).fetchall()

        if not rows:
            return "## Per-Category\n\nNo resolved predictions yet."

        lines = [
            "## Per-Category Performance\n",
            f"| Category | N | Brier | Accuracy |",
            f"|----------|---|-------|----------|",
        ]
        for r in rows:
            cat = r["category"] or "general"
            lines.append(f"| {cat} | {r['n']} | {r['brier']:.4f} | {r['accuracy']:.1%} |")

        return "\n".join(lines)

    def _section_fish(self) -> str:
        # Get per-fish predictions from JSON stored in predictions table
        rows = self.db.conn.execute("""
            SELECT p.fish_predictions, r.outcome
            FROM predictions p
            JOIN resolutions r ON p.market_id = r.market_id
            WHERE p.fish_predictions IS NOT NULL
        """).fetchall()

        if not rows:
            return "## Per-Fish\n\nNo resolved predictions with Fish data."

        import json
        fish_data: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            try:
                fish_list = json.loads(r["fish_predictions"])
                outcome = r["outcome"]
                for fp in fish_list:
                    persona = fp.get("persona", "unknown")
                    prob = fp.get("probability", 0.5)
                    fish_data.setdefault(persona, []).append((prob, outcome))
            except (json.JSONDecodeError, TypeError):
                continue

        lines = [
            "## Per-Fish Performance\n",
            f"| Rank | Persona | Brier | N |",
            f"|------|---------|-------|---|",
        ]
        fish_brier = {}
        for persona, data in fish_data.items():
            preds = [d[0] for d in data]
            outs = [d[1] for d in data]
            fish_brier[persona] = (compute_brier(preds, outs), len(data))

        for rank, (persona, (brier, n)) in enumerate(
            sorted(fish_brier.items(), key=lambda x: x[1][0]), 1
        ):
            lines.append(f"| {rank} | {persona} | {brier:.4f} | {n} |")

        return "\n".join(lines)

    def _section_positions(self) -> str:
        closed = self.db.get_closed_positions(limit=20)
        if not closed:
            return "## Recent Positions\n\nNo closed positions yet."

        lines = [
            "## Recent Positions (last 20)\n",
            f"| Side | P&L | Reason | Question |",
            f"|------|-----|--------|----------|",
        ]
        for p in closed:
            sym = "+" if (p.pnl_usd or 0) > 0 else ""
            lines.append(
                f"| {p.side} | ${sym}{p.pnl_usd or 0:.2f} | "
                f"{p.exit_reason or 'open'} | {p.question[:40]} |"
            )
        return "\n".join(lines)

    def _section_risk(self) -> str:
        record = self.db.get_track_record()
        open_pos = self.db.get_open_positions()

        lines = [
            "## Risk Metrics\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Open positions | {len(open_pos)} |",
            f"| Max drawdown | {record.max_drawdown_pct:.1%} |",
            f"| Consecutive losses | — |",
        ]

        # Open position exposure by category
        if open_pos:
            lines.append(f"\n### Open Exposure\n")
            for p in open_pos:
                lines.append(f"- {p.side} ${p.size_usd:.2f} — {p.question[:50]}")

        return "\n".join(lines)
