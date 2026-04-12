"""Alert manager — file-based alerting for trading events.

Logs all significant trading events to a structured alert file.
Extensible to Discord/Telegram/email in the future.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger


class AlertManager:
    """File-based alert system for trading events."""

    def __init__(
        self,
        alert_file: str = "data/alerts.jsonl",
        method: str = "file",
    ) -> None:
        self.alert_file = Path(alert_file)
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)
        self.method = method

    def _emit(self, level: str, event: str, details: dict) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            **details,
        }

        # File logging (always)
        with open(self.alert_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Console logging
        if level == "CRITICAL":
            logger.error(f"ALERT [{event}]: {details}")
        elif level == "WARNING":
            logger.warning(f"ALERT [{event}]: {details}")
        else:
            logger.info(f"ALERT [{event}]: {details}")

    def position_opened(
        self, market_id: str, question: str, side: str, size: float, edge: float,
    ) -> None:
        self._emit("INFO", "POSITION_OPENED", {
            "market_id": market_id,
            "question": question[:80],
            "side": side,
            "size_usd": size,
            "edge": round(edge, 4),
        })

    def position_closed(
        self, market_id: str, question: str, pnl: float, reason: str,
    ) -> None:
        level = "INFO" if pnl >= 0 else "WARNING"
        self._emit(level, "POSITION_CLOSED", {
            "market_id": market_id,
            "question": question[:80],
            "pnl_usd": round(pnl, 2),
            "reason": reason,
        })

    def drawdown_warning(self, current_pct: float, threshold_pct: float) -> None:
        self._emit("WARNING", "DRAWDOWN_WARNING", {
            "current_pct": round(current_pct, 4),
            "threshold_pct": threshold_pct,
        })

    def drawdown_halt(self, current_pct: float) -> None:
        self._emit("CRITICAL", "DRAWDOWN_HALT", {
            "current_pct": round(current_pct, 4),
            "message": "Trading halted. Manual --reset-drawdown required.",
        })

    def engine_error(self, error: str, market_id: str = "") -> None:
        self._emit("CRITICAL", "ENGINE_ERROR", {
            "error": error[:500],
            "market_id": market_id,
        })

    def calibrator_retrained(self, n_samples: int, method: str) -> None:
        self._emit("INFO", "CALIBRATOR_RETRAINED", {
            "n_samples": n_samples,
            "method": method,
        })

    def reconciliation_mismatch(self, details: str) -> None:
        self._emit("CRITICAL", "RECONCILIATION_MISMATCH", {
            "details": details[:500],
            "action": "Trading halted until resolved.",
        })

    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        """Read the last N alerts from the file."""
        if not self.alert_file.exists():
            return []
        lines = self.alert_file.read_text(encoding="utf-8").strip().split("\n")
        alerts = []
        for line in lines[-n:]:
            try:
                alerts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return alerts
