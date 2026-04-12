"""DatabaseManager — SQLite persistence for K-Fish v5 production.

All trading state persists across restarts:
- Predictions (every engine run)
- Positions (open/closed with P&L)
- Calibration data (rolling window)
- Resolutions (market outcomes)
- System state (bankroll, drawdown peak, etc.)

Usage:
    with DatabaseManager() as db:
        pred_id = db.log_prediction(result, market_price=0.45)
        db.open_position(position, pred_id, order_id="0x...", tx_hash="0x...")
        db.close_position(pos_id, exit_price=1.0, pnl=12.50, reason="resolved")
        preds, outs = db.get_calibration_data(limit=5000)
        record = db.get_track_record()
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class PositionRecord:
    """A position loaded from the database."""
    id: int
    market_id: str
    question: str
    side: str
    entry_price: float
    size_usd: float
    shares: float
    entry_timestamp: str
    prediction_id: int | None
    status: str
    order_id: str | None = None
    tx_hash: str | None = None
    token_id: str | None = None
    exit_price: float | None = None
    exit_timestamp: str | None = None
    pnl_usd: float | None = None
    pnl_pct: float | None = None
    exit_reason: str | None = None


@dataclass
class TrackRecord:
    """Aggregated performance statistics."""
    n_predictions: int
    n_positions: int
    n_closed: int
    n_wins: int
    n_losses: int
    win_rate: float
    total_pnl_usd: float
    avg_pnl_usd: float
    best_trade_usd: float
    worst_trade_usd: float
    avg_brier: float
    n_resolutions: int
    bankroll: float
    max_drawdown_pct: float


class DatabaseManager:
    """SQLite-based persistence for K-Fish production trading."""

    def __init__(self, db_path: str = "data/kfish.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "DatabaseManager":
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        logger.info(f"DB connected: {self.db_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DatabaseManager not entered as context manager")
        return self._conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            sql = schema_path.read_text(encoding="utf-8")
            self.conn.executescript(sql)
        else:
            logger.warning(f"Schema file not found: {schema_path}")

    # ── Predictions ──────────────────────────────────────────────

    def log_prediction(self, result: Any, market_price: float | None = None) -> int:
        """Log an engine prediction result. Returns the prediction ID."""
        fish_json = json.dumps([
            {
                "persona": fp.persona,
                "probability": fp.probability,
                "confidence": fp.confidence,
                "reasoning": fp.reasoning[:200] if fp.reasoning else "",
            }
            for fp in getattr(result, "fish_predictions", [])
        ])

        personas_json = json.dumps(
            getattr(result, "personas_used", [])
        )

        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO predictions (
                market_id, question, category, timestamp,
                raw_probability, extremized_probability, calibrated_probability,
                market_price,
                n_fish, n_rounds, spread, std_dev, effective_confidence,
                disagreement_flag, personas_used, model,
                total_elapsed_s, research_elapsed_s,
                fish_predictions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            getattr(result, "market_id", ""),
            getattr(result, "question", ""),
            getattr(result, "category", ""),
            datetime.now().isoformat(),
            getattr(result, "raw_probability", 0.5),
            getattr(result, "extremized_probability", 0.5),
            getattr(result, "calibrated_probability", 0.5),
            market_price,
            getattr(result, "n_fish", 0),
            getattr(result, "n_rounds", 0),
            getattr(result, "spread", 0.0),
            getattr(result, "std_dev", 0.0),
            getattr(result, "effective_confidence", 0.0),
            getattr(result, "disagreement_flag", False),
            personas_json,
            getattr(result, "model", ""),
            getattr(result, "total_elapsed_s", 0.0),
            getattr(result, "research_elapsed_s", 0.0),
            fish_json,
        ))
        self.conn.commit()
        return cursor.lastrowid

    # ── Positions ────────────────────────────────────────────────

    def open_position(
        self,
        position: Any,
        prediction_id: int,
        entry_price: float = 0.0,
        order_id: str = "",
        tx_hash: str = "",
        token_id: str = "",
    ) -> int:
        """Record a new open position. Returns the position ID.

        Args:
            position: Position object from engine.
            prediction_id: FK to predictions table.
            entry_price: actual market price at entry (NOT edge).
            order_id: exchange order ID.
            tx_hash: on-chain transaction hash.
            token_id: CTF token ID.
        """
        size_usd = getattr(position, "position_size_usd", 0.0)
        shares = size_usd / entry_price if entry_price > 0.01 else 0

        cursor = self.conn.execute("""
            INSERT INTO positions (
                market_id, question, side, entry_price, size_usd,
                shares, entry_timestamp, prediction_id,
                order_id, tx_hash, token_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (
            getattr(position, "market_id", ""),
            getattr(position, "question", ""),
            getattr(position, "side", "YES"),
            entry_price,
            size_usd,
            shares,
            datetime.now().isoformat(),
            prediction_id,
            order_id,
            tx_hash,
            token_id,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        pnl: float,
        reason: str,
    ) -> None:
        """Close an open position with P&L."""
        row = self.conn.execute(
            "SELECT entry_price, size_usd FROM positions WHERE id = ?",
            (position_id,)
        ).fetchone()

        pnl_pct = (pnl / row["size_usd"] * 100) if row and row["size_usd"] > 0 else 0

        self.conn.execute("""
            UPDATE positions SET
                exit_price = ?,
                exit_timestamp = ?,
                pnl_usd = ?,
                pnl_pct = ?,
                exit_reason = ?,
                status = 'closed'
            WHERE id = ?
        """, (exit_price, datetime.now().isoformat(), pnl, pnl_pct, reason, position_id))
        self.conn.commit()

    def get_open_positions(self) -> list[PositionRecord]:
        """Get all currently open positions."""
        rows = self.conn.execute(
            "SELECT * FROM positions WHERE status IN ('open', 'pending', 'filled')"
        ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_closed_positions(self, limit: int = 100) -> list[PositionRecord]:
        """Get recently closed positions."""
        rows = self.conn.execute(
            "SELECT * FROM positions WHERE status = 'closed' ORDER BY exit_timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_position(r) for r in rows]

    # ── Calibration ──────────────────────────────────────────────

    def log_calibration_point(self, prediction: float, outcome: float, market_id: str = "") -> None:
        """Add a single calibration data point."""
        self.conn.execute(
            "INSERT INTO calibration_data (prediction, outcome, market_id) VALUES (?, ?, ?)",
            (prediction, outcome, market_id),
        )
        self.conn.commit()

    def get_calibration_data(self, limit: int = 5000) -> tuple[list[float], list[float]]:
        """Get calibration training data (most recent first)."""
        rows = self.conn.execute(
            "SELECT prediction, outcome FROM calibration_data ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        if not rows:
            return [], []
        preds = [r["prediction"] for r in rows]
        outs = [r["outcome"] for r in rows]
        return preds, outs

    # ── Resolutions ──────────────────────────────────────────────

    def log_resolution(self, market_id: str, outcome: float, question: str = "") -> None:
        """Record a market resolution outcome."""
        # Find our prediction for this market
        pred_row = self.conn.execute(
            "SELECT calibrated_probability FROM predictions WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
            (market_id,)
        ).fetchone()

        our_pred = pred_row["calibrated_probability"] if pred_row else None
        brier = (our_pred - outcome) ** 2 if our_pred is not None else None

        self.conn.execute("""
            INSERT OR REPLACE INTO resolutions (market_id, question, outcome, resolved_at, our_prediction, brier_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (market_id, question, outcome, datetime.now().isoformat(), our_pred, brier))

        # Also add to calibration data
        if our_pred is not None:
            self.log_calibration_point(our_pred, outcome, market_id)

        self.conn.commit()

    # ── Track Record ─────────────────────────────────────────────

    def get_track_record(self, n: int = 100) -> TrackRecord:
        """Get aggregated performance statistics."""
        # Predictions count
        n_pred = self.conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]

        # Position stats
        pos_stats = self.conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed,
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl_usd), 0) as total_pnl,
                COALESCE(AVG(CASE WHEN status = 'closed' THEN pnl_usd END), 0) as avg_pnl,
                COALESCE(MAX(pnl_usd), 0) as best,
                COALESCE(MIN(pnl_usd), 0) as worst
            FROM positions
        """).fetchone()

        # Resolution stats
        res_stats = self.conn.execute("""
            SELECT COUNT(*) as n, COALESCE(AVG(brier_score), 0) as avg_brier
            FROM resolutions WHERE brier_score IS NOT NULL
        """).fetchone()

        # System state
        bankroll = float(self.get_system_state("bankroll") or "1000")
        max_dd = float(self.get_system_state("max_drawdown_pct") or "0")

        n_closed = pos_stats["closed"] or 0
        n_wins = pos_stats["wins"] or 0
        n_losses = pos_stats["losses"] or 0

        return TrackRecord(
            n_predictions=n_pred,
            n_positions=pos_stats["total"],
            n_closed=n_closed,
            n_wins=n_wins,
            n_losses=n_losses,
            win_rate=n_wins / max(n_closed, 1),
            total_pnl_usd=pos_stats["total_pnl"],
            avg_pnl_usd=pos_stats["avg_pnl"],
            best_trade_usd=pos_stats["best"],
            worst_trade_usd=pos_stats["worst"],
            avg_brier=res_stats["avg_brier"],
            n_resolutions=res_stats["n"],
            bankroll=bankroll,
            max_drawdown_pct=max_dd,
        )

    # ── System State ─────────────────────────────────────────────

    def get_system_state(self, key: str) -> str | None:
        """Get a system state value."""
        row = self.conn.execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_system_state(self, key: str, value: str) -> None:
        """Set a system state value."""
        self.conn.execute("""
            INSERT OR REPLACE INTO system_state (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value, datetime.now().isoformat()))
        self.conn.commit()

    # ── Seed from retrodiction JSON ──────────────────────────────

    def seed_from_retrodiction(self, retro_dir: str = "data/retrodiction") -> int:
        """Import retrodiction results into calibration_data table.

        Returns number of data points imported.
        """
        retro_path = Path(retro_dir)
        if not retro_path.exists():
            return 0

        count = 0
        for fp in sorted(retro_path.glob("retro_v2_*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                for pred in data.get("predictions", []):
                    p = pred.get("extremized_probability", pred.get("raw_probability"))
                    gt = pred.get("ground_truth")
                    mid = pred.get("market_id", "")
                    if p is not None and gt is not None:
                        # Check if already imported
                        existing = self.conn.execute(
                            "SELECT id FROM calibration_data WHERE market_id = ? AND prediction = ?",
                            (mid, p)
                        ).fetchone()
                        if not existing:
                            self.conn.execute(
                                "INSERT INTO calibration_data (prediction, outcome, market_id) VALUES (?, ?, ?)",
                                (p, gt, mid),
                            )
                            count += 1
            except Exception as e:
                logger.warning(f"Failed to load {fp.name}: {e}")

        self.conn.commit()
        logger.info(f"Seeded {count} calibration points from retrodiction")
        return count

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _row_to_position(row: sqlite3.Row) -> PositionRecord:
        return PositionRecord(
            id=row["id"],
            market_id=row["market_id"],
            question=row["question"],
            side=row["side"],
            entry_price=row["entry_price"],
            size_usd=row["size_usd"],
            shares=row["shares"],
            entry_timestamp=row["entry_timestamp"],
            prediction_id=row["prediction_id"],
            status=row["status"],
            order_id=row["order_id"],
            tx_hash=row["tx_hash"],
            token_id=row["token_id"],
            exit_price=row["exit_price"],
            exit_timestamp=row["exit_timestamp"],
            pnl_usd=row["pnl_usd"],
            pnl_pct=row["pnl_pct"],
            exit_reason=row["exit_reason"],
        )
