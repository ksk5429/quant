"""Tests for K-Fish v5 database persistence layer.

Phase 1 Gate requirement: 15+ tests covering all CRUD operations.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import pytest

from src.db.manager import DatabaseManager, PositionRecord, TrackRecord


@pytest.fixture
def db():
    """Create a temporary database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        with DatabaseManager(db_path) as manager:
            yield manager
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@dataclass
class MockResult:
    """Mock EngineV4Result for testing."""
    market_id: str = "test_123"
    question: str = "Will it rain?"
    category: str = "weather"
    raw_probability: float = 0.65
    extremized_probability: float = 0.72
    calibrated_probability: float = 0.68
    n_fish: int = 9
    n_rounds: int = 2
    spread: float = 0.15
    std_dev: float = 0.08
    effective_confidence: float = 0.7
    disagreement_flag: bool = False
    personas_used: list = None
    model: str = "haiku"
    total_elapsed_s: float = 135.0
    research_elapsed_s: float = 20.0
    fish_predictions: list = None

    def __post_init__(self):
        if self.personas_used is None:
            self.personas_used = ["calibrator", "contrarian"]
        if self.fish_predictions is None:
            self.fish_predictions = []


@dataclass
class MockPosition:
    """Mock Position for testing."""
    market_id: str = "test_123"
    question: str = "Will it rain?"
    side: str = "YES"
    edge: float = 0.08
    position_size_usd: float = 25.0


# ── Schema Tests ──

class TestSchemaCreation:
    def test_tables_exist(self, db):
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "predictions" in table_names
        assert "positions" in table_names
        assert "calibration_data" in table_names
        assert "resolutions" in table_names
        assert "system_state" in table_names

    def test_schema_idempotent(self, db):
        """Creating schema twice doesn't crash."""
        db._init_schema()
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len(tables) >= 5


# ── Prediction Tests ──

class TestPredictions:
    def test_log_prediction(self, db):
        result = MockResult()
        pred_id = db.log_prediction(result, market_price=0.50)
        assert pred_id > 0

    def test_log_prediction_stores_values(self, db):
        result = MockResult(raw_probability=0.65, calibrated_probability=0.68)
        pred_id = db.log_prediction(result, market_price=0.50)

        row = db.conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        assert row["raw_probability"] == 0.65
        assert row["calibrated_probability"] == 0.68
        assert row["market_price"] == 0.50
        assert row["n_fish"] == 9
        assert row["model"] == "haiku"

    def test_log_prediction_stores_fish_json(self, db):
        result = MockResult()
        pred_id = db.log_prediction(result)

        row = db.conn.execute(
            "SELECT fish_predictions FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        fish = json.loads(row["fish_predictions"])
        assert isinstance(fish, list)

    def test_log_multiple_predictions_same_market(self, db):
        r1 = MockResult(market_id="m1")
        r2 = MockResult(market_id="m1")
        id1 = db.log_prediction(r1, market_price=0.50)
        # Second prediction for same market should get new timestamp
        import time; time.sleep(0.01)
        id2 = db.log_prediction(r2, market_price=0.55)
        assert id2 > id1


# ── Position Tests ──

class TestPositions:
    def test_open_position(self, db):
        result = MockResult()
        pred_id = db.log_prediction(result)
        pos = MockPosition()
        pos_id = db.open_position(pos, pred_id, order_id="ord_1", tx_hash="0xabc")
        assert pos_id > 0

    def test_open_position_stored(self, db):
        result = MockResult()
        pred_id = db.log_prediction(result)
        pos = MockPosition(side="NO", position_size_usd=30.0)
        pos_id = db.open_position(pos, pred_id)

        positions = db.get_open_positions()
        assert len(positions) == 1
        assert positions[0].side == "NO"
        assert positions[0].size_usd == 30.0
        assert positions[0].status == "open"

    def test_close_position(self, db):
        result = MockResult()
        pred_id = db.log_prediction(result)
        pos = MockPosition(position_size_usd=25.0)
        pos_id = db.open_position(pos, pred_id)

        db.close_position(pos_id, exit_price=1.0, pnl=12.50, reason="resolved")

        open_pos = db.get_open_positions()
        assert len(open_pos) == 0

        closed = db.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].pnl_usd == 12.50
        assert closed[0].exit_reason == "resolved"
        assert closed[0].status == "closed"

    def test_multiple_open_positions(self, db):
        for i in range(3):
            result = MockResult(market_id=f"market_{i}")
            pred_id = db.log_prediction(result)
            pos = MockPosition(market_id=f"market_{i}")
            db.open_position(pos, pred_id)

        positions = db.get_open_positions()
        assert len(positions) == 3


# ── Calibration Tests ──

class TestCalibration:
    def test_log_and_retrieve_calibration(self, db):
        for p, o in [(0.7, 1.0), (0.3, 0.0), (0.5, 1.0)]:
            db.log_calibration_point(p, o)

        preds, outs = db.get_calibration_data()
        assert len(preds) == 3
        assert len(outs) == 3

    def test_calibration_data_order(self, db):
        """Most recent first."""
        db.log_calibration_point(0.1, 0.0, "first")
        db.log_calibration_point(0.9, 1.0, "second")

        preds, _ = db.get_calibration_data()
        assert preds[0] == 0.9  # most recent first

    def test_calibration_limit(self, db):
        for i in range(10):
            db.log_calibration_point(i / 10, float(i % 2))

        preds, _ = db.get_calibration_data(limit=5)
        assert len(preds) == 5


# ── Resolution Tests ──

class TestResolutions:
    def test_log_resolution(self, db):
        result = MockResult(market_id="res_1", calibrated_probability=0.75)
        db.log_prediction(result)

        db.log_resolution("res_1", outcome=1.0, question="Test?")

        row = db.conn.execute(
            "SELECT * FROM resolutions WHERE market_id = 'res_1'"
        ).fetchone()
        assert row["outcome"] == 1.0
        assert row["our_prediction"] == 0.75
        assert abs(row["brier_score"] - (0.75 - 1.0) ** 2) < 0.001

    def test_resolution_feeds_calibrator(self, db):
        result = MockResult(market_id="res_2", calibrated_probability=0.80)
        db.log_prediction(result)

        db.log_resolution("res_2", outcome=1.0)

        preds, outs = db.get_calibration_data()
        assert len(preds) >= 1
        assert 0.80 in preds


# ── System State Tests ──

class TestSystemState:
    def test_set_and_get(self, db):
        db.set_system_state("bankroll", "1000")
        assert db.get_system_state("bankroll") == "1000"

    def test_update_existing(self, db):
        db.set_system_state("bankroll", "1000")
        db.set_system_state("bankroll", "950")
        assert db.get_system_state("bankroll") == "950"

    def test_get_nonexistent(self, db):
        assert db.get_system_state("nonexistent") is None


# ── Track Record Tests ──

class TestTrackRecord:
    def test_empty_track_record(self, db):
        record = db.get_track_record()
        assert record.n_predictions == 0
        assert record.n_positions == 0
        assert record.win_rate == 0

    def test_track_record_with_data(self, db):
        db.set_system_state("bankroll", "1000")

        # Add predictions and positions
        for i in range(5):
            result = MockResult(market_id=f"tr_{i}")
            pred_id = db.log_prediction(result)
            pos = MockPosition(market_id=f"tr_{i}")
            pos_id = db.open_position(pos, pred_id)
            pnl = 10.0 if i % 2 == 0 else -5.0
            db.close_position(pos_id, exit_price=0.8, pnl=pnl, reason="resolved")

        record = db.get_track_record()
        assert record.n_predictions == 5
        assert record.n_positions == 5
        assert record.n_closed == 5
        assert record.n_wins == 3
        assert record.n_losses == 2
        assert record.total_pnl_usd == 20.0  # 3*10 - 2*5 = 20


# ── Seed from Retrodiction Tests ──

class TestSeedFromRetrodiction:
    def test_seed_no_dir(self, db):
        count = db.seed_from_retrodiction("nonexistent_dir")
        assert count == 0
