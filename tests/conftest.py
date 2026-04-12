"""Shared test fixtures for K-Fish test suite."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import pytest

from src.db.manager import DatabaseManager


@pytest.fixture
def tmp_db():
    """Temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        with DatabaseManager(path) as db:
            yield db
    finally:
        if os.path.exists(path):
            os.remove(path)


@dataclass
class MockEngineResult:
    """Mock EngineV4Result for testing."""
    market_id: str = "test_mkt"
    question: str = "Will X happen?"
    category: str = "general"
    raw_probability: float = 0.65
    extremized_probability: float = 0.72
    calibrated_probability: float = 0.68
    n_fish: int = 9
    n_rounds: int = 2
    spread: float = 0.15
    std_dev: float = 0.08
    effective_confidence: float = 0.7
    disagreement_flag: bool = False
    swarm_healthy: bool = True
    personas_used: list = field(default_factory=lambda: ["calibrator", "contrarian"])
    model: str = "haiku"
    total_elapsed_s: float = 135.0
    research_elapsed_s: float = 20.0
    fish_predictions: list = field(default_factory=list)


@dataclass
class MockPosition:
    """Mock Position for testing."""
    market_id: str = "test_mkt"
    question: str = "Will X happen?"
    side: str = "YES"
    edge: float = 0.08
    position_size_usd: float = 25.0
    expected_value: float = 0.12
    kelly_fraction: float = 0.02
    confidence: float = 0.7
    reason: str = "test"
