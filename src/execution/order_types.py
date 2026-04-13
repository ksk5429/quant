"""Order result and position data types for the execution layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    order_id: str
    status: str  # 'filled', 'partial', 'pending', 'rejected', 'simulated'
    filled_price: float
    filled_size: float
    side: str
    token_id: str
    timestamp: str
    tx_hash: str | None = None
    paper: bool = True
    rejection_reason: str | None = None

    @property
    def is_filled(self) -> bool:
        return self.status in ("filled", "simulated")


@dataclass
class ClosedPosition:
    """A position that has been resolved or closed."""
    position_id: int
    market_id: str
    question: str
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    market_outcome: float  # actual resolution: 1.0 (YES) or 0.0 (NO)
    reason: str  # 'resolved', 'stop_loss', 'manual', 'drawdown_halt'


@dataclass
class ReconciliationReport:
    """Comparison between DB state and on-chain holdings."""
    timestamp: str
    n_db_open: int
    n_chain_positions: int
    matches: int
    mismatches: int
    details: list[dict]
    is_clean: bool

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
