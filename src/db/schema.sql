-- K-Fish v5 Production Database Schema
-- SQLite, auto-created by DatabaseManager on first run

-- Predictions log (every engine run)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    question TEXT NOT NULL,
    category TEXT,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    -- Pipeline outputs
    raw_probability REAL NOT NULL,
    extremized_probability REAL NOT NULL,
    calibrated_probability REAL NOT NULL,
    market_price REAL,

    -- Swarm metadata
    n_fish INTEGER,
    n_rounds INTEGER,
    spread REAL,
    std_dev REAL,
    effective_confidence REAL,
    disagreement_flag BOOLEAN,
    personas_used TEXT,  -- JSON array
    model TEXT,

    -- Timing
    total_elapsed_s REAL,
    research_elapsed_s REAL,

    -- Fish details (JSON array of per-fish predictions)
    fish_predictions TEXT,

    UNIQUE(market_id, timestamp)
);

-- Positions (open and closed)
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    question TEXT NOT NULL,

    -- Entry
    side TEXT NOT NULL CHECK (side IN ('YES', 'NO')),
    entry_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    shares REAL NOT NULL DEFAULT 0,
    entry_timestamp TEXT NOT NULL,
    prediction_id INTEGER REFERENCES predictions(id),

    -- Exit (NULL while open)
    exit_price REAL,
    exit_timestamp TEXT,
    pnl_usd REAL,
    pnl_pct REAL,
    exit_reason TEXT,  -- 'resolved', 'stop_loss', 'manual', 'drawdown_halt'

    -- On-chain
    order_id TEXT,
    tx_hash TEXT,
    token_id TEXT,

    -- Status
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'pending', 'filled', 'closed', 'cancelled')),

    UNIQUE(market_id, entry_timestamp)
);

-- Calibration training data (rolling window)
CREATE TABLE IF NOT EXISTS calibration_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction REAL NOT NULL,
    outcome REAL NOT NULL,
    market_id TEXT,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Market resolution outcomes
CREATE TABLE IF NOT EXISTS resolutions (
    market_id TEXT PRIMARY KEY,
    question TEXT,
    outcome REAL NOT NULL,  -- 1.0 or 0.0
    resolved_at TEXT NOT NULL,
    our_prediction REAL,
    brier_score REAL
);

-- System state (key-value for bankroll, drawdown, etc.)
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_predictions_market ON predictions(market_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market_id);
CREATE INDEX IF NOT EXISTS idx_calibration_timestamp ON calibration_data(timestamp);
