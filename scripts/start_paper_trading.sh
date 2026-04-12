#!/bin/bash
# K-Fish Paper Trading Daemon
# Runs scan→analyze→execute cycles every 6 hours in paper mode.
# This is the default, safe entry point. No real money at risk.

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "K-Fish Paper Trading"
echo "Mode:     PAPER (no real orders)"
echo "Interval: 6 hours"
echo "Markets:  top 10 by volume"
echo "Bankroll: \$1,000 (simulated)"
echo "============================================"

python -m src.execution.live_loop \
    --paper \
    --top 10 \
    --interval 6 \
    --model haiku \
    --concurrent 3 \
    --max-position 50 \
    --max-exposure 300 \
    --bankroll 1000
