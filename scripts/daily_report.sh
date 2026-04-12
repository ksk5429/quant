#!/bin/bash
# K-Fish Daily Report
# Generates track record report and checks resolutions.

set -e
cd "$(dirname "$0")/.."

echo "=== K-Fish Daily Report ==="
echo "Date: $(date)"
echo ""

# Check resolutions
echo "--- Checking resolutions ---"
python -m src.execution.live_loop --resolve-only --db data/kfish.db 2>/dev/null

# Generate report
echo ""
echo "--- Generating track record ---"
python -c "
from src.db.manager import DatabaseManager
from src.reporting.dashboard import TrackRecordDashboard

with DatabaseManager('data/kfish.db') as db:
    dashboard = TrackRecordDashboard(db)
    path = dashboard.generate_report()
    print(open(path, encoding='utf-8').read())
" 2>/dev/null

echo ""
echo "Report saved to reports/track_record.md"
