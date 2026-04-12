#!/bin/bash
# K-Fish Weekly Review
# Full statistical report + calibration + go/no-go checklist.

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "K-Fish Weekly Review"
echo "Date: $(date)"
echo "=========================================="

python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.db.manager import DatabaseManager
from src.reporting.dashboard import TrackRecordDashboard
from src.prediction.calibration import compute_brier, compute_ece

with DatabaseManager('data/kfish.db') as db:
    # Generate full report
    dashboard = TrackRecordDashboard(db)
    path = dashboard.generate_report('reports/weekly_review.md')

    # Track record
    record = db.get_track_record()
    preds, outs = db.get_calibration_data()

    print('=== PERFORMANCE SUMMARY ===')
    print(f'Predictions:  {record.n_predictions}')
    print(f'Positions:    {record.n_positions} ({record.n_closed} closed)')
    print(f'Win rate:     {record.win_rate:.1%}')
    print(f'Total P&L:    \${record.total_pnl_usd:+.2f}')
    print(f'Avg Brier:    {record.avg_brier:.4f}')

    if preds:
        brier = compute_brier(preds, outs)
        ece = compute_ece(preds, outs)
        bss = 1 - brier / 0.25
        print(f'Calibration:  Brier={brier:.4f}, ECE={ece:.4f}, BSS={bss:+.4f}')

    print()
    print('=== GO/NO-GO CHECKLIST ===')
    n_resolved = record.n_closed
    pnl_positive = record.total_pnl_usd > 0
    brier_ok = record.avg_brier < 0.25 if record.avg_brier > 0 else False
    ece_ok = ece < 0.10 if preds else False

    checks = [
        (n_resolved >= 100, f'100+ positions resolved: {n_resolved}'),
        (pnl_positive, f'Paper P&L positive: \${record.total_pnl_usd:+.2f}'),
        (brier_ok, f'Brier < 0.25 (random): {record.avg_brier:.4f}'),
        (ece_ok, f'ECE < 0.10: {ece:.4f}' if preds else 'ECE < 0.10: no data'),
        (False, 'Human (KSK) approves live trading: PENDING'),
    ]

    for passed, desc in checks:
        mark = '✅' if passed else '❌'
        print(f'  {mark} {desc}')

    all_pass = all(p for p, _ in checks)
    print()
    if all_pass:
        print('>>> ALL CHECKS PASS — ready for live trading <<<')
    else:
        print('>>> NOT READY — continue paper trading <<<')
" 2>/dev/null

echo ""
echo "Full report: reports/weekly_review.md"
