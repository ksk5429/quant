"""Live trading loop — the production daemon.

Runs scan→analyze→execute cycles on interval. Handles:
- Paper trading (default) and live trading modes
- Resolution checking on open positions
- Drawdown monitoring and automatic halt
- Graceful error handling (never crashes)
- Human approval gate for live mode

Usage:
    # Paper trading (DEFAULT)
    python -m src.execution.live_loop --paper --top 10 --interval 6

    # Single cycle
    python -m src.execution.live_loop --paper --once --top 10

    # Live trading (requires --live + confirmation)
    python -m src.execution.live_loop --live --top 5 --max-position 25

    # Check resolutions only
    python -m src.execution.live_loop --resolve-only

    # Reconcile DB vs chain
    python -m src.execution.live_loop --reconcile
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

from src.db.manager import DatabaseManager
from src.execution.polymarket_executor import PolymarketExecutor
from src.execution.position_manager import PositionManager
from src.markets.scanner import MarketScanner
from src.mirofish.engine_v4 import PredictionEngineV4
from src.reporting.alerts import AlertManager
from src.risk.portfolio import DrawdownMonitor


@dataclass
class LiveConfig:
    """Configuration for the live trading loop."""
    paper_trading: bool = True
    top_n: int = 10
    interval_hours: float = 6.0
    model: str = "haiku"
    max_concurrent: int = 3
    max_position_usd: float = 50.0
    max_exposure_usd: float = 300.0
    min_volume_usd: float = 100_000
    min_liquidity_usd: float = 20_000
    bankroll_usd: float = 1000.0
    kelly_fraction: float = 0.25
    db_path: str = "data/kfish.db"


@dataclass
class CycleReport:
    """Report from a single scan-analyze-execute cycle."""
    timestamp: str
    n_scanned: int
    n_analyzed: int
    n_positions_opened: int
    n_positions_resolved: int
    total_pnl_resolved: float
    errors: list[str]
    elapsed_s: float


class LiveTradingLoop:
    """Production trading daemon."""

    def __init__(
        self,
        engine: PredictionEngineV4,
        executor: PolymarketExecutor,
        position_manager: PositionManager,
        db: DatabaseManager,
        config: LiveConfig,
    ) -> None:
        self.engine = engine
        self.executor = executor
        self.pm = position_manager
        self.db = db
        self.config = config
        self._drawdown = DrawdownMonitor(max_drawdown_pct=0.15)
        self._alerts = AlertManager()
        self._cycle_count = 0

        # Restore drawdown halt from DB
        halt_state = db.get_system_state("drawdown_halted")
        if halt_state == "true":
            self._drawdown.halted = True
            executor.set_drawdown_halt(True)
            logger.warning("Drawdown halt restored from DB")

    async def run_cycle(self) -> CycleReport:
        """Single scan→analyze→execute cycle."""
        t_start = time.monotonic()
        self._cycle_count += 1
        errors = []
        n_opened = 0
        n_resolved = 0
        total_pnl = 0.0

        logger.info(f"{'='*60}")
        logger.info(f"CYCLE {self._cycle_count} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"{'='*60}")

        # ── Step 1: Check resolutions on open positions ──
        try:
            closed = await self.pm.check_resolutions()
            n_resolved = len(closed)
            for c in closed:
                total_pnl += c.pnl_usd
                self._drawdown.record_pnl(c.pnl_usd)
                self.executor.release_exposure(c.size_usd)
                self._alerts.position_closed(
                    c.market_id, c.question, c.pnl_usd, c.reason,
                )
                # Feed calibrator with actual market outcome (not reconstructed from P&L)
                self.engine.record_outcome(
                    market_id=c.market_id,
                    prediction=c.entry_price,  # our entry price as prediction proxy
                    outcome=c.market_outcome,   # actual resolution from Gamma API
                )
            if self._drawdown.check_halt(self.config.bankroll_usd):
                self.executor.set_drawdown_halt(True)
                self.db.set_system_state("drawdown_halted", "true")
                self._alerts.drawdown_halt(
                    self._drawdown.current_drawdown / self.config.bankroll_usd
                )
        except Exception as e:
            logger.exception(f"Resolution check failed: {e}")
            self._alerts.engine_error(str(e))
            errors.append(f"resolution: {e}")

        # ── Step 2: Scan markets ──
        n_scanned = 0
        candidates = []
        try:
            scanner = MarketScanner(
                min_volume_usd=self.config.min_volume_usd,
                min_liquidity_usd=self.config.min_liquidity_usd,
            )
            candidates = await scanner.scan()
            n_scanned = len(candidates)
            candidates = candidates[:self.config.top_n]
            logger.info(f"Scanned {n_scanned} markets, analyzing top {len(candidates)}")
        except Exception as e:
            logger.exception(f"Scan failed: {e}")
            errors.append(f"scan: {e}")

        # ── Step 3: Filter out markets we already hold ──
        open_pos = self.db.get_open_positions()
        held_markets = {p.market_id for p in open_pos}
        candidates = [c for c in candidates if c.id not in held_markets]

        # ── Step 4: Analyze each market ──
        n_analyzed = 0
        for i, market in enumerate(candidates, 1):
            if self._drawdown.halted:
                logger.warning("Drawdown halt active — skipping remaining markets")
                break

            try:
                result = await self.engine.analyze(
                    question=market.question,
                    description=market.description,
                    outcomes=market.outcomes,
                    market_price=market.yes_price,
                    market_id=market.id,
                    volume_usd=market.volume_usd,
                )
                n_analyzed += 1

                # ── Step 5: Execute if edge found ──
                if result.position and result.swarm_healthy:
                    pred_row = self.db.conn.execute(
                        "SELECT id FROM predictions WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
                        (market.id,)
                    ).fetchone()
                    pred_id = pred_row["id"] if pred_row else 0

                    order = await self.pm.execute_position(
                        position=result.position,
                        market=market,
                        prediction_id=pred_id,
                    )
                    if order.is_filled:
                        n_opened += 1
                        self._alerts.position_opened(
                            market.id, market.question,
                            result.position.side,
                            result.position.position_size_usd,
                            result.edge,
                        )

                    logger.info(
                        f"  [{i}] {result.position.side} ${result.position.position_size_usd:.0f} "
                        f"edge={result.edge:.1%} → {order.status}"
                    )
                else:
                    edge_str = f"edge={result.edge:.1%}" if result.edge > 0 else "no edge"
                    logger.info(f"  [{i}] SKIP ({edge_str}) {market.question[:40]}")

            except Exception as e:
                logger.exception(f"Analysis failed for {market.question[:40]}: {e}")
                errors.append(f"analyze: {e}")

        elapsed = time.monotonic() - t_start

        report = CycleReport(
            timestamp=datetime.now().isoformat(),
            n_scanned=n_scanned,
            n_analyzed=n_analyzed,
            n_positions_opened=n_opened,
            n_positions_resolved=n_resolved,
            total_pnl_resolved=round(total_pnl, 2),
            errors=errors,
            elapsed_s=round(elapsed, 1),
        )

        self._print_cycle_report(report)
        return report

    async def run_daemon(self, interval_hours: float | None = None) -> None:
        """Run cycles on interval. Catches all exceptions."""
        interval = interval_hours or self.config.interval_hours
        interval_s = interval * 3600

        mode = "PAPER" if self.config.paper_trading else "LIVE"
        logger.info(
            f"K-Fish daemon starting: mode={mode}, "
            f"interval={interval}h, top_n={self.config.top_n}"
        )

        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.exception(f"Cycle crashed (continuing): {e}")

            logger.info(f"Next cycle in {interval:.1f} hours")

            # Sleep in small increments so we can be interrupted
            sleep_end = time.monotonic() + interval_s
            while time.monotonic() < sleep_end:
                await asyncio.sleep(min(60, sleep_end - time.monotonic()))

    async def resolve_only(self) -> None:
        """Just check resolutions, don't analyze or trade."""
        logger.info("Checking resolutions on open positions...")
        closed = await self.pm.check_resolutions()
        if closed:
            total = sum(c.pnl_usd for c in closed)
            logger.info(f"Resolved {len(closed)} positions, P&L=${total:+.2f}")
        else:
            logger.info("No positions resolved")

        open_pos = self.db.get_open_positions()
        logger.info(f"Open positions remaining: {len(open_pos)}")

    async def reconcile(self) -> None:
        """Run reconciliation check."""
        report = await self.pm.reconcile()
        logger.info(
            f"Reconciliation: DB={report.n_db_open}, Chain={report.n_chain_positions}, "
            f"Clean={report.is_clean}"
        )
        if not report.is_clean:
            logger.error(f"MISMATCH: {report.mismatches} discrepancies found")

    def _print_cycle_report(self, r: CycleReport) -> None:
        print(f"\n{'─'*50}")
        print(f"Cycle {self._cycle_count} complete ({r.elapsed_s:.0f}s)")
        print(f"  Scanned:   {r.n_scanned}")
        print(f"  Analyzed:  {r.n_analyzed}")
        print(f"  Opened:    {r.n_positions_opened}")
        print(f"  Resolved:  {r.n_positions_resolved} (P&L=${r.total_pnl_resolved:+.2f})")
        if r.errors:
            print(f"  Errors:    {len(r.errors)}")
        print(f"{'─'*50}")


# ── CLI Entry Point ──

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="K-Fish Live Trading Loop")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading (DEFAULT)")
    parser.add_argument("--live", action="store_true", help="Live trading (requires confirmation)")
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    parser.add_argument("--resolve-only", action="store_true", help="Check resolutions only")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile DB vs chain")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--interval", type=float, default=6.0, help="Hours between cycles")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--concurrent", type=int, default=3)
    parser.add_argument("--max-position", type=float, default=50.0)
    parser.add_argument("--max-exposure", type=float, default=300.0)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--db", type=str, default="data/kfish.db")
    parser.add_argument("--confirm", action="store_true", help="Skip live mode confirmation")
    parser.add_argument("--reset-drawdown", action="store_true", help="Reset drawdown halt")
    args = parser.parse_args()

    is_live = args.live and not args.paper
    if args.live:
        args.paper = False

    # RULE 1: Live requires confirmation
    if is_live and not args.confirm:
        print("\n" + "=" * 60)
        print("WARNING: LIVE TRADING MODE")
        print("This will place REAL orders with REAL money on Polymarket.")
        print("=" * 60)
        confirm = input("Type 'I understand this uses real money' to continue: ")
        if confirm.strip() != "I understand this uses real money":
            print("Aborted.")
            return

    config = LiveConfig(
        paper_trading=not is_live,
        top_n=args.top,
        interval_hours=args.interval,
        model=args.model,
        max_concurrent=args.concurrent,
        max_position_usd=args.max_position,
        max_exposure_usd=args.max_exposure,
        bankroll_usd=args.bankroll,
        db_path=args.db,
    )

    with DatabaseManager(config.db_path) as db:
        # Initialize system state
        if db.get_system_state("bankroll") is None:
            db.set_system_state("bankroll", str(config.bankroll_usd))

        engine = PredictionEngineV4(
            model=config.model,
            max_concurrent=config.max_concurrent,
            kelly_fraction=config.kelly_fraction,
            bankroll_usd=config.bankroll_usd,
            db=db,
        )

        executor = PolymarketExecutor(
            paper_trading=config.paper_trading,
            max_position_usd=config.max_position_usd,
            max_exposure_usd=config.max_exposure_usd,
        )

        pm = PositionManager(executor=executor, db=db)

        loop = LiveTradingLoop(
            engine=engine,
            executor=executor,
            position_manager=pm,
            db=db,
            config=config,
        )

        if args.reset_drawdown:
            loop._drawdown.halted = False
            executor.set_drawdown_halt(False)
            db.set_system_state("drawdown_halted", "false")
            logger.info("Drawdown halt reset (persisted to DB)")

        if args.resolve_only:
            await loop.resolve_only()
        elif args.reconcile:
            await loop.reconcile()
        elif args.once:
            await loop.run_cycle()
        else:
            await loop.run_daemon()


if __name__ == "__main__":
    asyncio.run(main())
