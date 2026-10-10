"""24/7 Standalone Cloud Worker for Virtual TES Gen0 (Phase 5.9).

Runs autonomously in a dedicated process (e.g. Railway WORKER service):
- Acquires and maintains single-active leader lock (PostgreSQL advisory lock / lease)
- Periodically polls live market data hierarchy (Litgrid primary, Elering fallback)
- Detects interval boundaries and executes completed shadow TES intervals
- Replays missed intervals upon startup recovery
- Progresses continuous physical SOC / sand temperature state
- Publishes worker heartbeats every 60 seconds
- Gracefully handles SIGTERM and SIGINT
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select, text, update
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.database.models import DayAheadPrice, RawMarketData, ShadowTESSession, WorkerHeartbeat
from app.database.session import init_db, make_engine, make_session_factory
from app.services.market_data_service import MarketDataService
from app.services.shadow_runtime import ShadowRuntimeService, get_market_interval_bounds
from app.services.worker_lock import DistributedWorkerLock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Worker) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("virtual_tes.worker")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VirtualTESWorker:
    """Autonomous 24/7 worker engine."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = make_engine(self.settings.database_url)
        init_db(self.engine)
        self.session_factory = make_session_factory(self.engine)

        self.lock = DistributedWorkerLock()
        self.market_service = MarketDataService(tz=self.settings.tz)
        self.shadow_service = ShadowRuntimeService(market_service=self.market_service, tz=self.settings.tz)

        self.shutdown_requested = False
        self.last_heartbeat_time: float = 0.0
        self.last_market_poll_time: float = 0.0
        self.market_poll_interval_sec: float = 300.0  # Poll market every 5 minutes
        self.heartbeat_interval_sec: float = 60.0    # Heartbeat every 60 seconds
        self.tick_sleep_sec: float = 4.0

        self.last_market_fetch_utc: datetime | None = None
        self.last_optimization_utc: datetime | None = None
        self.last_executed_interval_utc: datetime | None = None
        self.market_data_freshness: str = "LIVE"
        self.last_maintenance_time: float = 0.0

    def setup_signal_handlers(self) -> None:
        """Register graceful shutdown handlers for SIGINT and SIGTERM."""
        def _handle_signal(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received signal %s. Initiating graceful worker shutdown...", sig_name)
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def publish_heartbeat(self, session: Session, status: str = "RUNNING") -> None:
        """Persist worker health status to worker_heartbeats table."""
        try:
            active_session = session.execute(
                select(ShadowTESSession).order_by(ShadowTESSession.created_at_utc.desc()).limit(1)
            ).scalar_one_or_none()

            shadow_status = active_session.status if active_session else "STOPPED"
            if active_session:
                self.last_executed_interval_utc = active_session.last_executed_interval_start_utc
                if active_session.last_state_update_utc:
                    self.last_optimization_utc = active_session.last_state_update_utc

            now = utcnow()
            _, next_iv_end = get_market_interval_bounds(now)

            hb = WorkerHeartbeat(
                worker_id=self.lock.worker_id,
                timestamp_utc=now,
                status=status,
                app_env=self.settings.app_env,
                market_status=self.market_data_freshness,
                shadow_status=shadow_status,
                last_market_fetch_utc=self.last_market_fetch_utc,
                last_optimization_utc=self.last_optimization_utc,
                last_executed_interval_utc=self.last_executed_interval_utc,
                next_interval_utc=next_iv_end,
                details_json={
                    "is_leader": self.lock.is_leader,
                    "model_version": self.shadow_service.tes_params.capacity_semantics_version,
                    "bidding_zone": self.settings.bidding_zone,
                },
            )
            session.add(hb)
            # Prune stale heartbeats older than 3 days to prevent unbounded growth
            cutoff = now - timedelta(days=3)
            session.execute(delete(WorkerHeartbeat).where(WorkerHeartbeat.timestamp_utc < cutoff))
            session.commit()
            self.last_heartbeat_time = time.time()
            logger.debug("Published worker heartbeat: status=%s, leader=%s", status, self.lock.is_leader)
        except Exception as exc:
            session.rollback()
            logger.error("Failed to write worker heartbeat: %s", exc)

    def startup_catch_up(self, session: Session) -> None:
        """Recover and replay any missed intervals during downtime."""
        try:
            active = session.execute(
                select(ShadowTESSession).where(ShadowTESSession.status == "RUNNING").limit(1)
            ).scalar_one_or_none()
            if active:
                logger.info("Found active RUNNING shadow session %s. Replaying missed intervals...", active.id)
                caught_up = self.shadow_service.catch_up_missed_intervals(session, active)
                logger.info("Startup recovery complete. Replayed %d intervals.", caught_up)
                self.last_executed_interval_utc = active.last_executed_interval_start_utc
        except Exception as exc:
            session.rollback()
            logger.error("Error during startup catch-up: %s", exc)

    def check_market_data(self, session: Session) -> None:
        """Poll market data hierarchy and detect new day-ahead publications."""
        now = time.time()
        if now - self.last_market_poll_time < self.market_poll_interval_sec:
            return

        try:
            logger.info("Polling market data hierarchy (LITGRID primary, ELERING fallback)...")
            fetch_res = self.market_service.fetch_market_data(session=session)
            self.last_market_fetch_utc = utcnow()
            self.last_market_poll_time = now
            self.market_data_freshness = "LIVE" if not fetch_res.is_stale else "STALE"

            # Check if tomorrow prices just published and trigger re-optimization
            if fetch_res.status_snapshot.tomorrow_prices == "COMPLETE":
                active = session.execute(
                    select(ShadowTESSession).where(ShadowTESSession.status == "RUNNING").limit(1)
                ).scalar_one_or_none()
                if active:
                    logger.info("Tomorrow prices complete. Re-optimizing shadow session %s horizon...", active.id)
                    self.shadow_service.reoptimize(session, active, trigger_reason="DAY_AHEAD_PUBLISHED")
                    self.last_optimization_utc = utcnow()
        except Exception as exc:
            session.rollback()
            self.market_data_freshness = "STALE / ERROR"
            logger.warning("Market polling encountered issue (retaining last valid data): %s", exc)

    def run(self) -> None:
        """Main 24/7 worker execution loop."""
        logger.info("Starting Virtual TES Gen0 autonomous worker (id=%s)...", self.lock.worker_id)
        logger.info("Environment: %s | Database: %s", self.settings.app_env, self.settings.database_url.split("@")[-1])
        self.setup_signal_handlers()

        # Initial catch-up on start
        with self.session_factory() as session:
            if self.lock.acquire_or_renew(session):
                logger.info("Acquired primary worker leader lock.")
                self.startup_catch_up(session)
                self.run_daily_maintenance(session)
                self.publish_heartbeat(session, status="RUNNING")
            else:
                logger.warning("Primary lock held by another worker instance. Entering STANDBY mode.")
                self.publish_heartbeat(session, status="STANDBY")

        # Main 24/7 loop
        while not self.shutdown_requested:
            try:
                with self.session_factory() as session:
                    is_leader = self.lock.acquire_or_renew(session)

                    if is_leader:
                        # 1. Physical tick and interval boundary check
                        self.shadow_service.step_continuous_tick(session)

                        # 2. Market polling check
                        self.check_market_data(session)

                        # 3. Heartbeat check (every 60s)
                        if time.time() - self.last_heartbeat_time >= self.heartbeat_interval_sec:
                            self.publish_heartbeat(session, status="RUNNING")

                        # 4. Periodic automated maintenance & compaction (every 24h)
                        self.run_daily_maintenance(session)
                    else:
                        # Secondary standby instance: wait and publish standby heartbeat
                        if time.time() - self.last_heartbeat_time >= self.heartbeat_interval_sec:
                            self.publish_heartbeat(session, status="STANDBY")

            except Exception as loop_err:
                logger.error("Unexpected worker loop exception (recovering): %s", loop_err, exc_info=True)

            # Sleep between ticks
            time.sleep(self.tick_sleep_sec)

    def run_daily_maintenance(self, session: Session) -> None:
        """Run housekeeping to keep disk space minimal and prevent Railway volume bloat."""
        now = time.time()
        if now - self.last_maintenance_time < 21600.0:  # Run every 6 hours
            return
        self.last_maintenance_time = now
        try:
            logger.info("Running automated database maintenance & compaction...")
            # 1. Prune heartbeats older than 2 days
            cutoff_hb = utcnow() - timedelta(days=2)
            session.execute(delete(WorkerHeartbeat).where(WorkerHeartbeat.timestamp_utc < cutoff_hb))

            # 2. Prune raw market payloads older than 3 days (keeps disk usage < 100MB permanently)
            cutoff_raw = utcnow() - timedelta(days=3)
            old_raw_ids = session.execute(
                select(RawMarketData.id).where(RawMarketData.fetched_at < cutoff_raw)
            ).scalars().all()
            if old_raw_ids:
                session.execute(
                    update(DayAheadPrice)
                    .where(DayAheadPrice.raw_market_data_id.in_(old_raw_ids))
                    .values(raw_market_data_id=None)
                )
                session.execute(
                    delete(RawMarketData).where(RawMarketData.id.in_(old_raw_ids))
                )
                logger.info("Pruned %d stale raw market payloads older than 3 days.", len(old_raw_ids))

            session.commit()

            if self.engine.dialect.name == "postgresql":
                with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.execute(text("VACUUM ANALYZE raw_market_data;"))
                    conn.execute(text("VACUUM ANALYZE worker_heartbeats;"))
            elif self.engine.dialect.name == "sqlite":
                with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE);"))
            logger.info("Periodic database maintenance completed successfully.")
        except Exception as exc:
            session.rollback()
            logger.warning("Database maintenance encountered non-fatal error: %s", exc)

        # Clean shutdown
        logger.info("Shutting down worker %s...", self.lock.worker_id)
        try:
            with self.session_factory() as session:
                self.publish_heartbeat(session, status="STOPPED")
                self.lock.release(session)
        except Exception as shutdown_err:
            logger.warning("Error during final worker shutdown cleanup: %s", shutdown_err)
        logger.info("Virtual TES worker stopped cleanly.")


def main() -> None:
    worker = VirtualTESWorker()
    worker.run()


if __name__ == "__main__":
    main()
