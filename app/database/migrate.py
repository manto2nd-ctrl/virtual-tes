"""Database migration and verification script for deployment (Phase 5.9).

Runs idempotently on both PostgreSQL and SQLite:
- Creates all operational tables if not present
- Applies backwards-compatible schema column alterations
- Seeds default engineering scenarios if empty
- Verifies database connectivity and table inventory
- NEVER drops tables, clears data, or resets state
"""

from __future__ import annotations

import logging
import sys
from sqlalchemy import inspect, text

from app.config.settings import get_settings
from app.database.session import init_db, make_engine, make_session_factory
from app.services.scenario_service import seed_default_scenarios_if_empty

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("virtual_tes.migrate")

EXPECTED_TABLES = [
    "raw_market_data",
    "day_ahead_prices",
    "tariff_config",
    "tes_config",
    "tes_state",
    "heat_demand",
    "site_load",
    "optimization_runs",
    "optimization_schedule",
    "simulation_runs",
    "simulation_results",
    "run_status_events",
    "engineering_scenarios",
    "shadow_tes_sessions",
    "shadow_interval_records",
    "shadow_event_audits",
    "worker_heartbeats",
    "distributed_locks",
]


def run_migrations() -> None:
    settings = get_settings()
    logger.info("Connecting to database: %s", settings.database_url.split("@")[-1])

    engine = make_engine(settings.database_url)

    # 1. Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection established.")

    # 2. Run idempotent DDL creation
    logger.info("Applying schema migrations (Base.metadata.create_all)...")
    init_db(engine)

    # 3. Seed default engineering scenarios if table empty
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        seed_default_scenarios_if_empty(session)
    logger.info("Default engineering scenarios verified/seeded.")

    # 4. Verify all operational tables exist
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    logger.info("Found %d tables in database.", len(existing_tables))

    missing = [t for t in EXPECTED_TABLES if t not in existing_tables]
    if missing:
        logger.error("Migration incomplete. Missing expected tables: %s", missing)
        sys.exit(1)

    logger.info("All %d operational tables successfully verified:", len(EXPECTED_TABLES))
    for tbl in sorted(EXPECTED_TABLES):
        logger.info("  ✓ %s", tbl)

    logger.info("Database migration completed successfully with zero data loss.")


if __name__ == "__main__":
    run_migrations()
