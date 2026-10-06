"""Engine/session creation, schema initialisation and immutability triggers."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.database import models  # noqa: F401  (register tables)
from app.database.base import Base

#: Tables where UPDATE and DELETE are forbidden at the database level.
IMMUTABLE_TABLES: tuple[str, ...] = (
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
    "shadow_interval_records",
    "shadow_event_audits",
)


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.close()


def normalize_database_url(database_url: str) -> str:
    """Normalize database URL for SQLAlchemy and psycopg3 (supporting Railway postgres://)."""
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url[len("postgres://"):]
    if database_url.startswith("postgresql://") and "+psycopg" not in database_url and "+" not in database_url.split("://")[0]:
        return "postgresql+psycopg://" + database_url[len("postgresql://"):]
    return database_url


def make_engine(database_url: str, echo: bool = False) -> Engine:
    """Create an engine; creates parent directory for SQLite and configures pooling for PostgreSQL."""
    url = normalize_database_url(database_url)
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {"echo": echo, "future": True}
    if not url.startswith("sqlite:"):
        kwargs.update({
            "pool_pre_ping": True,
            "pool_size": 10,
            "max_overflow": 20,
        })

    engine = create_engine(url, **kwargs)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _sqlite_pragmas)
    return engine


def install_immutability_triggers(engine: Engine) -> None:
    with engine.begin() as conn:
        for table in IMMUTABLE_TABLES:
            for op in ("UPDATE", "DELETE"):
                conn.execute(text(
                    f"CREATE TRIGGER IF NOT EXISTS trg_{table}_no_{op.lower()} "
                    f"BEFORE {op} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, '{table} is append-only: {op} forbidden'); END;"
                ))


def _migrate_columns_sqlite(engine: Engine) -> None:
    """Safe backward-compatible column addition for existing SQLite databases."""
    with engine.begin() as conn:
        res = conn.execute(text("PRAGMA table_info(optimization_runs)")).fetchall()
        cols = {row[1] for row in res}
        if cols and "summary" not in cols:
            conn.execute(text("ALTER TABLE optimization_runs ADD COLUMN summary JSON"))

        res = conn.execute(text("PRAGMA table_info(optimization_schedule)")).fetchall()
        s_cols = {row[1] for row in res}
        if s_cols and "unmet_heat_kw" not in s_cols:
            conn.execute(text("ALTER TABLE optimization_schedule ADD COLUMN unmet_heat_kw FLOAT DEFAULT 0.0"))
        if s_cols and "auxiliary_load_kw" not in s_cols:
            conn.execute(text("ALTER TABLE optimization_schedule ADD COLUMN auxiliary_load_kw FLOAT DEFAULT 0.0"))

        res = conn.execute(text("PRAGMA table_info(tes_config)")).fetchall()
        t_cols = {row[1] for row in res}
        if t_cols and "charge_efficiency_is_provisional" not in t_cols:
            conn.execute(text("ALTER TABLE tes_config ADD COLUMN charge_efficiency_is_provisional BOOLEAN DEFAULT 1"))
        if t_cols and "discharge_efficiency_is_provisional" not in t_cols:
            conn.execute(text("ALTER TABLE tes_config ADD COLUMN discharge_efficiency_is_provisional BOOLEAN DEFAULT 1"))
        if t_cols and "auxiliary_power_kw" not in t_cols:
            conn.execute(text("ALTER TABLE tes_config ADD COLUMN auxiliary_power_kw FLOAT DEFAULT 0.05"))

        # Phase 5.8.3 optimization price basis & values
        if cols and "optimization_price_basis" not in cols:
            conn.execute(text("ALTER TABLE optimization_runs ADD COLUMN optimization_price_basis VARCHAR(32) DEFAULT 'EFFECTIVE_VARIABLE_PRICE'"))

        res = conn.execute(text("PRAGMA table_info(shadow_tes_sessions)")).fetchall()
        ss_cols = {row[1] for row in res}
        if ss_cols and "optimization_price_basis" not in ss_cols:
            conn.execute(text("ALTER TABLE shadow_tes_sessions ADD COLUMN optimization_price_basis VARCHAR(32) DEFAULT 'EFFECTIVE_VARIABLE_PRICE'"))
        if ss_cols and "optimization_price_eur_mwh" not in ss_cols:
            conn.execute(text("ALTER TABLE shadow_tes_sessions ADD COLUMN optimization_price_eur_mwh FLOAT"))

        res = conn.execute(text("PRAGMA table_info(shadow_interval_records)")).fetchall()
        sir_cols = {row[1] for row in res}
        if sir_cols and "optimization_price_basis" not in sir_cols:
            conn.execute(text("ALTER TABLE shadow_interval_records ADD COLUMN optimization_price_basis VARCHAR(32) DEFAULT 'EFFECTIVE_VARIABLE_PRICE'"))
        if sir_cols and "optimization_price_eur_mwh" not in sir_cols:
            conn.execute(text("ALTER TABLE shadow_interval_records ADD COLUMN optimization_price_eur_mwh FLOAT"))


def init_db(engine: Engine) -> None:
    """Create all tables (idempotent) and install append-only triggers."""
    Base.metadata.create_all(engine)
    if engine.dialect.name == "sqlite":
        _migrate_columns_sqlite(engine)
        install_immutability_triggers(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
