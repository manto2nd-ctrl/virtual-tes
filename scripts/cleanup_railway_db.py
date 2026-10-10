"""Railway Database Cleanup & Compaction Utility.

Identifies and cleans up historical disk bloat:
1. Deduplicates RawMarketData payloads (each redundant ~2MB API response is removed,
   updating any day_ahead_prices foreign key references to the canonical row).
2. Prunes obsolete worker heartbeats older than 3 days.
3. Reclaims physical disk space via VACUUM / wal_checkpoint.

Usage:
    DATABASE_URL="postgresql://..." python scripts/cleanup_railway_db.py
    # or for local SQLite:
    uv run python scripts/cleanup_railway_db.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.database.models import DayAheadPrice, RawMarketData, WorkerHeartbeat
from app.database.session import (
    IMMUTABLE_TABLES,
    init_db,
    install_immutability_triggers,
    make_engine,
    make_session_factory,
    normalize_database_url,
)


def get_sqlite_size(db_url: str) -> tuple[int, int]:
    """Return size in bytes of SQLite db and wal files."""
    if not db_url.startswith("sqlite:///"):
        return 0, 0
    path_str = db_url.removeprefix("sqlite:///")
    if ":memory:" in path_str:
        return 0, 0
    db_path = Path(path_str)
    wal_path = Path(f"{path_str}-wal")
    db_size = db_path.stat().st_size if db_path.exists() else 0
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    return db_size, wal_size


def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def run_cleanup() -> int:
    settings = get_settings()
    db_url = settings.database_url
    print("=" * 80)
    print("VIRTUAL TES GEN0 — DATABASE CLEANUP & DISK COMPACTION UTILITY")
    print("=" * 80)
    print(f"Target Database: {normalize_database_url(db_url).split('@')[-1]}")

    engine = make_engine(db_url)
    is_sqlite = engine.dialect.name == "sqlite"

    init_db(engine)
    session_factory = make_session_factory(engine)

    # 1. Initial Inspection
    with session_factory() as session:
        raw_count = session.query(func.count(RawMarketData.id)).scalar() or 0
        distinct_sha_count = session.query(func.count(func.distinct(RawMarketData.payload_sha256))).scalar() or 0
        hb_count = session.query(func.count(WorkerHeartbeat.id)).scalar() or 0
        price_count = session.query(func.count(DayAheadPrice.id)).scalar() or 0

        print(f"\n[1] Current Database Inventory:")
        print(f"  - raw_market_data rows: {raw_count:,} (Distinct payloads: {distinct_sha_count:,})")
        print(f"  - worker_heartbeats rows: {hb_count:,}")
        print(f"  - day_ahead_prices rows: {price_count:,}")

        if is_sqlite:
            db_size, wal_size = get_sqlite_size(db_url)
            print(f"  - SQLite DB file size:  {format_bytes(db_size)}")
            print(f"  - SQLite WAL file size: {format_bytes(wal_size)}")
            print(f"  - Total on disk:        {format_bytes(db_size + wal_size)}")

    # 2. RawMarketData Deduplication
    print("\n[2] Deduplicating RawMarketData payloads...")
    # For SQLite, temporarily drop DELETE trigger on raw_market_data
    if is_sqlite:
        with engine.begin() as conn:
            conn.execute(text("DROP TRIGGER IF EXISTS trg_raw_market_data_no_delete"))

    deduped_removed = 0
    with session_factory() as session:
        try:
            # Find all duplicate payload_sha256 groups
            dup_groups = session.execute(
                select(RawMarketData.payload_sha256, func.count(RawMarketData.id))
                .group_by(RawMarketData.payload_sha256)
                .having(func.count(RawMarketData.id) > 1)
            ).all()

            for sha, count in dup_groups:
                # Get all IDs for this SHA ordered by ID ascending
                rows = session.execute(
                    select(RawMarketData.id)
                    .where(RawMarketData.payload_sha256 == sha)
                    .order_by(RawMarketData.id.asc())
                ).scalars().all()

                canonical_id = rows[0]
                duplicate_ids = rows[1:]

                # Update any DayAheadPrice foreign keys pointing to duplicate IDs
                session.execute(
                    update(DayAheadPrice)
                    .where(DayAheadPrice.raw_market_data_id.in_(duplicate_ids))
                    .values(raw_market_data_id=canonical_id)
                )

                # Delete duplicate raw_market_data rows
                session.execute(
                    delete(RawMarketData).where(RawMarketData.id.in_(duplicate_ids))
                )
                deduped_removed += len(duplicate_ids)

            session.commit()
            print(f"  -> Removed {deduped_removed:,} duplicate ~2MB raw payload rows!")
        except Exception as exc:
            session.rollback()
            print(f"  [!] Deduplication error: {exc}")

    # 2b. Purge raw payloads older than 2 days
    print("\n[2b] Purging raw payloads older than 2 days...")
    purged_raw = 0
    with session_factory() as session:
        try:
            cutoff_raw = datetime.now(timezone.utc) - timedelta(days=2)
            old_ids = session.execute(
                select(RawMarketData.id).where(RawMarketData.fetched_at < cutoff_raw)
            ).scalars().all()
            if old_ids:
                session.execute(
                    update(DayAheadPrice)
                    .where(DayAheadPrice.raw_market_data_id.in_(old_ids))
                    .values(raw_market_data_id=None)
                )
                session.execute(
                    delete(RawMarketData).where(RawMarketData.id.in_(old_ids))
                )
                session.commit()
                purged_raw = len(old_ids)
                print(f"  -> Purged {purged_raw:,} older raw payload rows (> 2 days old)!")
            else:
                print("  -> No raw payloads older than 2 days.")
        except Exception as p_exc:
            session.rollback()
            print(f"  [!] Raw payload purge error: {p_exc}")

    # Re-enable immutability trigger for SQLite
    if is_sqlite:
        install_immutability_triggers(engine)

    # 3. Prune Old Worker Heartbeats
    print("\n[3] Pruning obsolete worker heartbeats (> 3 days old)...")
    with session_factory() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        res = session.execute(delete(WorkerHeartbeat).where(WorkerHeartbeat.timestamp_utc < cutoff))
        session.commit()
        deleted_hb = res.rowcount if hasattr(res, "rowcount") else 0
        print(f"  -> Removed {deleted_hb:,} stale heartbeat records.")

    # 4. Reclaim Disk Space
    print("\n[4] Reclaiming physical disk space...")
    with engine.begin() as conn:
        if is_sqlite:
            conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            conn.execute(text("VACUUM"))
            print("  -> Executed SQLite WAL checkpoint & VACUUM.")
        else:
            # PostgreSQL
            print("  -> PostgreSQL VACUUM ANALYZE executing...")
            conn.execute(text("VACUUM ANALYZE"))
            print("  -> Completed. Tip: Run 'VACUUM FULL raw_market_data;' in psql to immediately release OS disk space.")

    # 5. Final Report
    with session_factory() as session:
        final_raw = session.query(func.count(RawMarketData.id)).scalar() or 0
        final_hb = session.query(func.count(WorkerHeartbeat.id)).scalar() or 0

        print(f"\n[5] Cleanup Summary:")
        print(f"  - raw_market_data rows: {raw_count:,} -> {final_raw:,} ({deduped_removed:,} deleted)")
        print(f"  - worker_heartbeats:    {hb_count:,} -> {final_hb:,}")

        if is_sqlite:
            final_db, final_wal = get_sqlite_size(db_url)
            print(f"  - SQLite DB on disk:    {format_bytes(final_db + final_wal)}")
            reclaimed = (db_size + wal_size) - (final_db + final_wal)
            if reclaimed > 0:
                print(f"  - Space reclaimed:      {format_bytes(reclaimed)}")

    print("\nCleanup completed successfully!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_cleanup())
