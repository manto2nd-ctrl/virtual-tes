"""Distributed leader election locking for single-active worker guarantees (Phase 5.9).

Supports:
1. Native PostgreSQL session-level advisory locks (pg_try_advisory_lock)
2. Database lease table fallback (DistributedLock table) for SQLite / testing
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.database.models import DistributedLock

log = logging.getLogger(__name__)
POSTGRES_ADVISORY_LOCK_ID = 847293  # Deterministic 32-bit ID for Virtual TES worker leader


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DistributedWorkerLock:
    """Acquires and maintains a single-active worker lease across multiple instances."""

    def __init__(self, lock_name: str = "worker_master", lease_seconds: int = 90) -> None:
        self.lock_name = lock_name
        self.lease_seconds = lease_seconds
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self.is_leader = False
        self._pg_conn = None

    def acquire_or_renew(self, session: Session) -> bool:
        """Attempt to acquire or renew leadership. Returns True if this worker is active leader."""
        # Use distributed lease table for robust connection-pool-safe leader election
        return self._acquire_db_lease(session)

    def _acquire_pg_advisory(self, session: Session) -> bool:
        """PostgreSQL session advisory lock. True if lock acquired, False if already held."""
        try:
            res = session.execute(
                text("SELECT pg_try_advisory_lock(:lock_id)"),
                {"lock_id": POSTGRES_ADVISORY_LOCK_ID},
            ).scalar()
            self.is_leader = bool(res)
            return self.is_leader
        except Exception as exc:
            log.warning("Postgres advisory lock check failed, falling back to lease table: %s", exc)
            return self._acquire_db_lease(session)

    def _acquire_db_lease(self, session: Session) -> bool:
        """Generic database lease table for SQLite or standalone fallback."""
        now = utcnow()
        new_expiry = now + timedelta(seconds=self.lease_seconds)

        try:
            lock_row = session.execute(
                select(DistributedLock).where(DistributedLock.lock_name == self.lock_name)
            ).scalar_one_or_none()

            if lock_row is None:
                # No lock exists, create and claim
                lock_row = DistributedLock(
                    lock_name=self.lock_name,
                    owner_id=self.worker_id,
                    acquired_at_utc=now,
                    expires_at_utc=new_expiry,
                )
                session.add(lock_row)
                session.commit()
                self.is_leader = True
                return True

            if lock_row.owner_id == self.worker_id:
                # Renew our own lease
                lock_row.expires_at_utc = new_expiry
                session.commit()
                self.is_leader = True
                return True

            # Held by another worker: check if expired
            if lock_row.expires_at_utc < now:
                # Lease expired! Take over
                lock_row.owner_id = self.worker_id
                lock_row.acquired_at_utc = now
                lock_row.expires_at_utc = new_expiry
                session.commit()
                self.is_leader = True
                log.info("Previous worker lease expired. Took over leader lock as %s", self.worker_id)
                return True

            # Active lease held by someone else
            self.is_leader = False
            return False

        except Exception as exc:
            session.rollback()
            log.warning("Error acquiring distributed lock: %s", exc)
            self.is_leader = False
            return False

    def release(self, session: Session) -> None:
        """Explicitly release lock upon clean shutdown."""
        bind = session.get_bind()
        if bind.dialect.name == "postgresql":
            try:
                session.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": POSTGRES_ADVISORY_LOCK_ID},
                )
                session.commit()
            except Exception as exc:
                log.debug("Error releasing postgres advisory lock: %s", exc)

        try:
            lock_row = session.execute(
                select(DistributedLock).where(
                    DistributedLock.lock_name == self.lock_name,
                    DistributedLock.owner_id == self.worker_id,
                )
            ).scalar_one_or_none()
            if lock_row:
                session.delete(lock_row)
                session.commit()
        except Exception as exc:
            session.rollback()
            log.debug("Error releasing distributed lock row: %s", exc)
        self.is_leader = False
