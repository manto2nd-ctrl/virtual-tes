"""SQLAlchemy base and a strict UTC datetime column type."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator):
    """Stores aware datetimes as naive UTC in SQLite and returns aware UTC.

    SQLite has no native timezone support; this type guarantees that (a) naive
    datetimes are rejected on write and (b) values always come back as UTC-aware.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):  # type: ignore[override]
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"Naive datetime not allowed in database: {value!r}")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect):  # type: ignore[override]
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass
