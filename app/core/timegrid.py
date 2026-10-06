"""DST-safe time handling.

Rules enforced everywhere in the system:
  * All internal timestamps are timezone-aware UTC ``datetime`` objects.
  * Naive datetimes are rejected (they are the #1 source of DST bugs).
  * A "local day" is built from local midnight to next local midnight, converted to
    UTC. It can therefore contain 23, 24 or 25 hours (92/96/100 quarter-hours).
  * We never assume a fixed number of intervals per day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc


def ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` converted to UTC. Raises ``ValueError`` for naive datetimes."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"Naive datetime not allowed (need timezone-aware): {dt!r}")
    return dt.astimezone(UTC)


def to_local(dt: datetime, tz: ZoneInfo) -> datetime:
    """Convert an aware datetime to the given local timezone."""
    return ensure_utc(dt).astimezone(tz)


def local_day_bounds_utc(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """UTC start/end of a local calendar day ``[00:00, next 00:00)`` in ``tz``.

    Local midnight never falls inside a DST gap/fold for Europe/Vilnius (changes happen
    at 03:00/04:00 local), so constructing midnight directly is unambiguous.
    """
    start_local = datetime.combine(day, time(0, 0), tzinfo=tz)
    end_local = datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=tz)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open time interval ``[start_utc, end_utc)``."""

    start_utc: datetime
    end_utc: datetime

    def __post_init__(self) -> None:
        start = ensure_utc(self.start_utc)
        end = ensure_utc(self.end_utc)
        if end <= start:
            raise ValueError(f"Interval end must be after start: {start} .. {end}")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)

    @property
    def duration_h(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds() / 3600.0

    @property
    def duration_minutes(self) -> int:
        return int((self.end_utc - self.start_utc).total_seconds() // 60)

    def start_local(self, tz: ZoneInfo) -> datetime:
        return self.start_utc.astimezone(tz)


def build_intervals(start_utc: datetime, end_utc: datetime, resolution_minutes: int) -> list[Interval]:
    """Build contiguous intervals of ``resolution_minutes`` covering ``[start, end)``.

    Arithmetic is done in UTC, so DST transitions cannot create gaps or duplicates.
    """
    start = ensure_utc(start_utc)
    end = ensure_utc(end_utc)
    if resolution_minutes <= 0:
        raise ValueError("resolution_minutes must be positive")
    step = timedelta(minutes=resolution_minutes)
    if (end - start) % step != timedelta(0):
        raise ValueError(f"Range {start}..{end} is not a multiple of {resolution_minutes} min")
    out: list[Interval] = []
    t = start
    while t < end:
        out.append(Interval(t, t + step))
        t += step
    return out


def local_day_intervals(day: date, tz: ZoneInfo, resolution_minutes: int) -> list[Interval]:
    """All market intervals of a local calendar day (92/96/100 for 15-min in EU)."""
    start, end = local_day_bounds_utc(day, tz)
    return build_intervals(start, end, resolution_minutes)


from typing import Any


@dataclass(frozen=True, slots=True)
class DayCoverageResult:
    """Coverage and completeness validation for a local market calendar day."""

    local_date: date
    expected_intervals: int
    received_intervals: int
    missing_intervals: int
    duplicate_intervals: int
    coverage_fraction: float
    is_complete: bool
    status: str
    first_interval_local: str | None
    last_interval_local: str | None
    intervals: list[Any]


def expected_intervals_in_local_day(
    local_date: date,
    timezone: ZoneInfo,
    resolution_minutes: int = 15,
) -> int:
    """Calculate the exact number of intervals in a local calendar day taking DST into account."""
    day_start_utc, day_end_utc = local_day_bounds_utc(local_date, timezone)
    grid = build_intervals(day_start_utc, day_end_utc, resolution_minutes)
    return len(grid)


def validate_local_market_day_coverage(
    local_date: date,
    timezone: ZoneInfo,
    intervals: list[Any],
    resolution_minutes: int = 15,
) -> DayCoverageResult:
    """Validate completeness and continuity of market intervals for a local calendar day.

    Dynamically calculates expected interval count from the true local day duration
    (96 for normal 24h, 92 for spring DST 23h, 100 for autumn DST 25h).
    Detects gaps, missing intervals, and duplicates.
    """
    day_start_utc, day_end_utc = local_day_bounds_utc(local_date, timezone)
    expected_grid = build_intervals(day_start_utc, day_end_utc, resolution_minutes)
    expected_count = len(expected_grid)

    matching = [
        iv for iv in intervals
        if day_start_utc <= ensure_utc(iv.start_utc) < day_end_utc
    ]
    matching.sort(key=lambda x: ensure_utc(x.start_utc))

    seen_starts: set[datetime] = set()
    unique_matching = []
    duplicate_count = 0
    for iv in matching:
        st = ensure_utc(iv.start_utc)
        if st in seen_starts:
            duplicate_count += 1
        else:
            seen_starts.add(st)
            unique_matching.append(iv)

    expected_starts = {exp.start_utc for exp in expected_grid}
    received_starts = {ensure_utc(iv.start_utc) for iv in unique_matching}
    missing_count = len(expected_starts - received_starts)
    received_count = len(received_starts)

    coverage_fraction = round(received_count / expected_count, 4) if expected_count > 0 else 0.0
    is_complete = (missing_count == 0 and duplicate_count == 0 and received_count == expected_count)

    first_local = None
    last_local = None
    if unique_matching:
        f_st = ensure_utc(unique_matching[0].start_utc).astimezone(timezone).strftime("%H:%M")
        f_en = ensure_utc(unique_matching[0].end_utc).astimezone(timezone).strftime("%H:%M")
        first_local = f"{f_st} -> {f_en}"

        l_st = ensure_utc(unique_matching[-1].start_utc).astimezone(timezone).strftime("%H:%M")
        l_en = ensure_utc(unique_matching[-1].end_utc).astimezone(timezone).strftime("%H:%M")
        last_local = f"{l_st} -> {l_en}"

    if received_count == 0:
        status = "TOMORROW PRICES NOT YET AVAILABLE"
    elif is_complete:
        status = "TOMORROW COMPLETE"
    else:
        status = "PARTIAL TOMORROW DATA"

    return DayCoverageResult(
        local_date=local_date,
        expected_intervals=expected_count,
        received_intervals=received_count,
        missing_intervals=missing_count,
        duplicate_intervals=duplicate_count,
        coverage_fraction=coverage_fraction,
        is_complete=is_complete,
        status=status,
        first_interval_local=first_local,
        last_interval_local=last_local,
        intervals=unique_matching,
    )

