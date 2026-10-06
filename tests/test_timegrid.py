from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo
import pytest

from app.core.timegrid import (
    UTC,
    Interval,
    build_intervals,
    ensure_utc,
    local_day_bounds_utc,
    local_day_intervals,
)


def test_ensure_utc():
    naive = datetime(2026, 1, 1, 12, 0)
    with pytest.raises(ValueError, match="Naive datetime not allowed"):
        ensure_utc(naive)

    aware = datetime(2026, 1, 1, 14, 0, tzinfo=ZoneInfo("Europe/Vilnius"))
    utc = ensure_utc(aware)
    assert utc.tzinfo == UTC
    assert utc.hour == 12  # Vilnius is UTC+2 in winter


def test_interval_validation():
    t1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 10, 15, tzinfo=UTC)
    iv = Interval(t1, t2)
    assert iv.duration_minutes == 15
    assert iv.duration_h == 0.25

    with pytest.raises(ValueError, match="Interval end must be after start"):
        Interval(t2, t1)


def test_standard_day_intervals():
    tz = ZoneInfo("Europe/Vilnius")
    # Standard winter day
    d = date(2026, 1, 15)
    intervals = local_day_intervals(d, tz, resolution_minutes=15)
    assert len(intervals) == 96
    start, end = local_day_bounds_utc(d, tz)
    assert intervals[0].start_utc == start
    assert intervals[-1].end_utc == end


def test_dst_spring_forward_23h():
    """On EU spring forward Sunday (e.g. 2026-03-29), Vilnius skips 03:00->04:00, day has 23 hours = 92 intervals."""
    tz = ZoneInfo("Europe/Vilnius")
    d = date(2026, 3, 29)
    intervals = local_day_intervals(d, tz, resolution_minutes=15)
    assert len(intervals) == 92
    total_h = sum(iv.duration_h for iv in intervals)
    assert total_h == 23.0


def test_dst_fall_back_25h():
    """On EU fall back Sunday (e.g. 2026-10-25), 04:00 repeats, day has 25 hours = 100 intervals."""
    tz = ZoneInfo("Europe/Vilnius")
    d = date(2026, 10, 25)
    intervals = local_day_intervals(d, tz, resolution_minutes=15)
    assert len(intervals) == 100
    total_h = sum(iv.duration_h for iv in intervals)
    assert total_h == 25.0
