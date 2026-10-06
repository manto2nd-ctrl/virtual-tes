"""Generic time-varying power profiles (kW) used for heat demand and site loads.

Three implementations:
  * ``ConstantProfile``  - one value forever;
  * ``ScheduleProfile``  - daily repeating blocks defined in LOCAL wall-clock time;
  * ``CsvProfile``       - explicit per-interval values from a CSV file (UTC timestamps).

The value of an interval is looked up at the interval START (documented assumption:
schedule block boundaries must be aligned to the simulation resolution; this is validated).
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from bisect import bisect_right
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from app.core.timegrid import Interval, ensure_utc


class PowerProfile(ABC):
    """Abstract kW profile over time."""

    name: str = "profile"

    @abstractmethod
    def power_kw(self, interval: Interval) -> float:
        """Average power [kW] over ``interval``."""

    def series(self, intervals: list[Interval]) -> list[float]:
        return [self.power_kw(iv) for iv in intervals]

    def describe(self) -> dict:
        """JSON-serialisable description for run configuration snapshots."""
        return {"type": type(self).__name__, "name": self.name}


class ConstantProfile(PowerProfile):
    def __init__(self, value_kw: float, name: str = "constant") -> None:
        if value_kw < 0:
            raise ValueError("power must be >= 0")
        self.value_kw = float(value_kw)
        self.name = name

    def power_kw(self, interval: Interval) -> float:
        return self.value_kw

    def describe(self) -> dict:
        return {**super().describe(), "value_kw": self.value_kw}


class ScheduleProfile(PowerProfile):
    """Daily repeating schedule in local time.

    ``blocks`` is a list of ``(start_time_local, kW)``; each block lasts until the next
    block start; the last wraps around to the first. Example::

        ScheduleProfile([(time(0), 1.0), (time(6), 1.8), (time(18), 1.3)], tz)
    """

    def __init__(self, blocks: list[tuple[time, float]], tz: ZoneInfo, name: str = "schedule",
                 alignment_minutes: int = 15) -> None:
        if not blocks:
            raise ValueError("schedule needs at least one block")
        blocks = sorted(blocks, key=lambda b: b[0])
        starts = [b[0] for b in blocks]
        if len(set(starts)) != len(starts):
            raise ValueError("duplicate block start times")
        for t, v in blocks:
            if v < 0:
                raise ValueError("power must be >= 0")
            if (t.hour * 60 + t.minute) % alignment_minutes or t.second or t.microsecond:
                raise ValueError(f"block start {t} not aligned to {alignment_minutes} min")
        self._minutes = [t.hour * 60 + t.minute for t in starts]
        self._values = [float(v) for _, v in blocks]
        self.tz = tz
        self.name = name

    def power_kw(self, interval: Interval) -> float:
        local = interval.start_local(self.tz)
        minute_of_day = local.hour * 60 + local.minute
        idx = bisect_right(self._minutes, minute_of_day) - 1  # -1 -> wraps to last block
        return self._values[idx]

    def describe(self) -> dict:
        return {**super().describe(), "tz": str(self.tz),
                "blocks": [{"start_minute": m, "kw": v} for m, v in zip(self._minutes, self._values)]}


class CsvProfile(PowerProfile):
    """Per-interval values from CSV with columns ``timestamp_utc,<value_column>``.

    ``timestamp_utc`` must be ISO-8601 with offset (e.g. ``2026-10-05T00:00:00+00:00``)
    and marks the interval START. Missing intervals raise ``KeyError`` rather than being
    silently filled.
    """

    def __init__(self, path: str | Path, value_column: str = "power_kw", name: str | None = None) -> None:
        self.path = Path(path)
        self.value_column = value_column
        self.name = name or self.path.stem
        self._data: dict[datetime, float] = {}
        with self.path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ts = ensure_utc(datetime.fromisoformat(row["timestamp_utc"]))
                val = float(row[value_column])
                if val < 0:
                    raise ValueError(f"negative power at {ts}")
                if ts in self._data:
                    raise ValueError(f"duplicate timestamp {ts} in {self.path}")
                self._data[ts] = val

    def power_kw(self, interval: Interval) -> float:
        try:
            return self._data[interval.start_utc]
        except KeyError:
            raise KeyError(f"{self.path.name}: no value for interval starting {interval.start_utc}") from None

    def describe(self) -> dict:
        return {**super().describe(), "path": str(self.path), "rows": len(self._data)}
