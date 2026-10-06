"""Other site electrical loads (kW electrical), excluding the TES heaters.

Simulated now; in production a ``PowerMeterSiteLoad`` implementing the same
``PowerProfile`` interface will supply measured / forecast values.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from app.process.profiles import ConstantProfile, CsvProfile, PowerProfile, ScheduleProfile


class SiteLoadProfile(PowerProfile):
    """Marker base class: non-TES site electrical load."""


class ConstantSiteLoad(ConstantProfile, SiteLoadProfile):
    def __init__(self, value_kw: float = 2.0, name: str = "constant_site_load") -> None:
        super().__init__(value_kw, name)


class ScheduledSiteLoad(ScheduleProfile, SiteLoadProfile):
    def __init__(self, blocks: list[tuple[time, float]], tz: ZoneInfo, name: str = "scheduled_site_load") -> None:
        super().__init__(blocks, tz, name)


class CsvSiteLoad(CsvProfile, SiteLoadProfile):
    def __init__(self, path: str | Path, value_column: str = "site_load_kw", name: str | None = None) -> None:
        super().__init__(path, value_column, name)
