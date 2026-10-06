"""Industrial process heat demand profiles (thermal kW).

The demand is what the process REQUIRES; it is independent of electricity price.
Delivered heat may be lower only if physically impossible (tracked as unmet heat).
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from app.process.profiles import ConstantProfile, CsvProfile, PowerProfile, ScheduleProfile


class HeatDemandProfile(PowerProfile):
    """Marker base class: thermal demand profile."""


class ConstantHeatDemand(ConstantProfile, HeatDemandProfile):
    def __init__(self, value_kw: float = 1.5, name: str = "constant_heat") -> None:
        super().__init__(value_kw, name)


class ScheduledHeatDemand(ScheduleProfile, HeatDemandProfile):
    def __init__(self, blocks: list[tuple[time, float]], tz: ZoneInfo, name: str = "scheduled_heat") -> None:
        super().__init__(blocks, tz, name)


class CsvHeatDemand(CsvProfile, HeatDemandProfile):
    def __init__(self, path: str | Path, value_column: str = "heat_demand_kw", name: str | None = None) -> None:
        super().__init__(path, value_column, name)


from typing import Literal


def wood_drying_schedule(tz: ZoneInfo) -> ScheduledHeatDemand:
    """Example wood drying chamber profile from the specification (NOT MEASURED DRYER PROFILE)."""
    return ScheduledHeatDemand(
        [(time(0), 1.0), (time(6), 1.8), (time(18), 1.3)],
        tz,
        name="example_process_profile",
    )


def get_heat_demand_scenario(
    scenario: Literal["1.0kw_constant", "1.5kw_constant", "2.0kw_constant", "example_process_profile"] | str,
    tz: ZoneInfo,
) -> HeatDemandProfile:
    """Return configured heat demand profile.

    Status: SCENARIO ASSUMPTIONS -- NOT MEASURED DRYER PROFILE.
    """
    s = scenario.lower().strip()
    if s in ("1.0", "1.0kw", "1.0kw_constant"):
        return ConstantHeatDemand(value_kw=1.0, name="1.0kw_constant")
    elif s in ("1.5", "1.5kw", "1.5kw_constant"):
        return ConstantHeatDemand(value_kw=1.5, name="1.5kw_constant")
    elif s in ("2.0", "2.0kw", "2.0kw_constant"):
        return ConstantHeatDemand(value_kw=2.0, name="2.0kw_constant")
    elif s in ("variable", "dryer", "example_process_profile", "wood_drying"):
        return wood_drying_schedule(tz)
    else:
        raise ValueError(f"Unknown heat demand scenario: {scenario}")
