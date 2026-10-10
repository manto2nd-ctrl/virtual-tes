"""Hardware Provenance Contracts & Supervisory Safety Guards.

Formalizes operating modes, data provenance tagging, telemetry schema,
and strict lockout for physical actuation (ConnectedDeviceGuard).

Cloud Railway is SUPERVISORY ONLY. No direct physical actuator commands
are permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OperatingMode(str, Enum):
    """System operating modes."""
    VIRTUAL = "VIRTUAL"      # Physics-based simulated twin
    SHADOW = "SHADOW"        # Read-only external telemetry comparison; zero actuation
    CONNECTED = "CONNECTED"  # Explicitly disabled until physical commissioning & local PLC interlocks


class DataProvenance(str, Enum):
    """Provenance indicator for all signals and KPIs."""
    SIMULATED = "SIMULATED"
    SHADOW_PREDICTED = "SHADOW_PREDICTED"
    MEASURED = "MEASURED"
    HISTORICAL_BACKTEST = "HISTORICAL_BACKTEST"
    SYNTHETIC = "SYNTHETIC"


class ConnectedDeviceLockedError(RuntimeError):
    """Raised when any software attempt is made to command physical hardware."""
    pass


class ConnectedDeviceGuard:
    """Supervisory safety guard blocking unauthorized physical actuation."""

    @staticmethod
    def verify_mode_allowed(mode: OperatingMode | str) -> None:
        """Validate that operating mode does not violate the cloud supervisory policy."""
        mode_str = mode.value if isinstance(mode, OperatingMode) else str(mode).upper()
        if mode_str == OperatingMode.CONNECTED.value or "CONNECTED" in mode_str:
            raise ConnectedDeviceLockedError(
                "CONNECTED mode is strictly locked out. Physical hardware control is prohibited. "
                "Local PLC gateway, independent high-limit over-temperature cutoffs, airflow interlocks, "
                "and formal on-site commissioning are required before physical actuation is enabled."
            )

    @staticmethod
    def verify_actuation_permitted() -> None:
        """Validate whether physical actuation commands are permitted (always locked on cloud)."""
        raise ConnectedDeviceLockedError(
            "Physical hardware actuation is strictly locked out. Cloud platform is supervisory only."
        )

    @staticmethod
    def enforce_read_only(command_name: str) -> None:
        """Enforce read-only constraint on cloud services."""
        raise ConnectedDeviceLockedError(
            f"Physical actuation command '{command_name}' rejected. "
            "Railway cloud deployment operates in supervisory read-only / simulation mode only."
        )


@dataclass(frozen=True)
class DeviceTelemetry:
    """Universal telemetry contract shared between virtual twin and future physical adapters."""
    timestamp_utc: datetime
    site_id: str = "gen0-vilnius-demo"
    device_id: str = "dryer-v1"
    mode: OperatingMode = OperatingMode.VIRTUAL
    provenance: DataProvenance = DataProvenance.SIMULATED
    sensor_quality: str = "GOOD"

    # Electrical
    heater_electrical_power_kw: float = 0.0
    blower_power_kw: float = 0.0
    blower_running: bool = True
    backup_heater_power_kw: float = 0.0

    # Temperatures (°C)
    sand_temperature_c: float = 196.8
    hx_air_inlet_temp_c: float = 40.0
    hx_air_outlet_temp_c: float = 70.0
    dryer_supply_temp_c: float = 70.0
    dryer_exhaust_temp_c: float = 48.0

    # Flow & Humidity
    airflow_m3_h: float = 80.0
    inlet_rh_percent: float = 45.0
    exhaust_rh_percent: float = 82.0

    # Timber Batch Kinetics (Estimated)
    timber_mass_dry_kg: float = 500.0
    moisture_content_percent: float = 32.0
    water_removed_kg: float = 45.0

    # Alarms / Interlocks
    alarms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BESSStatus:
    """Proposed Battery Energy Storage System (BESS) representation.

    Explicitly declared as PROPOSED / NOT CONNECTED. Zero phantom capacity,
    zero phantom savings, zero active power in Gen0 results.
    """
    status: str = "PROPOSED / NOT CONNECTED"
    provenance: DataProvenance = DataProvenance.SYNTHETIC
    nominal_capacity_kwh: float = 12.0
    active_power_kw: float = 0.0
    soc_percent: float = 50.0
    phantom_savings_eur: float = 0.0
    is_active: bool = False
    policy_note: str = (
        "BESS is declared as PROPOSED / NOT CONNECTED. Reserved for future grid-import "
        "support; zero phantom capacity and zero electric savings are attributed to Gen0."
    )
