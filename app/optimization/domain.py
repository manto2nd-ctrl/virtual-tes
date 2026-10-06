"""Domain models for mathematical optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from app.config.parameters import TESParameters
from app.core.timegrid import ensure_utc
from app.tes.model import ConstantDischargeLimit, DischargeLimitCurve


@dataclass(frozen=True, slots=True)
class OptimizationIntervalInput:
    """Normalized input data for a single optimization interval."""

    start_utc: datetime
    end_utc: datetime
    spot_price_eur_mwh: float
    effective_price_eur_mwh: float
    heat_demand_kw: float
    other_site_load_kw: float = 0.0
    auxiliary_load_kw: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_utc", ensure_utc(self.start_utc))
        object.__setattr__(self, "end_utc", ensure_utc(self.end_utc))
        if self.end_utc <= self.start_utc:
            raise ValueError(f"end_utc must be after start_utc: {self.start_utc} .. {self.end_utc}")

    @property
    def duration_h(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds() / 3600.0


@dataclass(frozen=True)
class OptimizationProblemInput:
    """Normalized input model for the LP optimizer."""

    intervals: list[OptimizationIntervalInput]
    tes_params: TESParameters
    grid_connection_limit_kw: float
    initial_soc_kwh: float
    target_terminal_soc_kwh: float | None = None
    terminal_soc_condition: Literal["exact", "min"] = "exact"
    discharge_limit_curve: DischargeLimitCurve = field(default_factory=ConstantDischargeLimit)
    unmet_heat_penalty_eur_per_kwh: float = 10000.0
    throughput_penalty_eur_per_kwh: float = 1e-5
    solver_name: str = "PULP_CBC_CMD"
    lexicographic: bool = True
    backup_heat_enabled: bool = False
    backup_heater_efficiency: float = 0.98
    backup_heater_max_power_kw: float = 10.0

    def resolved_target_terminal_soc_kwh(self) -> float:
        if self.target_terminal_soc_kwh is not None:
            return self.target_terminal_soc_kwh
        return self.initial_soc_kwh


@dataclass(frozen=True, slots=True)
class OptimizationIntervalResult:
    """Optimization outcome for a single interval."""

    start_utc: datetime
    end_utc: datetime
    spot_price_eur_mwh: float
    effective_price_eur_mwh: float
    charge_power_kw: float
    discharge_power_kw: float
    unmet_heat_kw: float
    soc_kwh: float
    soc_percent: float
    heat_demand_kw: float
    other_site_load_kw: float
    auxiliary_load_kw: float
    grid_power_kw: float
    cost_eur: float
    standing_loss_kw: float
    backup_heat_kw: float = 0.0
    backup_electricity_kw: float = 0.0


@dataclass(frozen=True, slots=True)
class OptimizationMetrics:
    """Explicit metrics required for reporting, backtesting and performance evaluation."""

    total_unmet_heat_kwh: float
    heat_supply_reliability_percent: float
    terminal_soc_kwh: float
    total_charge_kwh: float
    total_storage_energy_withdrawn_kwh: float
    useful_heat_delivered_kwh: float
    charge_conversion_losses_kwh: float
    discharge_conversion_losses_kwh: float
    standing_losses_kwh: float
    electricity_cost_eur: float
    cost_eur_per_mwh_useful_heat: float
    energy_balance_residual_kwh: float
    total_backup_heat_kwh: float = 0.0
    total_backup_electricity_kwh: float = 0.0
    total_backup_cost_eur: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total_unmet_heat_kwh": round(self.total_unmet_heat_kwh, 4),
            "heat_supply_reliability_percent": round(self.heat_supply_reliability_percent, 2),
            "terminal_soc_kwh": round(self.terminal_soc_kwh, 4),
            "total_charge_kwh": round(self.total_charge_kwh, 4),
            "total_storage_energy_withdrawn_kwh": round(self.total_storage_energy_withdrawn_kwh, 4),
            "useful_heat_delivered_kwh": round(self.useful_heat_delivered_kwh, 4),
            "charge_conversion_losses_kwh": round(self.charge_conversion_losses_kwh, 4),
            "discharge_conversion_losses_kwh": round(self.discharge_conversion_losses_kwh, 4),
            "standing_losses_kwh": round(self.standing_losses_kwh, 4),
            "electricity_cost_eur": round(self.electricity_cost_eur, 4),
            "cost_eur_per_mwh_useful_heat": round(self.cost_eur_per_mwh_useful_heat, 2),
            "energy_balance_residual_kwh": round(self.energy_balance_residual_kwh, 6),
        }


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    solver: str
    objective_eur: float
    intervals: list[OptimizationIntervalResult]
    metrics: OptimizationMetrics
    initial_soc_kwh: float
    target_terminal_soc_kwh: float
