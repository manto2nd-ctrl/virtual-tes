"""Domain models for historical backtesting and baseline comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.tes.model import ConstantDischargeLimit, DischargeLimitCurve, GEN0_CURVE_NAME


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a multi-day historical backtest run."""

    start_date: date
    end_date: date
    mode: Literal["realistic_rolling", "perfect_foresight"] = "realistic_rolling"
    initial_soc_kwh: float = 7.5
    tes_params: TESParameters = field(default_factory=TESParameters)
    site_params: SiteParameters = field(default_factory=SiteParameters)
    tariff_params: TariffParameters = field(default_factory=TariffParameters)
    discharge_limit_curve: DischargeLimitCurve = field(default_factory=ConstantDischargeLimit)
    discharge_curve_name: str = GEN0_CURVE_NAME
    discharge_curve_mode: Literal["fixed_gen0_hx", "scaled_hx_benchmark", "normalized_benchmark", "constant"] = "fixed_gen0_hx"
    auxiliary_power_kw: float = 0.05
    direct_heater_efficiency: float = 0.98
    cheapest_n_hours: int = 4
    rolling_horizon_hours: int = 24
    rolling_step_hours: int = 24
    rolling_terminal_soc_mode: Literal["hold_initial", "soc_min", "free"] = "hold_initial"
    bidding_zone: str = "LT"
    price_source: str = "mock"
    heat_demand_profile_name: str = "1.5kw_constant"
    backup_heat_enabled: bool = False
    backup_heater_efficiency: float = 0.98
    backup_heater_max_power_kw: float = 10.0
    day_ahead_assumed_available_local_time: str = "14:00"
    market_timezone: str = "Europe/Vilnius"
    sand_mass_kg: float | None = None
    hx_area_m2: float = 1.55
    overall_u_w_m2k: float = 9.0
    airflow_m3_h: float = 80.0
    air_inlet_temperature_c: float = 40.0
    thermal_capacity_full_span_kwh: float = 15.0
    optimizer_soc_min_fraction: float = 0.10
    optimizer_soc_max_fraction: float = 1.00
    dispatchable_capacity_kwh: float = 13.5
    temperature_at_optimizer_min_soc_c: float = 104.71

    def __post_init__(self) -> None:
        if self.tes_params is not None:
            object.__setattr__(self, "thermal_capacity_full_span_kwh", self.tes_params.thermal_capacity_full_span_kwh)
            object.__setattr__(self, "optimizer_soc_min_fraction", self.tes_params.optimizer_soc_min_fraction)
            object.__setattr__(self, "optimizer_soc_max_fraction", self.tes_params.optimizer_soc_max_fraction)
            object.__setattr__(self, "dispatchable_capacity_kwh", self.tes_params.dispatchable_capacity_kwh)
            object.__setattr__(self, "temperature_at_optimizer_min_soc_c", self.tes_params.temperature_at_optimizer_min_soc_c)


@dataclass(frozen=True)
class BacktestStrategyMetrics:
    """Comprehensive performance and economic metrics for one strategy over the backtest."""

    name: str
    mode: str
    electricity_kwh: float
    useful_heat_kwh: float
    unmet_heat_kwh: float
    heat_supply_reliability_percent: float
    tes_auxiliary_kwh: float
    thermal_standing_loss_kwh: float
    conversion_loss_kwh: float
    average_paid_electricity_price: float
    raw_electricity_cost_eur: float
    terminal_inventory_adjustment_eur: float
    unmet_heat_cost_eur: float = 0.0
    total_cost_eur: float = 0.0
    cost_eur_per_mwh_heat: float = 0.0
    savings_vs_direct_eur: float = 0.0
    savings_vs_direct_percent: float = 0.0
    equivalent_discharge_cycles: float = 0.0
    average_soc_kwh: float = 0.0
    average_soc_percent: float = 0.0
    minimum_soc_kwh: float = 0.0
    minimum_soc_percent: float = 0.0
    maximum_soc_kwh: float = 0.0
    maximum_soc_percent: float = 0.0
    hours_charge_off_during_high_price_periods: float = 0.0
    charging_hours: float = 0.0
    final_soc_kwh: float = 0.0
    energy_balance_residual_kwh: float = 0.0
    is_physical_gen0_viable: bool = False
    status_label: str = "ENGINEERING ESTIMATE -- NOT YET CALIBRATED TO PHYSICAL GEN0"
    viability_caveats: list[str] = field(default_factory=list)
    backup_heat_kwh: float = 0.0
    backup_electricity_kwh: float = 0.0
    backup_cost_eur: float = 0.0
    sand_mass_kg: float | None = None
    hx_area_m2: float = 1.55
    overall_u_w_m2k: float = 9.0
    airflow_m3_h: float = 80.0
    objective_value: float = 0.0
    thermal_capacity_full_span_kwh: float = 15.0
    dispatchable_capacity_kwh: float = 13.5
    optimizer_soc_min_fraction: float = 0.10
    optimizer_soc_max_fraction: float = 1.00
    temperature_at_optimizer_min_soc_c: float = 104.71
    configured_charge_power_limit_kw: float = 9.0
    peak_actual_charge_power_kw: float = 0.0
    grid_connection_limit_kw: float = 12.0
    peak_total_grid_power_kw: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "electricity_kwh": round(self.electricity_kwh, 2),
            "useful_heat_kwh": round(self.useful_heat_kwh, 2),
            "unmet_heat_kwh": round(self.unmet_heat_kwh, 2),
            "heat_supply_reliability_percent": round(self.heat_supply_reliability_percent, 2),
            "tes_auxiliary_kwh": round(self.tes_auxiliary_kwh, 2),
            "thermal_standing_loss_kwh": round(self.thermal_standing_loss_kwh, 2),
            "conversion_loss_kwh": round(self.conversion_loss_kwh, 2),
            "average_paid_electricity_price": round(self.average_paid_electricity_price, 2),
            "raw_electricity_cost_eur": round(self.raw_electricity_cost_eur, 2),
            "terminal_inventory_adjustment_eur": round(self.terminal_inventory_adjustment_eur, 2),
            "unmet_heat_cost_eur": round(self.unmet_heat_cost_eur, 2),
            "total_cost_eur": round(self.total_cost_eur, 2),
            "cost_eur_per_mwh_heat": round(self.cost_eur_per_mwh_heat, 2),
            "savings_vs_direct_eur": round(self.savings_vs_direct_eur, 2),
            "savings_vs_direct_percent": round(self.savings_vs_direct_percent, 2),
            "equivalent_discharge_cycles": round(self.equivalent_discharge_cycles, 2),
            "average_soc_kwh": round(self.average_soc_kwh, 2),
            "average_soc_percent": round(self.average_soc_percent, 1),
            "minimum_soc_kwh": round(self.minimum_soc_kwh, 2),
            "minimum_soc_percent": round(self.minimum_soc_percent, 1),
            "maximum_soc_kwh": round(self.maximum_soc_kwh, 2),
            "maximum_soc_percent": round(self.maximum_soc_percent, 1),
            "hours_charge_off_during_high_price_periods": round(self.hours_charge_off_during_high_price_periods, 2),
            "charging_hours": round(self.charging_hours, 2),
            "final_soc_kwh": round(self.final_soc_kwh, 2),
            "energy_balance_residual_kwh": round(self.energy_balance_residual_kwh, 6),
            "is_physical_gen0_viable": self.is_physical_gen0_viable,
            "status_label": self.status_label,
            "viability_caveats": self.viability_caveats,
            "backup_heat_kwh": round(self.backup_heat_kwh, 2),
            "backup_electricity_kwh": round(self.backup_electricity_kwh, 2),
            "backup_cost_eur": round(self.backup_cost_eur, 2),
            "sand_mass_kg": round(self.sand_mass_kg, 1) if self.sand_mass_kg is not None else None,
            "objective_value": round(self.objective_value, 4),
            "thermal_capacity_full_span_kwh": round(self.thermal_capacity_full_span_kwh, 2),
            "dispatchable_capacity_kwh": round(self.dispatchable_capacity_kwh, 2),
            "optimizer_soc_min_fraction": round(self.optimizer_soc_min_fraction, 4),
            "optimizer_soc_max_fraction": round(self.optimizer_soc_max_fraction, 4),
            "temperature_at_optimizer_min_soc_c": round(self.temperature_at_optimizer_min_soc_c, 2),
            "configured_charge_power_limit_kw": round(self.configured_charge_power_limit_kw, 2),
            "peak_actual_charge_power_kw": round(self.peak_actual_charge_power_kw, 2),
            "grid_connection_limit_kw": round(self.grid_connection_limit_kw, 2),
            "peak_total_grid_power_kw": round(self.peak_total_grid_power_kw, 2),
        }


@dataclass(frozen=True)
class BacktestReport:
    """Full backtest outcome comparing all evaluated strategies."""

    config: BacktestConfig
    strategies: dict[str, BacktestStrategyMetrics]
    days_count: int
    total_intervals: int
    sensitivity_results: list[dict] | None = None


@dataclass(frozen=True)
class SizingCombinationResult:
    """Outcome for a specific (capacity, charge_power) sizing combination."""

    capacity_kwh: float
    charge_power_kw: float
    sizing_mode: str
    useful_heat_kwh: float
    heat_supply_reliability_percent: float
    electricity_consumed_kwh: float
    average_paid_electricity_price: float
    total_cost_eur: float
    cost_eur_per_mwh_heat: float
    savings_vs_direct_eur: float
    savings_vs_direct_percent: float
    equivalent_cycles: float
    average_soc_kwh: float
    average_soc_percent: float
    minimum_soc_kwh: float
    maximum_soc_kwh: float
    charging_hours: float
    hours_charge_off_during_high_price_periods: float
    energy_balance_residual_kwh: float
    sand_mass_kg: float = 0.0
    hx_area_m2: float = 1.55
    overall_u_w_m2k: float = 9.0
    airflow_m3_h: float = 80.0
    unmet_heat_kwh: float = 0.0
    standing_losses_kwh: float = 0.0
    conversion_losses_kwh: float = 0.0
    aux_electricity_kwh: float = 0.0
    thermal_capacity_full_span_kwh: float = 15.0
    dispatchable_capacity_kwh: float = 13.5
    optimizer_soc_min_fraction: float = 0.10
    optimizer_soc_max_fraction: float = 1.00
    temperature_at_optimizer_min_soc_c: float = 104.71
    configured_charge_power_limit_kw: float = 9.0
    peak_actual_charge_power_kw: float = 0.0
    grid_connection_limit_kw: float = 12.0
    peak_total_grid_power_kw: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity_kwh > 0:
            object.__setattr__(self, "thermal_capacity_full_span_kwh", self.capacity_kwh)
            dispatchable = self.capacity_kwh * (self.optimizer_soc_max_fraction - self.optimizer_soc_min_fraction)
            object.__setattr__(self, "dispatchable_capacity_kwh", dispatchable)
            object.__setattr__(self, "configured_charge_power_limit_kw", self.charge_power_kw)
            from app.tes.thermal import ThermalStateMapper
            mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
            object.__setattr__(self, "temperature_at_optimizer_min_soc_c", mapper.temperature_from_soc_fraction(self.optimizer_soc_min_fraction))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SizingStudyReport:
    """Collection of sizing combinations across capacities and charge powers."""

    combinations: list[SizingCombinationResult]
    lowest_cost_combination: SizingCombinationResult | None = None
    highest_reliability_combination: SizingCombinationResult | None = None
    lowest_capacity_meeting_99_5_rel: SizingCombinationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
