"""Sensitivity analysis and systematic parameter sizing studies for TES.

Supports:
1. Standing loss sensitivity sweep (1% to 30%/day) -- labeled as assumptions.
2. Round-trip efficiency sensitivity sweep (LOW 76.5%, NOMINAL 85.5%, HIGH 93.1%).
3. Auxiliary electrical load sensitivity sweep (0 W, 50 W, 100 W, 200 W).
4. Process heat demand scenario sweep (1.0 kW, 1.5 kW, 2.0 kW, and variable dryer profile).
5. Systematic TES sizing sweeps (10, 15, 20, 30 kWh x 6, 9, 12 kW) across:
   - Mode A: Normalized benchmark (same Pmax(SOC%) for all capacities)
   - Mode B: Fixed Gen0 HX (absolute Pmax(SOC_kwh) curve for 1.5-1.6 m2 coil).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from zoneinfo import ZoneInfo

from app.backtest.domain import (
    BacktestConfig,
    BacktestStrategyMetrics,
    SizingCombinationResult,
    SizingStudyReport,
)
from app.backtest.engine import BacktestRunner
from app.config.parameters import SiteParameters, TESParameters
from app.models.domain import PricePoint
from app.process.heat_demand import get_heat_demand_scenario
from app.process.profiles import PowerProfile
from app.tes.model import get_gen0_discharge_curve

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensitivityCaseResult:
    parameter_name: str
    scenario_name: str
    parameter_value: float
    total_cost_eur: float
    savings_vs_direct_eur: float
    savings_vs_direct_percent: float
    heat_supply_reliability_percent: float
    equivalent_cycles: float
    standing_losses_kwh: float
    total_electricity_kwh: float = 0.0
    total_useful_heat_kwh: float = 0.0
    cost_eur_per_mwh_heat: float = 0.0


class SensitivityAnalyzer:
    """Performs parametric sweeps over physical engineering assumptions."""

    def __init__(self, tz: ZoneInfo) -> None:
        self.tz = tz
        self.runner = BacktestRunner(tz=tz)

    def sweep_standing_losses(
        self,
        base_config: BacktestConfig,
        heat_profile: PowerProfile,
        site_profile: PowerProfile,
        prices: list[PricePoint],
        scenarios: dict[str, float] | None = None,
    ) -> list[SensitivityCaseResult]:
        """Sweep standing loss percentages per day across scenarios.
        
        Labels explicitly as: 'standing-loss assumption', NOT 'measured TES loss'.
        """
        loss_scenarios = scenarios or {
            "Optimistic (1%/day)": 1.0,
            "Reference (2%/day)": 2.0,
            "Nominal (3%/day)": 3.0,
            "Conservative (6%/day)": 6.0,
            "High (10%/day)": 10.0,
            "Elevated (15%/day)": 15.0,
            "Severe (20%/day)": 20.0,
            "Extreme (30%/day)": 30.0,
        }

        results: list[SensitivityCaseResult] = []
        for name, loss_pct in loss_scenarios.items():
            new_tes = base_config.tes_params.model_copy(
                update={"standing_loss_percent_per_day": loss_pct}
            )
            cfg = BacktestConfig(
                **{
                    **base_config.__dict__,
                    "tes_params": new_tes,
                }
            )
            report = self.runner.run(cfg, heat_profile, site_profile, prices)
            opt_m = report.strategies["optimized"]
            results.append(
                SensitivityCaseResult(
                    parameter_name="standing_loss_percent_per_day",
                    scenario_name=f"{name} [standing-loss assumption]",
                    parameter_value=loss_pct,
                    total_cost_eur=opt_m.total_cost_eur,
                    savings_vs_direct_eur=opt_m.savings_vs_direct_eur,
                    savings_vs_direct_percent=opt_m.savings_vs_direct_percent,
                    heat_supply_reliability_percent=opt_m.heat_supply_reliability_percent,
                    equivalent_cycles=opt_m.equivalent_discharge_cycles,
                    standing_losses_kwh=opt_m.thermal_standing_loss_kwh,
                    total_electricity_kwh=opt_m.electricity_kwh,
                    total_useful_heat_kwh=opt_m.useful_heat_kwh,
                    cost_eur_per_mwh_heat=opt_m.cost_eur_per_mwh_heat,
                )
            )
        return results

    def sweep_efficiencies(
        self,
        base_config: BacktestConfig,
        heat_profile: PowerProfile,
        site_profile: PowerProfile,
        prices: list[PricePoint],
        efficiency_pairs: list[tuple[str, float, float]] | None = None,
    ) -> list[SensitivityCaseResult]:
        """Sweep round-trip efficiency pairs (eta_c, eta_d).
        
        Labels explicitly as provisional engineering assumptions.
        """
        pairs = efficiency_pairs or [
            ("Low (eta_c=0.90, eta_d=0.85, RTE=76.5%)", 0.90, 0.85),
            ("Nominal (eta_c=0.95, eta_d=0.90, RTE=85.5%)", 0.95, 0.90),
            ("High (eta_c=0.98, eta_d=0.95, RTE=93.1%)", 0.98, 0.95),
        ]

        results: list[SensitivityCaseResult] = []
        for name, eta_c, eta_d in pairs:
            new_tes = base_config.tes_params.model_copy(
                update={"charge_efficiency": eta_c, "discharge_efficiency": eta_d}
            )
            cfg = BacktestConfig(
                **{
                    **base_config.__dict__,
                    "tes_params": new_tes,
                }
            )
            report = self.runner.run(cfg, heat_profile, site_profile, prices)
            opt_m = report.strategies["optimized"]
            results.append(
                SensitivityCaseResult(
                    parameter_name="round_trip_efficiency",
                    scenario_name=f"{name} [provisional assumption]",
                    parameter_value=round(eta_c * eta_d, 4),
                    total_cost_eur=opt_m.total_cost_eur,
                    savings_vs_direct_eur=opt_m.savings_vs_direct_eur,
                    savings_vs_direct_percent=opt_m.savings_vs_direct_percent,
                    heat_supply_reliability_percent=opt_m.heat_supply_reliability_percent,
                    equivalent_cycles=opt_m.equivalent_discharge_cycles,
                    standing_losses_kwh=opt_m.thermal_standing_loss_kwh,
                    total_electricity_kwh=opt_m.electricity_kwh,
                    total_useful_heat_kwh=opt_m.useful_heat_kwh,
                    cost_eur_per_mwh_heat=opt_m.cost_eur_per_mwh_heat,
                )
            )
        return results

    def sweep_auxiliary_power(
        self,
        base_config: BacktestConfig,
        heat_profile: PowerProfile,
        site_profile: PowerProfile,
        prices: list[PricePoint],
        aux_powers_kw: list[float] | None = None,
    ) -> list[SensitivityCaseResult]:
        """Sweep TES-specific continuous auxiliary electrical load (blowers/PLC)."""
        powers = aux_powers_kw or [0.0, 0.05, 0.10, 0.20]
        results: list[SensitivityCaseResult] = []
        for p_kw in powers:
            new_tes = base_config.tes_params.model_copy(update={"auxiliary_power_kw": p_kw})
            cfg = BacktestConfig(
                **{
                    **base_config.__dict__,
                    "tes_params": new_tes,
                    "auxiliary_power_kw": p_kw,
                }
            )
            report = self.runner.run(cfg, heat_profile, site_profile, prices)
            opt_m = report.strategies["optimized"]
            results.append(
                SensitivityCaseResult(
                    parameter_name="auxiliary_power_kw",
                    scenario_name=f"{int(p_kw*1000)} W Continuous Auxiliary",
                    parameter_value=p_kw,
                    total_cost_eur=opt_m.total_cost_eur,
                    savings_vs_direct_eur=opt_m.savings_vs_direct_eur,
                    savings_vs_direct_percent=opt_m.savings_vs_direct_percent,
                    heat_supply_reliability_percent=opt_m.heat_supply_reliability_percent,
                    equivalent_cycles=opt_m.equivalent_discharge_cycles,
                    standing_losses_kwh=opt_m.thermal_standing_loss_kwh,
                    total_electricity_kwh=opt_m.electricity_kwh + opt_m.tes_auxiliary_kwh,
                    total_useful_heat_kwh=opt_m.useful_heat_kwh,
                    cost_eur_per_mwh_heat=opt_m.cost_eur_per_mwh_heat,
                )
            )
        return results

    def sweep_heat_demands(
        self,
        base_config: BacktestConfig,
        site_profile: PowerProfile,
        prices: list[PricePoint],
        scenarios: list[str] | None = None,
    ) -> list[SensitivityCaseResult]:
        """Sweep configurable process heat-demand scenarios."""
        scen_list = scenarios or ["1.0kw_constant", "1.5kw_constant", "2.0kw_constant", "example_process_profile"]
        results: list[SensitivityCaseResult] = []
        for s_name in scen_list:
            profile = get_heat_demand_scenario(s_name, self.tz)
            cfg = BacktestConfig(
                **{
                    **base_config.__dict__,
                    "heat_demand_profile_name": s_name,
                }
            )
            report = self.runner.run(cfg, profile, site_profile, prices)
            opt_m = report.strategies["optimized"]
            results.append(
                SensitivityCaseResult(
                    parameter_name="heat_demand_scenario",
                    scenario_name=f"{s_name} [scenario assumption]",
                    parameter_value=0.0,
                    total_cost_eur=opt_m.total_cost_eur,
                    savings_vs_direct_eur=opt_m.savings_vs_direct_eur,
                    savings_vs_direct_percent=opt_m.savings_vs_direct_percent,
                    heat_supply_reliability_percent=opt_m.heat_supply_reliability_percent,
                    equivalent_cycles=opt_m.equivalent_discharge_cycles,
                    standing_losses_kwh=opt_m.thermal_standing_loss_kwh,
                    total_electricity_kwh=opt_m.electricity_kwh,
                    total_useful_heat_kwh=opt_m.useful_heat_kwh,
                    cost_eur_per_mwh_heat=opt_m.cost_eur_per_mwh_heat,
                )
            )
        return results

    def sweep_u_values(
        self,
        base_config: BacktestConfig,
        heat_profile: PowerProfile,
        site_profile: PowerProfile,
        prices: list[PricePoint],
        u_values: list[float] | None = None,
    ) -> list[SensitivityCaseResult]:
        """Sweep heat exchanger overall heat transfer coefficient U [W/(m2*K)]."""
        u_list = u_values or [6.0, 9.0, 12.0]
        results: list[SensitivityCaseResult] = []
        for u_val in u_list:
            curve = get_gen0_discharge_curve(
                mode=base_config.discharge_curve_mode,
                capacity_kwh=base_config.tes_params.capacity_kwh,
                overall_u_w_m2k=u_val,
                airflow_m3_h=base_config.airflow_m3_h,
                air_inlet_temperature_c=base_config.tes_params.air_inlet_temperature_c,
                hx_area_m2=base_config.hx_area_m2,
            )
            new_tes = base_config.tes_params.model_copy(update={"overall_u_w_m2k": u_val})
            cfg = BacktestConfig(
                **{
                    **base_config.__dict__,
                    "tes_params": new_tes,
                    "overall_u_w_m2k": u_val,
                    "discharge_limit_curve": curve,
                }
            )
            report = self.runner.run(cfg, heat_profile, site_profile, prices)
            opt_m = report.strategies["optimized"]
            results.append(
                SensitivityCaseResult(
                    parameter_name="overall_u_w_m2k",
                    scenario_name=f"U={u_val} W/(m2*K) [HX assumption]",
                    parameter_value=u_val,
                    total_cost_eur=opt_m.total_cost_eur,
                    savings_vs_direct_eur=opt_m.savings_vs_direct_eur,
                    savings_vs_direct_percent=opt_m.savings_vs_direct_percent,
                    heat_supply_reliability_percent=opt_m.heat_supply_reliability_percent,
                    equivalent_cycles=opt_m.equivalent_discharge_cycles,
                    standing_losses_kwh=opt_m.thermal_standing_loss_kwh,
                    total_electricity_kwh=opt_m.electricity_kwh,
                    total_useful_heat_kwh=opt_m.useful_heat_kwh,
                    cost_eur_per_mwh_heat=opt_m.cost_eur_per_mwh_heat,
                )
            )
        return results

    def sweep_airflows(
        self,
        base_config: BacktestConfig,
        heat_profile: PowerProfile,
        site_profile: PowerProfile,
        prices: list[PricePoint],
        airflows_m3_h: list[float] | None = None,
    ) -> list[SensitivityCaseResult]:
        """Sweep heat exchanger airflow rates [m3/h]."""
        af_list = airflows_m3_h or [60.0, 80.0, 100.0]
        results: list[SensitivityCaseResult] = []
        for af in af_list:
            curve = get_gen0_discharge_curve(
                mode=base_config.discharge_curve_mode,
                capacity_kwh=base_config.tes_params.capacity_kwh,
                overall_u_w_m2k=base_config.overall_u_w_m2k,
                airflow_m3_h=af,
                air_inlet_temperature_c=base_config.tes_params.air_inlet_temperature_c,
                hx_area_m2=base_config.hx_area_m2,
            )
            new_tes = base_config.tes_params.model_copy(update={"airflow_m3_h": af})
            cfg = BacktestConfig(
                **{
                    **base_config.__dict__,
                    "tes_params": new_tes,
                    "airflow_m3_h": af,
                    "discharge_limit_curve": curve,
                }
            )
            report = self.runner.run(cfg, heat_profile, site_profile, prices)
            opt_m = report.strategies["optimized"]
            results.append(
                SensitivityCaseResult(
                    parameter_name="airflow_m3_h",
                    scenario_name=f"Airflow={af} m3/h [HX assumption]",
                    parameter_value=af,
                    total_cost_eur=opt_m.total_cost_eur,
                    savings_vs_direct_eur=opt_m.savings_vs_direct_eur,
                    savings_vs_direct_percent=opt_m.savings_vs_direct_percent,
                    heat_supply_reliability_percent=opt_m.heat_supply_reliability_percent,
                    equivalent_cycles=opt_m.equivalent_discharge_cycles,
                    standing_losses_kwh=opt_m.thermal_standing_loss_kwh,
                    total_electricity_kwh=opt_m.electricity_kwh,
                    total_useful_heat_kwh=opt_m.useful_heat_kwh,
                    cost_eur_per_mwh_heat=opt_m.cost_eur_per_mwh_heat,
                )
            )
        return results


class SizingStudyAnalyzer:
    """Performs systematic TES sizing sweeps across storage capacity and heater charging power."""

    def __init__(self, tz: ZoneInfo) -> None:
        self.tz = tz
        self.runner = BacktestRunner(tz=tz)

    def run_sizing_sweep(
        self,
        base_config: BacktestConfig,
        heat_profile: PowerProfile,
        site_profile: PowerProfile,
        prices: list[PricePoint],
        capacities_kwh: list[float] | None = None,
        charge_powers_kw: list[float] | None = None,
        sizing_mode: Literal["fixed_gen0_hx", "scaled_hx_benchmark", "both", "normalized_benchmark"] = "both",
    ) -> dict[str, SizingStudyReport]:
        """Run parametric sizing matrix over capacities and charging powers.
        
        Fairness rules (Section 9):
          - Mode A (fixed_gen0_hx): Fixed physical Gen0 heat exchanger (1.55 m2 coil)
            with same instantaneous Pmax at same normalized SOC% across capacities.
          - Mode B (scaled_hx_benchmark): Scaled HX benchmark where HX area scales
            proportionally with storage capacity: A_HX = A_ref * (capacity / 15.0).
        """
        caps = capacities_kwh or [10.0, 15.0, 20.0, 30.0]
        powers = charge_powers_kw or [6.0, 9.0, 12.0]

        if sizing_mode == "both":
            modes = ["fixed_gen0_hx", "scaled_hx_benchmark"]
        else:
            modes = [sizing_mode]

        reports: dict[str, SizingStudyReport] = {}

        for m in modes:
            combos: list[SizingCombinationResult] = []
            for cap in caps:
                for p_chg in powers:
                    curve = get_gen0_discharge_curve(
                        mode=m,
                        capacity_kwh=cap,
                        reference_capacity_kwh=15.0,
                        overall_u_w_m2k=base_config.overall_u_w_m2k,
                        airflow_m3_h=base_config.airflow_m3_h,
                        air_inlet_temperature_c=base_config.tes_params.air_inlet_temperature_c,
                        hx_area_m2=base_config.hx_area_m2,
                    )

                    new_tes = base_config.tes_params.model_copy(
                        update={
                            "capacity_kwh": cap,
                            "max_charge_power_kw": p_chg,
                            "overall_u_w_m2k": base_config.overall_u_w_m2k,
                            "airflow_m3_h": base_config.airflow_m3_h,
                            "hx_area_m2": base_config.hx_area_m2,
                        }
                    )

                    # Initial SOC at 50% capacity
                    init_soc = cap * 0.50

                    cfg = BacktestConfig(
                        **{
                            **base_config.__dict__,
                            "initial_soc_kwh": init_soc,
                            "tes_params": new_tes,
                            "discharge_limit_curve": curve,
                            "discharge_curve_mode": m,
                            "rolling_terminal_soc_mode": "hold_initial",
                        }
                    )

                    report = self.runner.run(cfg, heat_profile, site_profile, prices)
                    opt_m = report.strategies["optimized"]

                    combos.append(
                        SizingCombinationResult(
                            capacity_kwh=cap,
                            charge_power_kw=p_chg,
                            sizing_mode=m,
                            useful_heat_kwh=opt_m.useful_heat_kwh,
                            heat_supply_reliability_percent=opt_m.heat_supply_reliability_percent,
                            electricity_consumed_kwh=opt_m.electricity_kwh + opt_m.tes_auxiliary_kwh,
                            average_paid_electricity_price=opt_m.average_paid_electricity_price,
                            total_cost_eur=opt_m.total_cost_eur,
                            cost_eur_per_mwh_heat=opt_m.cost_eur_per_mwh_heat,
                            savings_vs_direct_eur=opt_m.savings_vs_direct_eur,
                            savings_vs_direct_percent=opt_m.savings_vs_direct_percent,
                            equivalent_cycles=opt_m.equivalent_discharge_cycles,
                            average_soc_kwh=opt_m.average_soc_kwh,
                            average_soc_percent=opt_m.average_soc_percent,
                            minimum_soc_kwh=opt_m.minimum_soc_kwh,
                            maximum_soc_kwh=opt_m.maximum_soc_kwh,
                            charging_hours=opt_m.charging_hours,
                            hours_charge_off_during_high_price_periods=opt_m.hours_charge_off_during_high_price_periods,
                            energy_balance_residual_kwh=opt_m.energy_balance_residual_kwh,
                            sand_mass_kg=opt_m.sand_mass_kg,
                            hx_area_m2=opt_m.hx_area_m2,
                            overall_u_w_m2k=opt_m.overall_u_w_m2k,
                            airflow_m3_h=opt_m.airflow_m3_h,
                            unmet_heat_kwh=opt_m.unmet_heat_kwh,
                            standing_losses_kwh=opt_m.thermal_standing_loss_kwh,
                            conversion_losses_kwh=opt_m.conversion_loss_kwh,
                            aux_electricity_kwh=opt_m.tes_auxiliary_kwh,
                            configured_charge_power_limit_kw=opt_m.configured_charge_power_limit_kw,
                            peak_actual_charge_power_kw=opt_m.peak_actual_charge_power_kw,
                            grid_connection_limit_kw=opt_m.grid_connection_limit_kw,
                            peak_total_grid_power_kw=opt_m.peak_total_grid_power_kw,
                        )
                    )

            # Sort by capacity then charge power
            combos.sort(key=lambda c: (c.capacity_kwh, c.charge_power_kw))

            # Identify benchmark indicators (Section 16)
            lowest_cost = min(combos, key=lambda c: c.cost_eur_per_mwh_heat)
            highest_rel = max(combos, key=lambda c: c.heat_supply_reliability_percent)
            reliable_combos = [c for c in combos if c.heat_supply_reliability_percent >= 99.5]
            lowest_cap_rel = (
                min(reliable_combos, key=lambda c: (c.capacity_kwh, c.charge_power_kw))
                if reliable_combos
                else None
            )

            reports[m] = SizingStudyReport(
                combinations=combos,
                lowest_cost_combination=lowest_cost,
                highest_reliability_combination=highest_rel,
                lowest_capacity_meeting_99_5_rel=lowest_cap_rel,
            )

        return reports
