"""Backtesting engine for multi-day historical evaluation of TES operation.

Supports:
1. Mode A: realistic_rolling (receding-horizon execution preserving continuous SOC).
2. Mode B: perfect_foresight (theoretical upper-bound benchmark).
3. Fair baseline comparison against direct resistive electric heating and cheapest-N heuristic.
4. Terminal inventory normalization for fair economic comparisons.
5. Physical Gen0 viability gating.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from app.backtest.domain import (
    BacktestConfig,
    BacktestReport,
    BacktestStrategyMetrics,
)
from app.backtest.market_info import MarketInformationConfig, MarketInformationModel
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import Interval, local_day_bounds_utc, local_day_intervals
from app.economics.tariff import effective_price_eur_mwh
from app.models.domain import PricePoint
from app.optimization.domain import (
    OptimizationIntervalInput,
    OptimizationProblemInput,
)
from app.optimization.lp_optimizer import LPOptimizer
from app.process.profiles import PowerProfile
from app.providers.price_provider import PriceProvider
from app.tes.model import (
    ConstantDischargeLimit,
    DischargeLimitCurve,
    PiecewiseLinearDischargeLimit,
    VirtualTES,
    retention_factor,
)
from app.tes.thermal import ThermalStateMapper

log = logging.getLogger(__name__)


def get_viability_status_label(config: BacktestConfig, mode: str, is_viable: bool) -> str:
    """Classify backtest outcome per Phase 5.5 Section 15 rules."""
    if mode == "perfect_foresight":
        return "THEORETICAL BENCHMARK -- ASSUMES PERFECT FUTURE KNOWLEDGE"
    if mode == "baseline_direct":
        return "BENCHMARK -- DIRECT RESISTIVE ELECTRIC HEATING"
    if mode == "baseline_heuristic":
        return "HEURISTIC BENCHMARK -- CHEAPEST-N HOURS"
    if not is_viable or config.price_source.lower() == "mock":
        return "SOFTWARE TEST -- SIMPLIFIED ASSUMPTIONS"
    return "ENGINEERING ESTIMATE -- NOT YET CALIBRATED TO PHYSICAL GEN0"


def evaluate_direct_electric_heating(
    intervals: list[Interval],
    prices: list[PricePoint],
    heat_demands: list[float],
    tariff: TariffParameters,
    heater_efficiency: float = 0.98,
) -> BacktestStrategyMetrics:
    """Evaluate direct resistive heating baseline (no thermal storage)."""
    total_elec_kwh = 0.0
    total_cost_eur = 0.0
    total_heat_kwh = 0.0
    price_weighted_sum = 0.0

    for iv, p, hd in zip(intervals, prices, heat_demands, strict=True):
        dt = iv.duration_h
        eff_price = effective_price_eur_mwh(p.price_eur_mwh, tariff)
        elec_kw = hd / heater_efficiency
        elec_kwh = elec_kw * dt
        cost = (eff_price * elec_kwh) / 1000.0

        total_elec_kwh += elec_kwh
        total_cost_eur += cost
        total_heat_kwh += hd * dt
        price_weighted_sum += eff_price * elec_kwh

    avg_price = (price_weighted_sum / total_elec_kwh) if total_elec_kwh > 0 else 0.0
    cost_per_mwh = (total_cost_eur / (total_heat_kwh / 1000.0)) if total_heat_kwh > 0 else 0.0

    # Conversion losses: (1 - eta_heater) * elec_kwh
    conversion_losses = (1.0 - heater_efficiency) * total_elec_kwh

    return BacktestStrategyMetrics(
        name="Direct Electric Heating",
        mode="baseline_direct",
        electricity_kwh=total_elec_kwh,
        useful_heat_kwh=total_heat_kwh,
        unmet_heat_kwh=0.0,
        heat_supply_reliability_percent=100.0,
        tes_auxiliary_kwh=0.0,
        thermal_standing_loss_kwh=0.0,
        conversion_loss_kwh=conversion_losses,
        average_paid_electricity_price=avg_price,
        raw_electricity_cost_eur=total_cost_eur,
        terminal_inventory_adjustment_eur=0.0,
        unmet_heat_cost_eur=0.0,
        total_cost_eur=total_cost_eur,
        cost_eur_per_mwh_heat=cost_per_mwh,
        savings_vs_direct_eur=0.0,
        savings_vs_direct_percent=0.0,
        equivalent_discharge_cycles=0.0,
        average_soc_kwh=0.0,
        average_soc_percent=0.0,
        minimum_soc_kwh=0.0,
        minimum_soc_percent=0.0,
        maximum_soc_kwh=0.0,
        maximum_soc_percent=0.0,
        hours_charge_off_during_high_price_periods=0.0,
        charging_hours=0.0,
        final_soc_kwh=0.0,
        energy_balance_residual_kwh=0.0,
        is_physical_gen0_viable=True,
        status_label="BENCHMARK -- DIRECT RESISTIVE ELECTRIC HEATING",
        viability_caveats=[],
        thermal_capacity_full_span_kwh=0.0,
        dispatchable_capacity_kwh=0.0,
        optimizer_soc_min_fraction=0.0,
        optimizer_soc_max_fraction=0.0,
        temperature_at_optimizer_min_soc_c=0.0,
        sand_mass_kg=None,
        configured_charge_power_limit_kw=0.0,
        peak_actual_charge_power_kw=0.0,
        grid_connection_limit_kw=12.0,
        peak_total_grid_power_kw=max((hd / heater_efficiency for hd in heat_demands), default=0.0),
    )


def evaluate_cheapest_n_heuristic(
    intervals: list[Interval],
    prices: list[PricePoint],
    heat_demands: list[float],
    site_loads: list[float],
    config: BacktestConfig,
    tz: ZoneInfo,
) -> BacktestStrategyMetrics:
    """Evaluate simple cheapest-N hours heuristic with continuous physical simulation."""
    tes = VirtualTES(
        params=config.tes_params,
        initial_soc_kwh=config.initial_soc_kwh,
        discharge_limit_curve=config.discharge_limit_curve,
    )

    # Group intervals by local calendar day to identify daily cheapest hours
    day_indices: dict[date, list[int]] = {}
    for idx, iv in enumerate(intervals):
        loc_d = iv.start_local(tz).date()
        day_indices.setdefault(loc_d, []).append(idx)

    # For each day, rank by effective price and pick cheapest N hours
    cheap_set: set[int] = set()
    for d, idxs in day_indices.items():
        # Number of intervals to pick: cheapest_n_hours * (intervals per hour)
        n_pts = int(config.cheapest_n_hours * (60 / (intervals[0].duration_h * 60)))
        ranked = sorted(idxs, key=lambda i: prices[i].price_eur_mwh)
        for i in ranked[:n_pts]:
            cheap_set.add(i)

    # High price threshold (75th percentile) for metrics
    sorted_spot = sorted(p.price_eur_mwh for p in prices)
    p75_idx = int(0.75 * len(sorted_spot))
    high_price_threshold = sorted_spot[min(p75_idx, len(sorted_spot) - 1)]

    total_charge_kwh = 0.0
    total_heat_delivered_kwh = 0.0
    total_unmet_heat_kwh = 0.0
    total_aux_kwh = 0.0
    total_standing_losses_kwh = 0.0
    total_charge_losses_kwh = 0.0
    total_discharge_losses_kwh = 0.0
    total_raw_cost_eur = 0.0
    total_unmet_heat_cost_eur = 0.0
    charging_hours = 0.0
    hours_charge_off_high_price = 0.0

    soc_history: list[float] = [config.initial_soc_kwh]
    actual_charge_powers: list[float] = []
    total_grid_powers: list[float] = []

    for idx, (iv, p, hd, sl) in enumerate(zip(intervals, prices, heat_demands, site_loads, strict=True)):
        dt = iv.duration_h
        eff_price = effective_price_eur_mwh(p.price_eur_mwh, config.tariff_params)

        # Auxiliary power
        aux_kw = config.auxiliary_power_kw
        aux_kwh = aux_kw * dt
        total_aux_kwh += aux_kwh

        # Available grid headroom for charging
        grid_headroom = max(0.0, config.site_params.grid_connection_limit_kw - sl - aux_kw)

        # Setpoint
        if idx in cheap_set:
            req_charge = min(config.tes_params.max_charge_power_kw, grid_headroom)
        else:
            req_charge = 0.0

        step_res = tes.step(
            interval_start_utc=iv.start_utc,
            dt_h=dt,
            requested_charge_kw=req_charge,
            requested_discharge_kw=hd,
            external_charge_limit_kw=grid_headroom,
        )

        unmet = max(0.0, hd - step_res.discharge_power_kw)
        if config.backup_heat_enabled:
            b_heat = min(unmet, config.backup_heater_max_power_kw)
            unmet = max(0.0, unmet - b_heat)
            b_elec_kwh = (b_heat / config.backup_heater_efficiency) * dt
            b_cost = (eff_price * b_elec_kwh) / 1000.0
            b_elec_kw = b_heat / config.backup_heater_efficiency
        else:
            b_cost = 0.0
            b_elec_kw = 0.0

        charge_kwh = step_res.charge_power_kw * dt
        cost = (eff_price * (charge_kwh + aux_kwh)) / 1000.0 + b_cost

        total_charge_kwh += charge_kwh
        total_heat_delivered_kwh += step_res.discharge_power_kw * dt
        total_unmet_heat_kwh += unmet * dt
        total_unmet_heat_cost_eur += b_cost
        total_standing_losses_kwh += step_res.storage_loss_kwh
        total_charge_losses_kwh += step_res.charge_conversion_loss_kwh
        total_discharge_losses_kwh += step_res.discharge_conversion_loss_kwh
        total_raw_cost_eur += cost

        actual_charge_powers.append(step_res.charge_power_kw)
        total_grid_powers.append(step_res.charge_power_kw + sl + aux_kw + b_elec_kw)

        if step_res.charge_power_kw > 0.01:
            charging_hours += dt

        if p.price_eur_mwh >= high_price_threshold and step_res.charge_power_kw < 0.01:
            hours_charge_off_high_price += dt

        soc_history.append(step_res.soc_end_kwh)

    final_soc = tes.soc_kwh
    delta_soc = final_soc - config.initial_soc_kwh

    # Terminal inventory adjustment based on tariff-inclusive effective price
    mean_eff_price = sum(effective_price_eur_mwh(p.price_eur_mwh, config.tariff_params) for p in prices) / len(prices)
    if delta_soc < -1e-6:
        # Penalty: must buy back missing energy
        adjustment_eur = (-delta_soc / config.tes_params.charge_efficiency) * (mean_eff_price / 1000.0)
    elif delta_soc > 1e-6:
        # Credit: surplus useful energy left in storage
        adjustment_eur = -(delta_soc * config.tes_params.discharge_efficiency) * (mean_eff_price / 1000.0)
    else:
        adjustment_eur = 0.0

    total_cost_eur = total_raw_cost_eur + adjustment_eur + total_unmet_heat_cost_eur

    # Metrics
    total_demand_kwh = sum(hd * iv.duration_h for hd, iv in zip(heat_demands, intervals))
    reliability = (
        100.0 * (1.0 - total_unmet_heat_kwh / total_demand_kwh) if total_demand_kwh > 0 else 100.0
    )
    cycles = (
        total_heat_delivered_kwh / (config.tes_params.capacity_kwh * config.tes_params.discharge_efficiency)
        if config.tes_params.capacity_kwh > 0
        else 0.0
    )
    cost_per_mwh = (total_cost_eur / (total_heat_delivered_kwh / 1000.0)) if total_heat_delivered_kwh > 0 else 0.0

    # Energy balance check
    e_in = total_charge_kwh
    e_out = (
        total_heat_delivered_kwh
        + delta_soc
        + total_standing_losses_kwh
        + total_charge_losses_kwh
        + total_discharge_losses_kwh
    )
    residual = e_in - e_out

    # Viability check
    is_viable, caveats = check_physical_viability(config)

    avg_paid_price = (
        (total_raw_cost_eur / (total_charge_kwh + total_aux_kwh)) * 1000.0
        if (total_charge_kwh + total_aux_kwh) > 0
        else 0.0
    )

    cap = config.tes_params.capacity_kwh
    mapper = ThermalStateMapper(
        t_min_c=config.tes_params.physical_temperature_min_c,
        t_max_c=config.tes_params.physical_temperature_max_c,
    )
    sand_mass_kg = config.sand_mass_kg or mapper.equivalent_sand_mass_for_capacity(config.tes_params.thermal_capacity_full_span_kwh)

    return BacktestStrategyMetrics(
        name=f"Cheapest {config.cheapest_n_hours}h Heuristic",
        mode="baseline_heuristic",
        electricity_kwh=total_charge_kwh,
        useful_heat_kwh=total_heat_delivered_kwh,
        unmet_heat_kwh=total_unmet_heat_kwh,
        heat_supply_reliability_percent=reliability,
        tes_auxiliary_kwh=total_aux_kwh,
        thermal_standing_loss_kwh=total_standing_losses_kwh,
        conversion_loss_kwh=total_charge_losses_kwh + total_discharge_losses_kwh,
        average_paid_electricity_price=avg_paid_price,
        raw_electricity_cost_eur=total_raw_cost_eur,
        terminal_inventory_adjustment_eur=adjustment_eur,
        unmet_heat_cost_eur=total_unmet_heat_cost_eur,
        total_cost_eur=total_cost_eur,
        cost_eur_per_mwh_heat=cost_per_mwh,
        savings_vs_direct_eur=0.0,  # filled later in report
        savings_vs_direct_percent=0.0,
        equivalent_discharge_cycles=cycles,
        average_soc_kwh=sum(soc_history) / len(soc_history),
        average_soc_percent=(sum(soc_history) / len(soc_history)) / cap * 100.0 if cap > 0 else 0.0,
        minimum_soc_kwh=min(soc_history),
        minimum_soc_percent=min(soc_history) / cap * 100.0 if cap > 0 else 0.0,
        maximum_soc_kwh=max(soc_history),
        maximum_soc_percent=max(soc_history) / cap * 100.0 if cap > 0 else 0.0,
        hours_charge_off_during_high_price_periods=hours_charge_off_high_price,
        charging_hours=charging_hours,
        final_soc_kwh=final_soc,
        energy_balance_residual_kwh=residual,
        is_physical_gen0_viable=is_viable,
        status_label=get_viability_status_label(config, "baseline_heuristic", is_viable),
        viability_caveats=caveats,
        sand_mass_kg=sand_mass_kg,
        thermal_capacity_full_span_kwh=config.tes_params.thermal_capacity_full_span_kwh,
        dispatchable_capacity_kwh=config.tes_params.dispatchable_capacity_kwh,
        optimizer_soc_min_fraction=config.tes_params.optimizer_soc_min_fraction,
        optimizer_soc_max_fraction=config.tes_params.optimizer_soc_max_fraction,
        temperature_at_optimizer_min_soc_c=config.tes_params.temperature_at_optimizer_min_soc_c,
        configured_charge_power_limit_kw=config.tes_params.max_charge_power_kw,
        peak_actual_charge_power_kw=max(actual_charge_powers, default=0.0),
        grid_connection_limit_kw=config.site_params.grid_connection_limit_kw,
        peak_total_grid_power_kw=max(total_grid_powers, default=0.0),
    )


def evaluate_realistic_rolling_lp(
    intervals: list[Interval],
    prices: list[PricePoint],
    heat_demands: list[float],
    site_loads: list[float],
    config: BacktestConfig,
    tz: ZoneInfo,
) -> BacktestStrategyMetrics:
    """Mode A: True receding-horizon market operation preserving continuous SOC.

    Operates strictly on electricity-price information that would actually be
    available at each decision time, using MarketInformationModel.
    """
    tes = VirtualTES(
        params=config.tes_params,
        initial_soc_kwh=config.initial_soc_kwh,
        discharge_limit_curve=config.discharge_limit_curve,
    )
    opt = LPOptimizer()
    market_model = MarketInformationModel(
        MarketInformationConfig(
            day_ahead_assumed_available_local_time=config.day_ahead_assumed_available_local_time,
            market_timezone=config.market_timezone,
        )
    )

    total_charge_kwh = 0.0
    total_heat_delivered_kwh = 0.0
    total_unmet_heat_kwh = 0.0
    total_aux_kwh = 0.0
    total_standing_losses_kwh = 0.0
    total_charge_losses_kwh = 0.0
    total_discharge_losses_kwh = 0.0
    total_raw_cost_eur = 0.0
    total_backup_heat_kwh = 0.0
    total_backup_elec_kwh = 0.0
    total_backup_cost_eur = 0.0
    charging_hours = 0.0
    hours_charge_off_high_price = 0.0

    sorted_spot = sorted(p.price_eur_mwh for p in prices)
    p75_idx = int(0.75 * len(sorted_spot))
    high_price_threshold = sorted_spot[min(p75_idx, len(sorted_spot) - 1)]

    soc_history: list[float] = [config.initial_soc_kwh]
    actual_charge_powers: list[float] = []
    total_grid_powers: list[float] = []
    price_by_start = {p.delivery_start_utc: p for p in prices}

    cur_idx = 0
    total_intervals_count = len(intervals)

    while cur_idx < total_intervals_count:
        cur_time = intervals[cur_idx].start_utc
        # 1. Identify currently known prices as of cur_time
        known_prices = market_model.get_known_prices(prices, cur_time)
        known_starts = {p.delivery_start_utc for p in known_prices}

        # 2. Build optimization horizon starting from cur_idx as far as prices are known
        h_end = cur_idx
        while h_end < total_intervals_count and intervals[h_end].start_utc in known_starts:
            h_end += 1

        if h_end == cur_idx:
            # Fallback if start interval price availability timing is at or after start
            h_end = cur_idx + 1

        horizon_indices = list(range(cur_idx, h_end))

        # 3. Determine execution block: run only until next market information event arrives
        market_events = market_model.get_market_events(prices, cur_time, intervals[-1].end_utc)
        future_events = [e for e in market_events if e > cur_time]
        if future_events:
            next_event = future_events[0]
            exec_indices = [i for i in horizon_indices if intervals[i].start_utc < next_event]
            if not exec_indices:
                exec_indices = [cur_idx]
        else:
            exec_indices = list(horizon_indices)

        # 4. Terminal SOC condition: only on the very last horizon of the entire backtest!
        is_final_horizon = (horizon_indices[-1] == total_intervals_count - 1)
        if is_final_horizon and config.rolling_terminal_soc_mode == "hold_initial":
            target_terminal_soc_kwh = config.initial_soc_kwh
            terminal_soc_condition = "exact"
        else:
            target_terminal_soc_kwh = config.tes_params.soc_min_kwh
            terminal_soc_condition = "min"

        # Build optimization problem
        h_intervals = [intervals[i] for i in horizon_indices]
        h_heat = [heat_demands[i] for i in horizon_indices]
        h_site = [site_loads[i] for i in horizon_indices]
        h_prices = [price_by_start[intervals[i].start_utc] for i in horizon_indices]

        opt_inputs: list[OptimizationIntervalInput] = []
        for iv, p, hd, sl in zip(h_intervals, h_prices, h_heat, h_site, strict=True):
            eff_price = effective_price_eur_mwh(p.price_eur_mwh, config.tariff_params)
            opt_inputs.append(
                OptimizationIntervalInput(
                    start_utc=iv.start_utc,
                    end_utc=iv.end_utc,
                    spot_price_eur_mwh=p.price_eur_mwh,
                    effective_price_eur_mwh=eff_price,
                    heat_demand_kw=hd,
                    other_site_load_kw=sl,
                    auxiliary_load_kw=config.auxiliary_power_kw,
                )
            )

        problem = OptimizationProblemInput(
            intervals=opt_inputs,
            tes_params=config.tes_params,
            grid_connection_limit_kw=config.site_params.grid_connection_limit_kw,
            initial_soc_kwh=tes.soc_kwh,
            target_terminal_soc_kwh=target_terminal_soc_kwh,
            terminal_soc_condition=terminal_soc_condition,
            discharge_limit_curve=config.discharge_limit_curve,
            lexicographic=True,
            backup_heat_enabled=config.backup_heat_enabled,
            backup_heater_efficiency=config.backup_heater_efficiency,
            backup_heater_max_power_kw=config.backup_heater_max_power_kw,
        )

        res = opt.optimize(problem)

        # 5. Execute scheduled dispatch ONLY for exec_indices
        for i in exec_indices:
            offset = i - cur_idx
            iv_res = res.intervals[offset]
            iv = intervals[i]
            p = price_by_start[iv.start_utc]
            hd = heat_demands[i]
            sl = site_loads[i]

            dt = iv.duration_h
            eff_price = iv_res.effective_price_eur_mwh
            aux_kw = config.auxiliary_power_kw
            aux_kwh = aux_kw * dt
            total_aux_kwh += aux_kwh

            grid_headroom = max(0.0, config.site_params.grid_connection_limit_kw - sl - aux_kw)

            step_res = tes.step(
                interval_start_utc=iv.start_utc,
                dt_h=dt,
                requested_charge_kw=iv_res.charge_power_kw,
                requested_discharge_kw=iv_res.discharge_power_kw,
                external_charge_limit_kw=grid_headroom,
            )

            unmet = max(0.0, hd - step_res.discharge_power_kw)
            if config.backup_heat_enabled:
                b_heat = min(unmet, config.backup_heater_max_power_kw)
                unmet = max(0.0, unmet - b_heat)
                b_elec = (b_heat / config.backup_heater_efficiency) * dt
                b_cost = (eff_price * b_elec) / 1000.0
                total_backup_heat_kwh += b_heat * dt
                total_backup_elec_kwh += b_elec
                total_backup_cost_eur += b_cost
            else:
                b_cost = 0.0

            charge_kwh = step_res.charge_power_kw * dt
            charge_cost = (eff_price * (charge_kwh + aux_kwh)) / 1000.0

            total_charge_kwh += charge_kwh
            total_heat_delivered_kwh += step_res.discharge_power_kw * dt
            total_unmet_heat_kwh += unmet * dt
            total_standing_losses_kwh += step_res.storage_loss_kwh
            total_charge_losses_kwh += step_res.charge_conversion_loss_kwh
            total_discharge_losses_kwh += step_res.discharge_conversion_loss_kwh
            total_raw_cost_eur += charge_cost + b_cost

            b_elec_kw = b_heat / config.backup_heater_efficiency if config.backup_heat_enabled else 0.0
            actual_charge_powers.append(step_res.charge_power_kw)
            total_grid_powers.append(step_res.charge_power_kw + sl + aux_kw + b_elec_kw)

            if step_res.charge_power_kw > 0.01:
                charging_hours += dt

            if p.price_eur_mwh >= high_price_threshold and step_res.charge_power_kw < 0.01:
                hours_charge_off_high_price += dt

            soc_history.append(step_res.soc_end_kwh)

        cur_idx += len(exec_indices)

    final_soc = tes.soc_kwh
    delta_soc = final_soc - config.initial_soc_kwh

    mean_eff_price = sum(effective_price_eur_mwh(p.price_eur_mwh, config.tariff_params) for p in prices) / len(prices)
    if delta_soc < -1e-6:
        adjustment_eur = (-delta_soc / config.tes_params.charge_efficiency) * (mean_eff_price / 1000.0)
    elif delta_soc > 1e-6:
        adjustment_eur = -(delta_soc * config.tes_params.discharge_efficiency) * (mean_eff_price / 1000.0)
    else:
        adjustment_eur = 0.0

    total_cost_eur = total_raw_cost_eur + adjustment_eur

    total_demand_kwh = sum(hd * iv.duration_h for hd, iv in zip(heat_demands, intervals))
    reliability = (
        100.0 * (1.0 - total_unmet_heat_kwh / total_demand_kwh) if total_demand_kwh > 0 else 100.0
    )
    cycles = (
        total_heat_delivered_kwh / (config.tes_params.capacity_kwh * config.tes_params.discharge_efficiency)
        if config.tes_params.capacity_kwh > 0
        else 0.0
    )
    cost_per_mwh = (total_cost_eur / (total_heat_delivered_kwh / 1000.0)) if total_heat_delivered_kwh > 0 else 0.0

    e_in = total_charge_kwh
    e_out = (
        total_heat_delivered_kwh
        + delta_soc
        + total_standing_losses_kwh
        + total_charge_losses_kwh
        + total_discharge_losses_kwh
    )
    residual = e_in - e_out

    is_viable, caveats = check_physical_viability(config)
    avg_paid_price = (
        (total_raw_cost_eur / (total_charge_kwh + total_aux_kwh + total_backup_elec_kwh)) * 1000.0
        if (total_charge_kwh + total_aux_kwh + total_backup_elec_kwh) > 0
        else 0.0
    )

    mapper = ThermalStateMapper(
        t_min_c=config.tes_params.physical_temperature_min_c,
        t_max_c=config.tes_params.physical_temperature_max_c,
    )
    sand_mass_kg = config.sand_mass_kg or mapper.equivalent_sand_mass_for_capacity(config.tes_params.thermal_capacity_full_span_kwh)
    cap = config.tes_params.capacity_kwh

    return BacktestStrategyMetrics(
        name="Optimized TES (Realistic Rolling)",
        mode="realistic_rolling",
        electricity_kwh=total_charge_kwh,
        useful_heat_kwh=total_heat_delivered_kwh,
        unmet_heat_kwh=total_unmet_heat_kwh,
        heat_supply_reliability_percent=reliability,
        tes_auxiliary_kwh=total_aux_kwh,
        thermal_standing_loss_kwh=total_standing_losses_kwh,
        conversion_loss_kwh=total_charge_losses_kwh + total_discharge_losses_kwh,
        average_paid_electricity_price=avg_paid_price,
        raw_electricity_cost_eur=total_raw_cost_eur,
        terminal_inventory_adjustment_eur=adjustment_eur,
        unmet_heat_cost_eur=0.0,
        total_cost_eur=total_cost_eur,
        cost_eur_per_mwh_heat=cost_per_mwh,
        savings_vs_direct_eur=0.0,
        savings_vs_direct_percent=0.0,
        equivalent_discharge_cycles=cycles,
        average_soc_kwh=sum(soc_history) / len(soc_history),
        average_soc_percent=(sum(soc_history) / len(soc_history)) / cap * 100.0 if cap > 0 else 0.0,
        minimum_soc_kwh=min(soc_history),
        minimum_soc_percent=min(soc_history) / cap * 100.0 if cap > 0 else 0.0,
        maximum_soc_kwh=max(soc_history),
        maximum_soc_percent=max(soc_history) / cap * 100.0 if cap > 0 else 0.0,
        hours_charge_off_during_high_price_periods=hours_charge_off_high_price,
        charging_hours=charging_hours,
        final_soc_kwh=final_soc,
        energy_balance_residual_kwh=residual,
        is_physical_gen0_viable=is_viable,
        status_label=get_viability_status_label(config, "realistic_rolling", is_viable),
        viability_caveats=caveats,
        backup_heat_kwh=total_backup_heat_kwh,
        backup_electricity_kwh=total_backup_elec_kwh,
        backup_cost_eur=total_backup_cost_eur,
        sand_mass_kg=sand_mass_kg,
        hx_area_m2=config.hx_area_m2,
        overall_u_w_m2k=config.overall_u_w_m2k,
        airflow_m3_h=config.airflow_m3_h,
        objective_value=total_raw_cost_eur,
        thermal_capacity_full_span_kwh=config.tes_params.thermal_capacity_full_span_kwh,
        dispatchable_capacity_kwh=config.tes_params.dispatchable_capacity_kwh,
        optimizer_soc_min_fraction=config.tes_params.optimizer_soc_min_fraction,
        optimizer_soc_max_fraction=config.tes_params.optimizer_soc_max_fraction,
        temperature_at_optimizer_min_soc_c=config.tes_params.temperature_at_optimizer_min_soc_c,
        configured_charge_power_limit_kw=config.tes_params.max_charge_power_kw,
        peak_actual_charge_power_kw=max(actual_charge_powers, default=0.0),
        grid_connection_limit_kw=config.site_params.grid_connection_limit_kw,
        peak_total_grid_power_kw=max(total_grid_powers, default=0.0),
    )


def evaluate_perfect_foresight_lp(
    intervals: list[Interval],
    prices: list[PricePoint],
    heat_demands: list[float],
    site_loads: list[float],
    config: BacktestConfig,
) -> BacktestStrategyMetrics:
    """Mode B: Perfect foresight benchmark across the entire historical period."""
    opt_inputs: list[OptimizationIntervalInput] = []
    total_aux_kwh = sum(config.auxiliary_power_kw * iv.duration_h for iv in intervals)

    for iv, p, hd, sl in zip(intervals, prices, heat_demands, site_loads, strict=True):
        eff_price = effective_price_eur_mwh(p.price_eur_mwh, config.tariff_params)
        opt_inputs.append(
            OptimizationIntervalInput(
                start_utc=iv.start_utc,
                end_utc=iv.end_utc,
                spot_price_eur_mwh=p.price_eur_mwh,
                effective_price_eur_mwh=eff_price,
                heat_demand_kw=hd,
                other_site_load_kw=sl,
                auxiliary_load_kw=config.auxiliary_power_kw,
            )
        )

    problem = OptimizationProblemInput(
        intervals=opt_inputs,
        tes_params=config.tes_params,
        grid_connection_limit_kw=config.site_params.grid_connection_limit_kw,
        initial_soc_kwh=config.initial_soc_kwh,
        target_terminal_soc_kwh=config.initial_soc_kwh,
        terminal_soc_condition="exact",
        discharge_limit_curve=config.discharge_limit_curve,
        lexicographic=True,
        backup_heat_enabled=config.backup_heat_enabled,
        backup_heater_efficiency=config.backup_heater_efficiency,
        backup_heater_max_power_kw=config.backup_heater_max_power_kw,
    )

    opt = LPOptimizer()
    res = opt.optimize(problem)
    m = res.metrics

    # Add auxiliary energy cost
    aux_cost_eur = 0.0
    for iv, p in zip(intervals, prices, strict=True):
        eff = effective_price_eur_mwh(p.price_eur_mwh, config.tariff_params)
        aux_cost_eur += (eff * config.auxiliary_power_kw * iv.duration_h) / 1000.0

    raw_cost_eur = m.electricity_cost_eur + aux_cost_eur
    adjustment_eur = 0.0  # exact terminal condition SOC[T] == SOC[0]
    total_cost_eur = raw_cost_eur + adjustment_eur

    sorted_spot = sorted(p.price_eur_mwh for p in prices)
    p75_idx = int(0.75 * len(sorted_spot))
    high_price_threshold = sorted_spot[min(p75_idx, len(sorted_spot) - 1)]

    hours_charge_off_high_price = sum(
        iv.duration_h
        for r, iv, p in zip(res.intervals, intervals, prices, strict=True)
        if p.price_eur_mwh >= high_price_threshold and r.charge_power_kw < 0.01
    )

    soc_list = [r.soc_kwh for r in res.intervals]
    cap = config.tes_params.capacity_kwh

    is_viable, caveats = check_physical_viability(config)
    caveats.append("Theoretical upper-bound benchmark: assumes perfect visibility of future prices.")

    cycles = (
        m.useful_heat_delivered_kwh / (cap * config.tes_params.discharge_efficiency)
        if cap > 0
        else 0.0
    )
    cost_per_mwh = (total_cost_eur / (m.useful_heat_delivered_kwh / 1000.0)) if m.useful_heat_delivered_kwh > 0 else 0.0
    avg_paid_price = (
        (raw_cost_eur / (m.total_charge_kwh + total_aux_kwh + m.total_backup_electricity_kwh)) * 1000.0
        if (m.total_charge_kwh + total_aux_kwh + m.total_backup_electricity_kwh) > 0
        else 0.0
    )

    charging_hours = sum(
        iv.duration_h for r, iv in zip(res.intervals, intervals, strict=True) if r.charge_power_kw > 0.01
    )

    mapper = ThermalStateMapper(
        t_min_c=config.tes_params.physical_temperature_min_c,
        t_max_c=config.tes_params.physical_temperature_max_c,
    )
    sand_mass_kg = config.sand_mass_kg or mapper.equivalent_sand_mass_for_capacity(config.tes_params.thermal_capacity_full_span_kwh)

    return BacktestStrategyMetrics(
        name="Optimized TES (Perfect Foresight Benchmark)",
        mode="perfect_foresight",
        electricity_kwh=m.total_charge_kwh,
        useful_heat_kwh=m.useful_heat_delivered_kwh,
        unmet_heat_kwh=m.total_unmet_heat_kwh,
        heat_supply_reliability_percent=m.heat_supply_reliability_percent,
        tes_auxiliary_kwh=total_aux_kwh,
        thermal_standing_loss_kwh=m.standing_losses_kwh,
        conversion_loss_kwh=m.charge_conversion_losses_kwh + m.discharge_conversion_losses_kwh,
        average_paid_electricity_price=avg_paid_price,
        raw_electricity_cost_eur=raw_cost_eur,
        terminal_inventory_adjustment_eur=adjustment_eur,
        unmet_heat_cost_eur=0.0,
        total_cost_eur=total_cost_eur,
        cost_eur_per_mwh_heat=cost_per_mwh,
        savings_vs_direct_eur=0.0,
        savings_vs_direct_percent=0.0,
        equivalent_discharge_cycles=cycles,
        average_soc_kwh=sum(soc_list) / len(soc_list) if soc_list else 0.0,
        average_soc_percent=(sum(soc_list) / len(soc_list)) / cap * 100.0 if cap > 0 else 0.0,
        minimum_soc_kwh=min(soc_list) if soc_list else 0.0,
        minimum_soc_percent=min(soc_list) / cap * 100.0 if cap > 0 else 0.0,
        maximum_soc_kwh=max(soc_list) if soc_list else 0.0,
        maximum_soc_percent=max(soc_list) / cap * 100.0 if cap > 0 else 0.0,
        hours_charge_off_during_high_price_periods=hours_charge_off_high_price,
        charging_hours=charging_hours,
        final_soc_kwh=m.terminal_soc_kwh,
        energy_balance_residual_kwh=m.energy_balance_residual_kwh,
        is_physical_gen0_viable=is_viable,
        status_label=get_viability_status_label(config, "perfect_foresight", is_viable),
        viability_caveats=caveats,
        backup_heat_kwh=m.total_backup_heat_kwh,
        backup_electricity_kwh=m.total_backup_electricity_kwh,
        backup_cost_eur=m.total_backup_cost_eur,
        sand_mass_kg=sand_mass_kg,
        hx_area_m2=config.hx_area_m2,
        overall_u_w_m2k=config.overall_u_w_m2k,
        airflow_m3_h=config.airflow_m3_h,
        objective_value=res.objective_eur,
        thermal_capacity_full_span_kwh=config.tes_params.thermal_capacity_full_span_kwh,
        dispatchable_capacity_kwh=config.tes_params.dispatchable_capacity_kwh,
        optimizer_soc_min_fraction=config.tes_params.optimizer_soc_min_fraction,
        optimizer_soc_max_fraction=config.tes_params.optimizer_soc_max_fraction,
        temperature_at_optimizer_min_soc_c=config.tes_params.temperature_at_optimizer_min_soc_c,
        configured_charge_power_limit_kw=config.tes_params.max_charge_power_kw,
        peak_actual_charge_power_kw=max((r.charge_power_kw for r in res.intervals), default=0.0),
        grid_connection_limit_kw=config.site_params.grid_connection_limit_kw,
        peak_total_grid_power_kw=max((r.grid_power_kw for r in res.intervals), default=0.0),
    )


def check_physical_viability(config: BacktestConfig) -> tuple[bool, list[str]]:
    """Determine whether the backtest configuration qualifies as a physical Gen0 viability scenario."""
    caveats: list[str] = []
    is_viable = True

    # 1. Discharge curve check
    if isinstance(config.discharge_limit_curve, ConstantDischargeLimit):
        caveats.append("Discharge curve is constant (unrealistic for physical sand-to-air heat exchanger).")
        is_viable = False

    # 2. Standing losses check
    if config.tes_params.standing_loss_percent_per_day < 2.0 and config.tes_params.standing_loss_fixed_kw <= 0:
        caveats.append(f"Standing loss ({config.tes_params.standing_loss_percent_per_day}%/day) is optimistic.")
        is_viable = False

    # 3. Auxiliary power check
    if config.auxiliary_power_kw <= 0.0:
        caveats.append("Zero auxiliary electrical power assumed (real system requires blowers/pumps).")
        is_viable = False

    # 4. Market data check
    if config.price_source.lower() == "mock":
        caveats.append("Mock synthetic market data used instead of verified ENTSO-E market prices.")
        is_viable = False

    # 5. U-value check (Section 13)
    if config.overall_u_w_m2k >= 25.0:
        caveats.append("U=25 W/(m2*K) is a highly optimistic stress test (not validated).")
        is_viable = False

    # 6. Backup heater check (Section 25)
    if config.backup_heat_enabled:
        caveats.append("Architecture includes hypothetical electric backup heater (not present in standard Gen0).")

    return is_viable, caveats


class BacktestRunner:
    """Executes multi-day historical backtests and comparative economic evaluations."""

    def __init__(self, tz: ZoneInfo) -> None:
        self.tz = tz

    def run(
        self,
        config: BacktestConfig,
        heat_profile: PowerProfile,
        site_profile: PowerProfile,
        prices: list[PricePoint],
    ) -> BacktestReport:
        if config.end_date <= config.start_date:
            raise ValueError(f"end_date must be after start_date: {config.start_date} .. {config.end_date}")

        # Build contiguous intervals across entire backtest span
        intervals: list[Interval] = []
        cur_d = config.start_date
        while cur_d < config.end_date:
            d_ivs = local_day_intervals(cur_d, self.tz, 15)
            intervals.extend(d_ivs)
            cur_d += timedelta(days=1)

        # Map prices
        by_start = {p.delivery_start_utc: p for p in prices}
        aligned_prices: list[PricePoint] = []
        for iv in intervals:
            p = by_start.get(iv.start_utc)
            if p is None:
                # Fallback to hourly if available
                h_start = iv.start_utc.replace(minute=0, second=0, microsecond=0)
                p_h = by_start.get(h_start)
                if p_h is not None and p_h.resolution_minutes == 60:
                    p = PricePoint(
                        bidding_zone=p_h.bidding_zone,
                        delivery_start_utc=iv.start_utc,
                        delivery_end_utc=iv.end_utc,
                        price_eur_mwh=p_h.price_eur_mwh,
                        resolution_minutes=15,
                        source=p_h.source,
                        currency=p_h.currency,
                        original_resolution_minutes=60,
                        derived_from_id=p_h.id,
                    )
                else:
                    raise ValueError(f"Missing price for interval starting {iv.start_utc.isoformat()}")
            aligned_prices.append(p)

        heat_demands = heat_profile.series(intervals)
        site_loads = site_profile.series(intervals)

        # 1. Direct electric heating baseline
        direct_metrics = evaluate_direct_electric_heating(
            intervals=intervals,
            prices=aligned_prices,
            heat_demands=heat_demands,
            tariff=config.tariff_params,
            heater_efficiency=config.direct_heater_efficiency,
        )

        # 2. Heuristic cheapest-N hours baseline
        heuristic_metrics = evaluate_cheapest_n_heuristic(
            intervals=intervals,
            prices=aligned_prices,
            heat_demands=heat_demands,
            site_loads=site_loads,
            config=config,
            tz=self.tz,
        )

        # 3. Optimized TES (Mode A or Mode B)
        if config.mode == "realistic_rolling":
            opt_metrics = evaluate_realistic_rolling_lp(
                intervals=intervals,
                prices=aligned_prices,
                heat_demands=heat_demands,
                site_loads=site_loads,
                config=config,
                tz=self.tz,
            )
        else:
            opt_metrics = evaluate_perfect_foresight_lp(
                intervals=intervals,
                prices=aligned_prices,
                heat_demands=heat_demands,
                site_loads=site_loads,
                config=config,
            )

        # Compute savings relative to direct resistive heating
        c_direct = direct_metrics.total_cost_eur

        # Update heuristic savings
        h_sav_eur = c_direct - heuristic_metrics.total_cost_eur
        h_sav_pct = (h_sav_eur / c_direct * 100.0) if c_direct > 0 else 0.0
        heuristic_metrics = BacktestStrategyMetrics(
            **{
                **heuristic_metrics.__dict__,
                "savings_vs_direct_eur": h_sav_eur,
                "savings_vs_direct_percent": h_sav_pct,
            }
        )

        # Update optimizer savings
        o_sav_eur = c_direct - opt_metrics.total_cost_eur
        o_sav_pct = (o_sav_eur / c_direct * 100.0) if c_direct > 0 else 0.0
        opt_metrics = BacktestStrategyMetrics(
            **{
                **opt_metrics.__dict__,
                "savings_vs_direct_eur": o_sav_eur,
                "savings_vs_direct_percent": o_sav_pct,
            }
        )

        strategies = {
            "direct": direct_metrics,
            "heuristic": heuristic_metrics,
            "optimized": opt_metrics,
        }

        days_count = (config.end_date - config.start_date).days
        return BacktestReport(
            config=config,
            strategies=strategies,
            days_count=days_count,
            total_intervals=len(intervals),
        )
