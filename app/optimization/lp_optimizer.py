"""Linear Programming (LP) optimizer for Thermal Energy Storage.

Formulates and solves the optimal day-ahead thermal storage dispatch problem
using continuous linear programming with PuLP / CBC.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pulp

from app.optimization.domain import (
    OptimizationIntervalResult,
    OptimizationMetrics,
    OptimizationProblemInput,
    OptimizationResult,
)
from app.tes.model import retention_factor

log = logging.getLogger(__name__)


class OptimizationError(Exception):
    """Raised when optimization problem setup or solve fails."""


class LPOptimizer:
    """Linear Programming dispatch optimizer for Thermal Energy Storage."""

    def __init__(self, solver: Any | None = None) -> None:
        self._custom_solver = solver

    def optimize(self, problem: OptimizationInput | OptimizationProblemInput) -> OptimizationResult:
        if not problem.intervals:
            raise OptimizationError("Cannot optimize with an empty interval list.")

        params = problem.tes_params
        T = len(problem.intervals)
        lp = pulp.LpProblem("TES_DayAhead_Dispatch_LP", pulp.LpMinimize)

        # Capacity bounds
        soc_min_kwh = (params.soc_min_percent / 100.0) * params.capacity_kwh
        soc_max_kwh = (params.soc_max_percent / 100.0) * params.capacity_kwh
        target_terminal_soc = problem.resolved_target_terminal_soc_kwh()

        if problem.initial_soc_kwh < soc_min_kwh - 1e-6 or problem.initial_soc_kwh > soc_max_kwh + 1e-6:
            raise OptimizationError(
                f"Initial SOC {problem.initial_soc_kwh:.2f} kWh is outside allowable bounds "
                f"[{soc_min_kwh:.2f}, {soc_max_kwh:.2f}] kWh"
            )

        # --- Decision Variables ---
        # Charging electrical power [kW]
        P_charge = [
            pulp.LpVariable(
                f"P_charge_{t}",
                lowBound=0.0,
                upBound=params.max_charge_power_kw,
            )
            for t in range(T)
        ]

        # Useful thermal discharge power [kW]
        P_discharge = [
            pulp.LpVariable(
                f"P_discharge_{t}",
                lowBound=0.0,
                upBound=params.max_discharge_power_kw,
            )
            for t in range(T)
        ]

        # Unmet process heat [kW] (slack variable with high penalty or lexicographic priority)
        unmet_heat = [
            pulp.LpVariable(
                f"unmet_heat_{t}",
                lowBound=0.0,
            )
            for t in range(T)
        ]

        # Optional physical backup heater [kW thermal]
        if problem.backup_heat_enabled:
            P_backup = [
                pulp.LpVariable(
                    f"P_backup_{t}",
                    lowBound=0.0,
                    upBound=problem.backup_heater_max_power_kw,
                )
                for t in range(T)
            ]
        else:
            P_backup = None

        # State of charge at boundary points t = 0..T [kWh]
        SOC = [
            pulp.LpVariable(
                f"SOC_{t}",
                lowBound=soc_min_kwh,
                upBound=soc_max_kwh,
            )
            for t in range(T + 1)
        ]

        # --- Constraints ---
        # 1. Initial SOC
        lp += (SOC[0] == problem.initial_soc_kwh, "initial_soc")

        # 2. Terminal SOC
        if problem.terminal_soc_condition == "exact":
            lp += (SOC[T] == target_terminal_soc, "terminal_soc_exact")
        elif problem.terminal_soc_condition == "min":
            lp += (SOC[T] >= target_terminal_soc, "terminal_soc_min")
        else:
            raise OptimizationError(f"Unsupported terminal_soc_condition: {problem.terminal_soc_condition}")

        # Extract LP discharge upper bounds from curve
        lp_discharge_bounds = problem.discharge_limit_curve.get_lp_upper_bounds(params)

        use_fixed_loss = params.standing_loss_model == "provisional_fixed_rate_kw"

        for t, iv in enumerate(problem.intervals):
            dt = iv.duration_h

            # A. Process heat delivery balance (Requirement 2 & Section 23/25)
            if P_backup is not None:
                lp += (
                    P_discharge[t] + P_backup[t] + unmet_heat[t] == iv.heat_demand_kw,
                    f"heat_balance_{t}",
                )
            else:
                lp += (
                    P_discharge[t] + unmet_heat[t] == iv.heat_demand_kw,
                    f"heat_balance_{t}",
                )

            # B. Grid connection limit (Requirement 6 & Section 25)
            grid_headroom = problem.grid_connection_limit_kw - iv.other_site_load_kw - iv.auxiliary_load_kw
            grid_headroom = max(0.0, grid_headroom)
            if P_backup is not None:
                backup_elec_equiv = P_backup[t] * (1.0 / problem.backup_heater_efficiency)
                lp += (
                    P_charge[t] + backup_elec_equiv <= grid_headroom,
                    f"grid_limit_{t}",
                )
            else:
                lp += (
                    P_charge[t] <= grid_headroom,
                    f"grid_limit_{t}",
                )

            # C. Pluggable discharge limit upper bounds (Requirement 4)
            for bound_idx, (slope, intercept) in enumerate(lp_discharge_bounds):
                lp += (
                    P_discharge[t] <= slope * SOC[t] + intercept,
                    f"discharge_limit_{t}_{bound_idx}",
                )

            # D. TES state-of-charge dynamics (Requirement 1)
            if use_fixed_loss:
                loss_kwh = params.standing_loss_fixed_kw * dt
                lp += (
                    SOC[t + 1]
                    == SOC[t]
                    - loss_kwh
                    + params.charge_efficiency * P_charge[t] * dt
                    - (1.0 / params.discharge_efficiency) * P_discharge[t] * dt,
                    f"soc_dynamics_{t}",
                )
            else:
                ret = retention_factor(params.standing_loss_percent_per_day, dt)
                lp += (
                    SOC[t + 1]
                    == ret * SOC[t]
                    + params.charge_efficiency * P_charge[t] * dt
                    - (1.0 / params.discharge_efficiency) * P_discharge[t] * dt,
                    f"soc_dynamics_{t}",
                )

        solver = self._custom_solver or pulp.PULP_CBC_CMD(msg=False)

        # --- Objective Function & Solve (Lexicographic or Penalty) ---
        if problem.lexicographic:
            # PASS 1: Minimize total unmet heat
            pass1_terms = [unmet_heat[t] * problem.intervals[t].duration_h for t in range(T)]
            lp.setObjective(pulp.lpSum(pass1_terms))
            s1_status = lp.solve(solver)
            if pulp.LpStatus[s1_status] != "Optimal":
                raise OptimizationError(f"Lexicographic Pass 1 failed with status: {pulp.LpStatus[s1_status]}")

            u_star = sum(max(0.0, float(pulp.value(unmet_heat[t]))) * problem.intervals[t].duration_h for t in range(T))

            # PASS 2: Subject to optimal unmet heat, minimize electricity cost + small throughput penalty
            if u_star <= 1e-6:
                lp += (
                    pulp.lpSum(pass1_terms) <= 0.0,
                    "lexicographic_unmet_heat_bound",
                )
            else:
                lp += (
                    pulp.lpSum(pass1_terms) <= u_star * (1.0 + 1e-6) + 1e-6,
                    "lexicographic_unmet_heat_bound",
                )

            pass2_terms = []
            for t, iv in enumerate(problem.intervals):
                dt = iv.duration_h
                elec_coef = (iv.effective_price_eur_mwh / 1000.0) * dt
                throughput_coef = problem.throughput_penalty_eur_per_kwh * dt
                pass2_terms.append((elec_coef + throughput_coef) * P_charge[t])
                # Strongly penalize any unnecessary unmet heat within slack
                pass2_terms.append(problem.unmet_heat_penalty_eur_per_kwh * dt * unmet_heat[t])
                if P_backup is not None:
                    backup_elec_coef = (iv.effective_price_eur_mwh / 1000.0) * (dt / problem.backup_heater_efficiency)
                    pass2_terms.append(backup_elec_coef * P_backup[t])

            lp.setObjective(pulp.lpSum(pass2_terms))
            s2_status = lp.solve(solver)
            status_name = pulp.LpStatus[s2_status]
            if status_name != "Optimal":
                raise OptimizationError(f"Lexicographic Pass 2 failed with status: {status_name}")
        else:
            # Single-pass penalty objective
            objective_terms = []
            for t, iv in enumerate(problem.intervals):
                dt = iv.duration_h
                elec_coef = (iv.effective_price_eur_mwh / 1000.0) * dt
                unmet_coef = problem.unmet_heat_penalty_eur_per_kwh * dt
                throughput_coef = problem.throughput_penalty_eur_per_kwh * dt

                objective_terms.append((elec_coef + throughput_coef) * P_charge[t])
                objective_terms.append(unmet_coef * unmet_heat[t])
                if P_backup is not None:
                    backup_elec_coef = (iv.effective_price_eur_mwh / 1000.0) * (dt / problem.backup_heater_efficiency)
                    objective_terms.append(backup_elec_coef * P_backup[t])

            lp.setObjective(pulp.lpSum(objective_terms))
            solve_status_code = lp.solve(solver)
            status_name = pulp.LpStatus[solve_status_code]
            if status_name != "Optimal":
                raise OptimizationError(f"Optimization failed with solver status: {status_name}")

        # --- Extract Results ---
        interval_results: list[OptimizationIntervalResult] = []
        total_charge_kwh = 0.0
        total_useful_heat_kwh = 0.0
        total_storage_energy_withdrawn_kwh = 0.0
        total_charge_losses_kwh = 0.0
        total_discharge_losses_kwh = 0.0
        total_standing_losses_kwh = 0.0
        total_unmet_heat_kwh = 0.0
        total_electricity_cost_eur = 0.0
        total_heat_demand_kwh = 0.0
        total_backup_heat_kwh = 0.0
        total_backup_electricity_kwh = 0.0
        total_backup_cost_eur = 0.0

        for t, iv in enumerate(problem.intervals):
            dt = iv.duration_h
            p_c = float(pulp.value(P_charge[t]))
            p_d = float(pulp.value(P_discharge[t]))
            u_h = float(pulp.value(unmet_heat[t]))
            soc_val = float(pulp.value(SOC[t]))
            p_b = float(pulp.value(P_backup[t])) if P_backup is not None else 0.0

            # Clean tiny numerical noise
            if abs(p_c) < 1e-7:
                p_c = 0.0
            if abs(p_d) < 1e-7:
                p_d = 0.0
            if abs(u_h) < 1e-7:
                u_h = 0.0
            if abs(p_b) < 1e-7:
                p_b = 0.0

            soc_pct = (soc_val / params.capacity_kwh * 100.0) if params.capacity_kwh > 0 else 0.0
            charge_cost = (iv.effective_price_eur_mwh * p_c * dt) / 1000.0
            backup_elec_kwh = (p_b / problem.backup_heater_efficiency) * dt if p_b > 0 else 0.0
            backup_cost = (iv.effective_price_eur_mwh * backup_elec_kwh) / 1000.0
            total_interval_cost = charge_cost + backup_cost

            if use_fixed_loss:
                standing_loss_kwh = min(soc_val, params.standing_loss_fixed_kw * dt)
                standing_loss_kw = params.standing_loss_fixed_kw
            else:
                ret = retention_factor(params.standing_loss_percent_per_day, dt)
                standing_loss_kwh = (1.0 - ret) * soc_val
                standing_loss_kw = standing_loss_kwh / dt

            grid_power = p_c + (backup_elec_kwh / dt if dt > 0 else 0.0) + iv.other_site_load_kw + iv.auxiliary_load_kw

            interval_results.append(
                OptimizationIntervalResult(
                    start_utc=iv.start_utc,
                    end_utc=iv.end_utc,
                    spot_price_eur_mwh=iv.spot_price_eur_mwh,
                    effective_price_eur_mwh=iv.effective_price_eur_mwh,
                    charge_power_kw=round(p_c, 4),
                    discharge_power_kw=round(p_d, 4),
                    unmet_heat_kw=round(u_h, 4),
                    soc_kwh=round(soc_val, 4),
                    soc_percent=round(soc_pct, 2),
                    heat_demand_kw=round(iv.heat_demand_kw, 4),
                    other_site_load_kw=round(iv.other_site_load_kw, 4),
                    auxiliary_load_kw=round(iv.auxiliary_load_kw, 4),
                    grid_power_kw=round(grid_power, 4),
                    cost_eur=round(total_interval_cost, 6),
                    standing_loss_kw=round(standing_loss_kw, 4),
                    backup_heat_kw=round(p_b, 4),
                    backup_electricity_kw=round(backup_elec_kwh / dt if dt > 0 else 0.0, 4),
                )
            )

            # Accumulate metrics
            total_charge_kwh += p_c * dt
            total_useful_heat_kwh += p_d * dt
            total_storage_energy_withdrawn_kwh += (p_d / params.discharge_efficiency) * dt
            total_charge_losses_kwh += (1.0 - params.charge_efficiency) * p_c * dt
            total_discharge_losses_kwh += ((1.0 / params.discharge_efficiency) - 1.0) * p_d * dt
            total_standing_losses_kwh += standing_loss_kwh
            total_unmet_heat_kwh += u_h * dt
            total_electricity_cost_eur += total_interval_cost
            total_heat_demand_kwh += iv.heat_demand_kw * dt
            total_backup_heat_kwh += p_b * dt
            total_backup_electricity_kwh += backup_elec_kwh
            total_backup_cost_eur += backup_cost

        terminal_soc = float(pulp.value(SOC[T]))
        delta_soc = terminal_soc - problem.initial_soc_kwh

        # Energy balance check:
        # E_in = total_charge_kwh
        # E_out = useful_heat + delta_soc + standing_loss + charge_losses + discharge_losses
        e_out = (
            total_useful_heat_kwh
            + delta_soc
            + total_standing_losses_kwh
            + total_charge_losses_kwh
            + total_discharge_losses_kwh
        )
        residual = total_charge_kwh - e_out

        reliability = (
            100.0 * (1.0 - total_unmet_heat_kwh / total_heat_demand_kwh)
            if total_heat_demand_kwh > 0
            else 100.0
        )
        reliability = max(0.0, min(100.0, reliability))

        cost_per_mwh = (
            (total_electricity_cost_eur / (total_useful_heat_kwh / 1000.0))
            if total_useful_heat_kwh > 0
            else 0.0
        )

        metrics = OptimizationMetrics(
            total_unmet_heat_kwh=total_unmet_heat_kwh,
            heat_supply_reliability_percent=reliability,
            terminal_soc_kwh=terminal_soc,
            total_charge_kwh=total_charge_kwh,
            total_storage_energy_withdrawn_kwh=total_storage_energy_withdrawn_kwh,
            useful_heat_delivered_kwh=total_useful_heat_kwh,
            charge_conversion_losses_kwh=total_charge_losses_kwh,
            discharge_conversion_losses_kwh=total_discharge_losses_kwh,
            standing_losses_kwh=total_standing_losses_kwh,
            electricity_cost_eur=total_electricity_cost_eur,
            cost_eur_per_mwh_useful_heat=cost_per_mwh,
            energy_balance_residual_kwh=residual,
            total_backup_heat_kwh=total_backup_heat_kwh,
            total_backup_electricity_kwh=total_backup_electricity_kwh,
            total_backup_cost_eur=total_backup_cost_eur,
        )

        log.info(
            "optimization completed",
            extra={
                "ctx": {
                    "status": status_name,
                    "electricity_cost_eur": round(total_electricity_cost_eur, 4),
                    "unmet_heat_kwh": round(total_unmet_heat_kwh, 4),
                    "residual_kwh": round(residual, 7),
                }
            },
        )

        return OptimizationResult(
            status=status_name,
            solver="PULP_CBC_CMD",
            objective_eur=float(pulp.value(lp.objective)),
            intervals=interval_results,
            metrics=metrics,
            initial_soc_kwh=problem.initial_soc_kwh,
            target_terminal_soc_kwh=target_terminal_soc,
        )
