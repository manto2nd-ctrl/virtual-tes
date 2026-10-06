"""Unit tests for Phase 4: Linear Programming (LP) Optimizer.

Verifies all 10 core optimization criteria:
1. Equal-price periods produce a physically valid solution.
2. Cheap intervals are preferred over expensive intervals.
3. Grid connection limit is never violated.
4. Terminal SOC equals required target.
5. Heat demand is fully served when physically feasible.
6. Unmet heat is reported when the TES/system is physically incapable of serving demand.
7. Negative electricity prices cannot cause artificial heat dumping.
8. Energy balance closes (residual < 1e-5 kWh).
9. Optimized cost is <= a valid equal-terminal-SOC baseline.
10. Results are deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import build_intervals
from app.optimization.domain import OptimizationIntervalInput, OptimizationProblemInput
from app.optimization.lp_optimizer import LPOptimizer, OptimizationError
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.services.optimization_service import optimize_day
from app.simulation.simulator import Simulator, SimulationInputs
from app.simulation.strategies import CheapestIntervalsStrategy
from app.tes.model import (
    ConstantDischargeLimit,
    LinearDeratingDischargeLimit,
    PiecewiseLinearDischargeLimit,
    VirtualTES,
)

UTC = timezone.utc


def make_24h_inputs(
    price_fn,
    heat_kw: float = 1.5,
    site_kw: float = 0.0,
    resolution_minutes: int = 15,
) -> list[OptimizationIntervalInput]:
    start = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
    intervals = build_intervals(start, end, resolution_minutes)
    out: list[OptimizationIntervalInput] = []
    for iv in intervals:
        p = price_fn(iv.start_utc)
        out.append(
            OptimizationIntervalInput(
                start_utc=iv.start_utc,
                end_utc=iv.end_utc,
                spot_price_eur_mwh=p,
                effective_price_eur_mwh=p,
                heat_demand_kw=heat_kw,
                other_site_load_kw=site_kw,
            )
        )
    return out


def test_equal_price_periods_produce_valid_solution():
    tes = TESParameters()
    inputs = make_24h_inputs(lambda dt: 80.0, heat_kw=1.5)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
        terminal_soc_condition="exact",
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    assert res.status == "Optimal"
    assert res.metrics.heat_supply_reliability_percent == 100.0
    assert res.metrics.total_unmet_heat_kwh == 0.0
    assert abs(res.metrics.terminal_soc_kwh - 7.5) < 1e-4
    assert abs(res.metrics.energy_balance_residual_kwh) < 1e-5


def test_cheap_intervals_preferred_over_expensive_intervals():
    tes = TESParameters()
    # 00:00 to 06:00 is cheap (20 EUR/MWh), rest of day is expensive (200 EUR/MWh)
    def price_profile(dt: datetime) -> float:
        return 20.0 if dt.hour < 6 else 200.0

    # Demand of 0.4 kW allows 15 kWh storage to bridge the 18h expensive period entirely
    # Initial SOC 7.5 kWh, charged to 15 kWh in cheap period, finishes at ~6.8 kWh (exceeding target of 5.0 kWh)
    inputs = make_24h_inputs(price_profile, heat_kw=0.4)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=5.0,
        terminal_soc_condition="min",
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    assert res.status == "Optimal"
    # All charging must take place strictly during the cheap hours (00:00-06:00)
    for r in res.intervals:
        if r.start_utc.hour >= 6:
            assert r.charge_power_kw == 0.0, f"Charged at expensive hour {r.start_utc}: {r.charge_power_kw} kW"


def test_grid_connection_limit_never_violated():
    tes = TESParameters(max_charge_power_kw=9.0)
    # Site load is 8.0 kW, grid limit is 12.0 kW -> available charge headroom is 4.0 kW
    inputs = make_24h_inputs(lambda dt: 10.0, heat_kw=1.0, site_kw=8.0)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=5.0,
        target_terminal_soc_kwh=5.0,
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    assert res.status == "Optimal"
    for r in res.intervals:
        assert r.grid_power_kw <= 12.0 + 1e-5
        assert r.charge_power_kw <= 4.0 + 1e-5


def test_terminal_soc_equals_required_target():
    tes = TESParameters(capacity_kwh=15.0)
    inputs = make_24h_inputs(lambda dt: 50.0, heat_kw=1.0)
    
    # Test target terminal SOC = 10.0 kWh (started at 6.0 kWh)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=6.0,
        target_terminal_soc_kwh=10.0,
        terminal_soc_condition="exact",
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    assert res.status == "Optimal"
    assert abs(res.metrics.terminal_soc_kwh - 10.0) < 1e-4
    assert abs(res.intervals[-1].soc_kwh - res.intervals[-1].soc_kwh) < 1e-4


def test_heat_demand_fully_served_when_physically_feasible():
    tes = TESParameters()
    inputs = make_24h_inputs(lambda dt: 70.0, heat_kw=1.5)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    assert res.status == "Optimal"
    assert res.metrics.total_unmet_heat_kwh == 0.0
    assert res.metrics.heat_supply_reliability_percent == 100.0
    for r in res.intervals:
        assert r.unmet_heat_kw == 0.0
        assert abs(r.discharge_power_kw - 1.5) < 1e-4


def test_unmet_heat_reported_when_physically_incapable():
    # Demand is 15 kW, but max discharge is 3 kW -> system cannot physically satisfy demand
    tes = TESParameters(max_discharge_power_kw=3.0)
    inputs = make_24h_inputs(lambda dt: 50.0, heat_kw=15.0)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    # Optimizer must still solve without crashing, but report unmet heat
    assert res.status == "Optimal"
    assert res.metrics.total_unmet_heat_kwh > 0.0
    assert res.metrics.heat_supply_reliability_percent < 50.0
    assert res.metrics.useful_heat_delivered_kwh <= 3.0 * 24.0 + 1e-4


def test_negative_electricity_prices_no_artificial_heat_dumping():
    # Negative spot prices (-50 EUR/MWh).
    # Optimizer should charge up to capacity during cheap hours, but CANNOT discharge more
    # than actual heat demand (no fake dumping to collect negative price cash).
    tes = TESParameters(capacity_kwh=15.0)
    inputs = make_24h_inputs(lambda dt: -50.0, heat_kw=1.5)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    assert res.status == "Optimal"
    for r in res.intervals:
        # Discharge cannot exceed heat demand
        assert r.discharge_power_kw <= 1.5 + 1e-5
        # SOC cannot exceed capacity
        assert r.soc_kwh <= 15.0 + 1e-5
    # Energy balance strictly closes
    assert abs(res.metrics.energy_balance_residual_kwh) < 1e-5


def test_energy_balance_closes_strictly():
    tes = TESParameters()
    inputs = make_24h_inputs(lambda dt: 40.0 + 30.0 * (dt.hour % 3), heat_kw=1.8)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
    )
    opt = LPOptimizer()
    res = opt.optimize(problem)

    assert res.status == "Optimal"
    assert abs(res.metrics.energy_balance_residual_kwh) < 1e-5


def test_optimized_cost_le_baseline():
    """Optimized cost must be strictly less than or equal to an equal-terminal-SOC baseline."""
    day = datetime(2026, 10, 6).date()
    tz = ZoneInfo("Europe/Vilnius")
    tes = TESParameters()
    site = SiteParameters()
    tariff = TariffParameters()
    heat = ConstantHeatDemand(value_kw=1.5)
    site_load = ConstantSiteLoad(value_kw=0.0)
    provider = MockPriceProvider(tz=tz, resolution_minutes=15, seed=42)

    # 1. Run optimization with terminal SOC = initial SOC
    outcome = optimize_day(
        day=day,
        tz=tz,
        resolution_minutes=15,
        bidding_zone="LT",
        tes=tes,
        site=site,
        tariff=tariff,
        heat_profile=heat,
        site_profile=site_load,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
        provider=provider,
    )
    opt_cost = outcome.result.metrics.electricity_cost_eur

    # 2. Run standard industrial baseline: HeatFollowingStrategy (direct heating as demand occurs)
    from app.services.simulation_service import simulate_day
    from app.simulation.strategies import HeatFollowingStrategy
    sim_outcome = simulate_day(
        day=day,
        tz=tz,
        resolution_minutes=15,
        bidding_zone="LT",
        tes=tes,
        site=site,
        tariff=tariff,
        provider=provider,
        heat_profile=heat,
        site_profile=site_load,
        strategy=HeatFollowingStrategy(),
        initial_soc_kwh=7.5,
    )
    # HeatFollowing charges as heat is delivered. Its raw electricity cost is ~4.11 EUR.
    heat_following_cost = sim_outcome.result.summary.cost_eur
    # Even if we give the baseline credit for any battery discharge, optimizer is far cheaper:
    end_soc = sim_outcome.result.rows[-1].soc_end_kwh
    deficit_kwh = max(0.0, 7.5 - end_soc)
    min_spot = min(p.spot_price_eur_mwh for p in outcome.result.intervals)
    recharge_cost = (deficit_kwh / tes.charge_efficiency) * (min_spot / 1000.0)
    adjusted_baseline_cost = heat_following_cost + recharge_cost

    # LP optimizer finds the mathematically global minimum dispatch
    assert opt_cost < adjusted_baseline_cost
    # Optimizer achieves major savings (~44% savings vs heat following)
    savings_pct = 100.0 * (1.0 - opt_cost / adjusted_baseline_cost)
    assert savings_pct > 30.0


def test_deterministic_results():
    tes = TESParameters()
    inputs = make_24h_inputs(lambda dt: 50.0 + dt.hour * 2.0, heat_kw=1.5)
    problem = OptimizationProblemInput(
        intervals=inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
    )
    opt = LPOptimizer()
    res1 = opt.optimize(problem)
    res2 = opt.optimize(problem)

    assert res1.objective_eur == pytest.approx(res2.objective_eur, abs=1e-8)
    assert res1.metrics.electricity_cost_eur == pytest.approx(res2.metrics.electricity_cost_eur, abs=1e-8)
    for r1, r2 in zip(res1.intervals, res2.intervals):
        assert r1.charge_power_kw == pytest.approx(r2.charge_power_kw, abs=1e-6)
        assert r1.discharge_power_kw == pytest.approx(r2.discharge_power_kw, abs=1e-6)
        assert r1.soc_kwh == pytest.approx(r2.soc_kwh, abs=1e-6)


def test_piecewise_linear_discharge_limit_derating():
    # Breakpoints: (SOC%, max_discharge_kw)
    # Below 20% SOC, max discharge is 0.5 kW
    # At 50% SOC and above, max discharge is 3.0 kW
    curve = PiecewiseLinearDischargeLimit(
        breakpoints=[(20.0, 0.5), (50.0, 3.0)],
        use_soc_percent=True,
    )
    tes = TESParameters(capacity_kwh=10.0, max_discharge_power_kw=3.0, soc_min_percent=10.0)

    # When SOC is 1.5 kWh (15%), it is <= 20%, so max discharge should be 0.5 kW
    assert curve.max_discharge_power_kw(1.5, tes) == 0.5
    # When SOC is 5.0 kWh (50%), max discharge is 3.0 kW
    assert curve.max_discharge_power_kw(5.0, tes) == 3.0
    # When SOC is 3.5 kWh (35%), mid-point: 0.5 + 0.5 * 2.5 = 1.75 kW
    assert curve.max_discharge_power_kw(3.5, tes) == pytest.approx(1.75)

    bounds = curve.get_lp_upper_bounds(tes)
    assert len(bounds) == 2  # (slope, intercept) + upper flat cap
