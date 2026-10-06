from datetime import date
from zoneinfo import ZoneInfo
import pytest

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.services.simulation_service import build_day_inputs, simulate_day
from app.simulation.simulator import Simulator
from app.simulation.strategies import CheapestIntervalsStrategy, HeatFollowingStrategy


@pytest.fixture
def sim_setup():
    tz = ZoneInfo("Europe/Vilnius")
    tes = TESParameters(
        capacity_kwh=15.0,
        max_charge_power_kw=9.0,
        max_discharge_power_kw=3.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.90,
        standing_loss_percent_per_day=2.0,
        soc_min_percent=10.0,
        soc_max_percent=100.0,
        initial_soc_percent=50.0,
    )
    site = SiteParameters(
        grid_connection_limit_kw=12.0,
        other_loads_kw=2.0,
        process_heat_demand_kw=1.5,
    )
    tariff = TariffParameters(
        supplier_markup_eur_mwh=5.0,
        variable_grid_fee_eur_mwh=15.0,
        variable_tax_eur_mwh=2.0,
    )
    provider = MockPriceProvider(tz=tz, resolution_minutes=15, seed=42)
    heat = ConstantHeatDemand(value_kw=site.process_heat_demand_kw)
    site_load = ConstantSiteLoad(value_kw=site.other_loads_kw)
    return {
        "tz": tz, "tes": tes, "site": site, "tariff": tariff,
        "provider": provider, "heat": heat, "site_load": site_load,
    }


def test_24h_simulation_energy_balance_and_constraints(sim_setup):
    s = sim_setup
    d = date(2026, 10, 6)
    inputs, _ = build_day_inputs(d, s["tz"], 15, "LT", s["provider"], s["heat"], s["site_load"])

    assert len(inputs.intervals) == 96

    strategy = CheapestIntervalsStrategy(n_intervals=16)
    result = Simulator(s["tes"], s["site"], s["tariff"]).run(inputs, strategy)
    summary = result.summary

    assert summary.n_intervals == 96
    # Energy balance residual must be virtually zero
    assert abs(summary.energy_balance_residual_kwh) < 1e-10

    # Grid connection limit never exceeded
    assert summary.grid_limit_violations == 0
    assert summary.max_grid_power_kw <= s["site"].grid_connection_limit_kw + 1e-9

    # SOC bounds never violated
    assert summary.soc_min_kwh >= s["tes"].soc_min_kwh - 1e-9
    assert summary.soc_max_kwh <= s["tes"].soc_max_kwh + 1e-9

    # Heat demand was fully met throughout
    assert summary.unmet_heat_kwh == 0.0
    assert summary.heat_delivered_kwh == summary.heat_demand_kwh


def test_simulation_determinism(sim_setup):
    s = sim_setup
    d = date(2026, 10, 6)
    inputs1, _ = build_day_inputs(d, s["tz"], 15, "LT", s["provider"], s["heat"], s["site_load"])
    inputs2, _ = build_day_inputs(d, s["tz"], 15, "LT", s["provider"], s["heat"], s["site_load"])

    assert inputs1.sha256() == inputs2.sha256()

    strategy1 = CheapestIntervalsStrategy(n_intervals=16)
    strategy2 = CheapestIntervalsStrategy(n_intervals=16)

    res1 = Simulator(s["tes"], s["site"], s["tariff"]).run(inputs1, strategy1)
    res2 = Simulator(s["tes"], s["site"], s["tariff"]).run(inputs2, strategy2)

    assert res1.summary.electricity_kwh == res2.summary.electricity_kwh
    assert res1.summary.cost_eur == res2.summary.cost_eur
    assert res1.summary.soc_final_kwh == res2.summary.soc_final_kwh


def test_heat_following_strategy_maintains_heat_supply(sim_setup):
    s = sim_setup
    d = date(2026, 10, 6)
    inputs, _ = build_day_inputs(d, s["tz"], 15, "LT", s["provider"], s["heat"], s["site_load"])

    strategy = HeatFollowingStrategy()
    result = Simulator(s["tes"], s["site"], s["tariff"]).run(inputs, strategy)

    assert result.summary.unmet_heat_kwh == 0.0
    assert abs(result.summary.energy_balance_residual_kwh) < 1e-10


def test_terminal_soc_tracking_and_budgeting(sim_setup):
    s = sim_setup
    d = date(2026, 10, 6)
    inputs, _ = build_day_inputs(d, s["tz"], 15, "LT", s["provider"], s["heat"], s["site_load"])

    target_terminal = 10.0  # target 10.0 kWh (higher than initial 7.5 kWh)

    # Strategy with automatic budgeting for terminal target
    strategy = CheapestIntervalsStrategy(n_intervals=None)
    result = Simulator(s["tes"], s["site"], s["tariff"]).run(
        inputs,
        strategy,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=target_terminal,
    )
    summary = result.summary

    assert summary.target_terminal_soc_kwh == target_terminal
    assert summary.terminal_soc_deficit_kwh is not None
    assert summary.terminal_soc_surplus_kwh is not None

    if summary.soc_final_kwh >= target_terminal:
        assert summary.terminal_soc_deficit_kwh == 0.0
        assert abs(summary.terminal_soc_surplus_kwh - (summary.soc_final_kwh - target_terminal)) < 1e-9
    else:
        assert summary.terminal_soc_surplus_kwh == 0.0
        assert abs(summary.terminal_soc_deficit_kwh - (target_terminal - summary.soc_final_kwh)) < 1e-9

    assert abs(summary.energy_balance_residual_kwh) < 1e-10

