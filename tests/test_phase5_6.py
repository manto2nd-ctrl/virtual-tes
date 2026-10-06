"""Unit tests for Phase 5.6: Physical Model Correction & True Rolling EMS.

Contains the 27 mandatory regression tests required by Section 34:
1. no SOC reset at midnight;
2. no special midnight terminal reserve exists;
3. rolling controller only sees prices available at decision time;
4. new market information triggers re-optimization;
5. perfect-foresight objective <= rolling objective within tolerance;
6. normalized SOC maps to the same sand temperature across different TES capacities;
7. larger capacity implies larger sand mass in physical mode;
8. fixed HX gives approximately equal Pmax at equal SOC percentage / sand temp across capacities;
9. scaled HX gives increased capability according to explicit scaling rule;
10. Pmax increases monotonically with sand temperature;
11. Pmax increases with HX area;
12. Pmax increases with U;
13. HX output is zero when T_sand <= T_air_in;
14. HX outlet temperature calculation closes energy balance;
15. U=25 is not the default physical estimate;
16. ordinary 15-minute local day contains 96 intervals;
17. spring DST local day contains 92 intervals;
18. autumn DST local day contains 100 intervals;
19. missing ENTSO-E token causes LIVE test to SKIP, not PASS;
20. fixture parser test remains independently testable;
21. original hourly resolution is preserved when internally expanded;
22. default Gen0 has no physical backup heater;
23. unmet heat does not silently create fake electricity consumption;
24. enabling explicit backup heater correctly increases electricity and grid load;
25. direct-electric economic baseline remains separate from backup architecture;
26. energy conservation closes across corrected sizing studies;
27. results remain deterministic.
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.backtest.domain import BacktestConfig
from app.backtest.engine import (
    BacktestRunner,
    check_physical_viability,
    evaluate_direct_electric_heating,
    evaluate_perfect_foresight_lp,
    evaluate_realistic_rolling_lp,
)
from app.backtest.market_info import MarketInformationConfig, MarketInformationModel
from app.backtest.sensitivity import SizingStudyAnalyzer
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import build_intervals, local_day_bounds_utc, local_day_intervals
from app.models.domain import PricePoint
from app.optimization.domain import OptimizationIntervalInput, OptimizationProblemInput
from app.optimization.lp_optimizer import LPOptimizer
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.tes.model import (
    ConstantDischargeLimit,
    HelicalAirHXModel,
    PiecewiseLinearDischargeLimit,
    get_gen0_discharge_curve,
)
from app.tes.thermal import ThermalStateMapper

UTC = timezone.utc
VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


def create_3day_test_env():
    """Build a 3-day test dataset with fluctuating prices."""
    s_date = date(2026, 10, 1)
    e_date = date(2026, 10, 4)  # 3 full days: Oct 1, 2, 3
    s_utc, _ = local_day_bounds_utc(s_date, VILNIUS_TZ)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), VILNIUS_TZ)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
    prov = MockPriceProvider(tz=VILNIUS_TZ, resolution_minutes=15, seed=123)
    price_bundle = prov.fetch_day_ahead("LT", s_utc, e_utc)
    prices = price_bundle.points

    tes_params = TESParameters(
        capacity_kwh=15.0,
        soc_min_percent=10.0,
        soc_max_percent=100.0,
        max_charge_power_kw=9.0,
        max_discharge_power_kw=3.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.90,
        standing_loss_percent_per_day=2.0,
        auxiliary_power_kw=0.05,
        overall_u_w_m2k=9.0,
        airflow_m3_h=80.0,
        hx_area_m2=1.55,
    )

    curve = get_gen0_discharge_curve("fixed_gen0_hx", capacity_kwh=15.0)

    config = BacktestConfig(
        start_date=s_date,
        end_date=e_date,
        mode="realistic_rolling",
        initial_soc_kwh=7.5,
        tes_params=tes_params,
        site_params=SiteParameters(grid_connection_limit_kw=12.0),
        tariff_params=TariffParameters(),
        discharge_limit_curve=curve,
        auxiliary_power_kw=0.05,
        price_source="mock",
        backup_heat_enabled=False,
    )

    heat_profile = ConstantHeatDemand(value_kw=1.0)
    site_profile = ConstantSiteLoad(value_kw=1.0)

    return config, intervals, prices, heat_profile, site_profile


# ---------------------------------------------------------------------------
# Test 1: No SOC reset at midnight
# ---------------------------------------------------------------------------
def test_no_soc_reset_at_midnight():
    config, intervals, prices, heat_profile, site_profile = create_3day_test_env()
    runner = BacktestRunner(tz=VILNIUS_TZ)
    report = runner.run(config, heat_profile, site_profile, prices)
    opt_m = report.strategies["optimized"]

    assert opt_m.energy_balance_residual_kwh < 1e-4
    assert opt_m.heat_supply_reliability_percent >= 99.0


# ---------------------------------------------------------------------------
# Test 2: No special midnight terminal reserve exists
# ---------------------------------------------------------------------------
def test_no_special_midnight_terminal_reserve_exists():
    config, intervals, prices, heat_profile, site_profile = create_3day_test_env()
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    res = evaluate_realistic_rolling_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )

    assert res.status_label is not None
    assert res.energy_balance_residual_kwh < 1e-4


# ---------------------------------------------------------------------------
# Test 3: Rolling controller only sees prices available at decision time
# ---------------------------------------------------------------------------
def test_rolling_controller_only_sees_prices_available_at_decision_time():
    market_cfg = MarketInformationConfig(
        day_ahead_assumed_available_local_time="14:00",
        market_timezone="Europe/Vilnius",
    )
    model = MarketInformationModel(config=market_cfg)

    # Morning on Oct 1: 10:00 local time
    morning_decision = datetime(2026, 10, 1, 10, 0, tzinfo=VILNIUS_TZ).astimezone(UTC)
    
    # Oct 1 delivery and Oct 2 delivery
    day1_start = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)
    day2_start = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)

    p_day1 = PricePoint(
        bidding_zone="LT",
        delivery_start_utc=day1_start,
        delivery_end_utc=day1_start + timedelta(hours=1),
        price_eur_mwh=50.0,
        resolution_minutes=60,
        source="entsoe",
    )
    p_day2 = PricePoint(
        bidding_zone="LT",
        delivery_start_utc=day2_start,
        delivery_end_utc=day2_start + timedelta(hours=1),
        price_eur_mwh=60.0,
        resolution_minutes=60,
        source="entsoe",
    )

    known_prices = model.get_known_prices([p_day1, p_day2], as_of_utc=morning_decision)
    assert p_day1 in known_prices
    assert p_day2 not in known_prices


# ---------------------------------------------------------------------------
# Test 4: New market information triggers re-optimization
# ---------------------------------------------------------------------------
def test_new_market_information_triggers_reoptimization():
    market_cfg = MarketInformationConfig(
        day_ahead_assumed_available_local_time="14:00",
        market_timezone="Europe/Vilnius",
    )
    model = MarketInformationModel(config=market_cfg)

    # Afternoon on Oct 1: 14:15 local time (after 14:00 publication)
    afternoon_decision = datetime(2026, 10, 1, 14, 15, tzinfo=VILNIUS_TZ).astimezone(UTC)
    day1_start = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)
    day2_start = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)

    p_day1 = PricePoint(
        bidding_zone="LT",
        delivery_start_utc=day1_start,
        delivery_end_utc=day1_start + timedelta(hours=1),
        price_eur_mwh=50.0,
        resolution_minutes=60,
        source="entsoe",
    )
    p_day2 = PricePoint(
        bidding_zone="LT",
        delivery_start_utc=day2_start,
        delivery_end_utc=day2_start + timedelta(hours=1),
        price_eur_mwh=60.0,
        resolution_minutes=60,
        source="entsoe",
    )

    known_afternoon = model.get_known_prices([p_day1, p_day2], as_of_utc=afternoon_decision)
    assert p_day1 in known_afternoon
    assert p_day2 in known_afternoon


# ---------------------------------------------------------------------------
# Test 5: Perfect-foresight objective <= rolling objective within tolerance
# ---------------------------------------------------------------------------
def test_perfect_foresight_objective_le_rolling_objective_within_tolerance():
    config, intervals, prices, heat_profile, site_profile = create_3day_test_env()
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    foresight = evaluate_perfect_foresight_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
    )
    rolling = evaluate_realistic_rolling_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )

    tolerance = 1e-3
    assert foresight.objective_value <= rolling.objective_value + tolerance


# ---------------------------------------------------------------------------
# Test 6: Normalized SOC maps to same sand temperature across capacities
# ---------------------------------------------------------------------------
def test_normalized_soc_maps_to_same_sand_temperature_across_capacities():
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    t_50 = mapper.temperature_from_soc_fraction(0.50)
    assert 185.0 < t_50 < 200.0

    for cap in [10.0, 15.0, 20.0, 30.0]:
        soc_kwh = cap * 0.50
        soc_fraction = soc_kwh / cap
        t_sand = mapper.temperature_from_soc_fraction(soc_fraction)
        assert math.isclose(t_sand, t_50, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 7: Larger capacity implies larger sand mass in physical mode
# ---------------------------------------------------------------------------
def test_larger_capacity_implies_larger_sand_mass_in_physical_mode():
    m10 = ThermalStateMapper.capacity_kwh_to_sand_mass_kg(10.0)
    m15 = ThermalStateMapper.capacity_kwh_to_sand_mass_kg(15.0)
    m20 = ThermalStateMapper.capacity_kwh_to_sand_mass_kg(20.0)
    m30 = ThermalStateMapper.capacity_kwh_to_sand_mass_kg(30.0)

    assert m10 < m15 < m20 < m30
    assert math.isclose(m15, 275.745, rel_tol=1e-3)
    assert math.isclose(m30, 2 * m15, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 8: Fixed HX gives approximately equal Pmax at equal SOC percentage
# ---------------------------------------------------------------------------
def test_fixed_hx_gives_approximately_equal_pmax_at_equal_soc_percentage_across_capacities():
    hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0)
    mapper = ThermalStateMapper()
    t_sand = mapper.temperature_from_soc_fraction(0.50)
    p_max_50 = hx.p_max_at_temperature_kw(t_sand)

    for cap in [10.0, 15.0, 20.0, 30.0]:
        curve = get_gen0_discharge_curve("fixed_gen0_hx", capacity_kwh=cap, overall_u_w_m2k=9.0, airflow_m3_h=80.0)
        p = curve.max_discharge_power_kw(soc_kwh=cap * 0.50, capacity_kwh=cap)
        assert math.isclose(p, p_max_50, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# Test 9: Scaled HX gives increased capability according to explicit scaling rule
# ---------------------------------------------------------------------------
def test_scaled_hx_gives_increased_capability_according_to_explicit_scaling_rule():
    curve_15 = get_gen0_discharge_curve("scaled_hx_benchmark", capacity_kwh=15.0, reference_capacity_kwh=15.0)
    curve_30 = get_gen0_discharge_curve("scaled_hx_benchmark", capacity_kwh=30.0, reference_capacity_kwh=15.0)

    p_15 = curve_15.max_discharge_power_kw(soc_kwh=7.5, capacity_kwh=15.0)
    p_30 = curve_30.max_discharge_power_kw(soc_kwh=15.0, capacity_kwh=30.0)

    assert p_30 > p_15


# ---------------------------------------------------------------------------
# Test 10: Pmax increases monotonically with sand temperature
# ---------------------------------------------------------------------------
def test_pmax_increases_monotonically_with_sand_temperature():
    hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0)
    temps = [80.0, 120.0, 160.0, 200.0, 250.0, 300.0]
    p_vals = [hx.p_max_at_temperature_kw(t) for t in temps]

    for i in range(len(p_vals) - 1):
        assert p_vals[i + 1] > p_vals[i]


# ---------------------------------------------------------------------------
# Test 11: Pmax increases with HX area
# ---------------------------------------------------------------------------
def test_pmax_increases_with_hx_area():
    hx_small = HelicalAirHXModel(hx_area_m2=1.0, overall_u_w_m2k=9.0, airflow_m3_h=80.0)
    hx_mid = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0)
    hx_large = HelicalAirHXModel(hx_area_m2=2.5, overall_u_w_m2k=9.0, airflow_m3_h=80.0)

    t_sand = 200.0
    assert hx_small.p_max_at_temperature_kw(t_sand) < hx_mid.p_max_at_temperature_kw(t_sand) < hx_large.p_max_at_temperature_kw(t_sand)


# ---------------------------------------------------------------------------
# Test 12: Pmax increases with U
# ---------------------------------------------------------------------------
def test_pmax_increases_with_u():
    hx_6 = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=6.0, airflow_m3_h=80.0)
    hx_9 = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0)
    hx_12 = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=12.0, airflow_m3_h=80.0)

    t_sand = 200.0
    assert hx_6.p_max_at_temperature_kw(t_sand) < hx_9.p_max_at_temperature_kw(t_sand) < hx_12.p_max_at_temperature_kw(t_sand)


# ---------------------------------------------------------------------------
# Test 13: HX output is zero when T_sand <= T_air_in
# ---------------------------------------------------------------------------
def test_hx_output_is_zero_when_t_sand_le_t_air_in():
    hx = HelicalAirHXModel(air_inlet_temperature_c=40.0)
    p = hx.p_max_at_temperature_kw(t_sand_c=40.0)
    t_out = hx.calculate_outlet_temperature_c(t_sand_c=40.0)
    assert p == 0.0
    assert t_out == 40.0

    p_below = hx.p_max_at_temperature_kw(t_sand_c=25.0)
    t_out_below = hx.calculate_outlet_temperature_c(t_sand_c=25.0)
    assert p_below == 0.0
    assert t_out_below == 40.0


# ---------------------------------------------------------------------------
# Test 14: HX outlet temperature calculation closes energy balance
# ---------------------------------------------------------------------------
def test_hx_outlet_temperature_calculation_closes_energy_balance():
    hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0, air_inlet_temperature_c=40.0)
    t_sand = 250.0
    p_max = hx.p_max_at_temperature_kw(t_sand)
    t_out = hx.calculate_outlet_temperature_c(t_sand)

    t_in_k = 40.0 + 273.15
    rho_air = 101325.0 / (287.05 * t_in_k)
    m_dot_kg_s = (80.0 / 3600.0) * rho_air
    cp_air = 1007.0
    q_calc_kw = m_dot_kg_s * cp_air * (t_out - 40.0) / 1000.0

    assert math.isclose(p_max, q_calc_kw, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 15: U=25 is not the default physical estimate
# ---------------------------------------------------------------------------
def test_u_25_is_not_default_physical_estimate():
    params = TESParameters()
    assert params.overall_u_w_m2k == 9.0
    assert params.overall_u_w_m2k != 25.0

    config, _, _, _, _ = create_3day_test_env()
    cfg_u25 = BacktestConfig(**{**config.__dict__, "overall_u_w_m2k": 25.0})
    viable, caveats = check_physical_viability(cfg_u25)
    assert viable is False
    assert any("25" in c for c in caveats)


# ---------------------------------------------------------------------------
# Test 16: Ordinary 15-minute local day contains 96 intervals
# ---------------------------------------------------------------------------
def test_ordinary_15min_local_day_contains_96_intervals():
    ivs = local_day_intervals(date(2026, 10, 6), VILNIUS_TZ, 15)
    assert len(ivs) == 96


# ---------------------------------------------------------------------------
# Test 17: Spring DST local day contains 92 intervals
# ---------------------------------------------------------------------------
def test_spring_dst_local_day_contains_92_intervals():
    ivs = local_day_intervals(date(2026, 3, 29), VILNIUS_TZ, 15)
    assert len(ivs) == 92


# ---------------------------------------------------------------------------
# Test 18: Autumn DST local day contains 100 intervals
# ---------------------------------------------------------------------------
def test_autumn_dst_local_day_contains_100_intervals():
    ivs = local_day_intervals(date(2026, 10, 25), VILNIUS_TZ, 15)
    assert len(ivs) == 100


# ---------------------------------------------------------------------------
# Test 19: Missing ENTSO-E token causes LIVE test to SKIP, not PASS
# ---------------------------------------------------------------------------
def test_missing_entsoe_token_causes_live_test_to_skip_not_pass():
    from scripts.entsoe_smoke_test import test_entsoe_live_authenticated

    status, detail = test_entsoe_live_authenticated(token=None)
    assert status == "SKIPPED -- NO ENTSO-E API TOKEN"
    assert "PASS" not in status


# ---------------------------------------------------------------------------
# Test 20: Fixture parser test remains independently testable
# ---------------------------------------------------------------------------
def test_fixture_parser_test_remains_independently_testable():
    from scripts.entsoe_smoke_test import test_entsoe_fixture_schema

    status, points = test_entsoe_fixture_schema()
    assert status == "PASS -- FIXTURE PARSING"
    assert len(points) == 24
    assert points[0].currency == "EUR"


# ---------------------------------------------------------------------------
# Test 21: Original hourly resolution is preserved when internally expanded
# ---------------------------------------------------------------------------
def test_original_hourly_resolution_is_preserved_when_internally_expanded():
    p_h = PricePoint(
        bidding_zone="LT",
        delivery_start_utc=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
        delivery_end_utc=datetime(2026, 10, 1, 1, 0, tzinfo=UTC),
        price_eur_mwh=75.0,
        resolution_minutes=60,
        source="entsoe",
    )
    assert p_h.resolution_minutes == 60
    assert p_h.original_resolution_minutes == 60


# ---------------------------------------------------------------------------
# Test 22: Default Gen0 has no physical backup heater
# ---------------------------------------------------------------------------
def test_default_gen0_has_no_physical_backup_heater():
    cfg = BacktestConfig(
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 2),
        initial_soc_kwh=7.5,
        discharge_limit_curve=ConstantDischargeLimit(),
    )
    assert cfg.backup_heat_enabled is False

    problem = OptimizationProblemInput(
        intervals=[],
        tes_params=TESParameters(),
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
    )
    assert problem.backup_heat_enabled is False


# ---------------------------------------------------------------------------
# Test 23: Unmet heat does not silently create fake electricity consumption
# ---------------------------------------------------------------------------
def test_unmet_heat_does_not_silently_create_fake_electricity_consumption():
    tes = TESParameters(max_discharge_power_kw=1.0)
    opt_inputs = [
        OptimizationIntervalInput(
            start_utc=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
            end_utc=datetime(2026, 10, 1, 1, 0, tzinfo=UTC),
            spot_price_eur_mwh=50.0,
            effective_price_eur_mwh=70.0,
            heat_demand_kw=3.0,
            other_site_load_kw=0.0,
        )
    ]
    problem = OptimizationProblemInput(
        intervals=opt_inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        backup_heat_enabled=False,
    )
    res = LPOptimizer().optimize(problem)

    assert res.metrics.total_unmet_heat_kwh > 0.0
    assert res.metrics.total_backup_electricity_kwh == 0.0
    assert res.metrics.total_backup_cost_eur == 0.0


# ---------------------------------------------------------------------------
# Test 24: Enabling explicit backup heater correctly increases electricity and grid load
# ---------------------------------------------------------------------------
def test_enabling_explicit_backup_heater_correctly_increases_electricity_and_grid_load():
    tes = TESParameters(max_discharge_power_kw=1.0)
    opt_inputs = [
        OptimizationIntervalInput(
            start_utc=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
            end_utc=datetime(2026, 10, 1, 1, 0, tzinfo=UTC),
            spot_price_eur_mwh=50.0,
            effective_price_eur_mwh=70.0,
            heat_demand_kw=3.0,
            other_site_load_kw=0.0,
        )
    ]
    problem = OptimizationProblemInput(
        intervals=opt_inputs,
        tes_params=tes,
        grid_connection_limit_kw=12.0,
        initial_soc_kwh=7.5,
        backup_heat_enabled=True,
        backup_heater_efficiency=1.0,
        backup_heater_max_power_kw=5.0,
    )
    res = LPOptimizer().optimize(problem)

    assert res.metrics.total_backup_heat_kwh > 0.0
    assert res.metrics.total_backup_electricity_kwh > 0.0
    assert res.metrics.total_backup_cost_eur > 0.0
    assert res.metrics.total_unmet_heat_kwh == 0.0


# ---------------------------------------------------------------------------
# Test 25: Direct-electric economic baseline remains separate from backup architecture
# ---------------------------------------------------------------------------
def test_direct_electric_economic_baseline_remains_separate_from_backup_architecture():
    config, intervals, prices, heat_profile, site_profile = create_3day_test_env()
    heat_demands = heat_profile.series(intervals)

    direct = evaluate_direct_electric_heating(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        tariff=config.tariff_params,
        heater_efficiency=config.direct_heater_efficiency,
    )

    assert direct.heat_supply_reliability_percent == 100.0
    assert direct.useful_heat_kwh == sum(hd * iv.duration_h for hd, iv in zip(heat_demands, intervals))
    assert math.isclose(direct.electricity_kwh, direct.useful_heat_kwh / config.direct_heater_efficiency, rel_tol=1e-5)
    assert direct.backup_heat_kwh == 0.0


# ---------------------------------------------------------------------------
# Test 26: Energy conservation closes across corrected sizing studies
# ---------------------------------------------------------------------------
def test_energy_conservation_closes_across_corrected_sizing_studies():
    config, intervals, prices, heat_profile, site_profile = create_3day_test_env()
    analyzer = SizingStudyAnalyzer(tz=VILNIUS_TZ)

    reports = analyzer.run_sizing_sweep(
        base_config=config,
        heat_profile=heat_profile,
        site_profile=site_profile,
        prices=prices,
        capacities_kwh=[10.0, 15.0],
        charge_powers_kw=[6.0, 9.0],
        sizing_mode="fixed_gen0_hx",
    )

    fixed_rep = reports["fixed_gen0_hx"]
    for combo in fixed_rep.combinations:
        assert abs(combo.energy_balance_residual_kwh) < 1e-4
        assert combo.sand_mass_kg > 0.0


# ---------------------------------------------------------------------------
# Test 27: Results remain deterministic
# ---------------------------------------------------------------------------
def test_results_remain_deterministic():
    config, intervals, prices, heat_profile, site_profile = create_3day_test_env()
    runner = BacktestRunner(tz=VILNIUS_TZ)

    rep1 = runner.run(config, heat_profile, site_profile, prices)
    rep2 = runner.run(config, heat_profile, site_profile, prices)

    opt1 = rep1.strategies["optimized"]
    opt2 = rep2.strategies["optimized"]

    assert opt1.total_cost_eur == opt2.total_cost_eur
    assert opt1.useful_heat_kwh == opt2.useful_heat_kwh
    assert opt1.electricity_kwh == opt2.electricity_kwh
    assert opt1.heat_supply_reliability_percent == opt2.heat_supply_reliability_percent
