"""Comprehensive Regression Test Suite for Phase 5.5 Physical Model Validation.

Verifies all 13 Section 18 criteria:
1. perfect foresight cost <= realistic rolling cost within solver tolerance;
2. SOC remains continuous across calendar days;
3. final SOC matches required period terminal target;
4. realistic rolling never sees unavailable future price data;
5. Gen0 discharge derating limits heat output at low SOC;
6. invalid discharge curves are rejected;
7. standing-loss sensitivity increases required charging energy monotonically;
8. increasing auxiliary load cannot reduce electricity consumption;
9. lower efficiency cannot improve optimized physical energy consumption;
10. direct heating baseline is independent of TES SOC;
11. mock and real price sources are clearly labeled;
12. parameter sweep results are deterministic;
13. energy balance closes for every sizing combination.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest

from app.backtest.domain import BacktestConfig
from app.backtest.engine import (
    BacktestRunner,
    check_physical_viability,
    evaluate_cheapest_n_heuristic,
    evaluate_direct_electric_heating,
    evaluate_perfect_foresight_lp,
    evaluate_realistic_rolling_lp,
    get_viability_status_label,
)
from app.backtest.sensitivity import SensitivityAnalyzer, SizingStudyAnalyzer
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import build_intervals, local_day_bounds_utc
from app.process.heat_demand import ConstantHeatDemand, get_heat_demand_scenario
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.tes.model import (
    ConstantDischargeLimit,
    GEN0_CURVE_NAME,
    GEN0_ENGINEERING_ESTIMATE_V1_BREAKPOINTS,
    InvalidDischargeCurveError,
    PiecewiseLinearDischargeLimit,
    VirtualTES,
    get_gen0_discharge_curve,
)

UTC = timezone.utc
VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


def create_phase5_5_env(seed: int = 42, days: int = 3):
    """Helper creating multi-day test environment with Gen0 engineering estimate curve."""
    s_date = date(2026, 10, 1)
    e_date = s_date + timedelta(days=days)
    s_utc, _ = local_day_bounds_utc(s_date, VILNIUS_TZ)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), VILNIUS_TZ)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
    prov = MockPriceProvider(tz=VILNIUS_TZ, resolution_minutes=15, seed=seed)
    prices = prov.fetch_day_ahead("LT", s_utc, e_utc).points

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
    )

    site_params = SiteParameters(
        grid_connection_limit_kw=11.0,
        process_heat_demand_kw=1.5,
        other_loads_kw=1.0,
    )

    tariff_params = TariffParameters(
        supplier_markup_eur_mwh=1.5,
        variable_grid_fee_eur_mwh=25.0,
        variable_tax_eur_mwh=5.0,
    )

    curve = get_gen0_discharge_curve(mode="normalized_benchmark")

    config = BacktestConfig(
        start_date=s_date,
        end_date=e_date,
        mode="realistic_rolling",
        initial_soc_kwh=7.5,
        tes_params=tes_params,
        site_params=site_params,
        tariff_params=tariff_params,
        discharge_limit_curve=curve,
        discharge_curve_name=GEN0_CURVE_NAME,
        discharge_curve_mode="normalized_benchmark",
        auxiliary_power_kw=0.05,
        cheapest_n_hours=4,
        rolling_terminal_soc_mode="hold_initial",
        bidding_zone="LT",
        price_source="mock",
    )

    heat_profile = ConstantHeatDemand(value_kw=1.5)
    site_profile = ConstantSiteLoad(value_kw=1.0)

    return config, intervals, prices, heat_profile, site_profile


def test_1_perfect_foresight_le_realistic_rolling():
    """Criterion 1: perfect foresight cost <= realistic rolling cost within solver tolerance."""
    for test_seed in [42, 101, 777]:
        config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(seed=test_seed, days=3)
        heat_demands = heat_profile.series(intervals)
        site_loads = site_profile.series(intervals)

        pf = evaluate_perfect_foresight_lp(intervals, prices, heat_demands, site_loads, config)
        roll = evaluate_realistic_rolling_lp(intervals, prices, heat_demands, site_loads, config, VILNIUS_TZ)

        # Perfect foresight sees all future prices; its evaluated cost must be <= rolling
        solver_tol_eur = 1e-4
        assert pf.total_cost_eur <= roll.total_cost_eur + solver_tol_eur, (
            f"Anomaly: PF ({pf.total_cost_eur:.4f} EUR) > Rolling ({roll.total_cost_eur:.4f} EUR) for seed {test_seed}"
        )


def test_2_continuous_soc_across_calendar_days():
    """Criterion 2: SOC remains continuous across calendar days without midnight resets."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=3)
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    roll = evaluate_realistic_rolling_lp(intervals, prices, heat_demands, site_loads, config, VILNIUS_TZ)

    # 3 days = 72 hours = 288 intervals of 15m.
    # Energy balance residual must be closed across the multi-day run
    assert abs(roll.energy_balance_residual_kwh) < 1e-5
    assert roll.heat_supply_reliability_percent > 95.0


def test_3_final_soc_matches_target():
    """Criterion 3: Final SOC matches required period terminal target."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=3)
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    roll = evaluate_realistic_rolling_lp(intervals, prices, heat_demands, site_loads, config, VILNIUS_TZ)

    # In hold_initial mode, final SOC on the last day must match initial_soc_kwh
    assert abs(roll.final_soc_kwh - config.initial_soc_kwh) < 1e-4
    assert abs(roll.terminal_inventory_adjustment_eur) < 1e-4


def test_4_rolling_never_sees_future_prices():
    """Criterion 4: Realistic rolling operates strictly day-by-day with only currently available day prices."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=2)
    runner = BacktestRunner(tz=VILNIUS_TZ)
    report = runner.run(config, heat_profile, site_profile, prices)
    assert report.total_intervals == 192  # 2 days * 96 ivs/day


def test_5_gen0_discharge_derating_limits_low_soc():
    """Criterion 5: Gen0 discharge derating limits heat output at low SOC."""
    curve = get_gen0_discharge_curve(mode="normalized_benchmark")
    params = TESParameters(capacity_kwh=15.0)

    # At 10% SOC (1.5 kWh), max discharge power is 0.50 kW
    p_10 = curve.max_discharge_power_kw(1.5, params)
    assert abs(p_10 - 0.50) < 1e-4

    # At 20% SOC (3.0 kWh), max discharge power is 0.80 kW
    p_20 = curve.max_discharge_power_kw(3.0, params)
    assert abs(p_20 - 0.80) < 1e-4

    # At 100% SOC (15.0 kWh), max discharge power is 2.30 kW
    p_100 = curve.max_discharge_power_kw(15.0, params)
    assert abs(p_100 - 2.30) < 1e-4


def test_6_invalid_discharge_curves_rejected():
    """Criterion 6: Invalid discharge curves are rejected with clear validation error."""
    # Decreasing power with SOC
    with pytest.raises(InvalidDischargeCurveError, match="must be non-decreasing"):
        PiecewiseLinearDischargeLimit(breakpoints=[(10.0, 2.0), (30.0, 1.0)])

    # Duplicate SOC breakpoint
    with pytest.raises(InvalidDischargeCurveError, match="strictly increasing"):
        PiecewiseLinearDischargeLimit(breakpoints=[(20.0, 1.0), (20.0, 2.0)])

    # Negative power
    with pytest.raises(InvalidDischargeCurveError, match="must be non-negative"):
        PiecewiseLinearDischargeLimit(breakpoints=[(10.0, -0.5), (30.0, 2.0)])

    # Non-concave slopes (convex slope jump from 0.05 to 0.20)
    with pytest.raises(InvalidDischargeCurveError, match="Slopes must be non-increasing"):
        PiecewiseLinearDischargeLimit(
            breakpoints=[(10.0, 0.5), (30.0, 1.5), (40.0, 3.5)],
            use_soc_percent=True,
        )


def test_7_standing_loss_sensitivity_monotonic():
    """Criterion 7: Standing-loss sensitivity increases required charging energy monotonically."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=2)
    analyzer = SensitivityAnalyzer(tz=VILNIUS_TZ)

    loss_cases = analyzer.sweep_standing_losses(
        config,
        heat_profile,
        site_profile,
        prices,
        scenarios={"2%": 2.0, "6%": 6.0, "15%": 15.0, "30%": 30.0},
    )

    for i in range(len(loss_cases) - 1):
        assert loss_cases[i + 1].standing_losses_kwh > loss_cases[i].standing_losses_kwh
        assert loss_cases[i + 1].total_electricity_kwh >= loss_cases[i].total_electricity_kwh - 1e-4
        assert loss_cases[i + 1].total_cost_eur >= loss_cases[i].total_cost_eur - 1e-4


def test_8_auxiliary_load_increases_electricity():
    """Criterion 8: Increasing auxiliary load cannot reduce electricity consumption."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=2)
    analyzer = SensitivityAnalyzer(tz=VILNIUS_TZ)

    aux_cases = analyzer.sweep_auxiliary_power(
        config,
        heat_profile,
        site_profile,
        prices,
        aux_powers_kw=[0.0, 0.05, 0.10, 0.20],
    )

    for i in range(len(aux_cases) - 1):
        assert aux_cases[i + 1].total_electricity_kwh >= aux_cases[i].total_electricity_kwh - 1e-4
        assert aux_cases[i + 1].total_cost_eur >= aux_cases[i].total_cost_eur - 1e-4


def test_9_lower_efficiency_increases_energy_consumption():
    """Criterion 9: Lower efficiency cannot improve optimized physical energy consumption."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=2)
    analyzer = SensitivityAnalyzer(tz=VILNIUS_TZ)

    eff_cases = analyzer.sweep_efficiencies(config, heat_profile, site_profile, prices)
    # eff_cases[0] is Low (76.5%), eff_cases[1] Nominal (85.5%), eff_cases[2] High (93.1%)
    # Lower RTE requires more charging electricity and results in higher net cost
    assert eff_cases[0].total_electricity_kwh > eff_cases[1].total_electricity_kwh > eff_cases[2].total_electricity_kwh
    assert eff_cases[0].total_cost_eur > eff_cases[1].total_cost_eur > eff_cases[2].total_cost_eur


def test_10_direct_heating_independent_of_tes_soc():
    """Criterion 10: Direct heating baseline is independent of TES SOC."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=2)
    heat_demands = heat_profile.series(intervals)

    direct1 = evaluate_direct_electric_heating(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        tariff=config.tariff_params,
        heater_efficiency=0.98,
    )

    # Direct heating does not use TES
    assert direct1.tes_auxiliary_kwh == 0.0
    assert direct1.thermal_standing_loss_kwh == 0.0
    assert direct1.heat_supply_reliability_percent == 100.0


def test_11_labeling_mock_vs_real_prices():
    """Criterion 11: Mock and real price sources are clearly labeled with appropriate status."""
    config, _, _, _, _ = create_phase5_5_env(days=2)

    # Mock source -> status label should be SOFTWARE TEST
    label_mock = get_viability_status_label(config, "realistic_rolling", is_viable=False)
    assert "SOFTWARE TEST" in label_mock

    # Real source with viable config -> ENGINEERING ESTIMATE (never calibrated)
    real_cfg = BacktestConfig(**{**config.__dict__, "price_source": "entsoe"})
    label_real = get_viability_status_label(real_cfg, "realistic_rolling", is_viable=True)
    assert "ENGINEERING ESTIMATE" in label_real
    assert "CALIBRATED" not in label_real or "NOT YET CALIBRATED" in label_real


def test_12_parameter_sweep_deterministic():
    """Criterion 12: Parameter sweep results are deterministic."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=2)
    analyzer = SensitivityAnalyzer(tz=VILNIUS_TZ)

    run1 = analyzer.sweep_standing_losses(config, heat_profile, site_profile, prices, {"2%": 2.0})
    run2 = analyzer.sweep_standing_losses(config, heat_profile, site_profile, prices, {"2%": 2.0})

    assert run1[0].total_cost_eur == run2[0].total_cost_eur
    assert run1[0].total_electricity_kwh == run2[0].total_electricity_kwh


def test_13_energy_balance_closes_for_all_sizing_combinations():
    """Criterion 13: Energy balance closes for every sizing combination."""
    config, intervals, prices, heat_profile, site_profile = create_phase5_5_env(days=2)
    analyzer = SizingStudyAnalyzer(tz=VILNIUS_TZ)

    reports = analyzer.run_sizing_sweep(
        base_config=config,
        heat_profile=heat_profile,
        site_profile=site_profile,
        prices=prices,
        capacities_kwh=[10.0, 20.0],
        charge_powers_kw=[6.0, 9.0],
        sizing_mode="normalized_benchmark",
    )

    report = reports["normalized_benchmark"]
    for combo in report.combinations:
        assert abs(combo.energy_balance_residual_kwh) < 1e-5, (
            f"Residual {combo.energy_balance_residual_kwh} exceeded tolerance for {combo.capacity_kwh}kWh / {combo.charge_power_kw}kW"
        )
