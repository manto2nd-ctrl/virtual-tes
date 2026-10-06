"""Unit tests for Phase 5: Multi-day Historical Backtesting & Baseline Comparison.

Verifies:
1. Piecewise discharge curve validation (strict monotonicity, non-negativity, concavity).
2. Continuous SOC preservation across calendar-day boundaries without midnight resets.
3. Realistic rolling mode vs Perfect foresight benchmark.
4. Fair baseline comparisons (Direct electric, Cheapest-N heuristic, Optimized TES).
5. Terminal inventory adjustment accounting.
6. Auxiliary electrical load inclusion in costs and constraints.
7. Parametric sensitivity analysis (standing losses and conversion efficiencies).
8. Physical Gen0 viability gating.
9. Exact energy balance closure across multi-day horizons.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest

from app.backtest.domain import BacktestConfig, BacktestStrategyMetrics
from app.backtest.engine import (
    BacktestRunner,
    check_physical_viability,
    evaluate_cheapest_n_heuristic,
    evaluate_direct_electric_heating,
    evaluate_perfect_foresight_lp,
    evaluate_realistic_rolling_lp,
)
from app.backtest.sensitivity import SensitivityAnalyzer
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import build_intervals, local_day_bounds_utc
from app.models.domain import PricePoint
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.tes.model import (
    ConstantDischargeLimit,
    InvalidDischargeCurveError,
    PiecewiseLinearDischargeLimit,
)

UTC = timezone.utc
VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


def create_3day_test_environment():
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

    curve = PiecewiseLinearDischargeLimit(
        breakpoints=[(10.0, 0.5), (30.0, 3.0), (100.0, 3.0)],
        use_soc_percent=True,
    )

    config = BacktestConfig(
        start_date=s_date,
        end_date=e_date,
        mode="realistic_rolling",
        initial_soc_kwh=7.5,
        tes_params=tes_params,
        site_params=site_params,
        tariff_params=tariff_params,
        discharge_limit_curve=curve,
        auxiliary_power_kw=0.05,
        cheapest_n_hours=4,
        bidding_zone="LT",
        price_source="mock",
    )

    heat_profile = ConstantHeatDemand(value_kw=1.5)
    site_profile = ConstantSiteLoad(value_kw=1.0)

    return config, intervals, prices, heat_profile, site_profile


def test_piecewise_discharge_limit_validation():
    """Verifies strict mathematical validation of piecewise discharge limit curves."""
    # 1. Valid curve passes
    curve = PiecewiseLinearDischargeLimit(
        breakpoints=[(10.0, 0.5), (30.0, 3.0), (100.0, 3.0)],
        use_soc_percent=True,
    )
    assert len(curve.breakpoints) == 3

    # 2. Too few breakpoints (< 2)
    with pytest.raises(InvalidDischargeCurveError, match="at least 2 breakpoints"):
        PiecewiseLinearDischargeLimit(breakpoints=[(10.0, 1.0)])

    # 3. Non-strictly increasing SOC
    with pytest.raises(InvalidDischargeCurveError, match="strictly increasing"):
        PiecewiseLinearDischargeLimit(breakpoints=[(10.0, 1.0), (10.0, 2.0)])

    with pytest.raises(InvalidDischargeCurveError, match="strictly increasing"):
        PiecewiseLinearDischargeLimit(breakpoints=[(30.0, 1.0), (20.0, 2.0)])

    # 4. Negative discharge capability
    with pytest.raises(InvalidDischargeCurveError, match="must be non-negative"):
        PiecewiseLinearDischargeLimit(breakpoints=[(10.0, -0.5), (30.0, 2.0)])

    # 5. Decreasing capability with SOC (must be non-decreasing)
    with pytest.raises(InvalidDischargeCurveError, match="non-decreasing"):
        PiecewiseLinearDischargeLimit(breakpoints=[(10.0, 3.0), (30.0, 1.0)])

    # 6. Non-concave / strictly increasing slopes (convex upper-bound violation)
    # Slope 1: (2.0 - 0.5)/(30 - 10) = 1.5/20 = 0.075
    # Slope 2: (6.0 - 2.0)/(50 - 30) = 4.0/20 = 0.20 > 0.075 -> convex!
    with pytest.raises(InvalidDischargeCurveError, match="Slopes must be non-increasing"):
        PiecewiseLinearDischargeLimit(
            breakpoints=[(10.0, 0.5), (30.0, 2.0), (50.0, 6.0)],
            use_soc_percent=True,
        )


def test_realistic_rolling_preserves_soc_continuously():
    """Requirement 1: Backtests must preserve SOC continuously across calendar-day boundaries."""
    config, intervals, prices, heat_profile, site_profile = create_3day_test_environment()
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    metrics = evaluate_realistic_rolling_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )

    # Useful heat should be served with high reliability
    assert metrics.heat_supply_reliability_percent > 99.0
    # Energy balance residual must be negligible
    assert abs(metrics.energy_balance_residual_kwh) < 1e-5
    # Auxiliary load should be exactly 50W * 72h = 3.6 kWh
    assert abs(metrics.tes_auxiliary_kwh - (0.05 * 72.0)) < 1e-4


def test_baseline_direct_electric_heating():
    """Requirement 8: Fair baseline direct resistive heating."""
    config, intervals, prices, heat_profile, site_profile = create_3day_test_environment()
    heat_demands = heat_profile.series(intervals)

    direct = evaluate_direct_electric_heating(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        tariff=config.tariff_params,
        heater_efficiency=config.direct_heater_efficiency,
    )

    # 1.5 kW * 72h = 108.0 kWh useful heat delivered
    assert abs(direct.useful_heat_kwh - 108.0) < 1e-4
    assert direct.unmet_heat_kwh == 0.0
    assert direct.heat_supply_reliability_percent == 100.0
    assert direct.tes_auxiliary_kwh == 0.0
    assert direct.thermal_standing_loss_kwh == 0.0
    # Heater electricity = useful_heat / 0.98
    expected_elec = 108.0 / 0.98
    assert abs(direct.electricity_kwh - expected_elec) < 1e-4
    assert direct.total_cost_eur > 0.0
    assert direct.energy_balance_residual_kwh == 0.0


def test_baseline_cheapest_n_with_inventory_adjustment():
    """Requirement 8 & 9: Cheapest-N heuristic with fair terminal inventory adjustment."""
    config, intervals, prices, heat_profile, site_profile = create_3day_test_environment()
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    heuristic = evaluate_cheapest_n_heuristic(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )

    # Energy balance must close
    assert abs(heuristic.energy_balance_residual_kwh) < 1e-5
    # Inventory adjustment must be present if final SOC differs from initial
    delta_soc = heuristic.final_soc_kwh - config.initial_soc_kwh
    if abs(delta_soc) > 1e-3:
        expected_total = (
            heuristic.raw_electricity_cost_eur
            + heuristic.terminal_inventory_adjustment_eur
            + heuristic.unmet_heat_cost_eur
        )
        assert abs(heuristic.total_cost_eur - expected_total) < 1e-5


def test_perfect_foresight_benchmark_mode():
    """Requirement 2B: Perfect foresight optimizer benchmark across complete horizon."""
    config, intervals, prices, heat_profile, site_profile = create_3day_test_environment()
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    foresight = evaluate_perfect_foresight_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
    )

    assert foresight.mode == "perfect_foresight"
    assert foresight.heat_supply_reliability_percent == 100.0
    assert foresight.unmet_heat_kwh == 0.0
    assert abs(foresight.energy_balance_residual_kwh) < 1e-5
    assert "Theoretical upper-bound benchmark" in foresight.viability_caveats[0] or "Theoretical upper-bound benchmark" in foresight.viability_caveats[-1]


def test_backtest_runner_and_comparative_savings():
    """Requirement 8 & 9: Full BacktestRunner comparing direct, heuristic, and rolling."""
    config, intervals, prices, heat_profile, site_profile = create_3day_test_environment()

    runner = BacktestRunner(tz=VILNIUS_TZ)
    report = runner.run(config, heat_profile, site_profile, prices)

    assert "direct" in report.strategies
    assert "heuristic" in report.strategies
    assert "optimized" in report.strategies

    direct = report.strategies["direct"]
    opt = report.strategies["optimized"]

    # Optimized TES should achieve significant cost savings vs direct resistive heating
    assert opt.total_cost_eur < direct.total_cost_eur
    assert opt.savings_vs_direct_eur > 0.0
    assert opt.savings_vs_direct_percent > 20.0


def test_physical_gen0_viability_gating():
    """Requirement 10: Do not present backtest economics as physical Gen0 viability unless
    discharge derating, realistic losses, auxiliary loads, and real market prices are enabled.
    """
    config, _, _, _, _ = create_3day_test_environment()

    # Default config has mock price source -> must fail viability
    is_viable, caveats = check_physical_viability(config)
    assert not is_viable
    assert any("Mock synthetic market data" in c for c in caveats)

    # Config with ConstantDischargeLimit -> must fail viability
    bad_curve_cfg = BacktestConfig(
        **{**config.__dict__, "price_source": "entsoe", "discharge_limit_curve": ConstantDischargeLimit()}
    )
    is_viable, caveats = check_physical_viability(bad_curve_cfg)
    assert not is_viable
    assert any("Discharge curve is constant" in c for c in caveats)

    # Config with standing loss < 2.0%/day -> must fail viability
    low_loss_params = config.tes_params.model_copy(update={"standing_loss_percent_per_day": 0.5})
    low_loss_cfg = BacktestConfig(
        **{**config.__dict__, "price_source": "entsoe", "tes_params": low_loss_params}
    )
    is_viable, caveats = check_physical_viability(low_loss_cfg)
    assert not is_viable
    assert any("optimistic" in c for c in caveats)

    # Config with 0 auxiliary power -> must fail viability
    zero_aux_cfg = BacktestConfig(
        **{**config.__dict__, "price_source": "entsoe", "auxiliary_power_kw": 0.0}
    )
    is_viable, caveats = check_physical_viability(zero_aux_cfg)
    assert not is_viable
    assert any("Zero auxiliary" in c for c in caveats)

    # Viable physical Gen0 config
    viable_cfg = BacktestConfig(
        **{**config.__dict__, "price_source": "entsoe"}
    )
    is_viable, caveats = check_physical_viability(viable_cfg)
    assert is_viable
    assert len(caveats) == 0


def test_parametric_sensitivity_analysis():
    """Requirement 5 & 6: Sensitivity sweep across standing losses and conversion efficiencies."""
    config, intervals, prices, heat_profile, site_profile = create_3day_test_environment()

    analyzer = SensitivityAnalyzer(tz=VILNIUS_TZ)

    # 1. Standing loss sweep
    loss_results = analyzer.sweep_standing_losses(config, heat_profile, site_profile, prices)
    assert len(loss_results) == 8
    # Losses in kWh must strictly increase with loss %
    for i in range(len(loss_results) - 1):
        assert loss_results[i + 1].standing_losses_kwh > loss_results[i].standing_losses_kwh
        assert loss_results[i + 1].total_cost_eur >= loss_results[i].total_cost_eur

    # 2. Efficiency sweep
    eff_results = analyzer.sweep_efficiencies(config, heat_profile, site_profile, prices)
    assert len(eff_results) == 3
    # Higher RTE should lead to lower total cost
    assert eff_results[0].total_cost_eur > eff_results[1].total_cost_eur > eff_results[2].total_cost_eur
