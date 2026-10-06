"""Unit tests for Phase 5.6 reporting, grid headroom, enthalpy formulation, and legacy compatibility.

Verifies:
1. Cheapest-N heuristic reports the same sand mass as optimized TES (~275.7 kg for 15 kWh).
2. Direct electric baseline reports no sand mass (None / N/A).
3. configured_charge_power_limit_kw is distinct from peak_actual_charge_power_kw.
4. Actual charge power respects grid headroom (P_grid_headroom = P_grid_limit - P_site - P_aux).
5. Peak total grid power never exceeds grid limit across any strategy or interval.
6. usable_capacity_kwh == dispatchable_capacity_kwh (13.5 kWh for 15 kWh / 10% reserve).
7. usable_capacity_kwh emits a DeprecationWarning.
8. No canonical physical calculation uses usable_capacity_kwh internally.
9. Relative enthalpy at T_min is zero (H_rel(80 °C) = 0).
10. Relative enthalpy at T_max equals full-span enthalpy (~195,838 J/kg).
11. 10% relative enthalpy maps back to ~104.71 °C.
12. Existing 15 kWh / 13.5 kWh capacity semantics remain unchanged.
"""

from __future__ import annotations

from datetime import date, timedelta, timezone
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo
import pytest

from app.backtest.domain import BacktestConfig
from app.backtest.engine import (
    evaluate_cheapest_n_heuristic,
    evaluate_direct_electric_heating,
    evaluate_perfect_foresight_lp,
    evaluate_realistic_rolling_lp,
)
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import build_intervals, local_day_bounds_utc
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.tes.model import PiecewiseLinearDischargeLimit
from app.tes.thermal import ThermalStateMapper

UTC = timezone.utc
VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


@pytest.fixture
def test_env_1day():
    """Build a 1-day test environment with fluctuating prices."""
    s_date = date(2026, 10, 1)
    e_date = date(2026, 10, 2)
    s_utc, _ = local_day_bounds_utc(s_date, VILNIUS_TZ)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), VILNIUS_TZ)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
    prov = MockPriceProvider(tz=VILNIUS_TZ, resolution_minutes=15, seed=42)
    prices = prov.fetch_day_ahead("LT", s_utc, e_utc).points

    tes_params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        optimizer_soc_max_fraction=1.00,
        max_charge_power_kw=12.0,  # Request 12 kW to test grid headroom clipping
        max_discharge_power_kw=3.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.90,
        standing_loss_percent_per_day=2.0,
        auxiliary_power_kw=0.05,
        physical_temperature_min_c=80.0,
        physical_temperature_max_c=300.0,
    )

    site_params = SiteParameters(
        grid_connection_limit_kw=12.0,
        process_heat_demand_kw=1.5,
        other_loads_kw=2.0,  # 2 kW site load -> grid headroom = 12.0 - 2.0 - 0.05 = 9.95 kW
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
    )

    heat_profile = ConstantHeatDemand(value_kw=1.5)
    site_profile = ConstantSiteLoad(value_kw=2.0)
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    return intervals, prices, config, heat_demands, site_loads


def test_1_cheapest_n_heuristic_reports_same_sand_mass_as_optimized(test_env_1day):
    """1. Cheapest-N heuristic reports the same sand mass as optimized TES (~275.7 kg for 15 kWh)."""
    intervals, prices, config, heat_demands, site_loads = test_env_1day
    m_cheap = evaluate_cheapest_n_heuristic(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )
    m_opt = evaluate_realistic_rolling_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )
    m_fore = evaluate_perfect_foresight_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
    )

    assert m_cheap.sand_mass_kg is not None
    assert m_opt.sand_mass_kg is not None
    assert m_fore.sand_mass_kg is not None

    expected_mass = pytest.approx(275.7, abs=0.2)
    assert m_cheap.sand_mass_kg == expected_mass
    assert m_opt.sand_mass_kg == expected_mass
    assert m_fore.sand_mass_kg == expected_mass
    assert m_cheap.sand_mass_kg == pytest.approx(m_opt.sand_mass_kg, rel=1e-6)


def test_2_direct_electric_baseline_reports_no_sand_mass(test_env_1day):
    """2. Direct electric baseline reports no sand mass (None / N/A)."""
    intervals, prices, config, heat_demands, site_loads = test_env_1day
    m_direct = evaluate_direct_electric_heating(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        tariff=config.tariff_params,
        heater_efficiency=config.direct_heater_efficiency,
    )

    assert m_direct.sand_mass_kg is None
    d = m_direct.to_dict()
    assert d["sand_mass_kg"] is None


def test_3_configured_charge_power_limit_is_distinct_from_peak_actual_charge_power(test_env_1day):
    """3. configured_charge_power_limit_kw is distinct from peak_actual_charge_power_kw."""
    intervals, prices, config, heat_demands, site_loads = test_env_1day
    # Configured is 12 kW, but grid limit is 12 kW with 2 kW site load and 0.05 kW aux load.
    m_opt = evaluate_realistic_rolling_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )

    assert m_opt.configured_charge_power_limit_kw == pytest.approx(12.0, abs=1e-6)
    assert m_opt.peak_actual_charge_power_kw < m_opt.configured_charge_power_limit_kw
    assert m_opt.peak_actual_charge_power_kw == pytest.approx(9.95, abs=0.01)


def test_4_actual_charge_power_respects_grid_headroom(test_env_1day):
    """4. Actual charge power respects grid headroom (P_grid_headroom = P_grid_limit - P_site - P_aux)."""
    intervals, prices, config, heat_demands, site_loads = test_env_1day
    headroom = (
        config.site_params.grid_connection_limit_kw
        - config.site_params.other_loads_kw
        - config.auxiliary_power_kw
    )
    assert headroom == pytest.approx(9.95, abs=1e-6)

    m_opt = evaluate_realistic_rolling_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )
    m_cheap = evaluate_cheapest_n_heuristic(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )
    m_fore = evaluate_perfect_foresight_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
    )

    assert m_opt.peak_actual_charge_power_kw <= headroom + 1e-6
    assert m_cheap.peak_actual_charge_power_kw <= headroom + 1e-6
    assert m_fore.peak_actual_charge_power_kw <= headroom + 1e-6


def test_5_peak_total_grid_power_never_exceeds_grid_limit(test_env_1day):
    """5. Peak total grid power never exceeds grid limit across any strategy."""
    intervals, prices, config, heat_demands, site_loads = test_env_1day
    grid_limit = config.site_params.grid_connection_limit_kw

    m_direct = evaluate_direct_electric_heating(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        tariff=config.tariff_params,
        heater_efficiency=config.direct_heater_efficiency,
    )
    m_cheap = evaluate_cheapest_n_heuristic(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )
    m_opt = evaluate_realistic_rolling_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
        tz=VILNIUS_TZ,
    )
    m_fore = evaluate_perfect_foresight_lp(
        intervals=intervals,
        prices=prices,
        heat_demands=heat_demands,
        site_loads=site_loads,
        config=config,
    )

    for m in [m_direct, m_cheap, m_opt, m_fore]:
        assert m.peak_total_grid_power_kw is not None
        assert m.peak_total_grid_power_kw <= grid_limit + 1e-6
        assert m.grid_connection_limit_kw == pytest.approx(grid_limit, abs=1e-6)


def test_6_usable_capacity_kwh_equals_dispatchable_capacity_kwh():
    """6. usable_capacity_kwh == dispatchable_capacity_kwh (13.5 kWh for 15 kWh / 10% reserve)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        params = TESParameters(
            thermal_capacity_full_span_kwh=15.0,
            optimizer_soc_min_fraction=0.10,
            optimizer_soc_max_fraction=1.00,
        )
        assert params.usable_capacity_kwh == pytest.approx(13.5, abs=1e-6)
        assert params.dispatchable_capacity_kwh == pytest.approx(13.5, abs=1e-6)
        assert params.usable_capacity_kwh == params.dispatchable_capacity_kwh
        assert params.usable_capacity_kwh != params.thermal_capacity_full_span_kwh


def test_7_usable_capacity_kwh_emits_deprecation_warning():
    """7. usable_capacity_kwh emits a DeprecationWarning."""
    params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        optimizer_soc_max_fraction=1.00,
    )
    with pytest.deprecated_call():
        _ = params.usable_capacity_kwh


def test_8_no_canonical_physical_calculation_uses_usable_capacity_kwh():
    """8. No canonical physical calculation uses usable_capacity_kwh internally."""
    root_app = Path(__file__).resolve().parent.parent / "app"
    matches = []
    for py_file in root_app.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if "usable_capacity_kwh" in line:
                # parameters.py is allowed only where it defines the deprecated property
                if py_file.name == "parameters.py":
                    continue
                matches.append(f"{py_file.name}:{idx} -> {line.strip()}")

    assert matches == [], f"Found forbidden usable_capacity_kwh references: {matches}"


def test_9_relative_enthalpy_at_t_min_is_zero():
    """9. Relative enthalpy at T_min is zero (H_rel(80 °C) = 0)."""
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    assert mapper.relative_enthalpy_j_per_kg(80.0) == pytest.approx(0.0, abs=1e-6)
    assert mapper.relative_enthalpy_from_soc_fraction(0.0) == pytest.approx(0.0, abs=1e-6)


def test_10_relative_enthalpy_at_t_max_equals_full_span_enthalpy():
    """10. Relative enthalpy at T_max equals full-span enthalpy (~195,838 J/kg)."""
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    h_max = mapper.relative_enthalpy_j_per_kg(300.0)
    # delta H across 80 °C (353.15 K) to 300 °C (573.15 K)
    assert h_max == pytest.approx(195833.0, abs=10.0)
    assert mapper.relative_enthalpy_from_soc_fraction(1.0) == pytest.approx(h_max, abs=1e-6)


def test_11_ten_percent_relative_enthalpy_maps_back_to_approx_104_71c():
    """11. 10% relative enthalpy maps back to ~104.71 °C."""
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    h_10 = mapper.relative_enthalpy_from_soc_fraction(0.10)
    # 10% of ~195,833 J/kg = ~19,583.3 J/kg
    assert h_10 == pytest.approx(19583.3, abs=5.0)

    t_from_h = mapper.temperature_from_relative_enthalpy(h_10)
    t_from_soc = mapper.temperature_from_soc_fraction(0.10)

    assert t_from_h == pytest.approx(104.71, abs=0.1)
    assert t_from_soc == pytest.approx(104.71, abs=0.1)
    assert t_from_h == pytest.approx(t_from_soc, abs=1e-5)


def test_12_existing_15kwh_and_13_5kwh_capacity_semantics_remain_unchanged():
    """12. Existing 15 kWh / 13.5 kWh capacity semantics remain unchanged."""
    params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        optimizer_soc_max_fraction=1.00,
        physical_temperature_min_c=80.0,
        physical_temperature_max_c=300.0,
    )
    assert params.thermal_capacity_full_span_kwh == 15.0
    assert params.dispatchable_capacity_kwh == pytest.approx(13.5, abs=1e-6)
    assert params.soc_min_energy_kwh == pytest.approx(1.5, abs=1e-6)
    assert params.soc_max_energy_kwh == pytest.approx(15.0, abs=1e-6)
    assert params.sand_mass_kg == pytest.approx(275.7, abs=0.2)
    assert params.temperature_at_optimizer_min_soc_c == pytest.approx(104.71, abs=0.5)
