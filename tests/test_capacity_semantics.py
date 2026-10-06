"""Tests validating physical capacity semantics vs. dispatchable capacity.

Proves:
1. 15 kWh full-span capacity with 10% minimum SOC yields exactly 13.5 kWh dispatchable capacity.
2. Physical SOC = 0 maps to 80 °C.
3. Physical SOC = 1 maps to 300 °C.
4. Optimizer minimum SOC = 0.10 maps to > 80 °C and ~105 °C for quartz model.
5. Optimizer minimum energy for 15 kWh is 1.5 kWh.
6. Sand mass for 15 kWh full-span remains approximately 275.7 kg.
7. Changing optimizer_soc_min_fraction does NOT change physical sand mass.
8. Changing optimizer_soc_min_fraction DOES change dispatchable capacity.
9. Changing physical temperature span DOES change specific energy and sand mass.
10. HX Pmax at optimizer minimum SOC uses physical temperature corresponding to that SOC (~105 °C), not T_min_physical (80 °C).
11. Sizing studies interpret 10/15/20/30 kWh as full-span physical capacities.
12. Old ambiguous config fields are either migrated explicitly or rejected with a clear error.
"""

from __future__ import annotations

import pytest

from app.backtest.domain import SizingCombinationResult
from app.config.parameters import TESParameters
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper


def test_1_full_span_capacity_with_10pct_min_soc_yields_13_5kwh_dispatchable() -> None:
    """1. 15 kWh full-span capacity with 10% minimum SOC yields exactly 13.5 kWh dispatchable capacity."""
    params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        optimizer_soc_max_fraction=1.00,
    )
    assert params.thermal_capacity_full_span_kwh == 15.0
    assert params.optimizer_soc_min_fraction == 0.10
    assert params.optimizer_soc_max_fraction == 1.00
    assert params.dispatchable_capacity_kwh == pytest.approx(13.5, abs=1e-6)


def test_2_physical_soc_zero_maps_to_80c() -> None:
    """2. Physical SOC = 0 maps to 80 °C."""
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    t_0 = mapper.temperature_from_soc_fraction(0.0)
    assert t_0 == pytest.approx(80.0, abs=1e-4)


def test_3_physical_soc_one_maps_to_300c() -> None:
    """3. Physical SOC = 1 maps to 300 °C."""
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    t_1 = mapper.temperature_from_soc_fraction(1.0)
    assert t_1 == pytest.approx(300.0, abs=1e-4)


def test_4_optimizer_min_soc_maps_to_approx_105c() -> None:
    """4. Optimizer minimum SOC = 0.10 maps to > 80 °C and approximately ~105 °C for quartz model."""
    params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        physical_temperature_min_c=80.0,
        physical_temperature_max_c=300.0,
    )
    t_min_soc = params.temperature_at_optimizer_min_soc_c
    # Must be strictly hotter than physical zero datum (80 °C)
    assert t_min_soc > 80.0
    # Specifically ~104.71 °C
    assert t_min_soc == pytest.approx(104.71, abs=0.5)


def test_5_optimizer_min_energy_for_15kwh_is_1_5kwh() -> None:
    """5. Optimizer minimum energy for 15 kWh is 1.5 kWh."""
    params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        optimizer_soc_max_fraction=1.00,
    )
    assert params.soc_min_energy_kwh == pytest.approx(1.5, abs=1e-6)
    assert params.soc_max_energy_kwh == pytest.approx(15.0, abs=1e-6)
    assert (params.soc_max_energy_kwh - params.soc_min_energy_kwh) == pytest.approx(13.5, abs=1e-6)


def test_6_sand_mass_for_15kwh_full_span_is_approx_275_7kg() -> None:
    """6. Sand mass for 15 kWh full-span remains approximately 275.7 kg."""
    params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        physical_temperature_min_c=80.0,
        physical_temperature_max_c=300.0,
    )
    assert params.sand_mass_kg == pytest.approx(275.7, abs=0.2)


def test_7_changing_optimizer_soc_min_fraction_does_not_change_sand_mass() -> None:
    """7. Changing optimizer_soc_min_fraction does NOT change physical sand mass."""
    p_10 = TESParameters(thermal_capacity_full_span_kwh=15.0, optimizer_soc_min_fraction=0.10)
    p_20 = TESParameters(thermal_capacity_full_span_kwh=15.0, optimizer_soc_min_fraction=0.20)
    p_30 = TESParameters(thermal_capacity_full_span_kwh=15.0, optimizer_soc_min_fraction=0.30)

    assert p_10.sand_mass_kg == pytest.approx(275.7, abs=0.2)
    assert p_20.sand_mass_kg == pytest.approx(p_10.sand_mass_kg, rel=1e-9)
    assert p_30.sand_mass_kg == pytest.approx(p_10.sand_mass_kg, rel=1e-9)


def test_8_changing_optimizer_soc_min_fraction_does_change_dispatchable_capacity() -> None:
    """8. Changing optimizer_soc_min_fraction DOES change dispatchable capacity."""
    p_10 = TESParameters(thermal_capacity_full_span_kwh=15.0, optimizer_soc_min_fraction=0.10)
    p_20 = TESParameters(thermal_capacity_full_span_kwh=15.0, optimizer_soc_min_fraction=0.20)

    assert p_10.dispatchable_capacity_kwh == pytest.approx(13.5, abs=1e-6)
    assert p_20.dispatchable_capacity_kwh == pytest.approx(12.0, abs=1e-6)
    assert p_10.dispatchable_capacity_kwh != p_20.dispatchable_capacity_kwh


def test_9_changing_physical_temperature_span_changes_specific_energy_and_sand_mass() -> None:
    """9. Changing the physical temperature span DOES change specific energy and sand mass."""
    mapper_80_300 = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    mapper_100_250 = ThermalStateMapper(t_min_c=100.0, t_max_c=250.0)

    q_wide = mapper_80_300.specific_stored_energy_kwh_per_kg()
    q_narrow = mapper_100_250.specific_stored_energy_kwh_per_kg()
    assert q_narrow < q_wide

    p_wide = TESParameters(
        physical_temperature_min_c=80.0,
        physical_temperature_max_c=300.0,
        thermal_capacity_full_span_kwh=15.0,
    )
    p_narrow = TESParameters(
        physical_temperature_min_c=100.0,
        physical_temperature_max_c=250.0,
        thermal_capacity_full_span_kwh=15.0,
    )
    assert p_narrow.sand_mass_kg > p_wide.sand_mass_kg


def test_10_hx_pmax_at_optimizer_min_soc_uses_physical_temperature() -> None:
    """10. HX Pmax at optimizer minimum SOC uses physical temperature corresponding to that SOC (~105 °C), not T_min_physical (80 °C)."""
    params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        physical_temperature_min_c=80.0,
        physical_temperature_max_c=300.0,
        overall_u_w_m2k=9.0,
        airflow_m3_h=80.0,
        air_inlet_temperature_c=40.0,
    )
    hx = HelicalAirHXModel(
        hx_area_m2=params.hx_area_m2,
        overall_u_w_m2k=params.overall_u_w_m2k,
        airflow_m3_h=params.airflow_m3_h,
        air_inlet_temperature_c=params.air_inlet_temperature_c,
    )
    pmax_at_80c = hx.p_max_at_temperature_kw(params.physical_temperature_min_c)
    pmax_at_min_soc = hx.p_max_at_temperature_kw(params.temperature_at_optimizer_min_soc_c)
    pmax_from_fraction = hx.p_max_at_soc_fraction_kw(params.optimizer_soc_min_fraction)

    assert pmax_at_min_soc > pmax_at_80c
    assert pmax_at_80c == pytest.approx(0.429, abs=0.01)
    assert pmax_at_min_soc == pytest.approx(0.694, abs=0.01)
    assert pmax_from_fraction == pytest.approx(pmax_at_min_soc, rel=1e-5)


def test_11_sizing_studies_interpret_capacities_as_full_span() -> None:
    """11. Sizing studies interpret 10/15/20/30 kWh as full-span physical capacities."""
    res_10 = SizingCombinationResult(
        capacity_kwh=10.0,
        charge_power_kw=9.0,
        sizing_mode="fixed_gen0_hx",
        useful_heat_kwh=50.0,
        heat_supply_reliability_percent=100.0,
        electricity_consumed_kwh=55.0,
        average_paid_electricity_price=50.0,
        total_cost_eur=2.75,
        cost_eur_per_mwh_heat=55.0,
        savings_vs_direct_eur=1.0,
        savings_vs_direct_percent=20.0,
        equivalent_cycles=5.0,
        average_soc_kwh=5.0,
        average_soc_percent=50.0,
        minimum_soc_kwh=1.0,
        maximum_soc_kwh=10.0,
        charging_hours=5.0,
        hours_charge_off_during_high_price_periods=2.0,
        energy_balance_residual_kwh=0.0,
    )
    assert res_10.thermal_capacity_full_span_kwh == 10.0
    assert res_10.dispatchable_capacity_kwh == pytest.approx(9.0, abs=1e-6)

    res_30 = SizingCombinationResult(
        capacity_kwh=30.0,
        charge_power_kw=12.0,
        sizing_mode="fixed_gen0_hx",
        useful_heat_kwh=150.0,
        heat_supply_reliability_percent=100.0,
        electricity_consumed_kwh=160.0,
        average_paid_electricity_price=50.0,
        total_cost_eur=8.0,
        cost_eur_per_mwh_heat=53.3,
        savings_vs_direct_eur=3.0,
        savings_vs_direct_percent=25.0,
        equivalent_cycles=5.0,
        average_soc_kwh=15.0,
        average_soc_percent=50.0,
        minimum_soc_kwh=3.0,
        maximum_soc_kwh=30.0,
        charging_hours=12.0,
        hours_charge_off_during_high_price_periods=6.0,
        energy_balance_residual_kwh=0.0,
    )
    assert res_30.thermal_capacity_full_span_kwh == 30.0
    assert res_30.dispatchable_capacity_kwh == pytest.approx(27.0, abs=1e-6)


def test_12_config_migration_and_conflict_rejection() -> None:
    """12. Old ambiguous config fields are either migrated cleanly or rejected with a clear error."""
    # 1. Clean migration of legacy fields
    legacy = TESParameters(capacity_kwh=20.0, soc_min_percent=15.0, soc_max_percent=95.0)
    assert legacy.thermal_capacity_full_span_kwh == 20.0
    assert legacy.optimizer_soc_min_fraction == 0.15
    assert legacy.optimizer_soc_max_fraction == 0.95
    assert legacy.dispatchable_capacity_kwh == pytest.approx(20.0 * (0.95 - 0.15), abs=1e-6)

    # 2. Canonical initialization populates legacy fields
    canonical = TESParameters(
        thermal_capacity_full_span_kwh=25.0,
        optimizer_soc_min_fraction=0.12,
        optimizer_soc_max_fraction=0.98,
    )
    assert canonical.capacity_kwh == 25.0
    assert canonical.soc_min_percent == 12.0
    assert canonical.soc_max_percent == 98.0

    # 3. Conflicting capacity inputs raise ValueError
    with pytest.raises(ValueError, match="Conflicting capacity inputs"):
        TESParameters(capacity_kwh=15.0, thermal_capacity_full_span_kwh=20.0)

    # 4. Conflicting soc_min inputs raise ValueError
    with pytest.raises(ValueError, match="Conflicting soc_min inputs"):
        TESParameters(soc_min_percent=10.0, optimizer_soc_min_fraction=0.20)

    # 5. Conflicting soc_max inputs raise ValueError
    with pytest.raises(ValueError, match="Conflicting soc_max inputs"):
        TESParameters(soc_max_percent=90.0, optimizer_soc_max_fraction=1.00)
