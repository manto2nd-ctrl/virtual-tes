"""Stage 4 — Virtual Wood Dryer Validation Test Suite.

Validates:
  1. Operating Reference: Sensible air heating (C_air ~ 25.22 W/K, dT = 30 K -> P_air ~ 0.76 kW).
  2. Mode 1 (Thermal Test Rig): Pure thermal validation of Gen0 TES & HX on instrumented duct;
     zero fictitious moisture removal.
  3. Mode 2 (Wood Drying Process): Moisture mass balance on dry basis:
     Water_removed = M_dry * (MC_initial - MC_current).
     For 500 kg (50% -> 15%): 175 kg water, theoretical evaporation energy ~110.25 kWh.
  4. Timber Load Configuration: 5 kg, 10 kg, 25 kg, 50 kg experimental loads up to 500 kg design.
  5. Power Deficit Warning: Displayed when schedule duty exceeds combined heating capacity.
  6. Energy Allocation & Physical Shortfall: Sensible air heating takes priority,
     latent heat strictly clamped to available energy budget, temperature shortfall under thermal pinch.
  7. Blower Interlock Safety: When airflow is zero, backup heater is locked OFF.
"""

import pytest
from app.process.dryer import (
    DryerControlMode,
    DryerOperatingConfig,
    DryerSimulationMode,
    TimberLoadPreset,
    VirtualDryerModel,
    WoodSpecies,
)


def test_section_a_sensible_air_heating_reference():
    """Verify sensible air-heating power: C_air ~ 25.22 W/K, delta_T = 30 K -> P_air ~ 0.76 kW."""
    model = VirtualDryerModel()
    model.config.simulation_mode = DryerSimulationMode.THERMAL_TEST_RIG
    model.config.target_supply_temp_c = 70.0
    model.config.inlet_air_temp_c = 40.0
    model.config.airflow_m3_h = 80.0

    m_dot, c_air_kw_k = model.calculate_c_air_kw_k(80.0)
    # Mass flow at 80 m3/h: 80 * 1.134 / 3600 ~ 0.0252 kg/s
    assert 0.024 < m_dot < 0.026
    # Heat capacity rate C_air: ~ 0.0253 kW/K = 25.3 W/K (~ 25.22 W/K reference)
    c_air_w_k = c_air_kw_k * 1000.0
    assert 24.5 < c_air_w_k < 26.0

    # Sensible heating delta T = 70 - 40 = 30 K
    p_sensible = c_air_kw_k * (70.0 - 40.0)
    assert 0.74 < p_sensible < 0.78  # ~ 0.76 kW

    res = model.step(sand_temperature_c=300.0)
    assert abs(res.sensible_air_heat_kw - p_sensible) < 0.05
    assert res.achieved_supply_temp_c == pytest.approx(70.0, abs=0.5)


def test_section_b_mode1_thermal_test_rig():
    """Verify Mode 1 (Thermal Test Rig): Pure thermal duct validation, zero fictitious moisture removal."""
    model = VirtualDryerModel()
    model.config.simulation_mode = DryerSimulationMode.THERMAL_TEST_RIG
    model.config.reference_process_heat_load_kw = 1.50
    model.config.target_supply_temp_c = 70.0
    model.config.inlet_air_temp_c = 40.0
    model.config.airflow_m3_h = 80.0

    res = model.step(sand_temperature_c=250.0)

    # In test rig mode, no fictitious drying occurs
    assert res.simulation_mode == DryerSimulationMode.THERMAL_TEST_RIG
    assert res.current_moisture_percent is None
    assert res.batch_progress_percent is None
    assert res.estimated_remaining_hours is None
    assert res.water_removed_total_kg == 0.0
    assert res.evaporation_heat_delivered_kw == 0.0

    # Thermal heat delivered from TES is measured accurately
    assert res.useful_heat_delivered_kw > 0.0
    assert res.sensible_air_heat_kw > 0.70
    assert res.achieved_supply_temp_c > 65.0


def test_section_c_moisture_mass_balance_and_energy():
    """Verify moisture mass balance on dry basis:

    For 500 kg dry wood from 50% to 15%:
    Water to remove = 500 * (0.50 - 0.15) = 175 kg.
    Theoretical evaporation energy = 175 * 0.6278 kWh/kg = 110.25 kWh.
    """
    model = VirtualDryerModel()
    model.config.simulation_mode = DryerSimulationMode.WOOD_DRYING_PROCESS
    model.reset_batch(
        timber_mass_dry_kg=500.0,
        initial_moisture_percent=50.0,
        target_moisture_percent=15.0,
        wood_species=WoodSpecies.PINE,
    )

    # 1. Total water to remove
    assert model.total_water_to_remove_kg == pytest.approx(175.0, abs=0.1)

    # 2. Theoretical evaporation energy requirement: 175 kg * 0.6278 kWh/kg ~ 110.25 kWh
    theoretical_evap_kwh = model.total_water_to_remove_kg * model.config.latent_heat_evaporation_kwh_kg
    assert theoretical_evap_kwh == pytest.approx(110.25, abs=0.5)

    # 3. Time bounds at 1.5 kW total:
    # At 1.5 kW dedicated to evaporation alone: 110.25 / 1.5 ~ 73.5 hours (> 73 hours)
    t_evap_only = theoretical_evap_kwh / 1.50
    assert t_evap_only > 73.0

    # With continuous 0.76 kW sensible air heating from 1.5 kW budget, remaining ~0.74 kW
    # gives 110.25 / 0.74 ~ 149 hours
    t_with_sensible = theoretical_evap_kwh / (1.50 - 0.76)
    assert t_with_sensible > 145.0


def test_section_d_timber_load_presets_scaling():
    """Verify small experimental loads: 5 kg, 10 kg, 25 kg, 50 kg and 500 kg."""
    loads_to_test = [5.0, 10.0, 25.0, 50.0, 500.0]

    for mass in loads_to_test:
        model = VirtualDryerModel()
        model.config.simulation_mode = DryerSimulationMode.WOOD_DRYING_PROCESS
        model.reset_batch(
            timber_mass_dry_kg=mass,
            initial_moisture_percent=50.0,
            target_moisture_percent=15.0,
        )

        expected_water = mass * (0.50 - 0.15)
        assert model.total_water_to_remove_kg == pytest.approx(expected_water, abs=0.01)

        # Step simulation at nominal sand temperature (250 C)
        res = model.step(sand_temperature_c=250.0, dt_seconds=900.0)
        assert res.timber_mass_dry_kg == mass
        assert res.current_moisture_percent <= 50.0
        assert res.water_removed_total_kg > 0.0


def test_section_d_power_deficit_warning():
    """Verify warning when desired batch duration requires more thermal power than TES + backup can supply."""
    model = VirtualDryerModel()
    model.config.simulation_mode = DryerSimulationMode.WOOD_DRYING_PROCESS
    model.config.backup_heater_enabled = False  # No backup heater

    # Attempt to dry 500 kg timber (175 kg water) in just 4 hours
    # Required evaporation rate = 175 / 4 = 43.75 kg/h -> ~ 27.5 kW required duty!
    # Combined capacity at 300 C sand bed is ~ 2.79 kW.
    model.reset_batch(timber_mass_dry_kg=500.0, initial_moisture_percent=50.0, target_moisture_percent=15.0)
    model.config.requested_duration_hours = 4.0

    res = model.step(sand_temperature_c=300.0)

    assert res.power_deficit_warning is True
    assert res.warning_message is not None
    assert "POWER DEFICIT" in res.warning_message
    assert "exceeding combined heating capacity" in res.warning_message


def test_section_e_energy_allocation_sensible_priority_and_clamping():
    """Verify First Law energy allocation:

    1. Sensible air heating takes priority to heat airflow to achievable supply temp.
    2. Evaporation energy cannot exceed physically available heat budget.
    """
    model = VirtualDryerModel()
    model.config.simulation_mode = DryerSimulationMode.WOOD_DRYING_PROCESS
    model.reset_batch(timber_mass_dry_kg=50.0, initial_moisture_percent=50.0, target_moisture_percent=15.0)

    # Case A: High sand temperature (300 C) -> abundant HX capacity (~2.79 kW)
    res_abundant = model.step(sand_temperature_c=300.0, dt_seconds=900.0)
    # Sensible heating satisfied
    assert res_abundant.achieved_supply_temp_c == pytest.approx(70.0, abs=0.5)
    # Evaporation takes remaining heat
    assert res_abundant.evaporation_heat_delivered_kw > 0.0
    # First Law check: sensible + evap + losses <= delivered
    assert res_abundant.sensible_air_heat_kw + res_abundant.evaporation_heat_delivered_kw <= res_abundant.total_heat_delivered_kw + 0.01

    # Case B: Cold sand bed (90 C) -> HX capacity is very low (~0.15 kW)
    # Total available heat is way below 0.76 kW sensible duty!
    model_cold = VirtualDryerModel()
    model_cold.config.simulation_mode = DryerSimulationMode.WOOD_DRYING_PROCESS
    model_cold.config.backup_heater_enabled = False
    model_cold.reset_batch(timber_mass_dry_kg=50.0, initial_moisture_percent=50.0, target_moisture_percent=15.0)

    res_cold = model_cold.step(sand_temperature_c=90.0, dt_seconds=900.0)
    # Must exhibit temperature shortfall (cannot reach 70 C target; achieves ~61.2 C)
    assert res_cold.achieved_supply_temp_c < 70.0
    assert res_cold.achieved_supply_temp_c == pytest.approx(61.2, abs=1.0)
    # Zero or negligible evaporation because meager heat went into slight sensible heating
    assert res_cold.evaporation_heat_delivered_kw == pytest.approx(0.0, abs=0.01)
    assert res_cold.tes_pinch_derated is True

    # At 45 C sand (barely above 40 C inlet), temperature shortfall is severe (< 45 C)
    res_45 = model_cold.step(sand_temperature_c=45.0, dt_seconds=900.0)
    assert res_45.achieved_supply_temp_c < 45.0


def test_blower_interlock_safety():
    """Verify that when airflow is zero, backup heater is locked OFF and no evaporation occurs."""
    model = VirtualDryerModel()
    model.config.airflow_m3_h = 0.0
    model.config.backup_heater_enabled = True  # Enabled in config but must be interlocked

    res = model.step(sand_temperature_c=250.0)

    assert res.blower_interlock_active is True
    assert res.backup_heater_power_kw == 0.0
    assert res.total_heat_delivered_kw == 0.0
    assert res.evaporation_heat_delivered_kw == 0.0
    assert "BLOWER_STOPPED_AIRFLOW_ZERO" in res.alarms
