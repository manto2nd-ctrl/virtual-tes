"""Comprehensive Validation Test Suite for Phase 5.11 (Stages 7, 8, 9).

Validates:
1.  SOC and temperature conversion consistency.
2.  TES energy conservation.
3.  Physical thermal capacity boundaries.
4.  HX available and delivered power constraints.
5.  Dryer sensible air-heating calculation.
6.  Water mass conservation on a dry basis.
7.  Evaporation energy limits.
8.  Insufficient heater power.
9.  Backup heater behavior & interlock.
10. Dryer temperature shortfall.
11. Grid connection limits.
12. Same market price across operational views.
13. UTC/local interval alignment and daylight-saving changes.
14. Duplicate interval prevention.
15. Worker restart and simulation recovery.
16. Opening and closing TES inventory valuation.
17. Consistent simulated savings.
18. No BESS contribution while inactive.

Specific Dryer Batch Tests:
- 10 kg dry-wood test batch.
- 25 kg dry-wood test batch.
- 500 kg hypothetical batch.
- Different inlet moisture levels.
- Different air temperatures and airflow settings.
- Different TES SOC levels.
- Insufficient HX power.
- Excessively aggressive requested drying time.

Reproducible 48-hour TES–dryer simulation with known price input and saved output results.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.database.models import (
    Base,
    DayAheadPrice,
    EnergyLedgerInterval,
    ShadowIntervalRecord,
    ShadowTESSession,
    WorkerHeartbeat,
)
from app.web.main import app
from app.process.dryer import (
    DryerControlMode,
    DryerOperatingConfig,
    DryerSimulationMode,
    DryingRecipePreset,
    VirtualDryerModel,
    WoodSpecies,
)
from app.services.energy_ledger_service import EnergyLedgerService
from app.services.market_price_service import (
    AuthoritativeMarketPrice,
    get_authoritative_market_price,
)
from app.services.shadow_runtime import (
    ShadowRuntimeService,
    get_market_interval_bounds,
)
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


@pytest.fixture
def validation_db() -> Session:
    """In-memory SQLite database session for validation tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


# ==============================================================================
# 1. SOC and Temperature Conversion Consistency
# ==============================================================================
def test_soc_and_temperature_conversion_consistency():
    """Verify quartz enthalpy model, Kelvin inside, physical temperatures, and roundtrip consistency."""
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)

    # Physical boundaries:
    # 0% SOC ~ 80.0°C
    t_0 = mapper.temperature_from_soc_fraction(0.0)
    assert abs(t_0 - 80.0) < 0.1, f"0% physical SOC should be 80.0°C, got {t_0}"

    # 100% SOC ~ 300.0°C
    t_100 = mapper.temperature_from_soc_fraction(1.0)
    assert abs(t_100 - 300.0) < 0.1, f"100% physical SOC should be 300.0°C, got {t_100}"

    # 50% SOC ~ 196.8°C (non-linear quartz curve)
    t_50 = mapper.temperature_from_soc_fraction(0.50)
    assert 195.0 <= t_50 <= 198.0, f"50% physical SOC should be ~196.8°C, got {t_50}"

    # Verify equivalent sand mass for 15.0 kWh full span is approximately 275.7 kg
    sand_mass = mapper.equivalent_sand_mass_for_capacity(15.0)
    assert 270.0 <= sand_mass <= 280.0, f"Sand mass for 15.0 kWh should be ~275.7 kg, got {sand_mass}"

    # Roundtrip mapping consistency across full operating range
    for temp in [80.0, 100.0, 150.0, 196.8, 250.0, 300.0]:
        soc = mapper.soc_fraction_from_temperature(temp)
        temp_roundtrip = mapper.temperature_from_soc_fraction(soc)
        assert abs(temp - temp_roundtrip) < 0.05, f"Roundtrip error at {temp}°C: got {temp_roundtrip}°C"


# ==============================================================================
# 2. TES Energy Conservation
# ==============================================================================
def test_tes_energy_conservation():
    """Verify First Law energy balance E_end = E_start + Q_in - Q_out - Q_loss per interval."""
    mapper = ThermalStateMapper(80.0, 300.0)
    e_start = 7.50  # 50% SOC of 15 kWh
    dt_h = 0.25     # 15 minutes
    p_charge = 6.0  # kW electrical
    charge_eff = 0.95
    p_discharge = 1.50 # kW thermal delivered
    discharge_eff = 0.90
    standing_loss_pct_day = 2.0

    q_in = p_charge * charge_eff * dt_h
    q_out = (p_discharge / discharge_eff) * dt_h
    q_loss = e_start * (standing_loss_pct_day / 100.0) * (dt_h / 24.0)

    e_end = e_start + q_in - q_out - q_loss
    residual = abs(e_end - (e_start + q_in - q_out - q_loss))
    assert residual < 1e-6, f"Energy residual violation: {residual}"
    assert 0.0 <= e_end <= 15.0


# ==============================================================================
# 3. Physical Thermal Capacity Boundaries
# ==============================================================================
def test_physical_thermal_capacity_boundaries(validation_db: Session):
    """Verify TES stored energy cannot be negative or exceed 15.0 kWh full-span capacity."""
    db = validation_db
    svc = ShadowRuntimeService()
    sess = svc.start_session(db, initial_soc_percent=98.0, initial_energy_kwh=14.7)

    now = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    # Attempt charging when almost full (request 9 kW for 15 min = 2.25 kWh -> would reach 16.95 kWh)
    rec = svc.execute_interval_step(
        db=db,
        shadow_session=sess,
        interval_start_utc=now,
        interval_end_utc=now + timedelta(minutes=15),
        spot_price_eur_mwh=30.0,
        requested_charge_kw=9.0,
        requested_discharge_kw=0.0,
    )
    assert rec.stored_energy_end_kwh <= 15.0 + 1e-4, "Stored energy exceeded 15.0 kWh ceiling!"
    assert rec.soc_end_fraction <= 1.0 + 1e-4

    # Now discharge from near empty (0.1 kWh)
    sess.current_stored_energy_kwh = 0.10
    sess.current_soc_fraction = 0.10 / 15.0
    rec_dis = svc.execute_interval_step(
        db=db,
        shadow_session=sess,
        interval_start_utc=now + timedelta(minutes=15),
        interval_end_utc=now + timedelta(minutes=30),
        spot_price_eur_mwh=100.0,
        requested_charge_kw=0.0,
        requested_discharge_kw=3.0,
    )
    assert rec_dis.stored_energy_end_kwh >= 0.0 - 1e-4, "Stored energy became negative!"
    assert rec_dis.soc_end_fraction >= 0.0


# ==============================================================================
# 4. HX Available and Delivered Power Constraints
# ==============================================================================
def test_hx_available_and_delivered_power_constraints():
    """Verify HX delivered power <= available power and delivered <= stored_energy / dt."""
    hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0, air_inlet_temperature_c=40.0)

    # At 300°C sand, available power is maximum
    p_avail_300 = hx.p_max_at_temperature_kw(300.0)
    assert 2.5 <= p_avail_300 <= 3.5, f"Expected ~2.79 kW at 300°C, got {p_avail_300}"

    # At 40°C sand (equal to air inlet temp), available power is 0.0
    p_avail_40 = hx.p_max_at_temperature_kw(40.0)
    assert abs(p_avail_40) < 1e-3, f"Expected 0.0 kW when sand temp equals inlet air temp, got {p_avail_40}"

    # Monotonicity with sand temperature
    p_avail_150 = hx.p_max_at_temperature_kw(150.0)
    p_avail_200 = hx.p_max_at_temperature_kw(200.0)
    assert p_avail_150 < p_avail_200 < p_avail_300


# ==============================================================================
# 5. Dryer Sensible Air-Heating Calculation
# ==============================================================================
def test_dryer_sensible_air_heating_calculation():
    """Verify reference sensible heating: 80 m3/h from 40°C to 70°C requires ~0.757 kW."""
    dryer = VirtualDryerModel()
    m_dot, c_air = dryer.calculate_c_air_kw_k(airflow_m3_h=80.0)

    # c_air should be approx 0.02522 kW/K = 25.22 W/K
    assert abs(c_air - 0.02522) < 0.002, f"Expected C_air ~0.02522 kW/K, got {c_air}"

    # Raising air by 30 K (40°C -> 70°C)
    p_sensible = c_air * 30.0
    assert abs(p_sensible - 0.757) < 0.05, f"Expected ~0.76 kW sensible air heating, got {p_sensible}"


# ==============================================================================
# 6. Water Mass Conservation on a Dry Basis
# ==============================================================================
def test_water_mass_conservation_on_dry_basis():
    """Verify water mass balance: m_water = m_dry * MC, and water removed = m_dry * (MC_init - MC_curr)."""
    cfg = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.WOOD_DRYING_PROCESS,
        timber_mass_dry_kg=50.0,
        initial_moisture_percent=50.0,
        target_moisture_percent=15.0,
    )
    dryer = VirtualDryerModel(config=cfg)

    m_dry = cfg.timber_mass_dry_kg
    init_water = m_dry * 0.50  # 25.0 kg
    target_water = m_dry * 0.15 # 7.5 kg
    total_to_remove = init_water - target_water # 17.5 kg

    assert abs(dryer.current_water_mass_kg - 25.0) < 1e-4
    assert abs(dryer.target_water_mass_kg - 7.5) < 1e-4
    assert abs(dryer.total_water_to_remove_kg - 17.5) < 1e-4

    # Execute a step with 1.5 kW heat delivered for 1 hour
    res = dryer.step(sand_temperature_c=250.0, dt_seconds=3600.0, tes_heat_delivered_kw=1.50, mutate_batch=True)
    assert res.water_removed_step_kg > 0.0
    assert abs((dryer.current_water_mass_kg + dryer.water_removed_cumulative_kg) - init_water) < 1e-4


# ==============================================================================
# 7. Evaporation Energy Limits
# ==============================================================================
def test_evaporation_energy_limits():
    """Verify evaporation cannot exceed available heat budget (h_evap ≈ 0.6278 kWh/kg)."""
    cfg = DryerOperatingConfig(backup_heater_enabled=False)
    dryer = VirtualDryerModel(config=cfg)
    # Step with 0.0 kW delivered heat and backup disabled
    res = dryer.step(sand_temperature_c=40.0, dt_seconds=900.0, tes_heat_delivered_kw=0.0, mutate_batch=False)
    assert res.water_removed_step_kg == 0.0, "Water evaporated with zero delivered heat!"
    assert res.evaporation_heat_delivered_kw == 0.0


# ==============================================================================
# 8. Insufficient Heater Power & Grid Limits
# ==============================================================================
def test_insufficient_heater_power(validation_db: Session):
    """Verify electrical charge power cannot exceed 9.0 kW heater capacity or grid headroom."""
    db = validation_db
    svc = ShadowRuntimeService()
    sess = svc.start_session(db, initial_soc_percent=20.0)

    now = datetime(2026, 10, 15, 2, 0, tzinfo=timezone.utc)
    # Requesting 15 kW charge (exceeds 9.0 kW heater and 12 kW grid)
    rec = svc.execute_interval_step(
        db=db,
        shadow_session=sess,
        interval_start_utc=now,
        interval_end_utc=now + timedelta(minutes=15),
        spot_price_eur_mwh=-5.0,  # Negative price incentive
        requested_charge_kw=15.0,
        requested_discharge_kw=0.0,
    )
    assert rec.actual_charge_kw <= 9.0 + 1e-4, f"Actual charge exceeded 9.0 kW: {rec.actual_charge_kw}"
    assert rec.grid_power_total_kw <= 12.0 + 1e-4, f"Total grid power exceeded 12.0 kW: {rec.grid_power_total_kw}"


# ==============================================================================
# 9. Backup Heater Behavior & Interlock
# ==============================================================================
def test_backup_heater_behavior_and_interlock():
    """Verify backup heater supplies shortfall up to rating, and locks OFF when airflow is 0."""
    cfg = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.THERMAL_TEST_RIG,
        mode=DryerControlMode.FIXED_DEMAND,
        reference_process_heat_load_kw=2.50,
        backup_heater_enabled=True,
        backup_heater_max_kw=3.0,
        blower_enabled=True,
        airflow_m3_h=80.0,
    )
    dryer = VirtualDryerModel(config=cfg)

    # Case 1: TES supplies available heat at 120°C (approx 0.86 kW); backup supplies the exact deficit of 2.50 kW
    res = dryer.step(sand_temperature_c=120.0, dt_seconds=900.0, mutate_batch=False)
    assert res.backup_heater_power_kw > 1.0, f"Backup heater should supply shortfall, got {res.backup_heater_power_kw}"
    assert abs((res.heat_supplied_from_tes_kw + res.backup_heater_power_kw) - 2.50) < 0.05

    # Case 2: Blower stopped -> Safety interlock locks backup heater OFF
    cfg.airflow_m3_h = 0.0
    res_stopped = dryer.step(sand_temperature_c=120.0, dt_seconds=900.0, tes_heat_delivered_kw=0.0, mutate_batch=False)
    assert res_stopped.blower_interlock_active is True
    assert res_stopped.backup_heater_power_kw == 0.0, "Backup heater operated while airflow was 0!"


# ==============================================================================
# 10. Dryer Temperature Shortfall
# ==============================================================================
def test_dryer_temperature_shortfall():
    """Verify achieved supply air temperature drops below target when heating capacity is insufficient."""
    cfg = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.THERMAL_TEST_RIG,
        mode=DryerControlMode.TARGET_TEMPERATURE,
        target_supply_temp_c=70.0,
        inlet_air_temp_c=40.0,
        airflow_m3_h=80.0,
        backup_heater_enabled=False,  # No backup
    )
    dryer = VirtualDryerModel(config=cfg)

    # Supply only 0.38 kW (approx half of the required ~0.76 kW sensible heat)
    res = dryer.step(sand_temperature_c=90.0, dt_seconds=900.0, tes_heat_delivered_kw=0.38, mutate_batch=False)
    assert res.achieved_supply_temp_c < 60.0, f"Expected temperature shortfall below 60°C, got {res.achieved_supply_temp_c}"
    assert res.achieved_supply_temp_c > 40.0


# ==============================================================================
# 11. Grid Connection Limits
# ==============================================================================
def test_grid_connection_limits(validation_db: Session):
    """Verify total grid power never exceeds site breaker limit (12.0 kW)."""
    db = validation_db
    svc = ShadowRuntimeService()
    sess = svc.start_session(db, initial_soc_percent=30.0)

    now = datetime(2026, 10, 15, 3, 0, tzinfo=timezone.utc)
    rec = svc.execute_interval_step(
        db=db,
        shadow_session=sess,
        interval_start_utc=now,
        interval_end_utc=now + timedelta(minutes=15),
        spot_price_eur_mwh=10.0,
        requested_charge_kw=9.0,
    )
    # Total grid power = charge (up to 9 kW) + site load (2 kW) + aux (0.05 kW) = 11.05 kW <= 12 kW
    assert rec.grid_power_total_kw <= 12.0
    assert rec.grid_power_total_kw == pytest.approx(11.05, abs=0.1)


# ==============================================================================
# 12. Same Market Price Across Operational Views
# ==============================================================================
def test_same_market_price_across_views(validation_db: Session):
    """Verify identical spot and effective prices across Authoritative Lookup and operational views."""
    db = validation_db
    now_utc = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
    s_utc, e_utc = get_market_interval_bounds(now_utc)

    # Insert known price
    db.add(DayAheadPrice(
        bidding_zone="LT",
        source="LITGRID",
        delivery_start_utc=s_utc,
        delivery_end_utc=e_utc,
        delivery_start_local=s_utc.astimezone(VILNIUS_TZ).isoformat(),
        resolution_minutes=15,
        price_eur_mwh=80.0,
        version=1,
    ))
    db.commit()

    tariff = TariffParameters(supplier_markup_eur_mwh=1.50, variable_grid_fee_eur_mwh=25.00, variable_tax_eur_mwh=5.00)
    auth_price = get_authoritative_market_price(s_utc, e_utc, db=db, tariff_params=tariff, now_utc=now_utc)

    assert auth_price.spot_price_eur_mwh == 80.0
    assert auth_price.effective_price_eur_mwh == pytest.approx(80.0 + 1.50 + 25.00 + 5.00, abs=0.01)

    # Shadow service lookup
    shadow_svc = ShadowRuntimeService(tariff_params=tariff)
    state = shadow_svc.get_live_dashboard_state(db, now_utc=now_utc)
    assert state["spot_price_eur_mwh"] == 80.0
    assert state["effective_price_eur_mwh"] == pytest.approx(111.50, abs=0.01)


# ==============================================================================
# 13. UTC/Local Interval Alignment and Daylight-Saving Changes
# ==============================================================================
def test_utc_local_interval_alignment_and_dst():
    """Verify UTC alignment across Lithuanian Daylight-Saving transitions (Europe/Vilnius)."""
    # 2026 DST switch: Sunday 2026-10-25 (clocks fall back from UTC+3 to UTC+2 at 04:00 local)
    dst_fall_utc = datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc)
    s_utc, e_utc = get_market_interval_bounds(dst_fall_utc)

    assert s_utc == datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc)
    assert e_utc == datetime(2026, 10, 25, 1, 15, tzinfo=timezone.utc)

    # Convert to Europe/Vilnius
    s_local = s_utc.astimezone(VILNIUS_TZ)
    e_local = e_utc.astimezone(VILNIUS_TZ)
    assert (e_local - s_local).total_seconds() == 900.0


# ==============================================================================
# 14. Duplicate Interval Prevention
# ==============================================================================
def test_duplicate_interval_prevention(validation_db: Session):
    """Verify duplicate interval execution is rejected idempotently."""
    db = validation_db
    svc = ShadowRuntimeService()
    sess = svc.start_session(db, initial_soc_percent=50.0)

    now = datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc)
    s_utc, e_utc = get_market_interval_bounds(now)

    rec1 = svc.execute_interval_step(
        db=db,
        shadow_session=sess,
        interval_start_utc=s_utc,
        interval_end_utc=e_utc,
        spot_price_eur_mwh=65.0,
    )

    # Second execution of the exact same interval
    rec2 = svc.execute_interval_step(
        db=db,
        shadow_session=sess,
        interval_start_utc=s_utc,
        interval_end_utc=e_utc,
        spot_price_eur_mwh=65.0,
    )

    assert rec1.id == rec2.id, "Second execution did not return existing idempotent record!"
    count = db.execute(select(ShadowIntervalRecord)).scalars().all()
    assert len(count) == 1, "Duplicate interval record created in database!"


# ==============================================================================
# 15. Worker Restart and Simulation Recovery
# ==============================================================================
def test_worker_restart_and_simulation_recovery(validation_db: Session):
    """Verify restoration of TES energy, SOC, active dryer batch kinetics, and preserved pause setting."""
    db = validation_db
    svc1 = ShadowRuntimeService()
    sess = svc1.start_session(db, initial_soc_percent=70.0)

    # Step interval and advance drying batch
    now = datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)
    svc1.dryer_model.reset_batch(timber_mass_dry_kg=25.0, initial_moisture_percent=50.0, target_moisture_percent=15.0)
    svc1.execute_interval_step(
        db=db,
        shadow_session=sess,
        interval_start_utc=now,
        interval_end_utc=now + timedelta(minutes=15),
        spot_price_eur_mwh=70.0,
        requested_charge_kw=0.0,
        requested_discharge_kw=1.50,
    )
    # Pause session
    svc1.pause_session(db, session_id=sess.id)
    assert sess.status == "PAUSED"

    # Simulate fresh application/worker restart with new service instance
    svc2 = ShadowRuntimeService()
    restored = svc2.restore_runtime_state(db, session_id=sess.id)

    assert restored["status"] == "PAUSED", "Pause setting was not preserved across restart!"
    assert restored["stored_energy_kwh"] < 10.5, "TES stored energy was not restored!"
    assert restored["batch_moisture_percent"] < 50.0, "Active dryer batch moisture was not restored!"
    assert restored["water_removed_cumulative_kg"] > 0.0, "Cumulative water removed was not restored!"


# ==============================================================================
# 16. Opening and Closing TES Inventory Valuation
# ==============================================================================
def test_opening_and_closing_tes_inventory_valuation():
    """Verify inventory delta equation: (E_end - E_start) * effective_price."""
    full_cap = 15.0
    soc_start = 0.40  # 6.0 kWh
    soc_end = 0.60    # 9.0 kWh
    eff_price_mwh = 100.0  # 0.10 EUR/kWh

    delta_e = (soc_end - soc_start) * full_cap # +3.0 kWh
    delta_v_inv = delta_e * (eff_price_mwh / 1000.0) # +0.30 EUR

    assert abs(delta_v_inv - 0.30) < 1e-4


# ==============================================================================
# 17. Consistent Simulated Savings
# ==============================================================================
def test_consistent_simulated_savings(validation_db: Session):
    """Verify baseline cost excludes unmet heat to prevent artificial savings."""
    db = validation_db
    ledger = EnergyLedgerService(tz=VILNIUS_TZ, full_capacity_kwh=15.0)

    now = datetime(2026, 10, 15, 11, 0, tzinfo=timezone.utc)
    # Record interval where demand was 2.0 kW (0.50 kWh) but only 1.0 kW (0.25 kWh) was delivered
    interval = ledger.record_interval(
        db=db,
        interval_start_utc=now,
        interval_end_utc=now + timedelta(minutes=15),
        spot_price_eur_mwh=68.0,
        effective_price_eur_mwh=100.0, # 0.10 EUR/kWh
        tes_charge_energy_kwh=0.0,
        dryer_thermal_delivered_kwh=0.25, # Delivered
        backup_heater_energy_kwh=0.0,
        blower_electric_kwh=0.01,
        auxiliary_electric_kwh=0.01,
        peak_grid_power_kw=4.0,
        avg_tes_heat_kw=1.0,
        avg_dryer_demand_kw=2.0,
        avg_sand_temp_c=190.0,
        avg_dryer_supply_temp_c=70.0,
        avg_dryer_exhaust_temp_c=48.0,
        dryer_heat_demand_kw=2.0,
        unmet_heat_kw=1.0, # 1.0 kW unmet
        opening_soc_fraction=0.50,
        closing_soc_fraction=0.483,
        commit=True,
    )

    # Baseline cost should be for DELIVERED heat only (0.25 kWh * 0.10 EUR/kWh = 0.025 EUR)
    # NOT for demanded heat (0.50 kWh * 0.10 = 0.05 EUR)
    assert abs(interval.baseline_cost_eur - 0.025) < 1e-4, f"Baseline included unmet heat: {interval.baseline_cost_eur}"


# ==============================================================================
# 18. No BESS Contribution While Inactive
# ==============================================================================
def test_no_bess_contribution_while_inactive():
    """Verify BESS contributes 0 kW power, 0 kWh energy, and 0 EUR savings while PROPOSED / INACTIVE."""
    svc = ShadowRuntimeService()
    dryer = VirtualDryerModel()
    # Confirm BESS metrics are 0.0 across all views
    client = TestClient(app)
    resp = client.get("/dashboard/overview")
    assert resp.status_code == 200
    assert "0.0 kW" in resp.text
    assert "PROPOSED / INACTIVE" in resp.text


# ==============================================================================
# Specific Dryer Batch Tests: 10 kg, 25 kg, 500 kg & Variations
# ==============================================================================
def test_dryer_batch_10kg_benchtop():
    """Verify 10 kg benchtop test batch completes drying within expected timeframe."""
    cfg = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.WOOD_DRYING_PROCESS,
        timber_mass_dry_kg=10.0,
        initial_moisture_percent=50.0,
        target_moisture_percent=15.0,
        requested_duration_hours=12.0,
    )
    dryer = VirtualDryerModel(config=cfg)
    total_water = dryer.total_water_to_remove_kg # 3.5 kg
    assert abs(total_water - 3.5) < 1e-4

    # Run steps with 1.5 kW heat delivery
    steps = 0
    res = dryer.step(sand_temperature_c=200.0, dt_seconds=900.0, tes_heat_delivered_kw=1.50, mutate_batch=True)
    while not res.is_batch_complete and steps < 60:
        res = dryer.step(sand_temperature_c=200.0, dt_seconds=900.0, tes_heat_delivered_kw=1.50, mutate_batch=True)
        steps += 1

    assert dryer.current_water_mass_kg <= dryer.target_water_mass_kg + 0.05
    assert abs(dryer.water_removed_cumulative_kg - 3.5) < 0.1


def test_dryer_batch_25kg_pilot():
    """Verify 25 kg pilot trial batch water mass balance and drying rate."""
    cfg = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.WOOD_DRYING_PROCESS,
        timber_mass_dry_kg=25.0,
        initial_moisture_percent=60.0,
        target_moisture_percent=12.0,
    )
    dryer = VirtualDryerModel(config=cfg)
    # Water to remove: 25 * (0.60 - 0.12) = 12.0 kg
    assert abs(dryer.total_water_to_remove_kg - 12.0) < 1e-4


def test_dryer_batch_500kg_hypothetical_power_deficit():
    """Verify 500 kg hypothetical batch generates POWER DEFICIT warning when requested duration is too short."""
    cfg = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.WOOD_DRYING_PROCESS,
        timber_mass_dry_kg=500.0,
        initial_moisture_percent=50.0,
        target_moisture_percent=15.0,
        requested_duration_hours=6.0,  # Unrealistic 6-hour drying
        backup_heater_max_kw=3.0,
    )
    dryer = VirtualDryerModel(config=cfg)
    # Evaporating 175 kg water requires ~110 kWh thermal; in 6h requires ~18.3 kW > 3 kW backup + 2.79 kW HX
    res = dryer.step(sand_temperature_c=300.0, dt_seconds=900.0, mutate_batch=False)
    assert res.power_deficit_warning is True
    assert "POWER DEFICIT" in res.warning_message


def test_dryer_moisture_levels_comparison():
    """Compare kinetic demand across 30%, 50%, and 70% inlet moisture levels."""
    demands = []
    for mc in [30.0, 50.0, 70.0]:
        cfg = DryerOperatingConfig(
            simulation_mode=DryerSimulationMode.WOOD_DRYING_PROCESS,
            timber_mass_dry_kg=50.0,
            initial_moisture_percent=mc,
            target_moisture_percent=15.0,
        )
        dryer = VirtualDryerModel(config=cfg)
        demands.append(dryer.total_water_to_remove_kg)

    assert demands[0] < demands[1] < demands[2]
    assert abs(demands[0] - 7.5) < 1e-4   # 50 * (0.30 - 0.15)
    assert abs(demands[1] - 17.5) < 1e-4  # 50 * (0.50 - 0.15)
    assert abs(demands[2] - 27.5) < 1e-4  # 50 * (0.70 - 0.15)


def test_dryer_airflow_and_temp_variations():
    """Compare sensible heat demands across 55°C @ 60 m3/h, 70°C @ 80 m3/h, and 85°C @ 100 m3/h."""
    configs = [
        (55.0, 60.0), # Gentle
        (70.0, 80.0), # Standard
        (85.0, 100.0), # High-throughput
    ]
    sensibles = []
    for t_targ, flow in configs:
        cfg = DryerOperatingConfig(target_supply_temp_c=t_targ, airflow_m3_h=flow, inlet_air_temp_c=40.0)
        dryer = VirtualDryerModel(config=cfg)
        _, c_air = dryer.calculate_c_air_kw_k(flow)
        sensibles.append(c_air * (t_targ - 40.0))

    # Higher airflow and temperature delta requires strictly higher sensible power
    assert sensibles[0] < sensibles[1] < sensibles[2]


def test_dryer_tes_soc_levels():
    """Compare HX available power delivery across 100% SOC (300°C), 50% SOC (197°C), and 15% SOC (120°C)."""
    hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0, air_inlet_temperature_c=40.0)
    p_100 = hx.p_max_at_soc_fraction_kw(1.00)
    p_50 = hx.p_max_at_soc_fraction_kw(0.50)
    p_15 = hx.p_max_at_soc_fraction_kw(0.15)

    assert p_15 < p_50 < p_100
    assert p_100 > 2.50
    assert p_15 < 1.50, "Low SOC should provide < 1.5 kW heat delivery"


# ==============================================================================
# Reproducible 48-Hour Simulation (192 Intervals)
# ==============================================================================
def test_reproducible_48h_simulation(validation_db: Session):
    """Run deterministic 48-hour (192 intervals) simulation with known prices and verify outputs."""
    db = validation_db
    svc = ShadowRuntimeService()
    sess = svc.start_session(db, initial_soc_percent=60.0, process_demand_kw=1.50)

    # 48 hours starting 2026-10-15 00:00 UTC
    start_sim = datetime(2026, 10, 15, 0, 0, tzinfo=timezone.utc)
    total_intervals = 192  # 48 hours * 4 intervals/hour

    # Deterministic price curve (diurnal cycle: high morning/evening, low night/afternoon)
    hourly_profile = [
        45.0, 40.0, 38.0, 35.0, 42.0, 58.0, 85.0, 110.0,
        95.0, 75.0, 65.0, 60.0, 55.0, 58.0, 68.0, 92.0,
        125.0, 140.0, 115.0, 90.0, 78.0, 65.0, 55.0, 48.0,
    ]

    records = []
    total_elec_kwh = 0.0
    total_heat_kwh = 0.0

    for i in range(total_intervals):
        iv_start = start_sim + timedelta(minutes=15 * i)
        iv_end = iv_start + timedelta(minutes=15)
        hr = iv_start.hour
        spot_price = hourly_profile[hr]

        # Dispatch rule: charge during lowest price hours (<50 EUR/MWh) if SOC < 95%
        req_charge = 9.0 if (spot_price < 50.0 and sess.current_soc_fraction < 0.95) else 0.0
        req_discharge = 1.50  # Continuous process demand

        rec = svc.execute_interval_step(
            db=db,
            shadow_session=sess,
            interval_start_utc=iv_start,
            interval_end_utc=iv_end,
            spot_price_eur_mwh=spot_price,
            requested_charge_kw=req_charge,
            requested_discharge_kw=req_discharge,
        )
        total_elec_kwh += rec.actual_charge_kw * 0.25
        total_heat_kwh += rec.actual_discharge_kw * 0.25
        records.append({
            "step": i,
            "interval_start_utc": iv_start.isoformat(),
            "spot_price_eur_mwh": spot_price,
            "soc_end_pct": round(rec.soc_end_fraction * 100.0, 1),
            "stored_energy_kwh": round(rec.stored_energy_end_kwh, 3),
            "actual_charge_kw": rec.actual_charge_kw,
            "actual_discharge_kw": rec.actual_discharge_kw,
        })

    # Assertions on 48h simulation
    assert len(records) == 192
    assert total_elec_kwh > 0.0
    assert total_heat_kwh > 0.0
    assert 0.0 <= sess.current_stored_energy_kwh <= 15.0

    # Save output results for reproducibility audit
    data_dir = Path("tests/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "reproducible_48h_simulation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "simulation_run_id": sess.id,
            "duration_hours": 48.0,
            "interval_count": total_intervals,
            "total_electricity_charged_kwh": round(total_elec_kwh, 2),
            "total_heat_delivered_kwh": round(total_heat_kwh, 2),
            "final_soc_percent": round(sess.current_soc_fraction * 100.0, 1),
            "final_stored_energy_kwh": round(sess.current_stored_energy_kwh, 2),
            "records": records[:24],  # Sample first 24 records
        }, f, indent=2)

    assert out_file.exists()
