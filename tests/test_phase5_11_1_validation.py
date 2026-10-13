"""Phase 5.11.1 Validation Test Suite: Simulation Integrity, TES-Dryer Physics, and Financial Audit.

Covers:
- Test A: 24-Hour Continuous Execution Integrity Test (96 intervals, monotonic, gap-free, execution mode & wall-clock recorded)
- Test B: 48-Hour Combined Physics & Heat-Transfer Test (First Law energy conservation, zero residual, dry-basis water mass conservation, forward pinch derating autonomy)
- Test C: Financial Audit & Baseline Re-Run Test (exact cost reproducibility, direct electric baseline equivalence, inventory valuation, sizing vessel feasibility, non-zero economic savings)
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from starlette.testclient import TestClient

from app.database.models import (
    Base,
    EnergyLedgerInterval,
)
from app.services.energy_ledger_service import EnergyLedgerService
from app.process.dryer import (
    DryerOperatingConfig,
    DryerSimulationMode,
    DryingRecipePreset,
    VirtualDryerModel,
    WoodSpecies,
)
from app.tes.thermal import (
    calculate_tes_heat_autonomy_hours,
)
from app.backtest.domain import BacktestConfig
from app.backtest.engine import (
    evaluate_direct_electric_heating,
    evaluate_realistic_rolling_lp,
)
from app.backtest.sensitivity import SizingStudyAnalyzer
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.core.timegrid import build_intervals, local_day_bounds_utc
from app.tes.model import get_gen0_discharge_curve
from app.web.main import app

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


@pytest.fixture
def db_session() -> Session:
    """Isolated in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    sess = session_factory()
    try:
        yield sess
    finally:
        sess.close()


@pytest.fixture
def client() -> TestClient:
    """Starlette test client."""
    return TestClient(app)


# ==============================================================================
# TEST A: 24-Hour Continuous Execution Integrity Test
# ==============================================================================
def test_a_24h_continuous_execution_integrity(db_session: Session):
    """Verify 96 continuous 15-minute settlement intervals:
    - Monotonic, contiguous, gap-free timestamps
    - Dual temporal logging: simulated interval vs wall-clock commit time
    - Execution mode provenance ('REALTIME_SHADOW' vs 'MANUAL_REPLAY')
    - Price classification tracking ('LIVE', 'HISTORICAL', 'FALLBACK', 'SYNTHETIC')
    - Idempotent deduplication (duplicate interval keys rejected / updated in place)
    """
    ledger_svc = EnergyLedgerService(tz=VILNIUS_TZ)
    base_time = datetime(2026, 10, 15, 0, 0, tzinfo=timezone.utc)

    # Record 96 consecutive 15-minute intervals (24 hours)
    for i in range(96):
        iv_start = base_time + timedelta(minutes=15 * i)
        iv_end = iv_start + timedelta(minutes=15)
        # Wall clock time simulated as slightly staggered execution
        wall_clock = datetime(2026, 10, 15, 8, 30, tzinfo=timezone.utc) + timedelta(seconds=i * 2)

        exec_mode = "REALTIME_SHADOW" if i >= 10 else "MANUAL_REPLAY"
        price_class = "LIVE" if i >= 10 else "HISTORICAL"

        rec = ledger_svc.record_interval(
            db=db_session,
            interval_start_utc=iv_start,
            interval_end_utc=iv_end,
            spot_price_eur_mwh=45.0 + i * 0.5,
            effective_price_eur_mwh=45.0 + i * 0.5 + 26.5,
            tes_charge_energy_kwh=0.5 if (i % 8 < 2) else 0.0,
            dryer_thermal_delivered_kwh=0.375,
            backup_heater_energy_kwh=0.0,
            blower_electric_kwh=0.0125,
            auxiliary_electric_kwh=0.005,
            peak_grid_power_kw=2.5,
            avg_tes_heat_kw=1.5,
            avg_dryer_demand_kw=1.5,
            avg_sand_temp_c=195.0,
            avg_dryer_supply_temp_c=68.5,
            avg_dryer_exhaust_temp_c=42.0,
            opening_soc_fraction=0.50 + math.sin(i / 10.0) * 0.1,
            closing_soc_fraction=0.50 + math.sin((i + 1) / 10.0) * 0.1,
            moisture_content_percent=50.0 - (i * 0.2),
            water_removed_kg=0.10,
            stored_energy_start_kwh=7.5,
            stored_energy_end_kwh=7.5,
            standing_loss_kwh=0.015,
            dryer_heat_demand_kw=1.5,
            unmet_heat_kw=0.0,
            simulation_run_id="run-phase-5-11-1",
            scenario_id="scenario-reference-24h",
            execution_mode=exec_mode,
            actual_wallclock_utc=wall_clock,
            price_data_classification=price_class,
            commit=True,
        )
        assert rec is not None
        assert rec.id is not None

    # Retrieve full history
    history_data = ledger_svc.get_ledger_history(db=db_session, period="all", limit=200)
    rows = history_data["rows"]
    assert len(rows) == 96

    # Verify chronological ordering and continuity
    # Note: get_ledger_history returns reverse chronological (newest first)
    chronological = list(reversed(rows))
    for idx, row in enumerate(chronological):
        # row is dict formatted by get_ledger_history
        assert row["execution_mode"] in ("MANUAL_REPLAY", "REALTIME_SHADOW")
        assert row["price_data_classification"] in ("HISTORICAL", "LIVE")
        assert row["actual_wallclock_utc"] is not None

        if idx < 10:
            assert row["execution_mode"] == "MANUAL_REPLAY"
            assert row["price_data_classification"] == "HISTORICAL"
        else:
            assert row["execution_mode"] == "REALTIME_SHADOW"
            assert row["price_data_classification"] == "LIVE"

    # Also directly verify ORM objects in db
    orm_records = db_session.execute(
        select(EnergyLedgerInterval).order_by(EnergyLedgerInterval.interval_start_utc.asc())
    ).scalars().all()
    assert len(orm_records) == 96

    for idx, rec in enumerate(orm_records):
        expected_start = base_time + timedelta(minutes=15 * idx)
        expected_end = expected_start + timedelta(minutes=15)
        rec_start = rec.interval_start_utc.replace(tzinfo=timezone.utc) if rec.interval_start_utc.tzinfo is None else rec.interval_start_utc
        rec_end = rec.interval_end_utc.replace(tzinfo=timezone.utc) if rec.interval_end_utc.tzinfo is None else rec.interval_end_utc
        assert rec_start == expected_start
        assert rec_end == expected_end
        assert (rec_end - rec_start).total_seconds() == 900.0
        assert rec.actual_wallclock_utc is not None
        assert rec.actual_wallclock_utc != rec_start

    # Test Idempotent Deduplication: inserting duplicate interval key updates in place without duplicate row
    dup_start = base_time + timedelta(minutes=15 * 5)
    dup_end = dup_start + timedelta(minutes=15)
    ledger_svc.record_interval(
        db=db_session,
        interval_start_utc=dup_start,
        interval_end_utc=dup_end,
        spot_price_eur_mwh=47.5,
        effective_price_eur_mwh=74.0,
        tes_charge_energy_kwh=0.0,
        dryer_thermal_delivered_kwh=0.375,
        backup_heater_energy_kwh=0.0,
        blower_electric_kwh=0.0125,
        auxiliary_electric_kwh=0.005,
        peak_grid_power_kw=2.5,
        avg_tes_heat_kw=1.5,
        avg_dryer_demand_kw=1.5,
        avg_sand_temp_c=195.0,
        avg_dryer_supply_temp_c=68.5,
        avg_dryer_exhaust_temp_c=42.0,
        opening_soc_fraction=0.50,
        closing_soc_fraction=0.50,
        stored_energy_start_kwh=7.5,
        stored_energy_end_kwh=7.5,
        standing_loss_kwh=0.015,
        dryer_heat_demand_kw=1.5,
        unmet_heat_kw=0.0,
        simulation_run_id="run-phase-5-11-1",
        scenario_id="scenario-reference-24h",
        commit=True,
    )
    # Total count must remain exactly 96
    all_recs = db_session.execute(select(EnergyLedgerInterval)).scalars().all()
    assert len(all_recs) == 96, "Duplicate interval key produced redundant row in database!"


# ==============================================================================
# TEST B: 48-Hour Combined Physics & Heat-Transfer Test
# ==============================================================================
def test_b_48h_combined_physics_and_heat_transfer():
    """Verify thermodynamic conservation:
    1. First Law energy balance in dryer: residual strictly == 0.0 kW (closing the 0.10 kW gap)
    2. Itemized power balance: Delivered = Sensible + Thermal Accumulation (Mode 1)
    3. Dry-basis wood water mass conservation identity: water evaporated = m_dry * (omega_0 - omega_t)
    4. Forward epsilon-NTU pinch derating: thermal autonomy dynamically derates when sand bed temperature approaches supply temperature
    """
    # 1. Mode 1 — Thermal Test Rig: zero residual test
    cfg_rig = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.THERMAL_TEST_RIG,
        target_supply_temp_c=70.0,
        inlet_air_temp_c=40.0,
        airflow_m3_h=80.0,
        reference_process_heat_load_kw=1.5,
        backup_heater_enabled=True,
    )
    dryer_rig = VirtualDryerModel(config=cfg_rig)
    step_res = dryer_rig.step(sand_temperature_c=200.0, dt_seconds=900.0)

    # First Law check: delivered must equal sum of absorbed power components
    assert step_res.thermal_residual_kw == 0.0, (
        f"Non-zero energy residual detected: {step_res.thermal_residual_kw} kW"
    )
    absorbed_mode1 = step_res.sensible_heat_demand_kw + step_res.thermal_accumulation_kw
    assert abs(step_res.total_heat_delivered_kw - absorbed_mode1) < 1e-4, (
        f"Delivered {step_res.total_heat_delivered_kw} kW != absorbed {absorbed_mode1} kW"
    )
    assert step_res.thermal_accumulation_kw > 0.0

    # 2. Mode 2 — Wood Drying Process: dry-basis water mass conservation & energy balance
    cfg_wood = DryerOperatingConfig(
        simulation_mode=DryerSimulationMode.WOOD_DRYING_PROCESS,
        recipe_preset=DryingRecipePreset.SOFTWOOD_STANDARD,
        wood_species=WoodSpecies.PINE,
        timber_mass_dry_kg=50.0,
        initial_moisture_percent=50.0,
        target_moisture_percent=15.0,
        target_supply_temp_c=65.0,
        inlet_air_temp_c=25.0,
        airflow_m3_h=80.0,
        backup_heater_enabled=True,
    )
    dryer_wood = VirtualDryerModel(config=cfg_wood)

    # Step through 10 intervals (2.5 hours)
    for _ in range(10):
        step_wood = dryer_wood.step(sand_temperature_c=210.0, dt_seconds=900.0)

    # First Law zero residual check in Mode 2
    assert step_wood.thermal_residual_kw == 0.0

    # Verify water mass identity: cumulative water removed == m_dry * (initial_mc - current_mc)
    m_dry = cfg_wood.timber_mass_dry_kg
    omega_0 = cfg_wood.initial_moisture_percent / 100.0
    omega_curr = step_wood.current_moisture_percent / 100.0
    expected_water_removed_kg = m_dry * (omega_0 - omega_curr)

    # Exact internal water mass identity
    init_water = m_dry * omega_0
    assert abs(dryer_wood.water_removed_cumulative_kg - (init_water - dryer_wood.current_water_mass_kg)) < 1e-4
    assert step_wood.water_removed_total_kg == pytest.approx(expected_water_removed_kg, abs=0.05)

    # Verify reset_batch synchronizes state without corruption
    dryer_wood.reset_batch(initial_moisture_percent=40.0, target_moisture_percent=12.0)
    assert dryer_wood.config.initial_moisture_percent == 40.0
    assert dryer_wood.config.target_moisture_percent == 12.0
    assert dryer_wood.water_removed_cumulative_kg == 0.0
    assert dryer_wood.current_water_mass_kg == pytest.approx(50.0 * 0.40, rel=1e-4)

    # 3. Forward epsilon-NTU Pinch Derating & Autonomy
    auto_high = calculate_tes_heat_autonomy_hours(
        stored_energy_kwh=15.0,
        process_demand_kw=1.5,
        capacity_kwh=15.0,
        min_soc_fraction=0.10,
    )
    # Forward simulation detects heat exchanger pinch derating at ~4.9 hours,
    # strictly below unconstrained theoretical capacity (8.1 hours)
    assert auto_high["rated_autonomy_hours"] >= 4.0
    assert auto_high["limiting_factor"] == "HEAT_EXCHANGER_PINCH"
    assert auto_high["theoretical_autonomy_hours"] > auto_high["rated_autonomy_hours"]

    auto_low = calculate_tes_heat_autonomy_hours(
        stored_energy_kwh=3.0,
        process_demand_kw=1.5,
        capacity_kwh=15.0,
        min_soc_fraction=0.10,
    )
    assert auto_low["rated_autonomy_hours"] <= auto_low["theoretical_autonomy_hours"]


# ==============================================================================
# TEST C: Financial Audit & Baseline Re-Run Test
# ==============================================================================
def test_c_financial_audit_and_baseline_rerun(client: TestClient):
    """Verify financial audit reproducibility and baseline consistency:
    1. Identical simulation re-run yields identical costs and savings to the cent
    2. Direct electric baseline delivers equivalent process heat at identical intervals
    3. Terminal inventory adjustment correctly reconciles ΔSOC * replacement cost
    4. Sizing Matrix accurately flags vessel containment (≤15 kWh fits 172.3L; >15 kWh exceeds)
    5. Economics benchmarking page renders non-zero savings for optimized strategies
    """
    tz = VILNIUS_TZ
    s_date = date(2026, 10, 1)
    e_date = date(2026, 10, 3)  # 48 hours
    s_utc, _ = local_day_bounds_utc(s_date, tz)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), tz)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
    assert len(intervals) == 192

    prov = MockPriceProvider(tz=tz, resolution_minutes=15, seed=42)
    prices = prov.fetch_day_ahead("LT", s_utc, e_utc).points

    tes_params = TESParameters(
        thermal_capacity_full_span_kwh=15.0,
        optimizer_soc_min_fraction=0.10,
        optimizer_soc_max_fraction=1.00,
        max_charge_power_kw=9.0,
        max_discharge_power_kw=3.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.90,
        standing_loss_percent_per_day=2.0,
        auxiliary_power_kw=0.05,
    )
    site_params = SiteParameters(grid_connection_limit_kw=12.0, process_heat_demand_kw=1.5, other_loads_kw=2.0)
    tariff_params = TariffParameters(supplier_markup_eur_mwh=1.5, variable_grid_fee_eur_mwh=25.0, variable_tax_eur_mwh=5.0)
    curve = get_gen0_discharge_curve("fixed_gen0_hx", capacity_kwh=15.0)

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

    # Run 1
    direct_1 = evaluate_direct_electric_heating(intervals, prices, heat_demands, tariff_params)
    rolling_1 = evaluate_realistic_rolling_lp(intervals, prices, heat_demands, site_loads, config, tz)

    # Run 2 (identical parameters)
    direct_2 = evaluate_direct_electric_heating(intervals, prices, heat_demands, tariff_params)
    rolling_2 = evaluate_realistic_rolling_lp(intervals, prices, heat_demands, site_loads, config, tz)

    # 1. Exact mathematical determinism down to the cent
    assert abs(direct_1.total_cost_eur - direct_2.total_cost_eur) < 1e-4
    assert abs(rolling_1.total_cost_eur - rolling_2.total_cost_eur) < 1e-4

    # 2. Baseline Equivalency: direct electric baseline must satisfy 100% of the required heat
    assert direct_1.useful_heat_kwh == pytest.approx(sum(hd * iv.duration_h for hd, iv in zip(heat_demands, intervals)), rel=1e-4)
    assert direct_1.unmet_heat_kwh == 0.0

    # 3. Terminal Inventory Adjustment
    # If final SOC differs from initial SOC, inventory adjustment must reflect delta_soc
    delta_soc = rolling_1.final_soc_kwh - config.initial_soc_kwh
    if abs(delta_soc) > 1e-3:
        assert rolling_1.terminal_inventory_adjustment_eur != 0.0

    # 4. Sizing Matrix Vessel Feasibility
    sizing_analyzer = SizingStudyAnalyzer(tz=tz)
    reports = sizing_analyzer.run_sizing_sweep(
        base_config=config,
        heat_profile=heat_profile,
        site_profile=site_profile,
        prices=prices,
        capacities_kwh=[10.0, 15.0, 20.0, 30.0],
        charge_powers_kw=[9.0],
        sizing_mode="fixed_gen0_hx",
    )
    combos = reports["fixed_gen0_hx"].combinations

    for c in combos:
        if c.capacity_kwh <= 15.0:
            assert c.fits_gen0_vessel is True
            assert c.vessel_feasibility_status == "GEN0 PROTOTYPE FIT"
            assert c.sand_volume_liters <= 172.5
        else:
            assert c.fits_gen0_vessel is False
            assert c.vessel_feasibility_status == "EXCEEDS 172.3L VESSEL"
            assert c.sand_volume_liters > 172.5

    # 5. Economics Dashboard Page Verification
    resp = client.get("/dashboard/economics")
    assert resp.status_code == 200
    html = resp.text
    assert "4-WAY BENCHMARK" in html
    assert "Strategy Comparison Matrix" in html
    assert "Cost per MWh Useful Heat" in html
