"""Stage 6: Economic Accounting Test Suite (Phase 5.11).

Validates:
1. Answers all 8 core economic questions directly:
   - How much electricity did TES use? (tes_electricity_used_kwh)
   - What did charging cost? (tes_charging_cost_eur)
   - How much useful heat was delivered? (total_heat_delivered_kwh)
   - How much water was removed? (water_removed_kg)
   - How much did the dryer cost to operate? (dryer_operating_cost_eur & total_actual_cost_eur)
   - What would equivalent direct-electric heating have cost? (total_baseline_cost_eur)
   - How much was saved? (total_net_savings_eur)
   - How much energy and cost remain in TES storage? (remaining_stored_energy_kwh & remaining_stored_energy_value_eur)
2. Consistent Baseline Integrity:
   - Baseline calculates equivalent useful process heat delivered at the time of demand.
   - Unmet heat is strictly excluded from baseline, preventing false savings from under-heating.
3. Multi-Period Aggregation:
   - Correctly aggregates Today, 7 Days, 30 Days, and Lifetime (all records).
4. Dryer Specific Unit Economics:
   - Specific thermal energy per kg water (kWh_th / kg).
   - Specific cost per kg water (EUR / kg).
   - Cost per completed batch (50 kg pine = 17.5 kg water; 500 kg kiln batch = 175 kg water).
   - Cost per m³ dried timber (Scots pine 450 kg/m³, 50% -> 15% MC = 157.5 kg water).
5. Labeling and Web Dashboard:
   - VIRTUAL ESTIMATED SAVINGS badge and audit notice prominently displayed.
   - All periods render with HTTP 200 and audit CSV export contains expanded financial columns.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from app.database.models import EnergyLedgerInterval
from app.database.session import init_db
from app.web.main import app
from app.models.hardware_interfaces import DataProvenance, OperatingMode
from app.services.energy_ledger_service import EnergyLedgerService

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


@pytest.fixture
def test_db_session():
    """Isolated in-memory SQLite database session for Stage 6 testing."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with session_factory() as session:
        yield session


def test_calculate_interval_finances_expanded():
    """Verify calculate_interval_finances decomposes TES charging cost vs dryer operating cost."""
    svc = EnergyLedgerService(tz=VILNIUS_TZ, full_capacity_kwh=15.0)

    # Effective price: €120/MWh = €0.12/kWh
    fin = svc.calculate_interval_finances(
        heat_delivered_kwh=1.0,      # 4 kW * 0.25h
        tes_charge_kwh=2.25,         # 9.0 kW * 0.25h
        backup_heat_kwh=0.25,        # 1.0 kW * 0.25h
        blower_kwh=0.02,
        aux_kwh=0.01,
        opening_soc_fraction=0.50,
        closing_soc_fraction=0.60,   # Delta = +0.10 * 15 kWh = +1.5 kWh
        effective_price_eur_mwh=120.0,
    )

    tariff = 0.12  # EUR/kWh
    # TES Charging Cost: 2.25 kWh * 0.12 = 0.2700 EUR
    assert fin["tes_charging_cost_eur"] == pytest.approx(2.25 * tariff, rel=1e-4)
    # Dryer Operating Cost: (0.25 + 0.02 + 0.01) * 0.12 = 0.28 * 0.12 = 0.0336 EUR
    assert fin["dryer_operating_cost_eur"] == pytest.approx(0.28 * tariff, rel=1e-4)
    # Total Actual Cost: (2.25 + 0.28) * 0.12 = 2.53 * 0.12 = 0.3036 EUR
    assert fin["actual_cost_eur"] == pytest.approx(2.53 * tariff, rel=1e-4)
    # Direct Baseline Cost: 1.0 kWh * 0.12 = 0.1200 EUR
    assert fin["baseline_cost_eur"] == pytest.approx(1.0 * tariff, rel=1e-4)
    # Inventory Delta: +1.5 kWh * 0.12 = +0.1800 EUR
    assert fin["inventory_delta_eur"] == pytest.approx(1.5 * tariff, rel=1e-4)
    # Net Savings: Baseline - Actual + InvDelta = 0.12 - 0.3036 + 0.18 = -0.0036 EUR
    assert fin["net_savings_eur"] == pytest.approx(0.12 - 0.3036 + 0.18, rel=1e-4)


def test_baseline_excludes_unmet_heat():
    """Verify direct electric baseline does NOT reward unmet heat / insufficient drying."""
    svc = EnergyLedgerService(tz=VILNIUS_TZ, full_capacity_kwh=15.0)

    # If heat demand is 2.0 kWh, but TES only delivered 0.5 kWh and backup is 0.0 kWh (unmet = 1.5 kWh)
    # The direct electric baseline must strictly equal 0.5 kWh * tariff, NOT 2.0 kWh * tariff!
    fin = svc.calculate_interval_finances(
        heat_delivered_kwh=0.5,      # Useful delivered heat only
        tes_charge_kwh=0.0,
        backup_heat_kwh=0.0,
        blower_kwh=0.01,
        aux_kwh=0.01,
        opening_soc_fraction=0.50,
        closing_soc_fraction=0.45,
        effective_price_eur_mwh=100.0, # 0.10 EUR/kWh
    )

    assert fin["baseline_cost_eur"] == pytest.approx(0.5 * 0.10, rel=1e-4)
    # Actual cost: 0.02 * 0.10 = 0.002 EUR
    assert fin["actual_cost_eur"] == pytest.approx(0.002, rel=1e-4)
    # Inventory delta: -0.05 * 15 * 0.10 = -0.75 * 0.10 = -0.075 EUR
    assert fin["inventory_delta_eur"] == pytest.approx(-0.075, rel=1e-4)
    # Net Savings = 0.05 - 0.002 - 0.075 = -0.027 EUR
    assert fin["net_savings_eur"] == pytest.approx(-0.027, rel=1e-4)


def test_answers_all_8_economic_questions(test_db_session: Session):
    """Verify that get_ledger_history computes and answers all 8 core questions explicitly."""
    db = test_db_session
    svc = EnergyLedgerService(tz=VILNIUS_TZ, full_capacity_kwh=15.0)

    t0 = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    # Add an interval
    svc.record_interval(
        db=db,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=80.0,
        effective_price_eur_mwh=100.0,  # 0.10 EUR/kWh
        tes_charge_energy_kwh=2.0,       # Question 1: TES used 2.0 kWh
        dryer_thermal_delivered_kwh=1.2, # Question 3: Delivered heat 1.2 kWh
        backup_heater_energy_kwh=0.2,
        blower_electric_kwh=0.05,
        auxiliary_electric_kwh=0.05,
        peak_grid_power_kw=9.0,
        avg_tes_heat_kw=4.0,
        avg_dryer_demand_kw=4.8,
        avg_sand_temp_c=220.0,
        avg_dryer_supply_temp_c=70.0,
        avg_dryer_exhaust_temp_c=45.0,
        opening_soc_fraction=0.50,
        closing_soc_fraction=0.60,      # Delta = +1.5 kWh
        moisture_content_percent=25.0,
        water_removed_kg=1.8,           # Question 4: Water removed 1.8 kg
        stored_energy_start_kwh=7.5,
        stored_energy_end_kwh=9.0,
    )

    res = svc.get_ledger_history(db=db, period="all")
    summary = res["summary"]

    # 1. How much electricity did TES use?
    assert summary["tes_electricity_used_kwh"] == 2.0

    # 2. What did charging cost?
    # 2.0 kWh * 0.10 EUR/kWh = 0.20 EUR
    assert summary["tes_charging_cost_eur"] == 0.20

    # 3. How much useful heat was delivered?
    assert summary["total_heat_delivered_kwh"] == 1.2

    # 4. How much water was removed?
    assert summary["water_removed_kg"] == 1.8

    # 5. How much did the dryer cost to operate?
    # Direct dryer parasitics: (0.2 backup + 0.05 blower + 0.05 aux) * 0.10 = 0.3 * 0.10 = 0.03 EUR
    assert summary["dryer_operating_cost_eur"] == 0.03
    # Total system operating cost: 0.20 + 0.03 = 0.23 EUR
    assert summary["total_actual_cost_eur"] == 0.23

    # 6. What would equivalent direct-electric heating have cost?
    # 1.2 kWh * 0.10 EUR/kWh = 0.12 EUR
    assert summary["total_baseline_cost_eur"] == 0.12

    # 7. How much was saved?
    # Net savings = 0.12 baseline - 0.23 actual + 0.15 inv delta = 0.04 EUR
    assert summary["total_net_savings_eur"] == pytest.approx(0.04, abs=0.01)

    # 8. How much energy and cost remain in TES storage?
    assert summary["remaining_stored_energy_kwh"] == 9.0
    # Valuation: 9.0 kWh * 0.10 EUR/kWh = 0.90 EUR
    assert summary["remaining_stored_energy_value_eur"] == 0.90


def test_dryer_specific_unit_economics(test_db_session: Session):
    """Verify dryer unit economics: kWh_th/kg water, EUR/kg water, EUR/batch, and EUR/m³."""
    db = test_db_session
    svc = EnergyLedgerService(tz=VILNIUS_TZ, full_capacity_kwh=15.0)

    t0 = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)
    # Delivered 2.0 kWh useful heat, removed 2.5 kg water, total actual cost 0.20 EUR
    svc.record_interval(
        db=db,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=68.0,
        effective_price_eur_mwh=100.0,
        tes_charge_energy_kwh=1.8,
        dryer_thermal_delivered_kwh=2.0,
        backup_heater_energy_kwh=0.0,
        blower_electric_kwh=0.1,
        auxiliary_electric_kwh=0.1,
        peak_grid_power_kw=8.0,
        avg_tes_heat_kw=8.0,
        avg_dryer_demand_kw=8.0,
        avg_sand_temp_c=200.0,
        avg_dryer_supply_temp_c=70.0,
        avg_dryer_exhaust_temp_c=40.0,
        opening_soc_fraction=0.50,
        closing_soc_fraction=0.50,
        moisture_content_percent=20.0,
        water_removed_kg=2.5,
    )

    res = svc.get_ledger_history(db=db, period="all")
    summary = res["summary"]

    # Specific thermal energy: 2.0 kWh / 2.5 kg = 0.800 kWh_th/kg
    assert summary["thermal_energy_per_kg_water_kwh"] == pytest.approx(0.800, rel=1e-3)

    # Total grid = 1.8 + 0.1 + 0.1 = 2.0 kWh; Actual cost = 2.0 * 0.10 = 0.20 EUR
    # Cost per kg water = 0.20 EUR / 2.5 kg = 0.0800 EUR/kg
    assert summary["cost_per_kg_water_eur"] == pytest.approx(0.0800, rel=1e-3)

    # Cost per 50 kg batch (17.5 kg water removed) = 17.5 * 0.08 = 1.40 EUR
    assert summary["cost_per_completed_batch_eur"] == pytest.approx(1.40, abs=0.01)

    # Cost per 500 kg batch (175.0 kg water removed) = 175 * 0.08 = 14.00 EUR
    assert summary["cost_per_completed_batch_500kg_eur"] == pytest.approx(14.00, abs=0.02)

    # Cost per m³ dried timber (Scots pine 450 kg/m³, 157.5 kg water removed) = 157.5 * 0.08 = 12.60 EUR
    assert summary["cost_per_m3_timber_eur"] == pytest.approx(12.60, abs=0.02)


def test_multi_period_comparison(test_db_session: Session):
    """Verify period_comparison contains today, 7d, 30d, and all with consistent accounting."""
    db = test_db_session
    svc = EnergyLedgerService(tz=VILNIUS_TZ, full_capacity_kwh=15.0)

    now_local = datetime.now(VILNIUS_TZ)
    # Interval 1: Local Today
    t_today = now_local.replace(minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    # Interval 2: 3 days ago (in 7d, 30d, all)
    t_3d = t_today - timedelta(days=3)
    # Interval 3: 15 days ago (in 30d, all)
    t_15d = t_today - timedelta(days=15)
    # Interval 4: 45 days ago (in all only)
    t_45d = t_today - timedelta(days=45)

    for i, t in enumerate([t_today, t_3d, t_15d, t_45d]):
        svc.record_interval(
            db=db,
            interval_start_utc=t,
            interval_end_utc=t + timedelta(minutes=15),
            spot_price_eur_mwh=68.0,
            effective_price_eur_mwh=100.0,
            tes_charge_energy_kwh=1.0,
            dryer_thermal_delivered_kwh=0.5,
            backup_heater_energy_kwh=0.0,
            blower_electric_kwh=0.01,
            auxiliary_electric_kwh=0.01,
            peak_grid_power_kw=4.0,
            avg_tes_heat_kw=2.0,
            avg_dryer_demand_kw=2.0,
            avg_sand_temp_c=190.0,
            avg_dryer_supply_temp_c=70.0,
            avg_dryer_exhaust_temp_c=40.0,
            opening_soc_fraction=0.50,
            closing_soc_fraction=0.50,
            moisture_content_percent=20.0,
            water_removed_kg=1.0,
        )

    res = svc.get_ledger_history(db=db, period="today")
    period_comp = res["period_comparison"]

    assert "today" in period_comp
    assert "7d" in period_comp
    assert "30d" in period_comp
    assert "all" in period_comp

    assert period_comp["today"]["interval_count"] >= 1
    assert period_comp["7d"]["interval_count"] >= 2
    assert period_comp["30d"]["interval_count"] >= 3
    assert period_comp["all"]["interval_count"] >= 4

    # TES electricity used should monotonically grow across broader horizons
    assert period_comp["today"]["tes_electricity_used_kwh"] <= period_comp["7d"]["tes_electricity_used_kwh"]
    assert period_comp["7d"]["tes_electricity_used_kwh"] <= period_comp["30d"]["tes_electricity_used_kwh"]
    assert period_comp["30d"]["tes_electricity_used_kwh"] <= period_comp["all"]["tes_electricity_used_kwh"]


def test_dashboard_history_route_renders_stage6():
    """Verify /dashboard/history route renders all 8 questions, VIRTUAL ESTIMATED SAVINGS badge, and periods."""
    client = TestClient(app)

    # Test all period filters
    for period in ["today", "7d", "30d", "all"]:
        resp = client.get(f"/dashboard/history?range={period}")
        assert resp.status_code == 200
        html = resp.text

        # Label check
        assert "VIRTUAL ESTIMATED SAVINGS" in html

        # 8 Questions checks
        assert "Electricity Used by TES" in html
        assert "Charging Cost" in html
        assert "Useful Heat Delivered" in html
        assert "Water Removed" in html
        assert "Total Operating Cost" in html
        assert "Direct-Electric Baseline" in html
        assert "Net Virtual Savings" in html
        assert "Remaining TES Storage" in html

        # Multi-Period breakdown check
        assert "Multi-Period Economic Accounting" in html
        assert "Lifetime (All Records)" in html

        # Dryer Unit Economics check
        assert "Virtual Dryer V1" in html
        assert "Specific Thermal Energy" in html
        assert "Specific Drying Cost" in html
        assert "Cost Per Completed Batch" in html
        assert "Cost Per m&sup3; Dried Timber" in html


def test_csv_export_expanded_columns():
    """Verify /api/ledger/export CSV contains Stage 6 audit columns."""
    client = TestClient(app)
    resp = client.get("/api/ledger/export?range=all")
    assert resp.status_code == 200

    csv_reader = csv.reader(io.StringIO(resp.text))
    header = next(csv_reader)

    assert "TES Charge Electricity (kWh)" in header
    assert "TES Charging Cost (EUR)" in header
    assert "Dryer Operating Cost (EUR)" in header
    assert "Specific Thermal (kWh/kg)" in header
    assert "Baseline Cost (EUR)" in header
    assert "Actual Cost (EUR)" in header
    assert "Inventory Valuation Delta (EUR)" in header
    assert "Net Savings (EUR)" in header
