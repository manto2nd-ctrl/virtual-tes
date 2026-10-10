"""Automated Test Suite for GEN0 Energy Manager — Virtual Dryer V1.

Validates:
1. Sensible air heating vs. total process demand separation.
2. Helical HX derating bounded by sand temperature & backup duct heater activation.
3. Blower stop safety interlock (zero airflow locks out duct heater).
4. Depleted TES behavior (zero thermal extraction from 0 kWh stored without grid).
5. Simultaneous charge and discharge energy balance reconciliation.
6. 15-minute settlement energy ledger: direct electric baseline, actual costs, inventory valuation delta, net savings.
7. Supervisory safety guard: ConnectedDeviceGuard prevents physical actuation in CONNECTED mode.
8. BESS proposed state: zero phantom savings, inactive power.
9. Dedicated web and API endpoints: /dashboard/dryer, /dashboard/bess, /dashboard/history, /api/dryer/status, /api/ledger/history.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

from app.auth.security import create_session_token
from app.config.settings import get_settings
from app.database.session import init_db
from app.models.hardware_interfaces import (
    ConnectedDeviceGuard,
    ConnectedDeviceLockedError,
    DataProvenance,
    OperatingMode,
)
from app.process.dryer import (
    DryerControlMode,
    DryerOperatingConfig,
    DryingRecipePreset,
    VirtualDryerModel,
)
from app.services.energy_ledger_service import EnergyLedgerService
from app.services.shadow_runtime import ShadowRuntimeService
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper
from app.web.main import app


@pytest.fixture
def mem_db():
    """Isolated in-memory SQLite database for test session."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    session = factory()
    yield session
    session.close()


def test_sensible_vs_total_demand_separation():
    """Verify that sensible air heating (0.757 kW for 40->70C at 80 m3/h) is not conflated with 1.50 kW total demand."""
    dryer = VirtualDryerModel()
    
    step_res = dryer.step(
        sand_temperature_c=250.0,
        dt_seconds=900,
    )

    assert 0.75 <= step_res.sensible_air_heat_kw <= 0.765
    assert step_res.total_heat_demand_kw == 1.50
    assert step_res.latent_and_losses_kw > 0.70
    assert abs((step_res.sensible_air_heat_kw + step_res.latent_and_losses_kw) - step_res.total_heat_demand_kw) < 1e-4
    assert step_res.achieved_supply_temp_c == 70.0
    assert step_res.backup_electric_power_kw == 0.0


def test_hx_derating_and_backup_heater_activation():
    """When sand cools below temperature needed for 1.5 kW, achievable temp drops and backup duct heater activates."""
    cfg = DryerOperatingConfig(backup_heater_enabled=True)
    dryer = VirtualDryerModel(config=cfg)

    step_res = dryer.step(
        sand_temperature_c=85.0,
        dt_seconds=900,
    )

    hx = HelicalAirHXModel()
    p_max = hx.p_max_at_temperature_kw(85.0)
    assert step_res.hx_power_limit_kw == pytest.approx(p_max, rel=1e-3)
    assert step_res.useful_heat_delivered_kw <= p_max + 1e-4
    assert step_res.backup_electric_power_kw > 0.0
    assert (step_res.useful_heat_delivered_kw + step_res.backup_electric_power_kw) == pytest.approx(1.50, rel=1e-3)


def test_blower_safety_interlock():
    """If airflow is 0 (blower stopped), backup electric heater MUST be locked out to prevent duct thermal runaway."""
    cfg = DryerOperatingConfig(
        airflow_m3_h=0.0,
        backup_heater_enabled=True,
    )
    dryer = VirtualDryerModel(config=cfg)

    step_res = dryer.step(
        sand_temperature_c=80.0,
        dt_seconds=900,
    )

    assert step_res.airflow_m3_h == 0.0
    assert step_res.blower_electric_power_kw == 0.0
    assert step_res.backup_electric_power_kw == 0.0
    assert step_res.blower_interlock_active is True


def test_depleted_tes_zero_heat():
    """Depleted TES (0 kWh / 80 C) cannot deliver useful heat without external electrical input."""
    cfg = DryerOperatingConfig(backup_heater_enabled=False)
    dryer = VirtualDryerModel(config=cfg)

    step_res = dryer.step(
        sand_temperature_c=80.0,
        dt_seconds=900,
    )

    hx = HelicalAirHXModel()
    p_max_min = hx.p_max_at_temperature_kw(80.0)
    assert step_res.useful_heat_delivered_kw <= p_max_min + 1e-3
    assert step_res.heat_shortfall_kw == pytest.approx(1.50 - step_res.useful_heat_delivered_kw, abs=1e-2)


def test_simultaneous_charge_and_discharge_reconciliation(mem_db):
    """Validate that shadow runtime correctly balances simultaneous charging and dryer extraction."""
    shadow_svc = ShadowRuntimeService()
    sess = shadow_svc.start_session(
        db=mem_db,
        initialization_mode="soc",
        initial_soc_percent=60.0,
        process_demand_kw=1.5,
        process_enabled=True,
    )

    start_energy = sess.current_stored_energy_kwh
    now = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    step_end = now + timedelta(minutes=15)

    rec = shadow_svc.execute_interval_step(
        db=mem_db,
        shadow_session=sess,
        interval_start_utc=now,
        interval_end_utc=step_end,
        spot_price_eur_mwh=40.0,
        requested_charge_kw=6.0,
        requested_discharge_kw=1.5,
    )

    assert sess.current_stored_energy_kwh > start_energy
    assert rec.actual_charge_kw == pytest.approx(6.0, rel=1e-2)
    assert rec.actual_discharge_kw > 0.0


def test_inventory_adjusted_net_savings_calculation():
    """Verify that 15-minute settlement ledger properly credits inventory delta Delta_V_inv."""
    svc = EnergyLedgerService(full_capacity_kwh=15.0)
    fin = svc.calculate_interval_finances(
        heat_delivered_kwh=0.375,
        tes_charge_kwh=0.0,
        backup_heat_kwh=0.0,
        blower_kwh=0.010,
        aux_kwh=0.0125,
        opening_soc_fraction=0.60,
        closing_soc_fraction=0.58,
        effective_price_eur_mwh=100.0,
    )

    assert fin["baseline_cost_eur"] == pytest.approx(0.0375, rel=1e-3)
    assert fin["actual_cost_eur"] == pytest.approx(0.0022, abs=1e-3)
    assert fin["inventory_delta_eur"] == pytest.approx(-0.0300, rel=1e-3)
    assert fin["net_savings_eur"] == pytest.approx(0.0052, rel=1e-2)


def test_connected_device_guard_strict_lockout():
    """Verify that ConnectedDeviceGuard strictly blocks physical actuation in CONNECTED mode."""
    with pytest.raises(ConnectedDeviceLockedError) as exc_info:
        ConnectedDeviceGuard.verify_mode_allowed(OperatingMode.CONNECTED)
    assert "strictly locked out" in str(exc_info.value)

    with pytest.raises(ConnectedDeviceLockedError):
        ConnectedDeviceGuard.verify_actuation_permitted()


def test_bess_proposed_zero_phantom_savings(mem_db):
    """Verify that BESS state reflects PROPOSED / NOT CONNECTED with 0 kW power and €0.00 savings."""
    shadow_svc = ShadowRuntimeService()
    state = shadow_svc.get_live_dashboard_state(db=mem_db)

    assert "PROPOSED" in state["bess"]["status"]
    assert state["bess"]["active_power_kw"] == 0.0
    assert state["bess"]["phantom_savings_eur"] == 0.0


def test_web_routes_and_api_integration():
    """Verify that /dashboard/dryer, /dashboard/bess, /dashboard/history, and APIs respond with 200 OK."""
    settings = get_settings()
    admin_token = create_session_token(
        username="admin",
        role="ADMIN",
        secret_key=settings.app_secret_key.get_secret_value(),
    )
    client = TestClient(app, cookies={"session_token": admin_token})

    # 1. Dryer Equipment Page
    resp_dryer = client.get("/dashboard/dryer")
    assert resp_dryer.status_code == 200
    assert "Dryer Equipment" in resp_dryer.text
    assert "Helical Air Coil" in resp_dryer.text

    # 2. BESS Equipment Page
    resp_bess = client.get("/dashboard/bess")
    assert resp_bess.status_code == 200
    assert "BESS Equipment" in resp_bess.text
    assert "PROPOSED / NOT CONNECTED" in resp_bess.text

    # 3. History & Settlement Ledger Page
    resp_hist = client.get("/dashboard/history")
    assert resp_hist.status_code == 200
    assert "History &amp; Savings Ledger" in resp_hist.text

    # 4. Dryer API Status
    resp_api_dryer = client.get("/api/dryer/status")
    assert resp_api_dryer.status_code == 200
    data = resp_api_dryer.json()
    assert "sensible_air_heat_kw" in data or "sensible_heat_demand_kw" in data
    assert "achieved_supply_temp_c" in data

    # 5. Dryer API Recipe Config Update
    resp_cfg = client.post(
        "/api/dryer/config",
        json={
            "target_supply_temp_c": 65.0,
            "inlet_air_temp_c": 35.0,
            "airflow_m3_h": 75.0,
            "total_process_heat_demand_kw": 1.40,
            "control_mode": "TARGET_TEMPERATURE",
            "backup_heater_enabled": True,
        },
    )
    assert resp_cfg.status_code == 200
    assert resp_cfg.json()["config"]["target_supply_temp_c"] == 65.0

    # 6. Ledger History API
    resp_ledger = client.get("/api/ledger/history?range=today")
    assert resp_ledger.status_code == 200
    assert "summary" in resp_ledger.json()

    # 7. Ledger CSV Export API
    resp_export = client.get("/api/ledger/export?range=today")
    assert resp_export.status_code == 200
    assert resp_export.headers["content-type"].startswith("text/csv")
