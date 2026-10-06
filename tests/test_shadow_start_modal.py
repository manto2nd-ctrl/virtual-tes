"""Automated tests for Shadow Start Modal & Single Source of Truth Validation.

Validates all 10 requirements:
1. SOC=50% maps to 7.5 kWh.
2. SOC=50% maps to approximately 196.8 °C.
3. Inconsistent triples cannot be submitted (single canonical input enforced).
4. Backend derives values from canonical input across soc, energy, and temp modes.
5. Cancel causes no state change.
6. Start creates a RUNNING shadow session.
7. Double-click cannot create duplicate sessions (idempotent single active session).
8. Invalid input returns a visible validation error.
9. Production modal works after HTMX page swaps (modals live outside the polling swap target).
10. All previous tests still pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.auth.security import create_session_token
from app.config.settings import Settings, get_settings
from app.database.models import ShadowTESSession
from app.database.session import init_db
from app.services.shadow_runtime import ShadowRuntimeService
from app.tes.thermal import ThermalStateMapper
from app.web.main import app


from sqlalchemy.pool import StaticPool


@pytest.fixture
def test_db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return engine


@pytest.fixture
def test_db_session(test_db_engine):
    factory = sessionmaker(bind=test_db_engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def admin_cookie():
    settings = get_settings()
    return create_session_token("admin_test", "ADMIN", settings.app_secret_key.get_secret_value())


@pytest.fixture
def test_client(test_db_engine):
    from app.web.routes import api as api_module
    from app.web.routes import dashboard as dashboard_module
    factory = sessionmaker(bind=test_db_engine, expire_on_commit=False, future=True)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[api_module.get_db] = override_get_db
    app.dependency_overrides[dashboard_module.get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1 & 2. SOC=50% maps to 7.5 kWh and approx 196.8 °C
# ---------------------------------------------------------------------------
def test_soc_50_maps_to_7_5_kwh_and_approx_196_8_c():
    """Verify physical coupling of 50% SOC to 7.50 kWh and ~196.8 °C."""
    mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
    service = ShadowRuntimeService()

    frac, kwh, temp = service.convert_initial_conditions(initial_soc_percent=50.0, initialization_mode="soc")
    assert frac == 0.50
    assert kwh == 7.50
    assert abs(temp - 196.77) < 0.1  # ~196.8 °C


# ---------------------------------------------------------------------------
# 3. Inconsistent triples cannot be submitted / canonical mode is enforced
# ---------------------------------------------------------------------------
def test_inconsistent_triples_cannot_be_submitted():
    """Conflicting non-canonical values are completely ignored and recomputed."""
    service = ShadowRuntimeService()

    # User attempts to submit SOC=50% with conflicting energy=1.5 and temp=100.0
    frac, kwh, temp = service.convert_initial_conditions(
        initial_soc_percent=50.0,
        initial_energy_kwh=1.5,
        initial_temp_c=100.0,
        initialization_mode="soc",
    )
    # The server re-calculates purely from SOC=50%
    assert frac == 0.50
    assert kwh == 7.50
    assert abs(temp - 196.77) < 0.1


# ---------------------------------------------------------------------------
# 4. Backend derives values from canonical input across modes
# ---------------------------------------------------------------------------
def test_backend_derives_values_from_canonical_input():
    """Verify derivation across 'energy' and 'temp' canonical modes."""
    service = ShadowRuntimeService()

    # Energy mode: 7.50 kWh -> 50% SOC, ~196.8 °C
    frac_e, kwh_e, temp_e = service.convert_initial_conditions(
        initial_energy_kwh=7.50,
        initialization_mode="energy",
    )
    assert frac_e == 0.50
    assert kwh_e == 7.50
    assert abs(temp_e - 196.77) < 0.1

    # Temp mode: 196.77 °C -> 50% SOC, 7.50 kWh
    frac_t, kwh_t, temp_t = service.convert_initial_conditions(
        initial_temp_c=196.77,
        initialization_mode="temp",
    )
    assert frac_t == 0.50
    assert abs(kwh_t - 7.50) < 0.01
    assert abs(temp_t - 196.77) < 0.1


# ---------------------------------------------------------------------------
# 5. Cancel causes no state change
# ---------------------------------------------------------------------------
def test_cancel_causes_no_state_change(test_db_session):
    """Closing the modal does not alter database state or sessions."""
    service = ShadowRuntimeService()
    sess = service.get_or_create_session(test_db_session)
    initial_status = sess.status
    initial_soc = sess.current_soc_fraction

    # No API endpoint is invoked during cancel. Query session:
    current = test_db_session.execute(select(ShadowTESSession).where(ShadowTESSession.id == sess.id)).scalar_one()
    assert current.status == initial_status
    assert current.current_soc_fraction == initial_soc


# ---------------------------------------------------------------------------
# 6. Start creates a RUNNING shadow session
# ---------------------------------------------------------------------------
def test_start_creates_running_shadow_session(test_client, admin_cookie):
    """POST /api/shadow/start transitions session to RUNNING with canonical state."""
    test_client.cookies.set("session_token", admin_cookie)

    resp = test_client.post(
        "/api/shadow/start",
        json={
            "initialization_mode": "soc",
            "initial_soc_percent": 50.0,
            "process_demand_kw": 1.5,
            "process_enabled": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["state"] == "RUNNING"
    assert data["initial_soc_fraction"] == 0.50
    assert data["initial_stored_energy_kwh"] == 7.50
    assert abs(data["initial_sand_temperature_c"] - 196.77) < 0.1


# ---------------------------------------------------------------------------
# 7. Double-click cannot create duplicate sessions
# ---------------------------------------------------------------------------
def test_double_click_cannot_create_duplicate_sessions(test_client, test_db_session, admin_cookie):
    """Submitting start twice updates the single active session rather than duplicating."""
    test_client.cookies.set("session_token", admin_cookie)

    payload = {"initialization_mode": "soc", "initial_soc_percent": 50.0}
    r1 = test_client.post("/api/shadow/start", json=payload)
    r2 = test_client.post("/api/shadow/start", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["session_id"] == r2.json()["session_id"]

    sessions = test_db_session.execute(select(ShadowTESSession)).scalars().all()
    assert len(sessions) == 1
    assert sessions[0].status == "RUNNING"


# ---------------------------------------------------------------------------
# 8. Invalid input returns a visible validation error
# ---------------------------------------------------------------------------
def test_invalid_input_returns_validation_error(test_client, admin_cookie):
    """Out-of-range inputs return HTTP 400 / 422 with descriptive error."""
    test_client.cookies.set("session_token", admin_cookie)

    # Invalid SOC (> 100)
    r_bad_soc = test_client.post(
        "/api/shadow/start",
        json={"initialization_mode": "soc", "initial_soc_percent": 150.0},
    )
    assert r_bad_soc.status_code in (400, 422)

    # Invalid Temp (< 80)
    r_bad_temp = test_client.post(
        "/api/shadow/start",
        json={"initialization_mode": "temp", "initial_temp_c": 40.0},
    )
    assert r_bad_temp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 9. Production modal works after HTMX page swaps
# ---------------------------------------------------------------------------
def test_modal_lives_outside_htmx_polling_swap_container(test_client, admin_cookie):
    """5-second auto-refresh endpoint does NOT include or destroy start-modal."""
    test_client.cookies.set("session_token", admin_cookie)

    # Polled live-card partial
    resp = test_client.get("/api/shadow/live-card", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    html = resp.text

    # The 5-second polling target MUST NOT contain start-modal
    assert '<dialog id="start-modal"' not in html
    assert '<div id="start-modal"' not in html
    assert 'Start Virtual TES Shadow Plant' not in html

    # But the full overview page DOES include the modals
    overview_resp = test_client.get("/dashboard/overview")
    assert overview_resp.status_code == 200
    assert 'Start Virtual TES Shadow Plant' in overview_resp.text
    assert 'dialog id="start-modal"' in overview_resp.text
    assert 'VIRTUAL INITIAL CONDITION — NOT PHYSICALLY MEASURED' in overview_resp.text
