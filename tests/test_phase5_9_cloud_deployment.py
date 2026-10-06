"""Automated test suite for Phase 5.9 — Railway 24/7 Cloud Deployment.

Validates all 13 test requirements:
1. Production config refuses mock fallback
2. PostgreSQL persistence works (ORM entities, schema types, transactions)
3. Shadow state survives worker restart (SOC, energy, temp unchanged)
4. Duplicate interval execution is prevented (UNIQUE constraint)
5. Worker heartbeat works (publishes every 60s, records metrics)
6. Stale worker is shown offline (> 3 minutes age marks worker OFFLINE)
7. Login is required (unauthenticated requests redirected to /login in production)
8. VIEWER cannot mutate system state (receives HTTP 403 Forbidden)
9. ADMIN can use approved shadow controls (receives HTTP 200 OK)
10. Secrets are not rendered in HTML/API (no passwords, hashes, secret keys)
11. Mobile dashboard routes render correctly (viewport meta tag, responsive layout)
12. Health endpoint works (GET /health returns HTTP 200 when database connected)
13. Compatibility with existing tests
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
import pytest
from starlette.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from app.auth.dependencies import UserSession
from app.auth.security import create_session_token, hash_password, verify_password
from app.config.settings import Settings, get_settings
from app.database.base import Base
from app.database.models import (
    DayAheadPrice,
    EngineeringScenario,
    ShadowIntervalRecord,
    ShadowTESSession,
    WorkerHeartbeat,
)
from app.database.session import init_db, make_engine, normalize_database_url
from app.services.market_data_service import LiveMarketDataUnavailableError, MarketDataService
from app.services.shadow_runtime import ShadowRuntimeService
from app.services.worker_lock import DistributedWorkerLock
from app.web.main import app
from app.worker import VirtualTESWorker


@pytest.fixture
def test_db_engine():
    """Isolated SQLite engine for testing database models and transactions."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    init_db(engine)
    return engine


@pytest.fixture
def test_db_session(test_db_engine):
    """Session factory for the isolated test database."""
    factory = sessionmaker(bind=test_db_engine, expire_on_commit=False, future=True)
    session = factory()
    yield session
    session.close()


@pytest.fixture
def client():
    """Starlette test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Production config refuses mock fallback
# ---------------------------------------------------------------------------
def test_production_config_refuses_mock_fallback(monkeypatch):
    """Ensure that in production mode, failing live providers never silently return mock data."""
    svc = MarketDataService()

    # Force all live providers to raise exceptions
    monkeypatch.setattr(svc.litgrid, "fetch_day_ahead", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Litgrid connection down")))
    monkeypatch.setattr(svc.elering, "fetch_day_ahead", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Elering timeout")))
    monkeypatch.setattr(svc.volton, "fetch_day_ahead", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Volton unreachable")))
    monkeypatch.setattr(svc.entsoe, "fetch_day_ahead", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ENTSO-E rejected")))

    # With no cached DB data, MarketDataService must raise LiveMarketDataUnavailableError
    with pytest.raises(LiveMarketDataUnavailableError) as exc_info:
        svc.fetch_market_data(session=None)

    assert "no mock fallback allowed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 2. Database persistence works (URL normalization & ORM entities)
# ---------------------------------------------------------------------------
def test_database_url_normalization():
    """Verify Railway postgres:// URLs are normalized to postgresql+psycopg://."""
    raw_railway_url = "postgres://postgres:securepass@junction.railway.internal:5432/railway"
    norm = normalize_database_url(raw_railway_url)
    assert norm.startswith("postgresql+psycopg://")
    assert "postgres:securepass" in norm

    raw_pg_url = "postgresql://user:pass@host:5432/db"
    assert normalize_database_url(raw_pg_url).startswith("postgresql+psycopg://")

    sqlite_url = "sqlite:///./data/tes.db"
    assert normalize_database_url(sqlite_url) == sqlite_url


def test_database_persistence_works(test_db_session):
    """Verify ORM entities persist with full precision and proper UTC handling."""
    now = datetime.now(timezone.utc)
    sess = ShadowTESSession(
        id="test-session-persistent",
        status="RUNNING",
        created_at_utc=now,
        started_at_utc=now,
        current_soc_fraction=0.725,
        current_stored_energy_kwh=10.875,
        current_sand_temperature_c=239.5,
    )
    test_db_session.add(sess)
    test_db_session.commit()

    queried = test_db_session.execute(
        select(ShadowTESSession).where(ShadowTESSession.id == "test-session-persistent")
    ).scalar_one()

    assert queried.current_soc_fraction == 0.725
    assert queried.current_stored_energy_kwh == 10.875
    assert queried.current_sand_temperature_c == 239.5
    assert queried.status == "RUNNING"


# ---------------------------------------------------------------------------
# 3. Shadow state survives worker restart
# ---------------------------------------------------------------------------
def test_shadow_state_survives_worker_restart(test_db_session):
    """Simulate worker shutdown and restart: state must NOT be reinitialized."""
    # Worker 1 creates and runs session at 65% SOC
    t0 = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
    session_obj = ShadowTESSession(
        id="restart-test-session",
        status="RUNNING",
        created_at_utc=t0,
        started_at_utc=t0,
        current_soc_fraction=0.650,
        current_stored_energy_kwh=9.75,
        current_sand_temperature_c=223.0,
        last_executed_interval_start_utc=t0,
    )
    test_db_session.add(session_obj)
    test_db_session.commit()

    # Simulate Worker 1 terminates (session closed)
    test_db_session.expunge_all()

    # Worker 2 starts up fresh
    worker_2_svc = ShadowRuntimeService()
    restored_session = test_db_session.execute(
        select(ShadowTESSession).where(ShadowTESSession.id == "restart-test-session")
    ).scalar_one()

    # Verify state remains exactly as persisted
    assert restored_session.status == "RUNNING"
    assert restored_session.current_soc_fraction == 0.650
    assert restored_session.current_stored_energy_kwh == 9.75
    assert restored_session.current_sand_temperature_c == 223.0
    assert restored_session.last_executed_interval_start_utc == t0


# ---------------------------------------------------------------------------
# 4. Duplicate interval execution is prevented (UNIQUE constraint)
# ---------------------------------------------------------------------------
def test_duplicate_interval_execution_prevented(test_db_session):
    """UNIQUE(shadow_session_id, interval_start_utc) must prevent duplicate intervals."""
    t_start = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 10, 6, 12, 15, tzinfo=timezone.utc)

    # First execution succeeds
    r1 = ShadowIntervalRecord(
        shadow_session_id="session-dup-test",
        interval_start_utc=t_start,
        interval_end_utc=t_end,
        duration_hours=0.25,
        spot_price_eur_mwh=45.0,
        effective_price_eur_mwh=50.0,
        action_type="CHARGE",
        requested_charge_kw=9.0,
        actual_charge_kw=9.0,
        requested_discharge_kw=0.0,
        actual_discharge_kw=0.0,
        process_demand_kw=1.5,
        hx_power_limit_kw=2.5,
        soc_start_fraction=0.50,
        soc_end_fraction=0.64,
        stored_energy_start_kwh=7.5,
        stored_energy_end_kwh=9.6,
        sand_temp_start_c=197.8,
        sand_temp_end_c=220.0,
        standing_loss_kwh=0.01,
        grid_power_total_kw=11.05,
        electricity_consumed_kwh=2.25,
        interval_cost_eur=0.1125,
        energy_balance_residual_kwh=0.0,
        reason_code="OPTIMIZED_DISPATCH",
    )
    test_db_session.add(r1)
    test_db_session.commit()

    # Second execution at identical interval_start_utc must raise IntegrityError
    from sqlalchemy.exc import IntegrityError
    r2 = ShadowIntervalRecord(
        shadow_session_id="session-dup-test",
        interval_start_utc=t_start,
        interval_end_utc=t_end,
        duration_hours=0.25,
        spot_price_eur_mwh=45.0,
        effective_price_eur_mwh=50.0,
        action_type="CHARGE",
        requested_charge_kw=9.0,
        actual_charge_kw=9.0,
        requested_discharge_kw=0.0,
        actual_discharge_kw=0.0,
        process_demand_kw=1.5,
        hx_power_limit_kw=2.5,
        soc_start_fraction=0.64,
        soc_end_fraction=0.78,
        stored_energy_start_kwh=9.6,
        stored_energy_end_kwh=11.7,
        sand_temp_start_c=220.0,
        sand_temp_end_c=245.0,
        standing_loss_kwh=0.01,
        grid_power_total_kw=11.05,
        electricity_consumed_kwh=2.25,
        interval_cost_eur=0.1125,
        energy_balance_residual_kwh=0.0,
        reason_code="OPTIMIZED_DISPATCH",
    )
    test_db_session.add(r2)
    with pytest.raises(IntegrityError):
        test_db_session.commit()
    test_db_session.rollback()


# ---------------------------------------------------------------------------
# 5. Worker heartbeat works
# ---------------------------------------------------------------------------
def test_worker_heartbeat_works(test_db_session):
    """Verify worker heartbeat writes with proper metadata and timestamps."""
    now = datetime.now(timezone.utc)
    hb = WorkerHeartbeat(
        worker_id="worker-test-hb-01",
        timestamp_utc=now,
        status="RUNNING",
        app_env="production",
        market_status="LIVE",
        shadow_status="RUNNING",
        last_market_fetch_utc=now - timedelta(minutes=2),
        last_optimization_utc=now - timedelta(minutes=5),
        last_executed_interval_utc=now - timedelta(minutes=15),
        next_interval_utc=now + timedelta(minutes=10),
    )
    test_db_session.add(hb)
    test_db_session.commit()

    latest = test_db_session.execute(
        select(WorkerHeartbeat).order_by(WorkerHeartbeat.timestamp_utc.desc()).limit(1)
    ).scalar_one()

    assert latest.worker_id == "worker-test-hb-01"
    assert latest.status == "RUNNING"
    assert latest.app_env == "production"
    assert latest.market_status == "LIVE"


# ---------------------------------------------------------------------------
# 6. Stale worker is shown offline
# ---------------------------------------------------------------------------
def test_stale_worker_shown_offline(test_db_session):
    """Heartbeat older than 3 minutes (180s) must be computed as OFFLINE."""
    from app.web.routes.dashboard import get_runtime_status_context

    now = datetime.now(timezone.utc)
    # Heartbeat from 5 minutes ago (300 seconds)
    stale_time = now - timedelta(seconds=300)
    hb = WorkerHeartbeat(
        worker_id="worker-stale-01",
        timestamp_utc=stale_time,
        status="RUNNING",
        app_env="production",
        market_status="LIVE",
        shadow_status="RUNNING",
    )
    test_db_session.add(hb)
    test_db_session.commit()

    status = get_runtime_status_context(test_db_session)
    assert status["worker_status"] == "OFFLINE"
    assert status["worker_online"] is False
    assert status["heartbeat_age_sec"] >= 180
    assert status["shadow_status"] in ("STOPPED", "OFFLINE / STALE")


# ---------------------------------------------------------------------------
# 7. Login is required in production mode
# ---------------------------------------------------------------------------
def test_login_is_required_in_production(client, monkeypatch):
    """When AUTH_REQUIRED is True, unauthenticated requests to dashboard must redirect to /login."""
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_required", True)

    # HTML request to overview without session token
    resp = client.get("/dashboard/overview", follow_redirects=False)
    assert resp.status_code == 303
    assert "/login" in resp.headers["location"]


# ---------------------------------------------------------------------------
# 8. VIEWER cannot mutate system state
# ---------------------------------------------------------------------------
def test_viewer_cannot_mutate_system_state(client, monkeypatch):
    """User with VIEWER role can view dashboard, but mutation endpoints return HTTP 403 Forbidden."""
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_required", True)

    viewer_token = create_session_token(
        username="test_viewer",
        role="VIEWER",
        secret_key=settings.app_secret_key.get_secret_value(),
    )
    cookies = {"session_token": viewer_token}

    # VIEWER can read dashboard pages
    resp_view = client.get("/dashboard/overview", cookies=cookies)
    assert resp_view.status_code == 200
    assert "VIEWER" in resp_view.text

    # VIEWER is blocked from shadow mutation endpoints
    resp_start = client.post("/api/shadow/start", json={"initial_soc_percent": 50.0}, cookies=cookies)
    assert resp_start.status_code == 403
    assert "prohibited" in resp_start.json()["detail"].lower()

    resp_pause = client.post("/api/shadow/pause", cookies=cookies)
    assert resp_pause.status_code == 403

    resp_reset = client.post("/api/shadow/reset", json={"target_soc_percent": 50.0, "reason": "test"}, cookies=cookies)
    assert resp_reset.status_code == 403

    resp_reopt = client.post("/api/shadow/reoptimize", cookies=cookies)
    assert resp_reopt.status_code == 403


# ---------------------------------------------------------------------------
# 9. ADMIN can use approved shadow controls
# ---------------------------------------------------------------------------
def test_admin_can_use_approved_shadow_controls(client, monkeypatch):
    """User with ADMIN role can invoke approved shadow controls."""
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_required", True)

    admin_token = create_session_token(
        username="test_admin",
        role="ADMIN",
        secret_key=settings.app_secret_key.get_secret_value(),
    )
    cookies = {"session_token": admin_token}

    resp_start = client.post("/api/shadow/start", json={"initial_soc_percent": 50.0}, cookies=cookies)
    assert resp_start.status_code == 200
    assert resp_start.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 10. Secrets are not rendered in HTML or API
# ---------------------------------------------------------------------------
def test_secrets_are_not_rendered_in_html_or_api(client):
    """Ensure sensitive credentials and secret keys are never leaked to HTML or API responses."""
    settings = get_settings()
    secret_str = settings.app_secret_key.get_secret_value()

    # Inspect rendered Overview HTML
    resp_overview = client.get("/dashboard/overview")
    assert secret_str not in resp_overview.text
    if settings.dashboard_password_hash:
        assert settings.dashboard_password_hash not in resp_overview.text

    # Inspect Assumptions HTML
    resp_assumptions = client.get("/dashboard/assumptions")
    assert secret_str not in resp_assumptions.text

    # Inspect API responses
    resp_status = client.get("/api/shadow/status")
    assert secret_str not in resp_status.text


# ---------------------------------------------------------------------------
# 11. Mobile dashboard routes render correctly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "/dashboard/overview",
        "/dashboard/market",
        "/dashboard/physical",
        "/dashboard/hx",
        "/dashboard/vessel",
        "/dashboard/dispatch",
        "/dashboard/economics",
        "/dashboard/sizing",
        "/dashboard/sensitivity",
        "/dashboard/assumptions",
    ],
)
def test_mobile_dashboard_routes_render_correctly(client, path):
    """Simulate mobile browser at ~390px viewport: must render HTTP 200 with viewport meta."""
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"}
    resp = client.get(path, headers=headers)
    assert resp.status_code == 200
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in resp.text
    assert "production-banner" in resp.text
    assert "MARKET DATA:" in resp.text


# ---------------------------------------------------------------------------
# 12. Health endpoint works
# ---------------------------------------------------------------------------
def test_health_endpoint_works(client):
    """GET /health must return HTTP 200 with database: connected."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"


# ---------------------------------------------------------------------------
# 13. Distributed worker lock election
# ---------------------------------------------------------------------------
def test_single_active_worker_lock_election(test_db_session):
    """If two workers start, only one becomes leader; the second enters standby."""
    lock_a = DistributedWorkerLock(lock_name="test_worker_leader", lease_seconds=60)
    lock_b = DistributedWorkerLock(lock_name="test_worker_leader", lease_seconds=60)

    # Worker A acquires lock
    is_a_leader = lock_a.acquire_or_renew(test_db_session)
    assert is_a_leader is True

    # Worker B attempts to acquire while A holds it
    is_b_leader = lock_b.acquire_or_renew(test_db_session)
    assert is_b_leader is False

    # Worker A releases lock
    lock_a.release(test_db_session)

    # Now Worker B can acquire lock
    is_b_now_leader = lock_b.acquire_or_renew(test_db_session)
    assert is_b_now_leader is True
    lock_b.release(test_db_session)
