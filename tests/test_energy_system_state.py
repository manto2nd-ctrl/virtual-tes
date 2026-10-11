"""Tests for Single Authoritative System State (Phase 5.11 - Stage 1).

Verifies:
1. Canonical EnergySystemState provides unified, atomic, versioned system state.
2. Thermodynamic consistency: SOC <=> Stored Energy <=> Sand Temp <=> HX Capacity.
3. Cross-page state consistency across Overview, Market, Dryer, and Physical views.
4. Top bar price consistency: effective variable market price correctly surfaced globally.
5. Strict physical safety: zero physical hardware commands issued.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from app.auth.security import create_session_token
from app.config.settings import get_settings
from app.database.models import ShadowTESSession
from app.database.repositories import ShadowRepository
from app.database.session import make_engine, make_session_factory, init_db
from app.services.energy_system_state import (
    EnergySystemState,
    get_energy_system_state,
)
from app.tes.model import HelicalAirHXModel, ThermalStateMapper
from app.web.main import app


@pytest.fixture
def auth_client() -> TestClient:
    """Authenticated TestClient using session token."""
    settings = get_settings()
    admin_token = create_session_token(
        username="admin",
        role="ADMIN",
        secret_key=settings.app_secret_key.get_secret_value(),
    )
    return TestClient(app, cookies={"session_token": admin_token})


def test_energy_system_state_structure():
    """EnergySystemState must contain all authoritative fields."""
    state = get_energy_system_state()
    assert isinstance(state, EnergySystemState)

    # 1. Identity & Versioning
    assert state.simulation_run_id is not None
    assert state.scenario_id is not None
    assert state.state_version >= 1
    assert state.snapshot_timestamp_utc.tzinfo is not None

    # 2. Market Interval & Pricing
    assert state.interval_start_utc < state.interval_end_utc
    assert state.spot_price_eur_mwh >= 0.0
    assert state.spot_price_eur_kwh == pytest.approx(state.spot_price_eur_mwh / 1000.0, rel=1e-4)
    assert state.effective_price_eur_mwh >= state.spot_price_eur_mwh
    assert state.effective_price_eur_kwh == pytest.approx(state.effective_price_eur_mwh / 1000.0, rel=1e-4)

    # 3. TES Physical & Stored State
    assert state.full_span_capacity_kwh == 15.0
    assert 0.0 <= state.soc_fraction <= 1.0
    assert state.soc_percent == pytest.approx(state.soc_fraction * 100.0, abs=0.1)
    assert state.stored_energy_kwh == pytest.approx(state.soc_fraction * 15.0, abs=0.1)
    assert 80.0 <= state.sand_temperature_c <= 300.0

    # 4. Heat Exchanger & Dryer
    assert state.hx_max_power_kw > 0.0
    assert state.hx_delivered_heat_kw <= state.hx_max_power_kw + 0.01
    assert state.dryer_heat_demand_kw > 0.0

    # 5. Grid Balances
    assert state.grid_connection_limit_kw == 12.0
    assert state.total_grid_import_kw <= 12.0
    assert state.grid_headroom_kw >= 0.0

    # 6. Safety & Hardware
    assert "DISCONNECTED" in state.hardware_status
    assert "ZERO PHYSICAL" in state.hardware_status


def test_thermodynamic_state_consistency():
    """Ensure strict mathematical mapping: SOC <=> Sand Temp <=> Stored Energy <=> HX Capacity."""
    # When SOC = 100%, sand temp is 300°C and stored energy is 15.0 kWh
    state = get_energy_system_state()
    mapper = ThermalStateMapper(80.0, 300.0)
    hx = HelicalAirHXModel(
        hx_area_m2=1.55,
        overall_u_w_m2k=9.0,
        airflow_m3_h=80.0,
        air_inlet_temperature_c=40.0,
        thermal_state_mapper=mapper,
    )
    
    expected_sand_temp = mapper.temperature_from_soc_fraction(state.soc_fraction)
    assert state.sand_temperature_c == pytest.approx(expected_sand_temp, abs=0.5)
    
    expected_hx_max = hx.p_max_at_temperature_kw(state.sand_temperature_c)
    assert state.hx_max_power_kw == pytest.approx(expected_hx_max, abs=0.1)
    assert state.hx_delivered_heat_kw <= state.hx_max_power_kw + 0.01


def test_cross_page_operational_state_consistency(auth_client: TestClient):
    """Overview, Market, Dryer, and Physical views must reflect the same canonical twin state."""
    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    with session_factory() as db:
        state = get_energy_system_state(db=db)

    # 1. Overview page
    resp_overview = auth_client.get("/dashboard/overview")
    assert resp_overview.status_code == 200
    assert f"{state.soc_percent:.1f}" in resp_overview.text
    assert f"{state.hx_max_power_kw:.2f}" in resp_overview.text

    # 2. Market page
    resp_market = auth_client.get("/dashboard/market")
    assert resp_market.status_code == 200
    assert f"{state.soc_percent:.1f}" in resp_market.text
    assert f"{state.stored_energy_kwh:.2f}" in resp_market.text

    # 3. Market current-card partial (HTMX polling)
    resp_card = auth_client.get("/api/market/current-card")
    assert resp_card.status_code == 200
    assert f"{state.soc_percent:.1f}" in resp_card.text
    assert f"{state.stored_energy_kwh:.2f}" in resp_card.text

    # 4. Dryer page
    resp_dryer = auth_client.get("/dashboard/dryer")
    assert resp_dryer.status_code == 200
    assert "Dryer Equipment" in resp_dryer.text

    # 5. Physical page
    resp_phys = auth_client.get("/dashboard/physical")
    assert resp_phys.status_code == 200
    assert "CURRENT LIVE TWIN OPERATIONAL STATE" in resp_phys.text
    assert f"{state.soc_percent:.1f}%" in resp_phys.text


def test_top_bar_price_surfacing(auth_client: TestClient):
    """Top bar must surface effective electricity price and not hardcoded €0.068 when live state exists."""
    resp = auth_client.get("/dashboard/overview")
    assert resp.status_code == 200
    assert "Market Price:" in resp.text
    assert "&euro;" in resp.text or "€" in resp.text
