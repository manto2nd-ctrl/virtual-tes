"""Automated test suite for Phase 5.7 Engineering Dashboard & Scenario Explorer.

Validates all 15 requirements from Section 36 of the specification:
1. UI API thermal-state calculation matches ThermalStateMapper directly.
2. HX calculator output matches HelicalAirHXModel directly.
3. Vessel volume calculation is correct (~269 L for 600x950 mm).
4. Capacity-driven vessel calculation is internally consistent.
5. Geometry-driven capacity calculation is internally consistent.
6. 275.7 kg gives ~15.0 kWh full-span.
7. 300 kg gives ~16.3 kWh full-span.
8. 320 kg gives ~17.4 kWh full-span.
9. Scenario persistence is append-only (triggers block updates/deletes).
10. Scenario comparison uses backend physics (compare_scenarios).
11. Mock results are visibly labeled as mock.
12. Engineering estimates are never labeled measured.
13. Grid visualization values respect the 12 kW limit.
14. Dashboard optimization results match CLI optimization results for identical inputs.
15. No hardware-control route exists (scan routes for modbus, plc, relay, contactor, etc.).
"""

from __future__ import annotations

import math
import pytest
from datetime import date, timedelta
from zoneinfo import ZoneInfo
from starlette.testclient import TestClient
from sqlalchemy.exc import InternalError, OperationalError, IntegrityError

from app.web.main import app
from app.tes.thermal import ThermalStateMapper, HelicalAirHXModel
from app.tes.vessel import (
    VesselGeometryCalculator,
    VesselGeometryParameters,
    VesselCalculationResult,
)
from app.tes.model import get_gen0_discharge_curve
from app.database.session import make_engine, make_session_factory
from app.database.models import EngineeringScenario
from app.services.scenario_service import (
    save_scenario,
    list_scenarios,
    compare_scenarios,
)
from app.config.settings import get_settings
from app.config.parameters import TESParameters, SiteParameters, TariffParameters
from app.core.timegrid import build_intervals, local_day_bounds_utc
from app.providers.mock import MockPriceProvider
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.economics.tariff import effective_price_eur_mwh
from app.optimization.domain import OptimizationIntervalInput, OptimizationProblemInput
from app.optimization.lp_optimizer import LPOptimizer


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Starlette test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. HTML Pages Availability & Rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "/dashboard/overview",
        "/dashboard/physical",
        "/dashboard/hx",
        "/dashboard/vessel",
        "/dashboard/dispatch",
        "/dashboard/economics",
        "/dashboard/sizing",
        "/dashboard/sensitivity",
        "/dashboard/assumptions",
        "/dashboard/scenarios",
    ],
)
def test_dashboard_pages_render_ok(client: TestClient, path: str):
    """All 10 dashboard pages must render HTTP 200 with standard HTML structure."""
    resp = client.get(path)
    assert resp.status_code == 200, f"Path {path} returned status {resp.status_code}"
    assert "<!DOCTYPE html>" in resp.text
    assert "Virtual TES Gen0" in resp.text


# ---------------------------------------------------------------------------
# 2. Requirement 1: UI API Thermal State Calculation Matches ThermalStateMapper
# ---------------------------------------------------------------------------

def test_api_thermal_state_matches_mapper(client: TestClient):
    """Verify POST /api/calculate/thermal-state matches ThermalStateMapper directly."""
    mapper = ThermalStateMapper(
        t_min_c=80.0,
        t_max_c=300.0,
    )
    sand_mass = 275.7
    full_cap = mapper.capacity_for_sand_mass(sand_mass)
    disp_cap = full_cap * (1.00 - 0.10)
    t_at_soc_50 = mapper.temperature_from_soc_fraction(0.50)

    payload = {
        "physical_temperature_min_c": 80.0,
        "physical_temperature_max_c": 300.0,
        "thermal_capacity_full_span_kwh": full_cap,
        "optimizer_soc_min_fraction": 0.10,
        "optimizer_soc_max_fraction": 1.00,
        "soc_fraction": 0.50,
    }
    resp = client.post("/api/calculate/thermal-state", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert math.isclose(data["thermal_capacity_full_span_kwh"], full_cap, abs_tol=1e-2)
    assert math.isclose(data["dispatchable_capacity_kwh"], disp_cap, abs_tol=1e-2)
    assert math.isclose(data["current_soc_fraction"], 0.50, abs_tol=1e-2)
    assert math.isclose(data["current_sand_temperature_c"], t_at_soc_50, abs_tol=1e-2)
    assert math.isclose(data["sand_mass_kg"], sand_mass, abs_tol=0.2)


# ---------------------------------------------------------------------------
# 3. Requirement 2: HX Calculator Output Matches HelicalAirHXModel Directly
# ---------------------------------------------------------------------------

def test_api_hx_matches_model(client: TestClient):
    """Verify POST /api/calculate/hx matches HelicalAirHXModel directly."""
    hx = HelicalAirHXModel(
        hx_area_m2=1.55,
        overall_u_w_m2k=9.0,
        airflow_m3_h=80.0,
        air_inlet_temperature_c=40.0,
    )
    mapper = ThermalStateMapper()
    t_sand = mapper.temperature_from_soc_fraction(0.50)
    expected_pmax = hx.p_max_at_temperature_kw(t_sand)
    expected_tout = hx.calculate_outlet_temperature_c(t_sand)

    payload = {
        "soc_fraction": 0.50,
        "overall_u_w_m2k": 9.0,
        "hx_area_m2": 1.55,
        "airflow_m3_h": 80.0,
        "air_inlet_temperature_c": 40.0,
        "process_demand_kw": 1.5,
    }
    resp = client.post("/api/calculate/hx", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert math.isclose(data["ntu"], hx.ntu, abs_tol=1e-3)
    assert math.isclose(data["effectiveness"], hx.effectiveness, abs_tol=1e-3)
    assert math.isclose(data["p_max_kw"], expected_pmax, abs_tol=1e-2)
    assert math.isclose(data["predicted_outlet_air_temperature_c"], expected_tout, abs_tol=1e-2)


# ---------------------------------------------------------------------------
# 4. Requirement 3: Vessel Volume Calculation (~269 L for 600x950 mm)
# ---------------------------------------------------------------------------

def test_vessel_volume_600x950(client: TestClient):
    """Vessel 600 mm ID x 950 mm height gives ~269 L gross internal volume."""
    params = VesselGeometryParameters(
        internal_diameter_mm=600.0,
        straight_shell_height_mm=950.0,
        sand_bulk_density_kg_m3=1600.0,
        hx_displaced_volume_l=24.0,
        internals_displaced_volume_l=8.0,
    )
    calc = VesselGeometryCalculator(params=params)

    # pi * r^2 * h = pi * (0.3)^2 * 0.95 = 0.268606 m^3 = 268.61 L
    assert round(calc.gross_volume_liters, 1) == 268.6
    assert round(calc.gross_volume_liters) == 269
    assert round(calc.net_available_volume_liters, 1) == 236.6

    # Verify via API endpoint as well
    resp = client.post(
        "/api/calculate/vessel",
        json={
            "mode": "capacity_driven",
            "target_full_span_kwh": 15.0,
            "internal_diameter_mm": 600.0,
            "straight_shell_height_mm": 950.0,
            "sand_bulk_density_kg_m3": 1600.0,
            "hx_displaced_volume_l": 24.0,
            "internals_displaced_volume_l": 8.0,
            "min_freeboard_mm": 80.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert round(data["result"]["gross_volume_l"]) == 269
    assert round(data["result"]["net_available_volume_l"], 1) == 236.6


# ---------------------------------------------------------------------------
# 5. Requirement 4: Capacity-Driven Vessel Calculation is Internally Consistent
# ---------------------------------------------------------------------------

def test_capacity_driven_vessel_consistency():
    """Mode A: capacity-driven calculation maintains volumetric and mass balance."""
    calc = VesselGeometryCalculator()
    target_cap = 15.0
    res = calc.calculate_capacity_driven(target_full_span_kwh=target_cap)

    # Mass = Target Cap / specific_kwh_kg
    expected_mass = calc.mapper.equivalent_sand_mass_for_capacity(target_cap)
    assert math.isclose(res.sand_mass_kg, expected_mass, rel_tol=1e-3)

    # Sand volume = mass / density * 1000
    expected_sand_vol_l = (res.sand_mass_kg / calc.params.sand_bulk_density_kg_m3) * 1000.0
    assert math.isclose(res.sand_volume_l, expected_sand_vol_l, rel_tol=1e-3)

    # Bed height = sand_vol_l / (net_v / straight_shell_height_mm)
    net_v_per_mm = calc.net_available_volume_liters / calc.params.straight_shell_height_mm
    expected_bed_h = expected_sand_vol_l / net_v_per_mm
    assert math.isclose(res.sand_fill_height_mm, expected_bed_h, rel_tol=1e-3)

    # Freeboard = Total Height - Bed Height
    assert math.isclose(res.freeboard_height_mm, calc.params.straight_shell_height_mm - expected_bed_h, rel_tol=1e-3)
    assert res.freeboard_percent > 0.0
    assert not res.is_overflow


# ---------------------------------------------------------------------------
# 6. Requirement 5: Geometry-Driven Capacity Calculation is Internally Consistent
# ---------------------------------------------------------------------------

def test_geometry_driven_vessel_consistency():
    """Mode B: geometry-driven calculation fills available net volume consistently."""
    calc = VesselGeometryCalculator()
    fill_frac = 0.85
    res = calc.calculate_geometry_driven(fill_fraction=fill_frac)

    assert math.isclose(res.sand_volume_l, calc.net_available_volume_liters * fill_frac, rel_tol=1e-3)
    assert res.sand_mass_kg > 0.0
    assert res.thermal_capacity_full_span_kwh > 0.0
    assert res.dispatchable_capacity_kwh == pytest.approx(
        res.thermal_capacity_full_span_kwh * 0.90, rel=1e-3
    )
    assert not res.is_overflow


# ---------------------------------------------------------------------------
# 7. Requirements 6, 7, 8: Exact Reference Points (275.7, 300, 320 kg)
# ---------------------------------------------------------------------------

def test_reference_scenarios_capacities():
    """Verify exact full-span capacities:
    - 275.7 kg -> ~15.0 kWh
    - 300.0 kg -> ~16.3 kWh (16.32 kWh)
    - 320.0 kg -> ~17.4 kWh (17.41 kWh)
    """
    mapper = ThermalStateMapper(
        t_min_c=80.0,
        t_max_c=300.0,
    )

    cap_275 = mapper.capacity_for_sand_mass(275.7)
    assert math.isclose(cap_275, 15.0, abs_tol=0.05)

    cap_300 = mapper.capacity_for_sand_mass(300.0)
    assert math.isclose(cap_300, 16.32, abs_tol=0.05)

    cap_320 = mapper.capacity_for_sand_mass(320.0)
    assert math.isclose(cap_320, 17.41, abs_tol=0.05)


# ---------------------------------------------------------------------------
# 8. Requirement 9: Scenario Persistence is Append-Only
# ---------------------------------------------------------------------------

def test_scenario_persistence_append_only():
    """Engineering scenarios table must reject UPDATE and DELETE via SQLite triggers."""
    import uuid
    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    unique_name = f"TestAppendOnly_{uuid.uuid4().hex[:8]}"

    with session_factory() as session:
        # Save a test scenario
        scen1 = save_scenario(
            session=session,
            name=unique_name,
            parameters={
                "sand_mass_kg": 275.7,
                "thermal_capacity_full_span_kwh": 15.0,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
            },
            description="Testing immutability",
        )
        assert scen1.version == 1

        # Save again with same name -> version auto-increments to 2
        scen2 = save_scenario(
            session=session,
            name=unique_name,
            parameters={
                "sand_mass_kg": 300.0,
                "thermal_capacity_full_span_kwh": 16.32,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
            },
            description="Second version",
        )
        assert scen2.version == 2
        assert scen2.id != scen1.id

        # Verify UPDATE raises database error (trigger abort)
        with pytest.raises((OperationalError, InternalError, IntegrityError)):
            session.query(EngineeringScenario).filter(
                EngineeringScenario.id == scen1.id
            ).update({"description": "attempted edit"})
            session.flush()
        session.rollback()

        # Verify DELETE raises database error (trigger abort)
        with pytest.raises((OperationalError, InternalError, IntegrityError)):
            target = session.query(EngineeringScenario).filter(
                EngineeringScenario.id == scen1.id
            ).first()
            session.delete(target)
            session.flush()
        session.rollback()


# ---------------------------------------------------------------------------
# 9. Requirement 10: Scenario Comparison Uses Backend Physics
# ---------------------------------------------------------------------------

def test_scenario_comparison_physics():
    """Scenario comparison must compute deltas on mass, capacity, and vessel dimensions."""
    import uuid
    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    name_a = f"ScenarioA_{uuid.uuid4().hex[:8]}"
    name_b = f"ScenarioB_{uuid.uuid4().hex[:8]}"

    with session_factory() as session:
        s1 = save_scenario(
            session=session,
            name=name_a,
            parameters={
                "sand_mass_kg": 275.7,
                "thermal_capacity_full_span_kwh": 15.0,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
            },
        )
        s2 = save_scenario(
            session=session,
            name=name_b,
            parameters={
                "sand_mass_kg": 300.0,
                "thermal_capacity_full_span_kwh": 16.32,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
            },
        )

        cmp = compare_scenarios(s1, s2)
        assert cmp["scenario_a"]["name"] == f"{name_a} (v1)"
        assert cmp["scenario_b"]["name"] == f"{name_b} (v1)"
        
        diff_by_label = {row["label"]: row for row in cmp["comparison_rows"]}
        assert "Sand Mass" in diff_by_label
        assert diff_by_label["Sand Mass"]["diff"] == pytest.approx(24.3, abs=0.1)
        assert diff_by_label["Full-Span Capacity"]["diff"] == pytest.approx(1.32, abs=0.1)


# ---------------------------------------------------------------------------
# 10. Requirement 11: Mock Results Visibly Labeled as Mock
# ---------------------------------------------------------------------------

def test_mock_labeling_in_rendered_ui(client: TestClient):
    """UI pages displaying mock results must include visible MOCK badges/labels."""
    resp = client.get("/dashboard/dispatch")
    assert resp.status_code == 200
    assert "MOCK" in resp.text.upper()
    assert "SYNTHETIC / MOCK PRICES" in resp.text

    resp_econ = client.get("/dashboard/economics")
    assert resp_econ.status_code == 200
    assert "MOCK" in resp_econ.text.upper()


# ---------------------------------------------------------------------------
# 11. Requirement 12: Engineering Estimates are Never Labeled Measured
# ---------------------------------------------------------------------------

def test_engineering_caveat_labels(client: TestClient):
    """UI pages must display caveat banners and must not claim physical measurement."""
    resp_overview = client.get("/dashboard/overview")
    assert "ENGINEERING ESTIMATE ONLY" in resp_overview.text
    assert "NOT YET CALIBRATED TO PHYSICAL GEN0" in resp_overview.text

    resp_vessel = client.get("/dashboard/vessel")
    assert "PRELIMINARY GEN0 VESSEL" in resp_vessel.text
    assert "NOT FOR FABRICATION" in resp_vessel.text


# ---------------------------------------------------------------------------
# 12. Requirement 13: Grid Visualization Respects 12 kW Limit
# ---------------------------------------------------------------------------

def test_grid_visualization_respects_12kw_limit(client: TestClient):
    """Total grid power must not exceed 12.0 kW across all 96 dispatch intervals."""
    resp = client.get("/api/dispatch-demo")
    assert resp.status_code == 200
    data = resp.json()

    for item in data["chart_data"]:
        assert item["total_grid_power"] <= 12.0001, (
            f"Interval {item['time']} exceeded grid limit: {item['total_grid_power']} kW"
        )
        assert item["grid_limit"] == 12.0
        assert item["configured_charge_limit"] == 9.0


# ---------------------------------------------------------------------------
# 13. Requirement 14: Dashboard Optimization Matches CLI Optimization Exactly
# ---------------------------------------------------------------------------

def test_dashboard_opt_matches_cli_opt():
    """Optimization conducted by API endpoint produces identical objective to direct solver."""
    tz = ZoneInfo("Europe/Vilnius")
    s_date = date(2026, 10, 1)
    e_date = date(2026, 10, 2)
    s_utc, _ = local_day_bounds_utc(s_date, tz)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), tz)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
    prov = MockPriceProvider(tz=tz, resolution_minutes=15, seed=42)
    prices = prov.fetch_day_ahead("LT", s_utc, e_utc).points

    heat_profile = ConstantHeatDemand(value_kw=1.5)
    site_profile = ConstantSiteLoad(value_kw=2.0)
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    site_params = SiteParameters(
        grid_connection_limit_kw=12.0,
        process_heat_demand_kw=1.5,
        other_loads_kw=2.0,
    )
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
    tariff_params = TariffParameters(
        supplier_markup_eur_mwh=1.5,
        variable_grid_fee_eur_mwh=25.0,
        variable_tax_eur_mwh=5.0,
    )
    curve = get_gen0_discharge_curve("fixed_gen0_hx", capacity_kwh=15.0, overall_u_w_m2k=9.0, airflow_m3_h=80.0)

    opt_inputs = []
    for iv, p, hd, sl in zip(intervals, prices, heat_demands, site_loads, strict=True):
        eff = effective_price_eur_mwh(p.price_eur_mwh, tariff_params)
        opt_inputs.append(
            OptimizationIntervalInput(
                start_utc=iv.start_utc,
                end_utc=iv.end_utc,
                spot_price_eur_mwh=p.price_eur_mwh,
                effective_price_eur_mwh=eff,
                heat_demand_kw=hd,
                other_site_load_kw=sl,
                auxiliary_load_kw=0.05,
            )
        )

    problem = OptimizationProblemInput(
        intervals=opt_inputs,
        tes_params=tes_params,
        grid_connection_limit_kw=site_params.grid_connection_limit_kw,
        initial_soc_kwh=7.5,
        target_terminal_soc_kwh=7.5,
        terminal_soc_condition="exact",
        discharge_limit_curve=curve,
        lexicographic=True,
    )
    res_direct = LPOptimizer().optimize(problem)

    # Call the client endpoint
    client = TestClient(app)
    resp = client.get("/api/dispatch-demo")
    assert resp.status_code == 200
    api_data = resp.json()

    assert math.isclose(api_data["objective_eur"], res_direct.objective_eur, abs_tol=1e-2)
    assert len(api_data["timeline_table"]) == len(res_direct.intervals)


# ---------------------------------------------------------------------------
# 14. Requirement 15: No Hardware-Control Route Exists
# ---------------------------------------------------------------------------

def test_no_hardware_control_routes():
    """Verify that no routes exist for PLC, Modbus, contactor, or hardware commands."""
    forbidden_terms = [
        "modbus",
        "plc",
        "relay",
        "contactor",
        "vfd",
        "damper",
        "heater_on",
        "heater_off",
        "sensor/poll",
        "hardware",
        "actuator",
    ]

    for route in app.routes:
        path = getattr(route, "path", "").lower()
        name = getattr(route, "name", "").lower()
        endpoint = getattr(route, "endpoint", None)
        endpoint_name = endpoint.__name__.lower() if endpoint else ""

        for term in forbidden_terms:
            assert term not in path, f"Forbidden hardware term '{term}' in route path: {path}"
            assert term not in name, f"Forbidden hardware term '{term}' in route name: {name}"
            assert term not in endpoint_name, (
                f"Forbidden hardware term '{term}' in endpoint: {endpoint_name}"
            )


# ---------------------------------------------------------------------------
# 15. Regression Tests for Refined Gen0 Reference & UI Requirements
# ---------------------------------------------------------------------------

def test_gen0_reference_hx_parameters(client: TestClient):
    """Verify that UI pages display the validated provisional Gen0 HX geometry (no stale 22mm/300mm/12m)."""
    resp_overview = client.get("/dashboard/overview")
    assert resp_overview.status_code == 200
    assert "AISI 304L" in resp_overview.text
    assert "60.3" in resp_overview.text
    assert "420" in resp_overview.text
    assert "1.55" in resp_overview.text

    resp_hx = client.get("/dashboard/hx")
    assert resp_hx.status_code == 200
    assert "60.3" in resp_hx.text
    assert "420" in resp_hx.text
    assert "1.55" in resp_hx.text
    assert "56.3" in resp_hx.text

    resp_assumptions = client.get("/dashboard/assumptions")
    assert resp_assumptions.status_code == 200
    assert "60.3" in resp_assumptions.text
    assert "420" in resp_assumptions.text


def test_overview_grid_headroom_terminology(client: TestClient):
    """Verify that overview displays distinct Available TES Charging Headroom and Remaining Grid Margin."""
    resp = client.get("/dashboard/overview")
    assert resp.status_code == 200
    assert "Available TES Charging Headroom" in resp.text
    assert "9.95" in resp.text
    assert "Remaining Grid Margin" in resp.text
    assert "0.95" in resp.text


def test_sizing_matrix_full_span_and_clipping(client: TestClient):
    """Verify sizing matrix sweeps 10, 15, 20, 30 kWh and clips 12 kW charger to 9.95 kW."""
    resp = client.get("/api/sizing-demo")
    assert resp.status_code == 200
    data = resp.json()
    
    combos = data["combinations"]
    caps = sorted(list({round(c["thermal_capacity_full_span_kwh"]) for c in combos}))
    assert caps == [10, 15, 20, 30]

    powers = sorted(list({round(c["configured_charge_power_limit_kw"]) for c in combos}))
    assert powers == [6, 9, 12]

    # Find the 12 kW charger combination
    for c in combos:
        if round(c["configured_charge_power_limit_kw"]) == 12:
            assert c["peak_actual_charge_power_kw"] <= 9.9501
            assert c["peak_actual_charge_power_kw"] < c["configured_charge_power_limit_kw"]


def test_standing_loss_sensitivity_range(client: TestClient):
    """Verify sensitivity covers 1% to 30%/day and labels all as ASSUMPTION / SENSITIVITY."""
    resp = client.get("/dashboard/sensitivity")
    assert resp.status_code == 200
    assert "ASSUMPTION / SENSITIVITY" in resp.text
    for pct in ["1.0", "2.0", "3.0", "6.0", "10.0", "15.0", "20.0", "30.0"]:
        assert pct in resp.text

