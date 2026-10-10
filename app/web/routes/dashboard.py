"""HTML Page Routes for the Gen0 Engineering Dashboard (Phase 5.7).

Serves Jinja2 templates for all 8 main engineering navigation areas:
1. Overview
2. TES Physical Model
3. Heat Exchanger
4. Vessel / Sand Sizing
5. Market & Dispatch
6. Economic Comparison
7. Sizing & Sensitivity
8. Assumptions & Data Quality
(plus Saved Scenarios)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth.dependencies import UserSession, get_current_user, require_authenticated
from app.backtest.domain import BacktestConfig
from app.backtest.engine import (
    evaluate_cheapest_n_heuristic,
    evaluate_direct_electric_heating,
    evaluate_perfect_foresight_lp,
    evaluate_realistic_rolling_lp,
)
from app.backtest.sensitivity import SensitivityAnalyzer, SizingStudyAnalyzer
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.config.settings import get_settings
from app.core.timegrid import build_intervals, ensure_utc, local_day_bounds_utc
from app.database.session import init_db, make_engine, make_session_factory
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.services.scenario_service import list_scenarios, seed_default_scenarios_if_empty
from app.tes.model import get_gen0_discharge_curve
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper
from app.tes.vessel import VesselGeometryCalculator, VesselGeometryParameters
from sqlalchemy import select

dashboard_router = APIRouter(
    prefix="/dashboard",
    tags=["Engineering UI"],
    dependencies=[Depends(require_authenticated)],
)
templates = Jinja2Templates(directory="app/web/templates")

_engine = None
_session_factory = None


def get_db():
    global _engine, _session_factory
    if _session_factory is None:
        settings = get_settings()
        _engine = make_engine(settings.database_url)
        init_db(_engine)
        _session_factory = make_session_factory(_engine)
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


def get_common_context(request: Request, active_page: str, user: UserSession | None = None) -> dict[str, Any]:
    """Provide standard global header information across all engineering pages."""
    settings = get_settings()
    if user is None:
        user = get_current_user(request, settings=settings)
    return {
        "request": request,
        "active_page": active_page,
        "user": user,
        "app_title": "VIRTUAL TES GEN0",
        "model_badge": "ENGINEERING MODEL -- NOT YET CALIBRATED TO PHYSICAL GEN0",
        "model_version": "Phase 5.9 / Railway 24/7",
        "price_source": "LITGRID (Primary) / ELERING (Cross-Check)",
        "physical_status": "ENGINEERING ESTIMATE",
        "capacity_semantics_version": "full_span_v1",
        "grid_connection_limit_kw": settings.site.grid_connection_limit_kw,
    }


def get_runtime_status_context(db: Session) -> dict[str, Any]:
    """Aggregate 24/7 worker heartbeat, server, and database health for UI display."""
    from app.database.models import ShadowTESSession, WorkerHeartbeat
    settings = get_settings()
    now = datetime.now(timezone.utc)
    hb = db.execute(
        select(WorkerHeartbeat).order_by(WorkerHeartbeat.timestamp_utc.desc()).limit(1)
    ).scalar_one_or_none()

    active_sess = db.execute(
        select(ShadowTESSession).order_by(ShadowTESSession.created_at_utc.desc()).limit(1)
    ).scalar_one_or_none()

    worker_online = False
    heartbeat_age_sec = None
    if hb and hb.timestamp_utc:
        age = (now - ensure_utc(hb.timestamp_utc)).total_seconds()
        heartbeat_age_sec = int(age)
        if age <= 180 and hb.status in ("RUNNING", "STANDBY"):
            worker_online = True

    worker_display = "RUNNING" if worker_online else "OFFLINE"
    server_display = "ONLINE"
    db_display = "CONNECTED"
    env_display = settings.app_env.upper()

    shadow_status = "STOPPED"
    if active_sess:
        if not worker_online and active_sess.status == "RUNNING":
            shadow_status = "OFFLINE / STALE"
        else:
            shadow_status = active_sess.status

    return {
        "env": env_display,
        "server_status": server_display,
        "worker_status": worker_display,
        "worker_online": worker_online,
        "db_status": db_display,
        "shadow_status": shadow_status,
        "heartbeat_age_sec": heartbeat_age_sec,
        "last_heartbeat_utc": hb.timestamp_utc if hb else None,
        "last_market_fetch_utc": hb.last_market_fetch_utc if hb else None,
        "last_executed_interval_utc": hb.last_executed_interval_utc if hb else None,
        "last_optimization_utc": hb.last_optimization_utc if hb else None,
        "next_interval_utc": hb.next_interval_utc if hb else None,
    }


def get_market_status_context(db: Session) -> dict[str, Any]:
    """Retrieve status snapshot from MarketDataService for UI rendering."""
    from app.services.market_data_service import MarketDataService

    service = MarketDataService()
    try:
        fetch_res = service.fetch_market_data(session=db)
        snap = fetch_res.status_snapshot
        return {
            "primary_source": snap.primary_source,
            "primary_health": snap.primary_health,
            "primary_error": snap.primary_error,
            "cross_check_source": snap.cross_check_source,
            "cross_check_health": snap.cross_check_health,
            "cross_check_error": snap.cross_check_error,
            "latest_interval_str": snap.latest_interval_str,
            "tomorrow_prices": snap.tomorrow_prices,
            "resolution": snap.resolution,
            "validation": snap.validation,
            "data_freshness": snap.data_freshness,
            "active_source": snap.active_source,
            "is_stale": fetch_res.is_stale,
            "intervals_count": snap.intervals_count,
        }
    except Exception as exc:
        return {
            "primary_source": "LITGRID",
            "primary_health": "ERROR",
            "primary_error": str(exc),
            "cross_check_source": "ELERING",
            "cross_check_health": "ERROR",
            "cross_check_error": str(exc),
            "latest_interval_str": "None",
            "tomorrow_prices": "WAITING",
            "resolution": "15 min",
            "validation": "NOT CHECKED",
            "data_freshness": "LIVE MARKET DATA UNAVAILABLE",
            "active_source": "NONE",
            "is_stale": False,
            "intervals_count": 0,
        }


def get_market_full_context(db: Session, now_utc: datetime | None = None) -> dict[str, Any]:
    """Retrieve full market view model for Overview and Market pages (Phase 5.8.1)."""
    from app.services.market_data_service import MarketDataService

    service = MarketDataService()
    return service.get_market_view_context(session=db, now_utc=now_utc)


@dashboard_router.get("/", response_class=HTMLResponse)
@dashboard_router.get("/overview", response_class=HTMLResponse)
def page_overview(request: Request, db: Session = Depends(get_db)):
    """1. Overview: Reference Gen0 configuration, KPI cards, and hardware specs."""
    seed_default_scenarios_if_empty(db)
    ctx = get_common_context(request, "overview")
    ctx["runtime_status"] = get_runtime_status_context(db)
    market_view = get_market_full_context(db)
    ctx["market"] = market_view
    ctx["market_status"] = market_view["market_status"]

    from app.services.shadow_runtime import ShadowRuntimeService
    shadow_svc = ShadowRuntimeService()
    ctx["shadow"] = shadow_svc.get_live_dashboard_state(db)

    mapper = ThermalStateMapper(80.0, 300.0)
    hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0, air_inlet_temperature_c=40.0)
    vessel_calc = VesselGeometryCalculator()
    vessel_res = vessel_calc.calculate_for_sand_mass(275.7)

    # Reference KPI snapshot
    ctx.update({
        "full_span_capacity_kwh": 15.0,
        "dispatchable_capacity_kwh": 13.5,
        "sand_mass_kg": 275.7,
        "temp_min_c": 80.0,
        "temp_max_c": 300.0,
        "soc_min_pct": 10.0,
        "soc_max_pct": 100.0,
        "temp_at_min_soc_c": round(mapper.temperature_from_soc_fraction(0.10), 2),
        "heater_power_kw": 9.0,
        "heater_drywells": "6 × 1.5 kW cartridge heaters in closed drywells",
        "grid_limit_kw": 12.0,
        "site_load_kw": 2.0,
        "aux_power_kw": 0.05,
        "available_tes_charging_headroom_kw": 9.95,
        "remaining_grid_margin_kw": 0.95,
        "available_charge_headroom_kw": 9.95,
        "standing_loss_pct_day": 2.0,
        "charge_eff": 0.95,
        "discharge_eff": 0.90,
        "round_trip_eff_pct": 85.5,
        "hx_material": "AISI 304L / EN 1.4307",
        "hx_tube": "60.3 × 2.0 mm",
        "hx_coil_dia_mm": "~420 mm",
        "hx_turns": "~6",
        "hx_length_m": "~8.0–8.5 m",
        "hx_area_m2": 1.55,
        "hx_ref_u": 9.0,
        "hx_ref_airflow": 80.0,
        "hx_ref_inlet_temp": 40.0,
        "hx_pmax_100_soc_kw": round(hx.p_max_at_soc_fraction_kw(1.0), 2),
        "hx_pmax_50_soc_kw": round(hx.p_max_at_soc_fraction_kw(0.50), 2),
        "hx_pmax_min_soc_kw": round(hx.p_max_at_soc_fraction_kw(0.10), 2),
        "vessel_id_mm": 600.0,
        "vessel_height_mm": 950.0,
        "vessel_gross_vol_l": round(vessel_res.gross_volume_l, 1),
        "vessel_net_vol_l": round(vessel_res.net_available_volume_l, 1),
        "vessel_status": "PRELIMINARY -- NOT FOR FABRICATION",
    })
    return templates.TemplateResponse(request=request, name="overview.html", context=ctx)


@dashboard_router.get("/market", response_class=HTMLResponse)
def page_market(request: Request, db: Session = Depends(get_db)):
    """Market: Real-time Lithuanian day-ahead price monitoring, provenance, and shadow dispatch (Phase 5.8.1)."""
    ctx = get_common_context(request, "market")
    market_view = get_market_full_context(db)
    ctx["market"] = market_view
    ctx["market_status"] = market_view["market_status"]
    return templates.TemplateResponse(request=request, name="market.html", context=ctx)



@dashboard_router.get("/physical", response_class=HTMLResponse)
def page_physical(request: Request):
    """2. TES Physical Model: Interactive enthalpy calculator and non-linear charts."""
    ctx = get_common_context(request, "physical")
    mapper = ThermalStateMapper(80.0, 300.0)

    ctx.update({
        "t_min_c": 80.0,
        "t_max_c": 300.0,
        "full_span_kwh": 15.0,
        "soc_min_fraction": 0.10,
        "soc_max_fraction": 1.00,
        "default_soc": 0.50,
        "sand_mass_kg": round(mapper.equivalent_sand_mass_for_capacity(15.0), 1),
        "specific_kwh_kg": round(mapper.specific_stored_energy_kwh_per_kg(), 5),
        "temp_at_min_soc_c": round(mapper.temperature_from_soc_fraction(0.10), 2),
        "dispatchable_kwh": 13.5,
    })
    return templates.TemplateResponse(request=request, name="physical.html", context=ctx)


@dashboard_router.get("/hx", response_class=HTMLResponse)
def page_hx(request: Request):
    """3. Heat Exchanger: Interactive epsilon-NTU model, presets, and derating curves."""
    ctx = get_common_context(request, "hx")
    ctx.update({
        "ref_u": 9.0,
        "ref_airflow": 80.0,
        "ref_area": 1.55,
        "ref_inlet_temp": 40.0,
        "process_demand_kw": 1.5,
    })
    return templates.TemplateResponse(request=request, name="hx.html", context=ctx)


@dashboard_router.get("/vessel", response_class=HTMLResponse)
def page_vessel(request: Request):
    """4. Vessel / Sand Sizing: Containment geometry, fill heights, freeboard, and comparisons."""
    ctx = get_common_context(request, "vessel")
    calc = VesselGeometryCalculator()
    ref_scenarios = calc.get_reference_scenarios()
    ctx.update({
        "default_diameter_mm": 600.0,
        "default_height_mm": 950.0,
        "default_density_kg_m3": 1600.0,
        "default_hx_disp_l": 24.0,
        "default_internals_disp_l": 8.0,
        "gross_volume_l": round(calc.gross_volume_liters, 1),
        "net_volume_l": round(calc.net_available_volume_liters, 1),
        "ref_scenarios": {k: v.to_dict() for k, v in ref_scenarios.items()},
    })
    return templates.TemplateResponse(request=request, name="vessel.html", context=ctx)


@dashboard_router.get("/dispatch", response_class=HTMLResponse)
def page_dispatch(request: Request, db: Session = Depends(get_db)):
    """5. Market & Dispatch: 24h timeline, spot price overlay, heater schedule, and grid limit."""
    ctx = get_common_context(request, "dispatch")
    ctx["market_status"] = get_market_status_context(db)
    return templates.TemplateResponse(request=request, name="dispatch.html", context=ctx)


@dashboard_router.get("/economics", response_class=HTMLResponse)
def page_economics(request: Request):
    """6. Economic Comparison: 4-way baseline comparison with explicit caveat banners."""
    ctx = get_common_context(request, "economics")

    # Run quick 2-day backtest to populate table
    tz = ZoneInfo("Europe/Vilnius")
    s_date = date(2026, 10, 1)
    e_date = date(2026, 10, 3)
    s_utc, _ = local_day_bounds_utc(s_date, tz)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), tz)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
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
        mode="both",
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

    direct = evaluate_direct_electric_heating(intervals, prices, heat_demands, tariff_params)
    cheapest = evaluate_cheapest_n_heuristic(intervals, prices, heat_demands, site_loads, config, tz)
    rolling = evaluate_realistic_rolling_lp(intervals, prices, heat_demands, site_loads, config, tz)
    foresight = evaluate_perfect_foresight_lp(intervals, prices, heat_demands, site_loads, config)

    ctx.update({
        "direct": direct.to_dict(),
        "cheapest": cheapest.to_dict(),
        "rolling": rolling.to_dict(),
        "foresight": foresight.to_dict(),
    })
    return templates.TemplateResponse(request=request, name="economics.html", context=ctx)


@dashboard_router.get("/sizing", response_class=HTMLResponse)
def page_sizing(request: Request):
    """7. Sizing Matrix: TES capacity & charging power trade-offs with non-prescriptive benchmarks."""
    ctx = get_common_context(request, "sizing")
    tz = ZoneInfo("Europe/Vilnius")
    s_date = date(2026, 10, 1)
    e_date = date(2026, 10, 2)
    s_utc, _ = local_day_bounds_utc(s_date, tz)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), tz)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
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

    sizing_analyzer = SizingStudyAnalyzer(tz=tz)
    reports = sizing_analyzer.run_sizing_sweep(
        base_config=config,
        heat_profile=heat_profile,
        site_profile=site_profile,
        prices=prices,
        capacities_kwh=[10.0, 15.0, 20.0, 30.0],
        charge_powers_kw=[6.0, 9.0, 12.0],
        sizing_mode="fixed_gen0_hx",
    )
    report = reports["fixed_gen0_hx"]

    ctx.update({
        "report": report.to_dict(),
        "combinations": [c.to_dict() for c in report.combinations],
        "lowest_cost": report.lowest_cost_combination.to_dict() if report.lowest_cost_combination else None,
        "highest_rel": report.highest_reliability_combination.to_dict() if report.highest_reliability_combination else None,
        "lowest_cap_995": report.lowest_capacity_meeting_99_5_rel.to_dict() if report.lowest_capacity_meeting_99_5_rel else None,
    })
    return templates.TemplateResponse(request=request, name="sizing.html", context=ctx)


@dashboard_router.get("/sensitivity", response_class=HTMLResponse)
def page_sensitivity(request: Request):
    """8. Sensitivity: Standing losses (1-30%), RTE, aux load, and heat demand sweeps."""
    ctx = get_common_context(request, "sensitivity")
    tz = ZoneInfo("Europe/Vilnius")
    s_date = date(2026, 10, 1)
    e_date = date(2026, 10, 2)
    s_utc, _ = local_day_bounds_utc(s_date, tz)
    _, e_utc = local_day_bounds_utc(e_date - timedelta(days=1), tz)

    intervals = build_intervals(s_utc, e_utc, resolution_minutes=15)
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

    analyzer = SensitivityAnalyzer(tz=tz)
    loss_cases = analyzer.sweep_standing_losses(config, heat_profile, site_profile, prices)
    eff_cases = analyzer.sweep_efficiencies(config, heat_profile, site_profile, prices)
    aux_cases = analyzer.sweep_auxiliary_power(config, heat_profile, site_profile, prices)
    hd_cases = analyzer.sweep_heat_demands(config, site_profile, prices)

    ctx.update({
        "standing_losses": [
            {
                "scenario_value": c.parameter_value,
                "scenario_name": c.scenario_name,
                "total_cost_eur": c.total_cost_eur,
                "cost_eur_per_mwh_heat": c.cost_eur_per_mwh_heat,
                "savings_vs_direct_eur": c.savings_vs_direct_eur,
                "savings_vs_direct_percent": c.savings_vs_direct_percent,
                "thermal_standing_loss_kwh": c.standing_losses_kwh,
                "average_soc_kwh": 7.5,
                "average_soc_percent": 50.0,
                "heat_supply_reliability_percent": c.heat_supply_reliability_percent,
            }
            for c in loss_cases
        ],
        "conversion_efficiencies": [
            {
                "scenario_name": c.scenario_name,
                "cost_eur_per_mwh_heat": c.cost_eur_per_mwh_heat,
                "savings_vs_direct_percent": c.savings_vs_direct_percent,
                "conversion_loss_kwh": 6.0,
            }
            for c in eff_cases
        ],
        "aux_loads": [
            {
                "scenario_value": c.parameter_value,
                "cost_eur_per_mwh_heat": c.cost_eur_per_mwh_heat,
                "savings_vs_direct_percent": c.savings_vs_direct_percent,
                "tes_auxiliary_kwh": c.total_electricity_kwh,
            }
            for c in aux_cases
        ],
        "heat_demands": [
            {
                "scenario_name": c.scenario_name,
                "total_cost_eur": c.total_cost_eur,
                "cost_eur_per_mwh_heat": c.cost_eur_per_mwh_heat,
                "heat_supply_reliability_percent": c.heat_supply_reliability_percent,
            }
            for c in hd_cases
        ],
    })
    return templates.TemplateResponse(request=request, name="sensitivity.html", context=ctx)


@dashboard_router.get("/assumptions", response_class=HTMLResponse)
def page_assumptions(request: Request, db: Session = Depends(get_db)):
    """9. Engineering Assumptions & Data Quality Panel."""
    ctx = get_common_context(request, "assumptions")
    scenarios = list_scenarios(db)
    ctx.update({
        "scenarios_count": len(scenarios),
        "live_entsoe_authenticated": False,
        "physical_gen0_calibrated": False,
    })
    return templates.TemplateResponse(request=request, name="assumptions.html", context=ctx)


@dashboard_router.get("/scenarios", response_class=HTMLResponse)
def page_scenarios(request: Request, db: Session = Depends(get_db)):
    """10. Saved Scenarios & Side-by-Side Comparison."""
    seed_default_scenarios_if_empty(db)
    ctx = get_common_context(request, "scenarios")
    scenarios = list_scenarios(db)
    ctx.update({
        "scenarios": [
            {
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "version": sc.version,
                "created_at": sc.created_at.strftime("%Y-%m-%d %H:%M"),
                "description": sc.description,
                "summary": sc.summary,
            }
            for sc in scenarios
        ]
    })
    return templates.TemplateResponse(request=request, name="scenarios.html", context=ctx)


@dashboard_router.get("/dryer", response_class=HTMLResponse)
def page_dryer(request: Request, db: Session = Depends(get_db)):
    """Dedicated Virtual Dryer V1 Equipment Page."""
    ctx = get_common_context(request, "dryer")
    from app.services.shadow_runtime import ShadowRuntimeService
    shadow_svc = ShadowRuntimeService()
    ctx["shadow"] = shadow_svc.get_live_dashboard_state(db)
    ctx["dryer_state"] = shadow_svc.get_dryer_status()
    ctx["dryer_config"] = shadow_svc.dryer_model.config
    ctx["runtime_status"] = get_runtime_status_context(db)
    return templates.TemplateResponse(request=request, name="dryer.html", context=ctx)


@dashboard_router.get("/bess", response_class=HTMLResponse)
def page_bess(request: Request, db: Session = Depends(get_db)):
    """BESS (Battery Energy Storage System) Equipment Page - Proposed Specification."""
    ctx = get_common_context(request, "bess")
    from app.services.shadow_runtime import ShadowRuntimeService
    shadow_svc = ShadowRuntimeService()
    ctx["shadow"] = shadow_svc.get_live_dashboard_state(db)
    ctx["runtime_status"] = get_runtime_status_context(db)
    return templates.TemplateResponse(request=request, name="bess.html", context=ctx)


@dashboard_router.get("/history", response_class=HTMLResponse)
def page_history(
    request: Request,
    range: str = "today",
    tz_view: str = "Europe/Vilnius",
    db: Session = Depends(get_db)
):
    """15-Minute Settlement Energy Ledger & Savings History."""
    ctx = get_common_context(request, "history")
    from app.services.energy_ledger_service import EnergyLedgerService
    ledger_svc = EnergyLedgerService()
    ledger_data = ledger_svc.get_ledger_history(db=db, period=range)
    ctx.update({
        "range": range,
        "tz_view": tz_view,
        "ledger_summary": ledger_data.get("summary", {}),
        "ledger_intervals": ledger_data.get("intervals", []),
        "total_intervals": ledger_data.get("total_intervals", 0),
    })
    return templates.TemplateResponse(request=request, name="history.html", context=ctx)
