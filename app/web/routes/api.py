"""JSON API routes for engineering calculators and exploration (Phase 5.7).

Follows Core UI Principle:
No physical formulas in JavaScript. All engineering calculations are executed
by backend Python domain models (ThermalStateMapper, HelicalAirHXModel,
VesselGeometryCalculator, LP Optimizer, BacktestRunner, SensitivityAnalyzer).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth.dependencies import UserSession, get_current_user, require_admin
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
from app.core.timegrid import build_intervals, local_day_bounds_utc
from app.database.session import init_db, make_engine, make_session_factory
from app.economics.tariff import effective_price_eur_mwh
from app.optimization.domain import OptimizationIntervalInput, OptimizationProblemInput
from app.optimization.lp_optimizer import LPOptimizer
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.services.scenario_service import (
    compare_scenarios,
    get_scenario_by_id,
    list_scenarios,
    save_scenario,
    seed_default_scenarios_if_empty,
)
from app.tes.model import PiecewiseLinearDischargeLimit, get_gen0_discharge_curve
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper
from app.tes.vessel import (
    VesselCalculationResult,
    VesselGeometryCalculator,
    VesselGeometryParameters,
)
from app.web.schemas import (
    HXCalculateRequest,
    HXCalculateResponse,
    ScenarioCompareRequest,
    ScenarioSaveRequest,
    ThermalStateRequest,
    ThermalStateResponse,
    VesselCalculateRequest,
)

api_router = APIRouter(prefix="/api", tags=["Engineering API"])
templates = Jinja2Templates(directory="app/web/templates")

# Database dependency
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


@api_router.post("/calculate/thermal-state", response_model=ThermalStateResponse)
def calculate_thermal_state(req: ThermalStateRequest) -> ThermalStateResponse:
    """Calculate thermal state metrics and curve coordinates using ThermalStateMapper."""
    mapper = ThermalStateMapper(
        t_min_c=req.physical_temperature_min_c,
        t_max_c=req.physical_temperature_max_c,
    )

    sand_mass_kg = mapper.equivalent_sand_mass_for_capacity(req.thermal_capacity_full_span_kwh)
    specific_kwh_kg = mapper.specific_stored_energy_kwh_per_kg()
    usable_span = req.optimizer_soc_max_fraction - req.optimizer_soc_min_fraction
    dispatchable_kwh = req.thermal_capacity_full_span_kwh * usable_span
    soc_min_kwh = req.thermal_capacity_full_span_kwh * req.optimizer_soc_min_fraction
    soc_max_kwh = req.thermal_capacity_full_span_kwh * req.optimizer_soc_max_fraction
    t_min_soc = mapper.temperature_from_soc_fraction(req.optimizer_soc_min_fraction)

    # Current evaluation point
    soc = max(0.0, min(1.0, req.soc_fraction))
    stored_kwh = req.thermal_capacity_full_span_kwh * soc
    sand_temp_c = mapper.temperature_from_soc_fraction(soc)
    remaining_disp_kwh = max(0.0, stored_kwh - soc_min_kwh)
    rel_h = mapper.relative_enthalpy_from_soc_fraction(soc)

    # Reference HX model output at this state
    hx = HelicalAirHXModel(
        hx_area_m2=1.55,
        overall_u_w_m2k=9.0,
        airflow_m3_h=80.0,
        air_inlet_temperature_c=40.0,
        thermal_state_mapper=mapper,
    )
    hx_pmax = hx.p_max_at_temperature_kw(sand_temp_c)

    # 1. SOC vs Temperature curve (0% to 100% in 2% steps)
    soc_temp_curve: list[dict[str, float]] = []
    for step in range(0, 101, 2):
        s_frac = step / 100.0
        t_c = mapper.temperature_from_soc_fraction(s_frac)
        soc_temp_curve.append({
            "soc_percent": round(s_frac * 100.0, 1),
            "temperature_c": round(t_c, 2),
            "stored_energy_kwh": round(req.thermal_capacity_full_span_kwh * s_frac, 2),
        })

    # 2. Temperature vs Energy curve (from T_min to T_max in 2 °C steps)
    temp_energy_curve: list[dict[str, float]] = []
    t_curr = req.physical_temperature_min_c
    while t_curr <= req.physical_temperature_max_c + 0.1:
        s_frac = mapper.soc_fraction_from_temperature(t_curr)
        e_kwh = req.thermal_capacity_full_span_kwh * s_frac
        temp_energy_curve.append({
            "temperature_c": round(t_curr, 1),
            "stored_energy_kwh": round(e_kwh, 3),
            "is_reserve": 1.0 if e_kwh <= soc_min_kwh else 0.0,
        })
        t_curr += 2.0

    return ThermalStateResponse(
        physical_temperature_min_c=req.physical_temperature_min_c,
        physical_temperature_max_c=req.physical_temperature_max_c,
        thermal_capacity_full_span_kwh=req.thermal_capacity_full_span_kwh,
        optimizer_soc_min_fraction=req.optimizer_soc_min_fraction,
        optimizer_soc_max_fraction=req.optimizer_soc_max_fraction,
        dispatchable_capacity_kwh=round(dispatchable_kwh, 2),
        soc_min_energy_kwh=round(soc_min_kwh, 2),
        soc_max_energy_kwh=round(soc_max_kwh, 2),
        sand_mass_kg=round(sand_mass_kg, 1),
        specific_storage_kwh_per_kg=round(specific_kwh_kg, 5),
        temperature_at_optimizer_min_soc_c=round(t_min_soc, 2),
        current_soc_fraction=round(soc, 4),
        current_stored_energy_kwh=round(stored_kwh, 2),
        current_sand_temperature_c=round(sand_temp_c, 2),
        current_remaining_dispatchable_kwh=round(remaining_disp_kwh, 2),
        current_relative_enthalpy_j_per_kg=round(rel_h, 1),
        current_hx_pmax_kw=round(hx_pmax, 2),
        soc_temperature_curve=soc_temp_curve,
        temperature_energy_curve=temp_energy_curve,
    )


@api_router.post("/calculate/hx", response_model=HXCalculateResponse)
def calculate_hx(req: HXCalculateRequest) -> HXCalculateResponse:
    """Calculate heat exchanger thermal performance using HelicalAirHXModel."""
    mapper = ThermalStateMapper()
    hx = HelicalAirHXModel(
        hx_area_m2=req.hx_area_m2,
        overall_u_w_m2k=req.overall_u_w_m2k,
        airflow_m3_h=req.airflow_m3_h,
        air_inlet_temperature_c=req.air_inlet_temperature_c,
        air_pressure_pa=req.air_pressure_pa,
        thermal_state_mapper=mapper,
    )

    if req.sand_temperature_c is not None:
        sand_temp_c = req.sand_temperature_c
        soc_fraction = mapper.soc_fraction_from_temperature(sand_temp_c)
    else:
        soc_fraction = req.soc_fraction
        sand_temp_c = mapper.temperature_from_soc_fraction(soc_fraction)

    p_max = hx.p_max_at_temperature_kw(sand_temp_c)
    t_out = hx.calculate_outlet_temperature_c(sand_temp_c)
    can_satisfy = p_max >= req.process_demand_kw

    # Curves for U = 6, 9, 12 W/m²K at selected airflow
    hx_6 = HelicalAirHXModel(hx_area_m2=req.hx_area_m2, overall_u_w_m2k=6.0, airflow_m3_h=req.airflow_m3_h, air_inlet_temperature_c=req.air_inlet_temperature_c, thermal_state_mapper=mapper)
    hx_9 = HelicalAirHXModel(hx_area_m2=req.hx_area_m2, overall_u_w_m2k=9.0, airflow_m3_h=req.airflow_m3_h, air_inlet_temperature_c=req.air_inlet_temperature_c, thermal_state_mapper=mapper)
    hx_12 = HelicalAirHXModel(hx_area_m2=req.hx_area_m2, overall_u_w_m2k=12.0, airflow_m3_h=req.airflow_m3_h, air_inlet_temperature_c=req.air_inlet_temperature_c, thermal_state_mapper=mapper)

    power_curve: list[dict[str, Any]] = []
    temp_curve: list[dict[str, Any]] = []

    for step in range(0, 101, 5):
        s = step / 100.0
        t_sand = mapper.temperature_from_soc_fraction(s)
        p6 = hx_6.p_max_at_temperature_kw(t_sand)
        p9 = hx_9.p_max_at_temperature_kw(t_sand)
        p12 = hx_12.p_max_at_temperature_kw(t_sand)
        tout_selected = hx.calculate_outlet_temperature_c(t_sand)

        power_curve.append({
            "soc_percent": step,
            "sand_temperature_c": round(t_sand, 1),
            "p_u6_kw": round(p6, 2),
            "p_u9_kw": round(p9, 2),
            "p_u12_kw": round(p12, 2),
            "process_demand_kw": req.process_demand_kw,
        })

        temp_curve.append({
            "soc_percent": step,
            "sand_temperature_c": round(t_sand, 1),
            "outlet_temp_c": round(tout_selected, 1),
            "target_dryer_temp_c": 70.0,
        })

    return HXCalculateResponse(
        soc_fraction=round(soc_fraction, 4),
        sand_temperature_c=round(sand_temp_c, 2),
        overall_u_w_m2k=req.overall_u_w_m2k,
        hx_area_m2=req.hx_area_m2,
        airflow_m3_h=req.airflow_m3_h,
        air_inlet_temperature_c=req.air_inlet_temperature_c,
        ua_w_k=round(hx.ua_w_k, 2),
        air_mass_flow_kg_s=round(hx.mass_flow_kg_s, 4),
        air_heat_capacity_rate_w_k=round(hx.c_air_w_k, 2),
        ntu=round(hx.ntu, 3),
        effectiveness=round(hx.effectiveness, 4),
        p_max_kw=round(p_max, 2),
        predicted_outlet_air_temperature_c=round(t_out, 2),
        process_demand_kw=req.process_demand_kw,
        can_satisfy_process_demand=can_satisfy,
        hx_power_curve=power_curve,
        hx_temp_curve=temp_curve,
    )


@api_router.post("/calculate/vessel")
def calculate_vessel(req: VesselCalculateRequest) -> dict[str, Any]:
    """Calculate vessel containment metrics, sand fill, and freeboard."""
    vessel_params = VesselGeometryParameters(
        internal_diameter_mm=req.internal_diameter_mm,
        straight_shell_height_mm=req.straight_shell_height_mm,
        sand_bulk_density_kg_m3=req.sand_bulk_density_kg_m3,
        hx_displaced_volume_l=req.hx_displaced_volume_l,
        internals_displaced_volume_l=req.internals_displaced_volume_l,
        min_freeboard_mm=req.min_freeboard_mm,
    )
    calc = VesselGeometryCalculator(params=vessel_params)

    if req.mode == "capacity_driven":
        result: VesselCalculationResult = calc.calculate_capacity_driven(req.target_full_span_kwh)
    else:
        # Geometry-driven
        if req.sand_mass_kg is not None and req.sand_mass_kg > 0:
            result = calc.calculate_for_sand_mass(req.sand_mass_kg)
        elif req.fill_fraction is not None and req.fill_fraction > 0:
            result = calc.calculate_geometry_driven(req.fill_fraction)
        else:
            result = calc.calculate_for_sand_mass(275.7)

    ref_scenarios = calc.get_reference_scenarios()
    ref_dict = {k: v.to_dict() for k, v in ref_scenarios.items()}

    return {
        "result": result.to_dict(),
        "reference_scenarios": ref_dict,
    }


@api_router.post("/scenarios")
def api_save_scenario(
    req: ScenarioSaveRequest,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Save an engineering scenario (append-only; increments version on duplicate name). Requires ADMIN role."""
    sc = save_scenario(
        session=db,
        name=req.name,
        parameters=req.parameters,
        description=req.description,
    )
    return {
        "status": "saved",
        "scenario_id": sc.scenario_id,
        "name": sc.name,
        "version": sc.version,
        "created_at": sc.created_at.isoformat(),
        "summary": sc.summary,
    }


@api_router.get("/scenarios")
def api_list_scenarios(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List all saved scenarios with summary parameters."""
    scenarios = list_scenarios(db)
    return [
        {
            "scenario_id": sc.scenario_id,
            "name": sc.name,
            "version": sc.version,
            "created_at": sc.created_at.isoformat(),
            "description": sc.description,
            "summary": sc.summary,
            "parameters": sc.parameters,
        }
        for sc in scenarios
    ]


@api_router.post("/scenarios/compare")
def api_compare_scenarios(req: ScenarioCompareRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Compare two saved scenarios side-by-side."""
    sc_1 = get_scenario_by_id(db, req.scenario_id_1)
    sc_2 = get_scenario_by_id(db, req.scenario_id_2)
    if not sc_1 or not sc_2:
        raise HTTPException(status_code=404, detail="One or both scenarios not found")

    return compare_scenarios(sc_1, sc_2)


@api_router.get("/market/status")
def get_market_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return status of Lithuanian market data provider hierarchy and cross-source validation."""
    from app.services.market_data_service import MarketDataService

    service = MarketDataService()
    try:
        fetch_res = service.fetch_market_data(session=db)
        snap = fetch_res.status_snapshot
        return {
            "primary_source": snap.primary_source,
            "primary_health": snap.primary_health,
            "cross_check_source": snap.cross_check_source,
            "cross_check_health": snap.cross_check_health,
            "latest_interval": snap.latest_interval_str,
            "tomorrow_prices": snap.tomorrow_prices,
            "resolution": snap.resolution,
            "validation": snap.validation,
            "data_freshness": snap.data_freshness,
            "active_source": snap.active_source,
            "intervals_count": snap.intervals_count,
        }
    except Exception as exc:
        return {
            "primary_source": "LITGRID",
            "primary_health": "ERROR",
            "cross_check_source": "ELERING",
            "cross_check_health": "ERROR",
            "latest_interval": "None",
            "tomorrow_prices": "WAITING",
            "resolution": "15 min",
            "validation": "NOT CHECKED",
            "data_freshness": "LIVE MARKET DATA UNAVAILABLE",
            "active_source": "NONE",
            "error": str(exc),
        }


@api_router.get("/market/current")
def get_market_current(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return JSON model of current market interval, next interval, status, today/tomorrow stats, and TES action."""
    from app.services.market_data_service import MarketDataService

    service = MarketDataService()
    ctx = service.get_market_view_context(session=db)
    return {
        "current_time_local": ctx["current_time_local_str"],
        "current_time_utc": ctx["current_time_utc_str"],
        "current_interval": ctx["interval_str"],
        "price_eur_mwh": ctx["price_eur_mwh"],
        "price_eur_mwh_str": ctx["price_eur_mwh_str"],
        "source": ctx["source"],
        "active_source": ctx["active_source"],
        "primary_source": ctx["primary_source"],
        "data_status": ctx["data_status"],
        "last_fetched": ctx["last_fetched_str"],
        "age_minutes": ctx["age_minutes"],
        "resolution": ctx["resolution"],
        "original_resolution": ctx["original_resolution"],
        "is_derived": ctx["is_derived"],
        "fallback_used": ctx["fallback_used"],
        "fallback_reason": ctx["fallback_reason"],
        "next_interval": ctx["next_interval_str"],
        "next_price_eur_mwh": ctx["next_price_eur_mwh"],
        "next_price_eur_mwh_str": ctx["next_price_eur_mwh_str"],
        "price_diff_eur_mwh": ctx["price_diff_eur_mwh"],
        "price_diff_eur_mwh_str": ctx["price_diff_eur_mwh_str"],
        "effective_price_eur_mwh": ctx["effective_price_eur_mwh"],
        "effective_price_eur_mwh_str": ctx["effective_price_eur_mwh_str"],
        "total_adders_eur_mwh": ctx["total_adders_eur_mwh"],
        "today_stats": {
            "intervals_count": ctx["today_stats"]["intervals_count"],
            "min_price_eur_mwh": ctx["today_stats"]["min_price_eur_mwh"],
            "min_price_time_str": ctx["today_stats"]["min_price_time_str"],
            "max_price_eur_mwh": ctx["today_stats"]["max_price_eur_mwh"],
            "max_price_time_str": ctx["today_stats"]["max_price_time_str"],
            "avg_price_eur_mwh": ctx["today_stats"]["avg_price_eur_mwh"],
        },
        "tomorrow_stats": {
            "status": ctx["tomorrow_stats"]["status"],
            "available": ctx["tomorrow_stats"]["available"],
            "intervals_count": ctx["tomorrow_stats"]["intervals_count"],
            "min_price_eur_mwh": ctx["tomorrow_stats"]["min_price_eur_mwh"],
            "max_price_eur_mwh": ctx["tomorrow_stats"]["max_price_eur_mwh"],
            "avg_price_eur_mwh": ctx["tomorrow_stats"]["avg_price_eur_mwh"],
        },
        "tes_action": ctx["tes_action"],
    }


@api_router.get("/market/current-card", response_class=HTMLResponse)
def get_market_current_card(request: Request, db: Session = Depends(get_db)):
    """Render HTML fragment of the current-market and TES action cards for HTMX 60-second polling (Phase 5.8.1)."""
    from app.services.market_data_service import MarketDataService

    service = MarketDataService()
    ctx = service.get_market_view_context(session=db)
    return templates.TemplateResponse(
        request=request,
        name="components/market_current_card.html",
        context={"request": request, "market": ctx},
    )


@api_router.get("/market/today-chart")
def get_market_today_chart(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return all unaggregated 15-minute intervals for today for Chart.js rendering (Phase 5.8.1)."""
    from app.services.market_data_service import MarketDataService

    service = MarketDataService()
    ctx = service.get_market_view_context(session=db)
    return {
        "today_stats": ctx["today_stats"],
        "current_interval": ctx["interval_str"],
        "source": ctx["active_source"],
    }


@api_router.get("/market/shadow-dispatch")
def get_market_shadow_dispatch(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Execute receding-horizon LP optimizer on real Lithuanian market data (Shadow Operation)."""
    from app.services.market_data_service import LiveMarketDataUnavailableError, MarketDataService

    service = MarketDataService()
    try:
        shadow_res = service.run_shadow_operation(session=db)
        snap = shadow_res.market_status
        return {
            "status": shadow_res.optimization_result.status,
            "status_label": shadow_res.status_label,
            "active_source": shadow_res.active_source,
            "market_status": {
                "primary_source": snap.primary_source,
                "primary_health": snap.primary_health,
                "cross_check_source": snap.cross_check_source,
                "cross_check_health": snap.cross_check_health,
                "latest_interval": snap.latest_interval_str,
                "tomorrow_prices": snap.tomorrow_prices,
                "resolution": snap.resolution,
                "validation": snap.validation,
                "data_freshness": snap.data_freshness,
            },
            "objective_eur": round(shadow_res.optimization_result.objective_eur, 2)
            if shadow_res.optimization_result.objective_eur is not None
            else 0.0,
            "total_cost_eur": shadow_res.total_cost_eur,
            "total_charge_kwh": shadow_res.total_charge_kwh,
            "total_heat_kwh": shadow_res.total_heat_kwh,
            "max_grid_power_kw": shadow_res.max_grid_power_kw,
            "chart_data": shadow_res.chart_data,
            "timeline_table": shadow_res.timeline_rows,
        }
    except LiveMarketDataUnavailableError as err:
        raise HTTPException(
            status_code=503,
            detail="LIVE MARKET DATA UNAVAILABLE: All live market data providers failed (no mock fallback allowed)",
        ) from err
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Shadow operation failed: {exc}") from exc


@api_router.get("/dispatch-demo")
def get_dispatch_demo(source: str = "mock", db: Session = Depends(get_db)) -> dict[str, Any]:
    """Generate market & dispatch dataset.

    - source="shadow" or "live": Runs real Lithuanian shadow operation.
    - source="mock": Runs deterministic benchmark day (Phase 5.7 regression parity).
    """
    if source.lower() in ("shadow", "live"):
        return get_market_shadow_dispatch(db=db)

    # Deterministic reference benchmark day
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
    site_params = SiteParameters(
        grid_connection_limit_kw=12.0,
        process_heat_demand_kw=1.5,
        other_loads_kw=2.0,
    )
    tariff_params = TariffParameters(
        supplier_markup_eur_mwh=1.5,
        variable_grid_fee_eur_mwh=25.0,
        variable_tax_eur_mwh=5.0,
    )
    curve = get_gen0_discharge_curve("fixed_gen0_hx", capacity_kwh=15.0, overall_u_w_m2k=9.0, airflow_m3_h=80.0)

    heat_profile = ConstantHeatDemand(value_kw=1.5)
    site_profile = ConstantSiteLoad(value_kw=2.0)
    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    opt_inputs: list[OptimizationIntervalInput] = []
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
    opt = LPOptimizer()
    res = opt.optimize(problem)

    mapper = ThermalStateMapper()
    timeline_rows = []
    chart_series = []

    for item in res.intervals:
        dt_local = item.start_utc.astimezone(tz)
        local_time_str = dt_local.strftime("%H:%M")
        sand_temp = mapper.temperature_from_soc_fraction(item.soc_percent / 100.0)
        p_max_hx = curve.max_discharge_power_kw(item.soc_kwh, tes_params)
        p_grid_total = item.charge_power_kw + item.other_site_load_kw + item.auxiliary_load_kw

        row = {
            "local_time": local_time_str,
            "spot_price_eur_mwh": round(item.spot_price_eur_mwh, 2),
            "effective_price_eur_mwh": round(item.effective_price_eur_mwh, 2),
            "requested_charge_kw": round(item.charge_power_kw, 2),
            "actual_charge_kw": round(item.charge_power_kw, 2),
            "process_demand_kw": round(item.heat_demand_kw, 2),
            "hx_max_available_kw": round(p_max_hx, 2),
            "actual_discharge_kw": round(item.discharge_power_kw, 2),
            "unmet_heat_kw": round(item.unmet_heat_kw, 2),
            "soc_kwh": round(item.soc_kwh, 2),
            "soc_percent": round(item.soc_percent, 1),
            "sand_temperature_c": round(sand_temp, 1),
            "grid_total_kw": round(p_grid_total, 2),
            "interval_cost_eur": round(item.cost_eur, 3),
        }
        timeline_rows.append(row)

        chart_series.append({
            "time": local_time_str,
            "spot_price": round(item.spot_price_eur_mwh, 2),
            "charge_power": round(item.charge_power_kw, 2),
            "soc_percent": round(item.soc_percent, 1),
            "sand_temperature_c": round(sand_temp, 1),
            "total_grid_power": round(p_grid_total, 2),
            "site_load": round(item.other_site_load_kw, 2),
            "aux_load": round(item.auxiliary_load_kw, 3),
            "grid_limit": 12.0,
            "configured_charge_limit": 9.0,
        })

    return {
        "status": res.status,
        "status_label": "REFERENCE BENCHMARK (MOCK PRICES)",
        "active_source": "MOCK",
        "objective_eur": round(res.objective_eur, 2) if res.objective_eur is not None else 0.0,
        "chart_data": chart_series,
        "timeline_table": timeline_rows,
    }


@api_router.get("/sizing-demo")
def get_sizing_demo() -> dict[str, Any]:
    """Execute the systematic sizing matrix and return formatted chart/table data."""
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
    return reports["fixed_gen0_hx"].to_dict()


@api_router.get("/sensitivity-demo")
def get_sensitivity_demo() -> dict[str, Any]:
    """Execute standing loss and RTE sensitivity sweeps and return data."""
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
    return {
        "standing_losses": [
            {
                "parameter_value": c.parameter_value,
                "cost_eur_per_mwh_heat": c.cost_eur_per_mwh_heat,
                "savings_vs_direct_percent": c.savings_vs_direct_percent,
                "standing_losses_kwh": c.standing_losses_kwh,
            }
            for c in loss_cases
        ]
    }


# --------------------------------------------------------------------------- Live Virtual TES Shadow Runtime (Phase 5.8.2)
from app.services.shadow_runtime import ShadowRuntimeService
from app.web.schemas import (
    ShadowProcessDemandRequest,
    ShadowResetRequest,
    ShadowStartRequest,
)

_shadow_service = ShadowRuntimeService()


@api_router.post("/shadow/start")
async def shadow_start(
    request: Request,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Start or reinitialize the live Virtual TES shadow runtime. Requires ADMIN role."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            req = ShadowStartRequest(**body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request JSON: {exc}")
    else:
        # Form submission (e.g. from HTMX hx-post or standard HTML form)
        form_data = await request.form()
        data = dict(form_data)
        mode = data.get("initialization_mode", "soc")

        def _get_float(name: str) -> float | None:
            raw = data.get(name)
            if raw is None or raw == "":
                return None
            try:
                return float(raw)
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid numeric value for {name}: {raw}")

        demand = _get_float("process_demand_kw")
        if demand is None:
            demand = 1.5
        enabled = str(data.get("process_enabled", "true")).lower() in ("true", "1", "on", "yes")

        req = ShadowStartRequest(
            initialization_mode=mode,
            initial_soc_percent=_get_float("initial_soc_percent"),
            initial_energy_kwh=_get_float("initial_energy_kwh"),
            initial_temp_c=_get_float("initial_temp_c"),
            process_demand_kw=demand,
            process_enabled=enabled,
        )

    try:
        sess = _shadow_service.start_session(
            db=db,
            initial_soc_percent=req.initial_soc_percent,
            initial_energy_kwh=req.initial_energy_kwh,
            initial_temp_c=req.initial_temp_c,
            initialization_mode=req.initialization_mode,
            process_demand_kw=req.process_demand_kw,
            process_enabled=req.process_enabled,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    if "HX-Request" in request.headers:
        state = _shadow_service.get_live_dashboard_state(db)
        resp = templates.TemplateResponse(
            request=request,
            name="components/shadow_runtime_card.html",
            context={"request": request, "shadow": state, "user": _admin},
        )
        resp.headers["HX-Trigger"] = "shadowStarted"
        return resp

    return {
        "status": "ok",
        "session_id": sess.id,
        "state": sess.status,
        "initial_soc_fraction": sess.initial_soc_fraction,
        "initial_stored_energy_kwh": sess.initial_stored_energy_kwh,
        "initial_sand_temperature_c": sess.initial_sand_temperature_c,
    }


@api_router.post("/shadow/pause")
def shadow_pause(
    request: Request,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Pause live execution. Requires ADMIN role."""
    sess = _shadow_service.pause_session(db=db)
    if "HX-Request" in request.headers:
        state = _shadow_service.get_live_dashboard_state(db)
        return templates.TemplateResponse(
            request=request,
            name="components/shadow_runtime_card.html",
            context={"request": request, "shadow": state, "user": _admin},
        )
    return {"status": "ok", "state": sess.status}


@api_router.post("/shadow/resume")
def shadow_resume(
    request: Request,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Resume paused live execution. Requires ADMIN role."""
    sess = _shadow_service.resume_session(db=db)
    if "HX-Request" in request.headers:
        state = _shadow_service.get_live_dashboard_state(db)
        return templates.TemplateResponse(
            request=request,
            name="components/shadow_runtime_card.html",
            context={"request": request, "shadow": state, "user": _admin},
        )
    return {"status": "ok", "state": sess.status}


@api_router.post("/shadow/stop")
def shadow_stop(
    request: Request,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Stop live shadow execution. Requires ADMIN role."""
    sess = _shadow_service.stop_session(db=db)
    if "HX-Request" in request.headers:
        state = _shadow_service.get_live_dashboard_state(db)
        return templates.TemplateResponse(
            request=request,
            name="components/shadow_runtime_card.html",
            context={"request": request, "shadow": state, "user": _admin},
        )
    return {"status": "ok", "state": sess.status}


@api_router.post("/shadow/reset")
async def shadow_reset(
    request: Request,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Reset physical virtual state with an immutable audit event. Requires ADMIN role."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        req = ShadowResetRequest(**body)
    else:
        form = await request.form()
        target_soc = float(form.get("target_soc_percent", 50.0))
        reason = str(form.get("reason", "MANUAL_RESET"))
        req = ShadowResetRequest(target_soc_percent=target_soc, reason=reason)

    sess = _shadow_service.reset_state(db=db, target_soc_percent=req.target_soc_percent, reason=req.reason)
    if "HX-Request" in request.headers:
        state = _shadow_service.get_live_dashboard_state(db)
        return templates.TemplateResponse(
            request=request,
            name="components/shadow_runtime_card.html",
            context={"request": request, "shadow": state, "user": _admin},
        )
    return {"status": "ok", "state": sess.status, "current_soc_percent": sess.current_soc_fraction * 100.0}


@api_router.post("/shadow/process-demand")
def shadow_process_demand(
    req: ShadowProcessDemandRequest,
    request: Request,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Adjust virtual process heat demand without modifying market data. Requires ADMIN role."""
    sess = _shadow_service.update_process_demand(db=db, demand_kw=req.process_demand_kw, enabled=req.process_enabled)
    if "HX-Request" in request.headers:
        state = _shadow_service.get_live_dashboard_state(db)
        return templates.TemplateResponse(
            request=request,
            name="components/shadow_runtime_card.html",
            context={"request": request, "shadow": state, "user": _admin},
        )
    return {
        "status": "ok",
        "process_heat_demand_kw": sess.process_heat_demand_kw,
        "process_enabled": sess.process_enabled,
    }


@api_router.post("/shadow/reoptimize")
def shadow_reoptimize(
    request: Request,
    _admin: UserSession = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Manually trigger receding-horizon optimization against current market prices. Requires ADMIN role."""
    sess = _shadow_service.get_or_create_session(db)
    _shadow_service.reoptimize(db, sess, trigger_reason="MANUAL_REOPTIMIZE")
    if "HX-Request" in request.headers:
        state = _shadow_service.get_live_dashboard_state(db)
        return templates.TemplateResponse(
            request=request,
            name="components/shadow_runtime_card.html",
            context={"request": request, "shadow": state, "user": _admin},
        )
    return {"status": "ok", "active_schedule_version": sess.active_schedule_version}


@api_router.get("/shadow/status")
def shadow_status(db: Session = Depends(get_db)):
    """Return JSON live dashboard state of the shadow runtime."""
    return _shadow_service.get_live_dashboard_state(db)


@api_router.get("/shadow/live-card", response_class=HTMLResponse)
def shadow_live_card(
    request: Request,
    user: UserSession | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Render HTML partial for HTMX 5-second auto-refresh."""
    state = _shadow_service.get_live_dashboard_state(db)
    return templates.TemplateResponse(
        request=request,
        name="components/shadow_runtime_card.html",
        context={"request": request, "shadow": state, "user": user},
    )



@api_router.get("/shadow/history")
def shadow_history(db: Session = Depends(get_db)):
    """Return executed interval history."""
    from app.database.repositories import ShadowRepository
    sess = _shadow_service.get_or_create_session(db)
    repo = ShadowRepository(db)
    records = repo.get_intervals(sess.id, limit=200, ascending=True)
    return [
        {
            "start_utc": r.interval_start_utc.isoformat(),
            "end_utc": r.interval_end_utc.isoformat(),
            "spot_price_eur_mwh": r.spot_price_eur_mwh,
            "effective_price_eur_mwh": r.effective_price_eur_mwh,
            "action_type": r.action_type,
            "charge_power_kw": r.actual_charge_kw,
            "discharge_power_kw": r.actual_discharge_kw,
            "soc_end_percent": round(r.soc_end_fraction * 100.0, 1),
            "sand_temperature_c": round(r.sand_temp_end_c, 1),
            "residual_kwh": r.energy_balance_residual_kwh,
            "reason_code": r.reason_code,
        }
        for r in records
    ]

