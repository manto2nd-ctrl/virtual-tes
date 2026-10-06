"""Service for persisting, retrieving, and comparing named engineering scenarios (Phase 5.7).

Scenarios snapshot:
- Physical storage parameters (temperature span, full-span capacity, SOC window, sand mass)
- Vessel geometry (diameter, straight height, bulk density, internal displacements)
- Heat exchanger parameters (area, U, airflow, inlet temp)
- Electrical limits (heaters, grid connection limit, site load, aux power)
- Process thermal demand
- Efficiencies and loss assumptions

Persistence is append-only in SQLite; updating an existing scenario name increments the version.
"""

from __future__ import annotations

import uuid
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import EngineeringScenario
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper
from app.tes.vessel import VesselGeometryCalculator, VesselGeometryParameters


def get_default_scenario_definitions() -> list[dict[str, Any]]:
    """Reference presets for initial seed scenarios."""
    return [
        {
            "name": "Gen0_15kWh_reference",
            "description": "Gen0 reference: 15 kWh full-span, 275.7 kg quartz sand, 9.0 W/m²K HX, 9 kW charging.",
            "parameters": {
                "physical_temperature_min_c": 80.0,
                "physical_temperature_max_c": 300.0,
                "thermal_capacity_full_span_kwh": 15.0,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
                "sand_mass_kg": 275.7,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
                "sand_bulk_density_kg_m3": 1600.0,
                "hx_displaced_volume_l": 24.0,
                "internals_displaced_volume_l": 8.0,
                "hx_area_m2": 1.55,
                "overall_u_w_m2k": 9.0,
                "airflow_m3_h": 80.0,
                "air_inlet_temperature_c": 40.0,
                "heater_power_kw": 9.0,
                "grid_connection_limit_kw": 12.0,
                "other_site_loads_kw": 2.0,
                "auxiliary_power_kw": 0.05,
                "process_heat_demand_kw": 1.5,
                "standing_loss_percent_per_day": 2.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.90,
            },
        },
        {
            "name": "Gen0_300kg",
            "description": "Gen0 300 kg sand fill variant (~16.3 kWh full-span / 14.7 kWh dispatchable).",
            "parameters": {
                "physical_temperature_min_c": 80.0,
                "physical_temperature_max_c": 300.0,
                "thermal_capacity_full_span_kwh": 16.32,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
                "sand_mass_kg": 300.0,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
                "sand_bulk_density_kg_m3": 1600.0,
                "hx_displaced_volume_l": 24.0,
                "internals_displaced_volume_l": 8.0,
                "hx_area_m2": 1.55,
                "overall_u_w_m2k": 9.0,
                "airflow_m3_h": 80.0,
                "air_inlet_temperature_c": 40.0,
                "heater_power_kw": 9.0,
                "grid_connection_limit_kw": 12.0,
                "other_site_loads_kw": 2.0,
                "auxiliary_power_kw": 0.05,
                "process_heat_demand_kw": 1.5,
                "standing_loss_percent_per_day": 2.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.90,
            },
        },
        {
            "name": "Gen0_320kg",
            "description": "Gen0 320 kg sand fill variant (~17.4 kWh full-span / 15.7 kWh dispatchable).",
            "parameters": {
                "physical_temperature_min_c": 80.0,
                "physical_temperature_max_c": 300.0,
                "thermal_capacity_full_span_kwh": 17.41,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
                "sand_mass_kg": 320.0,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
                "sand_bulk_density_kg_m3": 1600.0,
                "hx_displaced_volume_l": 24.0,
                "internals_displaced_volume_l": 8.0,
                "hx_area_m2": 1.55,
                "overall_u_w_m2k": 9.0,
                "airflow_m3_h": 80.0,
                "air_inlet_temperature_c": 40.0,
                "heater_power_kw": 9.0,
                "grid_connection_limit_kw": 12.0,
                "other_site_loads_kw": 2.0,
                "auxiliary_power_kw": 0.05,
                "process_heat_demand_kw": 1.5,
                "standing_loss_percent_per_day": 2.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.90,
            },
        },
        {
            "name": "HX_conservative",
            "description": "Conservative HX estimate: U = 6.0 W/m²K, airflow = 60 m³/h.",
            "parameters": {
                "physical_temperature_min_c": 80.0,
                "physical_temperature_max_c": 300.0,
                "thermal_capacity_full_span_kwh": 15.0,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
                "sand_mass_kg": 275.7,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
                "sand_bulk_density_kg_m3": 1600.0,
                "hx_displaced_volume_l": 24.0,
                "internals_displaced_volume_l": 8.0,
                "hx_area_m2": 1.55,
                "overall_u_w_m2k": 6.0,
                "airflow_m3_h": 60.0,
                "air_inlet_temperature_c": 40.0,
                "heater_power_kw": 9.0,
                "grid_connection_limit_kw": 12.0,
                "other_site_loads_kw": 2.0,
                "auxiliary_power_kw": 0.05,
                "process_heat_demand_kw": 1.5,
                "standing_loss_percent_per_day": 2.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.90,
            },
        },
        {
            "name": "HX_reference",
            "description": "Reference HX baseline: U = 9.0 W/m²K, airflow = 80 m³/h.",
            "parameters": {
                "physical_temperature_min_c": 80.0,
                "physical_temperature_max_c": 300.0,
                "thermal_capacity_full_span_kwh": 15.0,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
                "sand_mass_kg": 275.7,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
                "sand_bulk_density_kg_m3": 1600.0,
                "hx_displaced_volume_l": 24.0,
                "internals_displaced_volume_l": 8.0,
                "hx_area_m2": 1.55,
                "overall_u_w_m2k": 9.0,
                "airflow_m3_h": 80.0,
                "air_inlet_temperature_c": 40.0,
                "heater_power_kw": 9.0,
                "grid_connection_limit_kw": 12.0,
                "other_site_loads_kw": 2.0,
                "auxiliary_power_kw": 0.05,
                "process_heat_demand_kw": 1.5,
                "standing_loss_percent_per_day": 2.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.90,
            },
        },
        {
            "name": "HX_optimistic",
            "description": "Optimistic HX estimate: U = 12.0 W/m²K, airflow = 100 m³/h.",
            "parameters": {
                "physical_temperature_min_c": 80.0,
                "physical_temperature_max_c": 300.0,
                "thermal_capacity_full_span_kwh": 15.0,
                "optimizer_soc_min_fraction": 0.10,
                "optimizer_soc_max_fraction": 1.00,
                "sand_mass_kg": 275.7,
                "vessel_internal_diameter_mm": 600.0,
                "vessel_straight_shell_height_mm": 950.0,
                "sand_bulk_density_kg_m3": 1600.0,
                "hx_displaced_volume_l": 24.0,
                "internals_displaced_volume_l": 8.0,
                "hx_area_m2": 1.55,
                "overall_u_w_m2k": 12.0,
                "airflow_m3_h": 100.0,
                "air_inlet_temperature_c": 40.0,
                "heater_power_kw": 9.0,
                "grid_connection_limit_kw": 12.0,
                "other_site_loads_kw": 2.0,
                "auxiliary_power_kw": 0.05,
                "process_heat_demand_kw": 1.5,
                "standing_loss_percent_per_day": 2.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.90,
            },
        },
    ]


def compute_scenario_summary(params: dict[str, Any]) -> dict[str, Any]:
    """Compute derived physics metrics for a scenario snapshot."""
    t_min = float(params.get("physical_temperature_min_c", 80.0))
    t_max = float(params.get("physical_temperature_max_c", 300.0))
    mapper = ThermalStateMapper(t_min_c=t_min, t_max_c=t_max)

    sand_mass = float(params.get("sand_mass_kg", 275.7))
    full_span_kwh = mapper.capacity_for_sand_mass(sand_mass)
    soc_min = float(params.get("optimizer_soc_min_fraction", 0.10))
    soc_max = float(params.get("optimizer_soc_max_fraction", 1.00))
    dispatchable_kwh = full_span_kwh * (soc_max - soc_min)
    t_min_soc = mapper.temperature_from_soc_fraction(soc_min)

    # Vessel metrics
    vessel_params = VesselGeometryParameters(
        internal_diameter_mm=float(params.get("vessel_internal_diameter_mm", 600.0)),
        straight_shell_height_mm=float(params.get("vessel_straight_shell_height_mm", 950.0)),
        sand_bulk_density_kg_m3=float(params.get("sand_bulk_density_kg_m3", 1600.0)),
        hx_displaced_volume_l=float(params.get("hx_displaced_volume_l", 24.0)),
        internals_displaced_volume_l=float(params.get("internals_displaced_volume_l", 8.0)),
    )
    vessel_calc = VesselGeometryCalculator(
        params=vessel_params,
        thermal_mapper=mapper,
        optimizer_soc_min_fraction=soc_min,
        optimizer_soc_max_fraction=soc_max,
    )
    vessel_res = vessel_calc.calculate_for_sand_mass(sand_mass)

    # HX metrics
    hx = HelicalAirHXModel(
        hx_area_m2=float(params.get("hx_area_m2", 1.55)),
        overall_u_w_m2k=float(params.get("overall_u_w_m2k", 9.0)),
        airflow_m3_h=float(params.get("airflow_m3_h", 80.0)),
        air_inlet_temperature_c=float(params.get("air_inlet_temperature_c", 40.0)),
        thermal_state_mapper=mapper,
    )
    hx_pmax_max_soc = hx.p_max_at_soc_fraction_kw(soc_max)
    hx_pmax_min_soc = hx.p_max_at_soc_fraction_kw(soc_min)
    hx_pmax_50_soc = hx.p_max_at_soc_fraction_kw(0.50)

    # Grid headroom
    grid_limit = float(params.get("grid_connection_limit_kw", 12.0))
    site_load = float(params.get("other_site_loads_kw", 2.0))
    aux = float(params.get("auxiliary_power_kw", 0.05))
    charge_power_config = float(params.get("heater_power_kw", 9.0))
    grid_headroom = max(0.0, grid_limit - site_load - aux)
    actual_peak_charge = min(charge_power_config, grid_headroom)
    peak_grid_power = site_load + aux + actual_peak_charge

    return {
        "sand_mass_kg": round(sand_mass, 1),
        "thermal_capacity_full_span_kwh": round(full_span_kwh, 2),
        "dispatchable_capacity_kwh": round(dispatchable_kwh, 2),
        "temperature_at_optimizer_min_soc_c": round(t_min_soc, 2),
        "gross_vessel_volume_l": round(vessel_res.gross_volume_l, 1),
        "net_vessel_volume_l": round(vessel_res.net_available_volume_l, 1),
        "sand_volume_l": round(vessel_res.sand_volume_l, 1),
        "sand_fill_height_mm": round(vessel_res.sand_fill_height_mm, 1),
        "freeboard_height_mm": round(vessel_res.freeboard_height_mm, 1),
        "freeboard_percent": round(vessel_res.freeboard_percent, 1),
        "hx_pmax_at_100_soc_kw": round(hx_pmax_max_soc, 2),
        "hx_pmax_at_50_soc_kw": round(hx_pmax_50_soc, 2),
        "hx_pmax_at_min_soc_kw": round(hx_pmax_min_soc, 2),
        "hx_effectiveness": round(hx.effectiveness, 4),
        "configured_charge_power_kw": round(charge_power_config, 2),
        "available_charge_headroom_kw": round(grid_headroom, 2),
        "peak_actual_charge_power_kw": round(actual_peak_charge, 2),
        "grid_connection_limit_kw": round(grid_limit, 2),
        "peak_total_grid_power_kw": round(peak_grid_power, 2),
    }


def seed_default_scenarios_if_empty(session: Session) -> list[EngineeringScenario]:
    """Seed the initial standard scenarios if none exist."""
    existing_count = session.execute(select(EngineeringScenario.id)).scalars().first()
    if existing_count is not None:
        return list(session.execute(select(EngineeringScenario)).scalars().all())

    created: list[EngineeringScenario] = []
    for item in get_default_scenario_definitions():
        sc = save_scenario(
            session=session,
            name=item["name"],
            parameters=item["parameters"],
            description=item["description"],
        )
        created.append(sc)
    return created


def save_scenario(
    session: Session,
    name: str,
    parameters: dict[str, Any],
    description: str | None = None,
) -> EngineeringScenario:
    """Save an engineering scenario (append-only; increments version if name exists)."""
    # Find max version for this name
    stmt = (
        select(EngineeringScenario.version)
        .where(EngineeringScenario.name == name)
        .order_by(EngineeringScenario.version.desc())
    )
    latest_ver = session.execute(stmt).scalars().first()
    next_ver = (latest_ver or 0) + 1

    summary = compute_scenario_summary(parameters)
    scenario_obj = EngineeringScenario(
        scenario_id=str(uuid.uuid4()),
        name=name,
        version=next_ver,
        description=description,
        parameters=parameters,
        summary=summary,
    )
    session.add(scenario_obj)
    session.commit()
    session.refresh(scenario_obj)
    return scenario_obj


def list_scenarios(session: Session) -> list[EngineeringScenario]:
    """Return all saved scenarios ordered by name and descending version."""
    stmt = select(EngineeringScenario).order_by(EngineeringScenario.name.asc(), EngineeringScenario.version.desc())
    return list(session.execute(stmt).scalars().all())


def get_scenario_by_id(session: Session, scenario_id: str) -> EngineeringScenario | None:
    """Retrieve a single scenario by UUID."""
    stmt = select(EngineeringScenario).where(EngineeringScenario.scenario_id == scenario_id)
    return session.execute(stmt).scalars().first()


def compare_scenarios(scen_a: EngineeringScenario, scen_b: EngineeringScenario) -> dict[str, Any]:
    """Generate side-by-side comparison between two engineering scenarios using backend physics."""
    p_a, s_a = scen_a.parameters, scen_a.summary
    p_b, s_b = scen_b.parameters, scen_b.summary

    fields = [
        ("Sand Mass", s_a.get("sand_mass_kg"), s_b.get("sand_mass_kg"), "kg"),
        ("Full-Span Capacity", s_a.get("thermal_capacity_full_span_kwh"), s_b.get("thermal_capacity_full_span_kwh"), "kWh"),
        ("Dispatchable Capacity", s_a.get("dispatchable_capacity_kwh"), s_b.get("dispatchable_capacity_kwh"), "kWh"),
        ("Temp at Min SOC (10%)", s_a.get("temperature_at_optimizer_min_soc_c"), s_b.get("temperature_at_optimizer_min_soc_c"), "°C"),
        ("Sand Fill Height", s_a.get("sand_fill_height_mm"), s_b.get("sand_fill_height_mm"), "mm"),
        ("Freeboard Height", s_a.get("freeboard_height_mm"), s_b.get("freeboard_height_mm"), "mm"),
        ("HX Overall U", p_a.get("overall_u_w_m2k"), p_b.get("overall_u_w_m2k"), "W/m²K"),
        ("HX Airflow", p_a.get("airflow_m3_h"), p_b.get("airflow_m3_h"), "m³/h"),
        ("HX Max Power @ 100% SOC", s_a.get("hx_pmax_at_100_soc_kw"), s_b.get("hx_pmax_at_100_soc_kw"), "kW"),
        ("HX Max Power @ 50% SOC", s_a.get("hx_pmax_at_50_soc_kw"), s_b.get("hx_pmax_at_50_soc_kw"), "kW"),
        ("HX Max Power @ Min SOC", s_a.get("hx_pmax_at_min_soc_kw"), s_b.get("hx_pmax_at_min_soc_kw"), "kW"),
        ("Configured Charge Power", s_a.get("configured_charge_power_kw"), s_b.get("configured_charge_power_kw"), "kW"),
        ("Peak Actual Charge Power", s_a.get("peak_actual_charge_power_kw"), s_b.get("peak_actual_charge_power_kw"), "kW"),
        ("Grid Connection Limit", s_a.get("grid_connection_limit_kw"), s_b.get("grid_connection_limit_kw"), "kW"),
        ("Peak Total Grid Power", s_a.get("peak_total_grid_power_kw"), s_b.get("peak_total_grid_power_kw"), "kW"),
        ("Process Heat Demand", p_a.get("process_heat_demand_kw"), p_b.get("process_heat_demand_kw"), "kW"),
        ("Standing Loss Rate", p_a.get("standing_loss_percent_per_day"), p_b.get("standing_loss_percent_per_day"), "%/day"),
    ]

    diff_table = []
    for label, val_a, val_b, unit in fields:
        diff = (val_b - val_a) if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)) else None
        diff_str = f"{diff:+.2f}" if diff is not None else "N/A"
        diff_table.append({
            "label": label,
            "val_a": val_a,
            "val_b": val_b,
            "diff": diff,
            "diff_str": diff_str,
            "unit": unit,
        })

    return {
        "scenario_a": {"id": scen_a.scenario_id, "name": f"{scen_a.name} (v{scen_a.version})", "summary": s_a},
        "scenario_b": {"id": scen_b.scenario_id, "name": f"{scen_b.name} (v{scen_b.version})", "summary": s_b},
        "comparison_rows": diff_table,
    }
