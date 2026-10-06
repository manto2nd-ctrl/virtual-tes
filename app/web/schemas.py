"""Pydantic schemas for the Phase 5.7 Engineering Dashboard API."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ThermalStateRequest(BaseModel):
    physical_temperature_min_c: float = Field(default=80.0, description="Physical minimum temperature [°C]")
    physical_temperature_max_c: float = Field(default=300.0, description="Physical maximum temperature [°C]")
    thermal_capacity_full_span_kwh: float = Field(default=15.0, description="Total thermal capacity across physical span [kWh]")
    optimizer_soc_min_fraction: float = Field(default=0.10, ge=0.0, le=1.0, description="Optimizer minimum reserve SOC fraction")
    optimizer_soc_max_fraction: float = Field(default=1.00, ge=0.0, le=1.0, description="Optimizer maximum SOC fraction")
    soc_fraction: float = Field(default=0.50, ge=0.0, le=1.0, description="Current SOC fraction to evaluate [0-1]")


class ThermalStateResponse(BaseModel):
    physical_temperature_min_c: float
    physical_temperature_max_c: float
    thermal_capacity_full_span_kwh: float
    optimizer_soc_min_fraction: float
    optimizer_soc_max_fraction: float
    dispatchable_capacity_kwh: float
    soc_min_energy_kwh: float
    soc_max_energy_kwh: float
    sand_mass_kg: float
    specific_storage_kwh_per_kg: float
    temperature_at_optimizer_min_soc_c: float
    current_soc_fraction: float
    current_stored_energy_kwh: float
    current_sand_temperature_c: float
    current_remaining_dispatchable_kwh: float
    current_relative_enthalpy_j_per_kg: float
    current_hx_pmax_kw: float
    soc_temperature_curve: list[dict[str, float]]
    temperature_energy_curve: list[dict[str, float]]


class HXCalculateRequest(BaseModel):
    soc_fraction: float = Field(default=0.50, ge=0.0, le=1.0)
    sand_temperature_c: float | None = Field(default=None)
    overall_u_w_m2k: float = Field(default=9.0, gt=0.0)
    hx_area_m2: float = Field(default=1.55, gt=0.0)
    airflow_m3_h: float = Field(default=80.0, gt=0.0)
    air_inlet_temperature_c: float = Field(default=40.0)
    air_pressure_pa: float = Field(default=101325.0)
    process_demand_kw: float = Field(default=1.5)


class HXCalculateResponse(BaseModel):
    soc_fraction: float
    sand_temperature_c: float
    overall_u_w_m2k: float
    hx_area_m2: float
    airflow_m3_h: float
    air_inlet_temperature_c: float
    ua_w_k: float
    air_mass_flow_kg_s: float
    air_heat_capacity_rate_w_k: float
    ntu: float
    effectiveness: float
    p_max_kw: float
    predicted_outlet_air_temperature_c: float
    process_demand_kw: float
    can_satisfy_process_demand: bool
    hx_power_curve: list[dict[str, Any]]
    hx_temp_curve: list[dict[str, Any]]


class VesselCalculateRequest(BaseModel):
    mode: str = Field(default="capacity_driven", description="'capacity_driven' or 'geometry_driven'")
    internal_diameter_mm: float = Field(default=600.0, gt=0.0)
    straight_shell_height_mm: float = Field(default=950.0, gt=0.0)
    sand_bulk_density_kg_m3: float = Field(default=1600.0, gt=0.0)
    hx_displaced_volume_l: float = Field(default=24.0, ge=0.0)
    internals_displaced_volume_l: float = Field(default=8.0, ge=0.0)
    target_full_span_kwh: float = Field(default=15.0, gt=0.0)
    sand_mass_kg: float | None = Field(default=None)
    fill_fraction: float | None = Field(default=None)
    min_freeboard_mm: float = Field(default=80.0, ge=0.0)


class ScenarioSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str | None = None
    parameters: dict[str, Any]


class ScenarioCompareRequest(BaseModel):
    scenario_id_1: str
    scenario_id_2: str


class ShadowStartRequest(BaseModel):
    initialization_mode: Literal["soc", "energy", "temp"] = Field(
        default="soc", description="Single physical source of truth: 'soc', 'energy', or 'temp'"
    )
    initial_soc_percent: float | None = Field(default=50.0, ge=0.0, le=100.0)
    initial_energy_kwh: float | None = Field(default=None, ge=0.0, le=15.0)
    initial_temp_c: float | None = Field(default=None, ge=80.0, le=300.0)
    process_demand_kw: float = Field(default=1.5, ge=0.0, le=20.0)
    process_enabled: bool = Field(default=True)


class ShadowResetRequest(BaseModel):
    target_soc_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    reason: str = Field(default="MANUAL_RESET")


class ShadowProcessDemandRequest(BaseModel):
    process_demand_kw: float = Field(default=1.5, ge=0.0)
    process_enabled: bool = Field(default=True)

