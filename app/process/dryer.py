"""Virtual Dryer V1 Thermodynamic Engine & Timber Kinetics (Phase 5.10).

Follows Core Physical Principles:
1. Strict separation of sensible air heating (m_dot * cp * deltaT) from latent evaporation & losses.
2. Heat capacity rate of air at nominal conditions (80 m3/h, 40°C inlet):
   C_air = 25.22 W/K = 0.02522 kW/K. Heating 40°C -> 70°C is ~0.757 kW sensible heat.
3. Physical coil extraction bounded by HelicalAirHXModel.p_max_at_temperature_kw(T_sand).
4. Achievable supply air temperature derived directly from available HX heat.
5. Backup duct electric heater activates only on shortfall; interlocked OFF if blower airflow is 0.
6. Moisture removal tagged as SIMULATED ESTIMATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.models.hardware_interfaces import DataProvenance, OperatingMode
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper


class DryerControlMode(str, Enum):
    """Dryer thermal control modes."""
    FIXED_DEMAND = "FIXED_DEMAND"            # Fixed total process heat demand (e.g. 1.50 kW)
    TARGET_TEMPERATURE = "TARGET_TEMPERATURE"  # Target supply temperature drives demand


class DryingRecipePreset(str, Enum):
    """Standard industrial drying recipes."""
    SOFTWOOD_STANDARD = "SOFTWOOD_STANDARD"  # 70°C, 80 m3/h, 1.50 kW
    HARDWOOD_GENTLE = "HARDWOOD_GENTLE"      # 55°C, 60 m3/h, 1.00 kW (slow dry to prevent checking)
    HIGH_THROUGHPUT = "HIGH_THROUGHPUT"      # 85°C, 100 m3/h, 2.50 kW
    STANDBY = "STANDBY"                      # 40°C, 20 m3/h, 0.20 kW


@dataclass
class DryerOperatingConfig:
    """Operating settings for Virtual Dryer V1."""
    mode: DryerControlMode = DryerControlMode.FIXED_DEMAND
    target_supply_temp_c: float = 70.0
    inlet_air_temp_c: float = 40.0
    airflow_m3_h: float = 80.0
    total_process_demand_kw: float = 1.50
    air_density_kg_m3: float = 1.13
    air_cp_kj_kg_k: float = 1.005

    # Backup Duct Electric Heater
    backup_heater_enabled: bool = True
    backup_heater_max_kw: float = 3.0

    # Blower Fan
    blower_enabled: bool = True
    blower_rated_power_kw: float = 0.05

    # Timber Batch Parameters
    timber_mass_dry_kg: float = 500.0
    initial_moisture_percent: float = 50.0
    target_moisture_percent: float = 15.0
    latent_heat_evaporation_kwh_kg: float = 0.6278  # ~2260 kJ/kg = 0.6278 kWh/kg

    # Active Recipe
    recipe_preset: DryingRecipePreset = DryingRecipePreset.SOFTWOOD_STANDARD

    @property
    def control_mode(self) -> DryerControlMode:
        return self.mode

    @property
    def total_process_heat_demand_kw(self) -> float:
        return self.total_process_demand_kw

    @property
    def batch_dry_mass_kg(self) -> float:
        return self.timber_mass_dry_kg

    @property
    def batch_initial_moisture_fraction(self) -> float:
        return self.initial_moisture_percent / 100.0

    @property
    def batch_target_moisture_fraction(self) -> float:
        return self.target_moisture_percent / 100.0


@dataclass
class DryerStepResult:
    """Detailed thermodynamic output of a single dryer simulation step."""
    # Control Mode
    mode: DryerControlMode
    recipe_preset: DryingRecipePreset

    # Airflow & Capacity Rates
    airflow_m3_h: float
    air_mass_flow_kg_s: float
    heat_capacity_rate_kw_k: float  # C_air (kW/K)

    # Temperatures (°C)
    inlet_air_temp_c: float
    target_supply_temp_c: float
    achieved_supply_temp_c: float
    exhaust_air_temp_c: float
    sand_bed_temp_c: float

    # Heat Balances (kW)
    sensible_heat_demand_kw: float
    latent_and_loss_demand_kw: float
    total_process_demand_kw: float

    # Thermal Deliveries (kW)
    hx_max_available_kw: float
    heat_supplied_from_tes_kw: float
    backup_heater_power_kw: float
    heat_shortfall_unmet_kw: float
    total_heat_delivered_kw: float

    # Electrical Consumption (kW)
    blower_power_kw: float
    total_parasitic_electric_kw: float

    # Batch Progress (Simulated Estimate)
    current_moisture_percent: float
    water_removed_total_kg: float
    batch_progress_percent: float
    estimated_remaining_hours: float
    is_batch_complete: bool

    # Alarms & Status
    blower_interlock_active: bool
    tes_pinch_derated: bool
    alarms: list[str] = field(default_factory=list)
    provenance: DataProvenance = DataProvenance.SIMULATED

    @property
    def sensible_air_heat_kw(self) -> float:
        return self.sensible_heat_demand_kw

    @property
    def latent_and_losses_kw(self) -> float:
        return self.latent_and_loss_demand_kw

    @property
    def total_heat_demand_kw(self) -> float:
        return self.total_process_demand_kw

    @property
    def useful_heat_delivered_kw(self) -> float:
        return self.heat_supplied_from_tes_kw

    @property
    def backup_electric_power_kw(self) -> float:
        return self.backup_heater_power_kw

    @property
    def heat_shortfall_kw(self) -> float:
        return self.heat_shortfall_unmet_kw

    @property
    def blower_electric_power_kw(self) -> float:
        return self.blower_power_kw

    @property
    def total_electricity_kw(self) -> float:
        return self.total_parasitic_electric_kw

    @property
    def batch_current_moisture_fraction(self) -> float:
        return self.current_moisture_percent / 100.0

    @property
    def batch_initial_moisture_fraction(self) -> float:
        return 0.50

    @property
    def batch_target_moisture_fraction(self) -> float:
        return 0.15

    @property
    def batch_water_removed_kg(self) -> float:
        return self.water_removed_total_kg

    @property
    def batch_est_remaining_hours(self) -> float:
        return self.estimated_remaining_hours

    @property
    def batch_dry_mass_kg(self) -> float:
        return 500.0

    @property
    def hx_power_limit_kw(self) -> float:
        return self.hx_max_available_kw

    @property
    def status(self) -> str:
        return "IDLE" if self.blower_interlock_active else "RUNNING"


class VirtualDryerModel:
    """Virtual Dryer V1 physical simulator and batch kinetics tracker."""

    def __init__(
        self,
        config: DryerOperatingConfig | None = None,
        hx_model: HelicalAirHXModel | None = None,
        thermal_mapper: ThermalStateMapper | None = None,
    ) -> None:
        self.config = config or DryerOperatingConfig()
        self.thermal_mapper = thermal_mapper or ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
        self.hx_model = hx_model or HelicalAirHXModel(
            hx_area_m2=1.55,
            overall_u_w_m2k=9.0,
            airflow_m3_h=max(1.0, self.config.airflow_m3_h),
            air_inlet_temperature_c=self.config.inlet_air_temp_c,
            thermal_state_mapper=self.thermal_mapper,
        )

        # Batch state tracking
        self.current_water_mass_kg = (
            self.config.timber_mass_dry_kg * (self.config.initial_moisture_percent / 100.0)
        )
        self.target_water_mass_kg = (
            self.config.timber_mass_dry_kg * (self.config.target_moisture_percent / 100.0)
        )
        self.total_water_to_remove_kg = max(0.0, self.current_water_mass_kg - self.target_water_mass_kg)
        self.water_removed_cumulative_kg = 0.0

    def apply_recipe(self, preset: DryingRecipePreset) -> None:
        """Apply a standard drying recipe preset."""
        self.config.recipe_preset = preset
        if preset == DryingRecipePreset.SOFTWOOD_STANDARD:
            self.config.target_supply_temp_c = 70.0
            self.config.airflow_m3_h = 80.0
            self.config.total_process_demand_kw = 1.50
            self.config.target_moisture_percent = 15.0
        elif preset == DryingRecipePreset.HARDWOOD_GENTLE:
            self.config.target_supply_temp_c = 55.0
            self.config.airflow_m3_h = 60.0
            self.config.total_process_demand_kw = 1.00
            self.config.target_moisture_percent = 12.0
        elif preset == DryingRecipePreset.HIGH_THROUGHPUT:
            self.config.target_supply_temp_c = 85.0
            self.config.airflow_m3_h = 100.0
            self.config.total_process_demand_kw = 2.50
            self.config.target_moisture_percent = 15.0
        elif preset == DryingRecipePreset.STANDBY:
            self.config.target_supply_temp_c = 40.0
            self.config.airflow_m3_h = 20.0
            self.config.total_process_demand_kw = 0.20

    def calculate_c_air_kw_k(self, airflow_m3_h: float | None = None) -> tuple[float, float]:
        """Compute mass flow rate (kg/s) and heat capacity rate C_air (kW/K)."""
        flow = self.config.airflow_m3_h if airflow_m3_h is None else airflow_m3_h
        m_dot = (flow * self.config.air_density_kg_m3) / 3600.0
        c_air = m_dot * self.config.air_cp_kj_kg_k  # kW/K
        return m_dot, c_air

    def step(
        self,
        sand_temperature_c: float,
        dt_seconds: float = 900.0,  # 15 minutes default
    ) -> DryerStepResult:
        """Execute a physical simulation step of the dryer given current TES sand temperature."""
        cfg = self.config
        alarms: list[str] = []

        # 1. Blower Interlock check
        blower_running = cfg.blower_enabled and cfg.airflow_m3_h > 0.1
        if not blower_running:
            alarms.append("BLOWER_STOPPED_AIRFLOW_ZERO")
            m_dot = 0.0
            c_air = 0.0
            blower_kw = 0.0
        else:
            m_dot, c_air = self.calculate_c_air_kw_k(cfg.airflow_m3_h)
            # Fan power affinity scaling
            blower_kw = cfg.blower_rated_power_kw * (cfg.airflow_m3_h / 80.0) ** 2.5

        # 2. Demand calculations: Sensible Air Heating vs. Total Process Demand
        if not blower_running:
            sensible_demand_kw = 0.0
            total_demand_kw = 0.0
            latent_and_loss_kw = 0.0
        elif cfg.mode == DryerControlMode.TARGET_TEMPERATURE:
            delta_t_target = max(0.0, cfg.target_supply_temp_c - cfg.inlet_air_temp_c)
            sensible_demand_kw = c_air * delta_t_target
            # Base loss + evaporation enthalpy
            latent_and_loss_kw = max(0.2, cfg.total_process_demand_kw - sensible_demand_kw)
            total_demand_kw = sensible_demand_kw + latent_and_loss_kw
        else:
            # FIXED_DEMAND mode
            total_demand_kw = cfg.total_process_demand_kw
            delta_t_target = max(0.0, cfg.target_supply_temp_c - cfg.inlet_air_temp_c)
            nominal_sensible = c_air * delta_t_target
            sensible_demand_kw = min(total_demand_kw, nominal_sensible)
            latent_and_loss_kw = max(0.0, total_demand_kw - sensible_demand_kw)

        # 3. Physically available coil heat from sand bed
        if blower_running and sand_temperature_c > cfg.inlet_air_temp_c:
            hx_max_available_kw = self.hx_model.p_max_at_temperature_kw(sand_temperature_c)
        else:
            hx_max_available_kw = 0.0

        # TES delivered heat
        tes_supplied_kw = min(total_demand_kw, hx_max_available_kw)
        tes_pinch_derated = tes_supplied_kw < total_demand_kw and blower_running

        if tes_pinch_derated:
            alarms.append("TES_OUTPUT_DERATED_PINCH")

        # 4. Supply air temperature achieved purely from TES coil
        if blower_running and c_air > 0:
            sensible_from_tes = min(sensible_demand_kw, tes_supplied_kw)
            t_supply_coil = cfg.inlet_air_temp_c + (sensible_from_tes / c_air)
        else:
            t_supply_coil = cfg.inlet_air_temp_c

        # 5. Backup electric duct heater (interlocked OFF if blower stopped!)
        heat_shortfall_kw = max(0.0, total_demand_kw - tes_supplied_kw)
        if blower_running and cfg.backup_heater_enabled and heat_shortfall_kw > 0.001:
            backup_kw = min(cfg.backup_heater_max_kw, heat_shortfall_kw)
        else:
            backup_kw = 0.0

        unmet_kw = max(0.0, heat_shortfall_kw - backup_kw)
        total_delivered_kw = tes_supplied_kw + backup_kw

        # Final achieved supply temperature after backup heater
        if blower_running and c_air > 0:
            additional_sensible = min(max(0.0, sensible_demand_kw - sensible_from_tes), backup_kw)
            t_supply_achieved = min(cfg.target_supply_temp_c, t_supply_coil + (additional_sensible / c_air))
        else:
            t_supply_achieved = cfg.inlet_air_temp_c

        # 6. Exhaust air temperature (after sensible absorption in chamber)
        if blower_running and c_air > 0:
            temp_drop = min(25.0, latent_and_loss_kw / c_air)
            t_exhaust = max(cfg.inlet_air_temp_c, t_supply_achieved - temp_drop)
        else:
            t_exhaust = cfg.inlet_air_temp_c

        # 7. Timber Batch Kinetics (Simulated Estimate)
        hours_elapsed = dt_seconds / 3600.0
        # Useful latent energy delivered in kWh
        useful_latent_kwh = (min(latent_and_loss_kw, total_delivered_kw)) * hours_elapsed
        water_removed_step_kg = useful_latent_kwh / cfg.latent_heat_evaporation_kwh_kg if useful_latent_kwh > 0 else 0.0

        # Update cumulative removal
        if self.current_water_mass_kg > self.target_water_mass_kg:
            self.water_removed_cumulative_kg += min(
                water_removed_step_kg,
                self.current_water_mass_kg - self.target_water_mass_kg
            )
            self.current_water_mass_kg = max(
                self.target_water_mass_kg,
                self.current_water_mass_kg - water_removed_step_kg
            )

        current_mc = (self.current_water_mass_kg / cfg.timber_mass_dry_kg) * 100.0
        batch_complete = current_mc <= (cfg.target_moisture_percent + 0.1)

        # Progress %
        if self.total_water_to_remove_kg > 0:
            progress_pct = min(100.0, (self.water_removed_cumulative_kg / self.total_water_to_remove_kg) * 100.0)
        else:
            progress_pct = 100.0

        # Remaining hours estimate
        water_remaining = max(0.0, self.current_water_mass_kg - self.target_water_mass_kg)
        hourly_rate = (latent_and_loss_kw / cfg.latent_heat_evaporation_kwh_kg) if latent_and_loss_kw > 0 else 0.0
        est_remaining_h = round(water_remaining / hourly_rate, 1) if hourly_rate > 0.01 else 0.0

        res = DryerStepResult(
            mode=cfg.mode,
            recipe_preset=cfg.recipe_preset,
            airflow_m3_h=cfg.airflow_m3_h if blower_running else 0.0,
            air_mass_flow_kg_s=round(m_dot, 5),
            heat_capacity_rate_kw_k=round(c_air, 5),
            inlet_air_temp_c=round(cfg.inlet_air_temp_c, 1),
            target_supply_temp_c=round(cfg.target_supply_temp_c, 1),
            achieved_supply_temp_c=round(t_supply_achieved, 1),
            exhaust_air_temp_c=round(t_exhaust, 1),
            sand_bed_temp_c=round(sand_temperature_c, 1),
            sensible_heat_demand_kw=round(sensible_demand_kw, 3),
            latent_and_loss_demand_kw=round(latent_and_loss_kw, 3),
            total_process_demand_kw=round(total_demand_kw, 3),
            hx_max_available_kw=round(hx_max_available_kw, 3),
            heat_supplied_from_tes_kw=round(tes_supplied_kw, 3),
            backup_heater_power_kw=round(backup_kw, 3),
            heat_shortfall_unmet_kw=round(unmet_kw, 3),
            total_heat_delivered_kw=round(total_delivered_kw, 3),
            blower_power_kw=round(blower_kw, 3),
            total_parasitic_electric_kw=round(blower_kw + backup_kw, 3),
            current_moisture_percent=round(current_mc, 1),
            water_removed_total_kg=round(self.water_removed_cumulative_kg, 1),
            batch_progress_percent=round(progress_pct, 1),
            estimated_remaining_hours=est_remaining_h,
            is_batch_complete=batch_complete,
            blower_interlock_active=not blower_running,
            tes_pinch_derated=tes_pinch_derated,
            alarms=alarms,
            provenance=DataProvenance.SIMULATED,
        )
        self._latest_result = res
        return res

    def get_status(self) -> dict[str, Any]:
        """Return serializable status dictionary of latest or default dryer step."""
        res = getattr(self, "_latest_result", None)
        if res is None:
            res = self.step(sand_temperature_c=197.0, dt_seconds=900.0)
            self._latest_result = res
        return {
            "status": res.status,
            "mode": res.mode.value,
            "recipe_preset": res.recipe_preset.value,
            "target_supply_temp_c": res.target_supply_temp_c,
            "inlet_temp_c": res.inlet_air_temp_c,
            "achieved_supply_temp_c": res.achieved_supply_temp_c,
            "exhaust_temp_c": res.exhaust_air_temp_c,
            "sand_bed_temp_c": res.sand_bed_temp_c,
            "airflow_m3_h": res.airflow_m3_h,
            "sensible_air_heat_kw": res.sensible_air_heat_kw,
            "latent_and_losses_kw": res.latent_and_losses_kw,
            "total_heat_demand_kw": res.total_heat_demand_kw,
            "hx_power_limit_kw": res.hx_power_limit_kw,
            "useful_heat_delivered_kw": res.useful_heat_delivered_kw,
            "backup_electric_power_kw": res.backup_electric_power_kw,
            "heat_shortfall_kw": res.heat_shortfall_kw,
            "blower_electric_power_kw": res.blower_electric_power_kw,
            "total_electricity_kw": res.total_electricity_kw,
            "batch_dry_mass_kg": res.batch_dry_mass_kg,
            "batch_initial_moisture_fraction": res.batch_initial_moisture_fraction,
            "batch_target_moisture_fraction": res.batch_target_moisture_fraction,
            "batch_current_moisture_fraction": res.batch_current_moisture_fraction,
            "batch_water_removed_kg": res.batch_water_removed_kg,
            "batch_progress_percent": res.batch_progress_percent,
            "batch_est_remaining_hours": res.batch_est_remaining_hours,
            "is_batch_complete": res.is_batch_complete,
            "blower_interlock_active": res.blower_interlock_active,
            "tes_pinch_derated": res.tes_pinch_derated,
            "alarms": res.alarms,
            "provenance": res.provenance.value,
        }
