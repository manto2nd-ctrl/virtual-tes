"""Virtual Dryer V1 Thermodynamic Engine & Timber Kinetics (Phase 5.11 - Stage 4).

Follows Core Physical Principles:
1. 1.5 kW is a configurable reference process heat load for Gen0 experiments, NOT a universal requirement.
2. Two simulation modes:
   - Mode 1 (Thermal Test Rig): Pure thermal validation of Gen0 TES & helical HX on an instrumented duct.
     No fictitious moisture removal. Calculates heat delivered, shortages, and achievable air temperatures.
   - Mode 2 (Wood Drying Process): Simulates real timber batch drying across configurable loads (5 kg, 10 kg,
     25 kg, 50 kg, 500 kg, Custom), wood species, board thickness, moisture content, ventilation & losses.
3. Strict separation of sensible air heating (C_air * deltaT) from fresh ventilation loss, chamber losses, and latent evaporation.
4. Distinguishes total circulation airflow (80 m3/h nominal) from fresh outdoor makeup / exhaust airflow via recirculation fraction.
   In recirculating operation, heating the entire circulating air stream is NOT repeatedly counted as a fresh ventilation loss.
5. First Law energy allocation: available heat first meets sensible air heating, then ventilation & chamber losses, then latent evaporation.
6. Energy conservation: evaporation energy cannot exceed the physically available heat budget.
7. Moisture mass balance on dry basis: Water removed = Dry_wood_mass * (MC_initial - MC_current).
8. Theoretical evaporation energy requirement: 175 kg * 0.63 kWh/kg ≈ 110.25 kWh for 500 kg dry wood from 50% to 15%.
9. Power deficit warning when requested drying duration exceeds available heating capacity.
10. All drying kinetics are labeled as SIMULATED ESTIMATE until calibrated against physical sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from app.models.hardware_interfaces import DataProvenance, OperatingMode
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper


class DryerSimulationMode(str, Enum):
    """Dryer primary simulation modes."""
    THERMAL_TEST_RIG = "THERMAL_TEST_RIG"        # Mode 1: Validate Gen0 TES/HX on instrumented duct (no timber)
    WOOD_DRYING_PROCESS = "WOOD_DRYING_PROCESS"  # Mode 2: Timber batch drying process with moisture kinetics


class DryerControlMode(str, Enum):
    """Dryer thermal control modes."""
    FIXED_DEMAND = "FIXED_DEMAND"              # Fixed reference process heat demand (e.g. 1.50 kW)
    TARGET_TEMPERATURE = "TARGET_TEMPERATURE"  # Temperature target drives sensible + process demand
    CALCULATED_PROCESS = "CALCULATED_PROCESS"  # Full multi-component heat calculation (air + vent + loss + evap)


class DryingRecipePreset(str, Enum):
    """Standard industrial drying recipes."""
    SOFTWOOD_STANDARD = "SOFTWOOD_STANDARD"  # 70°C, 80 m3/h, 50 kg load, 48h duration
    HARDWOOD_GENTLE = "HARDWOOD_GENTLE"      # 55°C, 60 m3/h, 50 kg load, 72h duration (gentle to prevent checking)
    HIGH_THROUGHPUT = "HIGH_THROUGHPUT"      # 85°C, 100 m3/h, 50 kg load, 24h duration
    STANDBY = "STANDBY"                      # 40°C, 20 m3/h, 0.20 kW idle holding


class WoodSpecies(str, Enum):
    """Wood species with physical drying characteristics."""
    PINE = "PINE"        # Scots pine (Pinus sylvestris) - standard softwood, rho ~450 kg/m3, FSP=30%
    SPRUCE = "SPRUCE"    # Norway spruce (Picea abies) - softwood, rho ~420 kg/m3, FSP=30%
    BIRCH = "BIRCH"      # Silver birch (Betula pendula) - hardwood, rho ~620 kg/m3, FSP=30%
    OAK = "OAK"          # European oak (Quercus robur) - dense hardwood, rho ~700 kg/m3, FSP=28%, slow diffusion


class TimberLoadPreset(str, Enum):
    """Selectable timber load presets."""
    LOAD_5KG = "5_KG"       # Small lab specimen test
    LOAD_10KG = "10_KG"     # Benchtop rig test
    LOAD_25KG = "25_KG"     # Pilot trial
    LOAD_50KG = "50_KG"     # Standard experimental batch
    LOAD_500KG = "500_KG"   # Industrial design scenario
    CUSTOM = "CUSTOM"       # Custom dry wood mass


@dataclass
class DryerOperatingConfig:
    """Operating settings and physical parameters for Virtual Dryer V1."""
    simulation_mode: DryerSimulationMode = DryerSimulationMode.WOOD_DRYING_PROCESS
    mode: DryerControlMode = DryerControlMode.FIXED_DEMAND
    recipe_preset: DryingRecipePreset = DryingRecipePreset.SOFTWOOD_STANDARD

    # Mode 1 Experimental Reference Load:
    reference_process_heat_load_kw: float = 1.50
    total_process_demand_kw: float = 1.50  # Maintained for backward compatibility

    # Temperatures & Pneumatics:
    target_supply_temp_c: float = 70.0
    inlet_air_temp_c: float = 40.0
    ambient_temp_c: float = 15.0
    airflow_m3_h: float = 80.0
    air_density_kg_m3: float = 1.13
    air_cp_kj_kg_k: float = 1.005

    # Ventilation & Chamber Losses:
    recirculation_fraction: float = 0.85  # 85% recirculated, 15% fresh air exchange
    chamber_loss_factor_kw_k: float = 0.004  # UA_chamber ~ 4 W/K
    chamber_base_loss_kw: float = 0.10      # Base standing conduction loss

    # Backup Duct Electric Heater:
    backup_heater_enabled: bool = True
    backup_heater_max_kw: float = 3.0

    # Blower Fan:
    blower_enabled: bool = True
    blower_rated_power_kw: float = 0.050

    # Mode 2 Timber Batch Parameters:
    wood_species: WoodSpecies = WoodSpecies.PINE
    timber_mass_dry_kg: float = 50.0  # Default to 50 kg experimental batch
    board_thickness_mm: float = 32.0  # Nominal 32 mm timber boards
    initial_moisture_percent: float = 50.0  # Dry basis %
    target_moisture_percent: float = 15.0   # Dry basis %
    inlet_humidity_percent: float = 60.0    # Ambient RH %
    requested_duration_hours: float = 48.0  # Requested drying batch duration

    latent_heat_evaporation_kwh_kg: float = 0.6278  # ~2260 kJ/kg = 0.6278 kWh/kg

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
    # Simulation & Control Mode
    simulation_mode: DryerSimulationMode
    mode: DryerControlMode
    recipe_preset: DryingRecipePreset

    # Pneumatics & Airflow Distinction
    airflow_m3_h: float            # Total circulation airflow
    fresh_airflow_m3_h: float      # Outdoor makeup & exhaust airflow
    recirculation_fraction: float  # Fraction recirculated in chamber
    air_mass_flow_kg_s: float
    heat_capacity_rate_kw_k: float # C_air (kW/K)

    # Temperatures (°C)
    inlet_air_temp_c: float
    target_supply_temp_c: float
    achieved_supply_temp_c: float
    exhaust_air_temp_c: float
    timber_temperature_c: float
    sand_bed_temp_c: float

    # Heat Demand Breakdown (kW)
    sensible_heat_demand_kw: float     # Sensible air heating across coil (C_air * deltaT)
    ventilation_loss_kw: float         # Fresh makeup air heating loss
    chamber_loss_kw: float             # Envelope conductive loss
    latent_and_loss_demand_kw: float   # Non-sensible total demand
    evaporation_heat_demand_kw: float  # Kinetic latent evaporation requirement
    total_process_demand_kw: float     # Total required process heat

    # Thermal Deliveries (kW)
    hx_max_available_kw: float
    heat_supplied_from_tes_kw: float
    backup_heater_power_kw: float
    heat_shortfall_unmet_kw: float
    total_heat_delivered_kw: float
    evaporation_heat_delivered_kw: float
    timber_sensible_heat_kw: float      # Sensible heat absorbed by timber charge (m_wood * cp * dT)
    thermal_accumulation_kw: float     # Thermal mass accumulation in duct & internal structure
    thermal_residual_kw: float         # First Law energy balance residual (delivered - sum(absorbed) == 0)

    # Electrical Consumption (kW)
    blower_power_kw: float
    total_parasitic_electric_kw: float

    # Batch Kinetics (Mode 2: Wood Drying Process)
    timber_mass_dry_kg: float
    wood_species: WoodSpecies
    current_moisture_percent: float | None
    water_removed_step_kg: float
    water_removed_total_kg: float
    batch_progress_percent: float | None
    estimated_remaining_hours: float | None
    is_batch_complete: bool
    thermal_energy_per_kg_water_kwh: float | None

    # Alarms, Interlocks & Warnings
    blower_interlock_active: bool
    tes_pinch_derated: bool
    power_deficit_warning: bool
    warning_message: str | None = None
    alarms: list[str] = field(default_factory=list)
    provenance: DataProvenance = DataProvenance.SIMULATED

    # Backward Compatibility Properties
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
        return (self.current_moisture_percent / 100.0) if self.current_moisture_percent is not None else 0.0

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
        return self.estimated_remaining_hours if self.estimated_remaining_hours is not None else 0.0

    @property
    def batch_dry_mass_kg(self) -> float:
        return self.timber_mass_dry_kg

    @property
    def hx_power_limit_kw(self) -> float:
        return self.hx_max_available_kw

    @property
    def status(self) -> str:
        return "IDLE" if self.blower_interlock_active else "RUNNING"


class VirtualDryerModel:
    """Virtual Dryer V1 physical simulator supporting Mode 1 (Thermal Test Rig) & Mode 2 (Wood Drying Process)."""

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

        # Cumulative tracking
        self.water_removed_cumulative_kg = 0.0
        self.cumulative_thermal_kwh = 0.0
        self.cumulative_electric_kwh = 0.0

        # Batch state tracking
        self.reset_batch(
            timber_mass_dry_kg=self.config.timber_mass_dry_kg,
            initial_moisture_percent=self.config.initial_moisture_percent,
            target_moisture_percent=self.config.target_moisture_percent,
        )

    def reset_batch(
        self,
        timber_mass_dry_kg: float | None = None,
        initial_moisture_percent: float | None = None,
        target_moisture_percent: float | None = None,
        wood_species: WoodSpecies | None = None,
    ) -> None:
        """Reset timber batch moisture states for Mode 2."""
        if wood_species is not None:
            self.config.wood_species = wood_species
        if timber_mass_dry_kg is not None:
            self.config.timber_mass_dry_kg = float(timber_mass_dry_kg)
        if initial_moisture_percent is not None:
            self.config.initial_moisture_percent = float(initial_moisture_percent)
        if target_moisture_percent is not None:
            self.config.target_moisture_percent = float(target_moisture_percent)

        m_dry = self.config.timber_mass_dry_kg
        init_mc = self.config.initial_moisture_percent / 100.0
        targ_mc = self.config.target_moisture_percent / 100.0

        self.current_water_mass_kg = m_dry * init_mc
        self.target_water_mass_kg = m_dry * targ_mc
        self.total_water_to_remove_kg = max(0.0, self.current_water_mass_kg - self.target_water_mass_kg)
        self.water_removed_cumulative_kg = 0.0
        self.cumulative_thermal_kwh = 0.0
        self.cumulative_electric_kwh = 0.0

    def apply_recipe(self, preset: DryingRecipePreset) -> None:
        """Apply a standard drying recipe preset."""
        self.config.recipe_preset = preset
        if preset == DryingRecipePreset.SOFTWOOD_STANDARD:
            self.config.wood_species = WoodSpecies.PINE
            self.config.target_supply_temp_c = 70.0
            self.config.airflow_m3_h = 80.0
            self.config.total_process_demand_kw = 1.50
            self.config.reference_process_heat_load_kw = 1.50
            self.config.target_moisture_percent = 15.0
            self.config.requested_duration_hours = 48.0
        elif preset == DryingRecipePreset.HARDWOOD_GENTLE:
            self.config.wood_species = WoodSpecies.OAK
            self.config.target_supply_temp_c = 55.0
            self.config.airflow_m3_h = 60.0
            self.config.total_process_demand_kw = 1.00
            self.config.reference_process_heat_load_kw = 1.00
            self.config.target_moisture_percent = 12.0
            self.config.requested_duration_hours = 72.0
        elif preset == DryingRecipePreset.HIGH_THROUGHPUT:
            self.config.wood_species = WoodSpecies.SPRUCE
            self.config.target_supply_temp_c = 85.0
            self.config.airflow_m3_h = 100.0
            self.config.total_process_demand_kw = 2.50
            self.config.reference_process_heat_load_kw = 2.50
            self.config.target_moisture_percent = 15.0
            self.config.requested_duration_hours = 24.0
        elif preset == DryingRecipePreset.STANDBY:
            self.config.target_supply_temp_c = 40.0
            self.config.airflow_m3_h = 20.0
            self.config.total_process_demand_kw = 0.20
            self.config.reference_process_heat_load_kw = 0.20

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
        tes_heat_delivered_kw: float | None = None,
        available_stored_energy_kwh: float | None = None,
        discharge_efficiency: float = 0.90,
        mutate_batch: bool = True,
    ) -> DryerStepResult:
        """Execute a physically consistent simulation step of the dryer given current TES sand temperature.

        Enforces:
          1. Blower interlock (if airflow is 0, backup duct heater is locked OFF).
          2. Distinction of sensible air heating from fresh air ventilation and chamber losses.
          3. First Law energy allocation: sensible air heating takes priority, remaining heat powers evaporation.
          4. Evaporation energy cannot exceed physically available heat budget.
          5. Mode 1 (Thermal Test Rig) skips moisture kinetics (no fictitious drying).
          6. Mode 2 (Wood Drying Process) simulates material kinetics on dry basis.
        """
        cfg = self.config
        alarms: list[str] = []
        dt_hours = dt_seconds / 3600.0

        # -----------------------------------------------------------------------
        # 1. Blower Interlock & Pneumatics
        # -----------------------------------------------------------------------
        blower_running = cfg.blower_enabled and cfg.airflow_m3_h > 0.1
        if not blower_running:
            alarms.append("BLOWER_STOPPED_AIRFLOW_ZERO")
            m_dot = 0.0
            c_air = 0.0
            blower_kw = 0.0
            fresh_flow_m3_h = 0.0
        else:
            m_dot, c_air = self.calculate_c_air_kw_k(cfg.airflow_m3_h)
            # Fan power affinity scaling
            blower_kw = cfg.blower_rated_power_kw * (cfg.airflow_m3_h / 80.0) ** 2.5
            fresh_flow_m3_h = cfg.airflow_m3_h * (1.0 - cfg.recirculation_fraction)

        # -----------------------------------------------------------------------
        # 2. Demand Breakdown: Sensible Air Heating vs. Process Loads
        # -----------------------------------------------------------------------
        delta_t_target = max(0.0, cfg.target_supply_temp_c - cfg.inlet_air_temp_c)
        nominal_sensible = c_air * delta_t_target  # Sensible air heating across coil

        if not blower_running:
            sensible_demand_kw = 0.0
            ventilation_loss_kw = 0.0
            chamber_loss_kw = 0.0
            evap_demand_kw = 0.0
            total_demand_kw = 0.0
            latent_and_loss_kw = 0.0
            m_evap_kinetic_rate = 0.0
        elif cfg.simulation_mode == DryerSimulationMode.THERMAL_TEST_RIG:
            # MODE 1: Thermal Test Rig (Validate Gen0 TES & HX on instrumented duct)
            ventilation_loss_kw = 0.0
            chamber_loss_kw = 0.0
            evap_demand_kw = 0.0
            m_evap_kinetic_rate = 0.0

            if cfg.mode == DryerControlMode.TARGET_TEMPERATURE:
                sensible_demand_kw = nominal_sensible
                total_demand_kw = nominal_sensible
                latent_and_loss_kw = 0.0
            else:
                # FIXED_DEMAND / Configured Reference Load (1.50 kW nominal)
                total_demand_kw = cfg.reference_process_heat_load_kw
                sensible_demand_kw = min(total_demand_kw, nominal_sensible)
                latent_and_loss_kw = max(0.0, total_demand_kw - sensible_demand_kw)
        else:
            # MODE 2: Wood Drying Process (Timber Charge & Kinetics)
            sensible_demand_kw = nominal_sensible

            # Fresh makeup air ventilation loss
            m_dot_fresh = (fresh_flow_m3_h * cfg.air_density_kg_m3) / 3600.0
            c_air_fresh = m_dot_fresh * cfg.air_cp_kj_kg_k
            est_exhaust_t = max(cfg.inlet_air_temp_c, cfg.target_supply_temp_c - 20.0)
            ventilation_loss_kw = c_air_fresh * max(0.0, est_exhaust_t - cfg.ambient_temp_c)

            # Chamber envelope conduction loss
            est_chamber_avg_t = 0.5 * (cfg.target_supply_temp_c + est_exhaust_t)
            chamber_loss_kw = cfg.chamber_base_loss_kw + (
                cfg.chamber_loss_factor_kw_k * max(0.0, est_chamber_avg_t - cfg.ambient_temp_c)
            )

            # Target drying rate from requested duration
            water_to_remove_total = max(
                0.0,
                cfg.timber_mass_dry_kg * ((cfg.initial_moisture_percent - cfg.target_moisture_percent) / 100.0),
            )
            dur_h = max(1.0, cfg.requested_duration_hours)
            m_evap_rate_req = water_to_remove_total / dur_h  # kg/h required on schedule

            # Species drying diffusion factor
            fsp = 28.0 if cfg.wood_species == WoodSpecies.OAK else 30.0
            cur_mc_pct = (self.current_water_mass_kg / cfg.timber_mass_dry_kg) * 100.0 if cfg.timber_mass_dry_kg > 0 else 0.0

            if cur_mc_pct > fsp:
                k_diffusion = 1.0  # Constant drying rate stage (free capillary water)
                h_evap = cfg.latent_heat_evaporation_kwh_kg
            else:
                # Falling rate stage (bound water diffusion resistance)
                emc = 8.0  # Equilibrium moisture content approx 8%
                k_mc = max(0.12, (cur_mc_pct - emc) / max(1.0, fsp - emc))
                thickness_factor = (32.0 / max(10.0, cfg.board_thickness_mm)) ** 0.5
                species_factor = 0.80 if cfg.wood_species == WoodSpecies.OAK else 1.0
                k_diffusion = k_mc * thickness_factor * species_factor
                h_evap = cfg.latent_heat_evaporation_kwh_kg * (1.0 + 0.18 * ((fsp - cur_mc_pct) / fsp))

            m_evap_kinetic_rate = m_evap_rate_req * k_diffusion  # kg/h
            evap_demand_kw = m_evap_kinetic_rate * h_evap

            if cfg.mode == DryerControlMode.FIXED_DEMAND:
                # When in fixed demand mode, bound total by configured setting (1.5 kW reference)
                total_demand_kw = cfg.total_process_demand_kw
                sensible_demand_kw = min(total_demand_kw, nominal_sensible)
                latent_and_loss_kw = max(0.0, total_demand_kw - sensible_demand_kw)
            else:
                # Pure physical multi-component process demand
                total_demand_kw = sensible_demand_kw + ventilation_loss_kw + chamber_loss_kw + evap_demand_kw
                latent_and_loss_kw = ventilation_loss_kw + chamber_loss_kw + evap_demand_kw

        # -----------------------------------------------------------------------
        # 3. Available TES Heat Exchanger Power & Delivery
        # -----------------------------------------------------------------------
        if blower_running and sand_temperature_c > cfg.inlet_air_temp_c:
            hx_max_available_kw = self.hx_model.p_max_at_temperature_kw(sand_temperature_c)
        else:
            hx_max_available_kw = 0.0

        if tes_heat_delivered_kw is not None:
            tes_supplied_kw = min(total_demand_kw, hx_max_available_kw, max(0.0, float(tes_heat_delivered_kw)))
        elif available_stored_energy_kwh is not None:
            max_p_energy = (max(0.0, float(available_stored_energy_kwh)) * discharge_efficiency) / dt_hours if dt_hours > 0 else 0.0
            tes_supplied_kw = min(total_demand_kw, hx_max_available_kw, max_p_energy)
        else:
            tes_supplied_kw = min(total_demand_kw, hx_max_available_kw)

        tes_pinch_derated = tes_supplied_kw < total_demand_kw and blower_running
        if tes_pinch_derated:
            alarms.append("TES_OUTPUT_DERATED_PINCH")

        # -----------------------------------------------------------------------
        # 4. Backup Duct Electric Heater (Airflow Interlocked!)
        # -----------------------------------------------------------------------
        heat_shortfall_kw = max(0.0, total_demand_kw - tes_supplied_kw)
        if blower_running and cfg.backup_heater_enabled and heat_shortfall_kw > 0.001:
            backup_kw = min(cfg.backup_heater_max_kw, heat_shortfall_kw)
        else:
            backup_kw = 0.0

        unmet_kw = max(0.0, heat_shortfall_kw - backup_kw)
        total_delivered_kw = tes_supplied_kw + backup_kw

        # Power deficit check
        total_combined_capacity_kw = hx_max_available_kw + (cfg.backup_heater_max_kw if cfg.backup_heater_enabled else 0.0)
        required_schedule_duty_kw = (
            sensible_demand_kw + ventilation_loss_kw + chamber_loss_kw + evap_demand_kw
            if cfg.simulation_mode == DryerSimulationMode.WOOD_DRYING_PROCESS
            else total_demand_kw
        )
        duty_to_compare = max(total_demand_kw, required_schedule_duty_kw)
        power_deficit = blower_running and (duty_to_compare > total_combined_capacity_kw + 0.05)
        warning_msg = None
        if power_deficit:
            warning_msg = (
                f"POWER DEFICIT: Requested drying duty of {cfg.timber_mass_dry_kg:.0f} kg {cfg.wood_species.value} "
                f"requires {duty_to_compare:.2f} kW thermal power, exceeding combined heating capacity "
                f"({total_combined_capacity_kw:.2f} kW). Drying will take longer than scheduled."
            )
            alarms.append("POWER_DEFICIT_WARNING")

        # -----------------------------------------------------------------------
        # 5. First Law Energy Allocation & Temperature Distribution
        # -----------------------------------------------------------------------
        # Sensible air heating across coil takes priority to establish supply air temperature
        if blower_running and c_air > 0:
            sensible_actual = min(sensible_demand_kw, total_delivered_kw)
            t_supply_achieved = cfg.inlet_air_temp_c + (sensible_actual / c_air)
            t_supply_achieved = min(cfg.target_supply_temp_c, t_supply_achieved)
        else:
            sensible_actual = 0.0
            t_supply_achieved = cfg.inlet_air_temp_c

        # Remaining delivered heat allocated to ventilation, chamber losses, and latent evaporation
        heat_remaining_for_process = max(0.0, total_delivered_kw - sensible_actual)

        if cfg.simulation_mode == DryerSimulationMode.THERMAL_TEST_RIG:
            # Mode 1: No evaporation
            evap_delivered_kw = 0.0
            water_removed_step_kg = 0.0
            current_mc = None
            progress_pct = None
            est_remaining_h = None
            sec_thermal = None
            batch_complete = False
            t_exhaust = t_supply_achieved
            t_timber = t_supply_achieved
            timber_sensible_kw = 0.0
            thermal_accumulation_kw = max(0.0, total_delivered_kw - sensible_actual)
            thermal_residual_kw = round(total_delivered_kw - (sensible_actual + thermal_accumulation_kw), 6)
        else:
            # Mode 2: Wood Drying Process
            # Losses are met first from remaining heat
            losses_actual = min(heat_remaining_for_process, ventilation_loss_kw + chamber_loss_kw)
            heat_rem_after_losses = max(0.0, heat_remaining_for_process - losses_actual)

            # Timber sensible warming (m_wood * cp * dT / t)
            t_chamber_est = 0.5 * (t_supply_achieved + max(cfg.inlet_air_temp_c, t_supply_achieved - 20.0))
            cp_wood_kj = 1.7  # Scots pine timber specific heat ~ 1.7 kJ/(kg*K)
            dur_sec = max(3600.0, cfg.requested_duration_hours * 3600.0)
            timber_sensible_demand_kw = (cfg.timber_mass_dry_kg * cp_wood_kj * max(0.0, t_chamber_est - cfg.ambient_temp_c)) / dur_sec
            timber_sensible_kw = min(heat_rem_after_losses, timber_sensible_demand_kw)
            heat_avail_for_evap = max(0.0, heat_rem_after_losses - timber_sensible_kw)

            # If in FIXED_DEMAND mode with historical 1.5 kW reference:
            if cfg.mode == DryerControlMode.FIXED_DEMAND:
                heat_avail_for_evap = min(latent_and_loss_kw, heat_avail_for_evap)

            # Evaporation cannot exceed physically available heat budget!
            h_evap_curr = cfg.latent_heat_evaporation_kwh_kg
            water_rem_budget_kg = (heat_avail_for_evap * dt_hours) / h_evap_curr if heat_avail_for_evap > 0 else 0.0
            water_rem_kinetic_kg = m_evap_kinetic_rate * dt_hours

            remaining_water_in_wood = max(0.0, self.current_water_mass_kg - self.target_water_mass_kg)
            water_removed_step_kg = min(water_rem_budget_kg, water_rem_kinetic_kg, remaining_water_in_wood)
            evap_delivered_kw = (water_removed_step_kg * h_evap_curr) / dt_hours if dt_hours > 0 else 0.0

            # Thermal accumulation in casing, structure, and internal circulating air enthalpy:
            absorbed_sum = sensible_actual + losses_actual + timber_sensible_kw + evap_delivered_kw
            thermal_accumulation_kw = max(0.0, total_delivered_kw - absorbed_sum)
            thermal_residual_kw = round(total_delivered_kw - (absorbed_sum + thermal_accumulation_kw), 6)

            # Update batch state (or compute preview values)
            if mutate_batch:
                self.water_removed_cumulative_kg += water_removed_step_kg
                self.current_water_mass_kg = max(
                    self.target_water_mass_kg,
                    self.current_water_mass_kg - water_removed_step_kg,
                )
                effective_water_mass = self.current_water_mass_kg
                effective_cum_water = self.water_removed_cumulative_kg
                self.cumulative_thermal_kwh += total_delivered_kw * dt_hours
                effective_cum_thermal = self.cumulative_thermal_kwh
            else:
                effective_water_mass = max(
                    self.target_water_mass_kg,
                    self.current_water_mass_kg - water_removed_step_kg,
                )
                effective_cum_water = self.water_removed_cumulative_kg + water_removed_step_kg
                effective_cum_thermal = self.cumulative_thermal_kwh + (total_delivered_kw * dt_hours)

            current_mc = (effective_water_mass / cfg.timber_mass_dry_kg) * 100.0 if cfg.timber_mass_dry_kg > 0 else 0.0
            batch_complete = current_mc <= (cfg.target_moisture_percent + 0.05)

            if self.total_water_to_remove_kg > 0:
                progress_pct = min(100.0, (effective_cum_water / self.total_water_to_remove_kg) * 100.0)
            else:
                progress_pct = 100.0

            # Remaining hours estimate based on actual evaporation rate
            water_left_to_remove = max(0.0, effective_water_mass - self.target_water_mass_kg)
            evap_hourly_rate = (water_removed_step_kg / dt_hours) if dt_hours > 0 else 0.0
            if batch_complete:
                est_remaining_h = 0.0
            elif evap_hourly_rate > 0.005:
                est_remaining_h = round(water_left_to_remove / evap_hourly_rate, 1)
            else:
                est_remaining_h = 999.0

            # Temperatures in chamber
            if blower_running and c_air > 0:
                delta_t_evap = evap_delivered_kw / c_air
                t_exhaust = max(cfg.inlet_air_temp_c, t_supply_achieved - delta_t_evap - 2.0)
            else:
                t_exhaust = cfg.inlet_air_temp_c

            t_timber = round(0.5 * (t_supply_achieved + t_exhaust), 1)

            # Specific Energy Consumption (kWh_th / kg water removed)
            if effective_cum_water > 0.05:
                sec_thermal = round(effective_cum_thermal / effective_cum_water, 2)
            else:
                sec_thermal = round(cfg.latent_heat_evaporation_kwh_kg, 2)

        res = DryerStepResult(
            simulation_mode=cfg.simulation_mode,
            mode=cfg.mode,
            recipe_preset=cfg.recipe_preset,
            airflow_m3_h=cfg.airflow_m3_h if blower_running else 0.0,
            fresh_airflow_m3_h=round(fresh_flow_m3_h, 1),
            recirculation_fraction=round(cfg.recirculation_fraction, 2),
            air_mass_flow_kg_s=round(m_dot, 5),
            heat_capacity_rate_kw_k=round(c_air, 5),
            inlet_air_temp_c=round(cfg.inlet_air_temp_c, 1),
            target_supply_temp_c=round(cfg.target_supply_temp_c, 1),
            achieved_supply_temp_c=round(t_supply_achieved, 1),
            exhaust_air_temp_c=round(t_exhaust, 1),
            timber_temperature_c=round(t_timber, 1),
            sand_bed_temp_c=round(sand_temperature_c, 1),
            sensible_heat_demand_kw=round(sensible_demand_kw, 3),
            ventilation_loss_kw=round(ventilation_loss_kw, 3),
            chamber_loss_kw=round(chamber_loss_kw, 3),
            latent_and_loss_demand_kw=round(latent_and_loss_kw, 3),
            evaporation_heat_demand_kw=round(evap_demand_kw, 3),
            total_process_demand_kw=round(total_demand_kw, 3),
            hx_max_available_kw=round(hx_max_available_kw, 3),
            heat_supplied_from_tes_kw=round(tes_supplied_kw, 3),
            backup_heater_power_kw=round(backup_kw, 3),
            heat_shortfall_unmet_kw=round(unmet_kw, 3),
            total_heat_delivered_kw=round(total_delivered_kw, 3),
            evaporation_heat_delivered_kw=round(evap_delivered_kw, 3),
            timber_sensible_heat_kw=round(timber_sensible_kw, 3),
            thermal_accumulation_kw=round(thermal_accumulation_kw, 3),
            thermal_residual_kw=thermal_residual_kw,
            blower_power_kw=round(blower_kw, 3),
            total_parasitic_electric_kw=round(blower_kw + backup_kw, 3),
            timber_mass_dry_kg=round(cfg.timber_mass_dry_kg, 1),
            wood_species=cfg.wood_species,
            current_moisture_percent=round(current_mc, 1) if current_mc is not None else None,
            water_removed_step_kg=round(water_removed_step_kg, 4),
            water_removed_total_kg=round(effective_cum_water if cfg.simulation_mode != DryerSimulationMode.THERMAL_TEST_RIG else 0.0, 2),
            batch_progress_percent=round(progress_pct, 1) if progress_pct is not None else None,
            estimated_remaining_hours=est_remaining_h if est_remaining_h is not None else None,
            is_batch_complete=batch_complete,
            thermal_energy_per_kg_water_kwh=sec_thermal,
            blower_interlock_active=not blower_running,
            tes_pinch_derated=tes_pinch_derated,
            power_deficit_warning=power_deficit,
            warning_message=warning_msg,
            alarms=alarms,
            provenance=DataProvenance.SIMULATED,
        )
        if mutate_batch:
            self._latest_result = res
        return res

    def restore_batch_state(
        self,
        current_moisture_percent: float | None = None,
        water_removed_cumulative_kg: float | None = None,
    ) -> None:
        """Restore active timber batch state from committed database records after restart."""
        m_dry = self.config.timber_mass_dry_kg
        if current_moisture_percent is not None:
            self.current_water_mass_kg = max(
                self.target_water_mass_kg,
                m_dry * (current_moisture_percent / 100.0),
            )
        if water_removed_cumulative_kg is not None:
            self.water_removed_cumulative_kg = max(0.0, float(water_removed_cumulative_kg))
        else:
            init_water = m_dry * (self.config.initial_moisture_percent / 100.0)
            self.water_removed_cumulative_kg = max(0.0, init_water - self.current_water_mass_kg)

    def get_status(self) -> dict[str, Any]:
        """Return serializable status dictionary of latest or default dryer step."""
        res = getattr(self, "_latest_result", None)
        if res is None:
            res = self.step(sand_temperature_c=197.0, dt_seconds=900.0)
            self._latest_result = res
        return {
            "status": res.status,
            "simulation_mode": res.simulation_mode.value,
            "mode": res.mode.value,
            "recipe_preset": res.recipe_preset.value,
            "wood_species": res.wood_species.value,
            "timber_mass_dry_kg": res.timber_mass_dry_kg,
            "target_supply_temp_c": res.target_supply_temp_c,
            "inlet_temp_c": res.inlet_air_temp_c,
            "achieved_supply_temp_c": res.achieved_supply_temp_c,
            "exhaust_temp_c": res.exhaust_air_temp_c,
            "timber_temperature_c": res.timber_temperature_c,
            "sand_bed_temp_c": res.sand_bed_temp_c,
            "airflow_m3_h": res.airflow_m3_h,
            "fresh_airflow_m3_h": res.fresh_airflow_m3_h,
            "recirculation_fraction": res.recirculation_fraction,
            "sensible_air_heat_kw": res.sensible_air_heat_kw,
            "ventilation_loss_kw": res.ventilation_loss_kw,
            "chamber_loss_kw": res.chamber_loss_kw,
            "latent_and_losses_kw": res.latent_and_losses_kw,
            "evaporation_heat_demand_kw": res.evaporation_heat_demand_kw,
            "total_heat_demand_kw": res.total_heat_demand_kw,
            "hx_power_limit_kw": res.hx_power_limit_kw,
            "useful_heat_delivered_kw": res.useful_heat_delivered_kw,
            "backup_electric_power_kw": res.backup_electric_power_kw,
            "heat_shortfall_kw": res.heat_shortfall_kw,
            "evaporation_heat_delivered_kw": res.evaporation_heat_delivered_kw,
            "timber_sensible_heat_kw": res.timber_sensible_heat_kw,
            "thermal_accumulation_kw": res.thermal_accumulation_kw,
            "thermal_residual_kw": res.thermal_residual_kw,
            "total_heat_delivered_kw": res.total_heat_delivered_kw,
            "blower_electric_power_kw": res.blower_electric_power_kw,
            "total_electricity_kw": res.total_electricity_kw,
            "current_moisture_percent": res.current_moisture_percent,
            "batch_dry_mass_kg": res.batch_dry_mass_kg,
            "batch_initial_moisture_fraction": res.batch_initial_moisture_fraction,
            "batch_target_moisture_fraction": res.batch_target_moisture_fraction,
            "batch_current_moisture_fraction": res.batch_current_moisture_fraction,
            "batch_water_removed_kg": res.batch_water_removed_kg,
            "batch_progress_percent": res.batch_progress_percent,
            "batch_est_remaining_hours": res.batch_est_remaining_hours,
            "is_batch_complete": res.is_batch_complete,
            "thermal_energy_per_kg_water_kwh": res.thermal_energy_per_kg_water_kwh,
            "blower_interlock_active": res.blower_interlock_active,
            "tes_pinch_derated": res.tes_pinch_derated,
            "power_deficit_warning": res.power_deficit_warning,
            "warning_message": res.warning_message,
            "alarms": res.alarms,
            "provenance": res.provenance.value,
        }
