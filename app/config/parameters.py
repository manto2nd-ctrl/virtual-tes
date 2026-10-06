"""Physical / economic parameter models.

These are plain, immutable Pydantic models with validation. They are independent of
where the values come from (.env, database ``tes_config`` row, API request), so the TES
model, simulator and optimizer never depend on the settings mechanism.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TESParameters(BaseModel):
    """Energy-based TES parameters.

    Conventions (see design doc, assumptions A1-A4):
      * ``max_charge_power_kw`` is ELECTRICAL heater input power.
      * ``max_discharge_power_kw`` is USEFUL HEAT delivered to the process.
      * ``capacity_kwh`` is TOTAL thermal capacity; usable window is
        [soc_min_percent, soc_max_percent] of it.
      * ``standing_loss_percent_per_day`` is relative to the currently stored energy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capacity_semantics_version: str = Field(
        "full_span_v1",
        description="Capacity semantics version identifier",
    )
    physical_temperature_min_c: float = Field(
        80.0,
        description="Physical reference minimum temperature [°C] where sensible energy = 0",
    )
    physical_temperature_max_c: float = Field(
        300.0,
        description="Physical reference maximum temperature [°C] where sensible energy = full span",
    )
    thermal_capacity_full_span_kwh: float = Field(
        15.0,
        gt=0,
        description="Total sensible thermal energy [kWh] stored across full physical temperature span",
    )
    optimizer_soc_min_fraction: float = Field(
        0.10,
        ge=0,
        le=1.0,
        description="Operational reserve floor fraction: controller does not intentionally discharge below this physical SOC fraction",
    )
    optimizer_soc_max_fraction: float = Field(
        1.00,
        ge=0,
        le=1.0,
        description="Operational ceiling fraction: controller does not charge above this physical SOC fraction",
    )

    # Legacy fields for backward compatibility (kept synchronized with canonical fields)
    capacity_kwh: float = Field(15.0, gt=0)
    soc_min_percent: float = Field(10.0, ge=0, le=100)
    soc_max_percent: float = Field(100.0, ge=0, le=100)
    initial_soc_percent: float = Field(50.0, ge=0, le=100)
    max_charge_power_kw: float = Field(9.0, ge=0)
    max_discharge_power_kw: float = Field(3.0, ge=0)
    charge_efficiency: float = Field(0.95, gt=0, le=1)
    charge_efficiency_is_provisional: bool = Field(
        True,
        description="Flag marking charge efficiency as provisional engineering assumption",
    )
    discharge_efficiency: float = Field(0.90, gt=0, le=1)
    discharge_efficiency_is_provisional: bool = Field(
        True,
        description="Flag marking discharge efficiency as provisional engineering assumption",
    )
    auxiliary_power_kw: float = Field(
        0.05,
        ge=0,
        description="TES continuous auxiliary electrical load [kW] (blowers, pumps, control PLC)",
    )
    standing_loss_percent_per_day: float = Field(2.0, ge=0, lt=100)
    standing_loss_is_provisional: bool = Field(
        True,
        description="Flag indicating standing loss model is provisional until empirical sand data is available",
    )
    standing_loss_model: str = Field(
        "provisional_fractional",
        description="Loss model: 'provisional_fractional' (% per day) or 'provisional_fixed_rate_kw'",
    )
    standing_loss_fixed_kw: float = Field(
        0.0,
        ge=0,
        description="Fixed standing loss rate [kW], active if standing_loss_model is 'provisional_fixed_rate_kw'",
    )
    hx_model_mode: str = Field(
        "fixed_gen0_hx",
        description="HX mode: 'fixed_gen0_hx' (Mode A), 'scaled_hx_benchmark' (Mode B), or 'constant'",
    )
    hx_area_m2: float = Field(1.55, gt=0, description="Heat exchanger external area [m2] (provisional)")
    overall_u_w_m2k: float = Field(9.0, gt=0, description="Overall U [W/(m2*K)], reference=9.0 (provisional)")
    airflow_m3_h: float = Field(80.0, gt=0, description="Process airflow [m3/h], reference=80.0 (provisional)")
    air_inlet_temperature_c: float = Field(40.0, description="Process return air inlet temp [°C] (provisional)")

    @model_validator(mode="before")
    @classmethod
    def _migrate_and_check_consistency(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check for conflicting capacity inputs
            if "capacity_kwh" in data and "thermal_capacity_full_span_kwh" in data:
                if abs(float(data["capacity_kwh"]) - float(data["thermal_capacity_full_span_kwh"])) > 1e-6:
                    raise ValueError(
                        f"Conflicting capacity inputs: capacity_kwh={data['capacity_kwh']} vs "
                        f"thermal_capacity_full_span_kwh={data['thermal_capacity_full_span_kwh']}. "
                        "Use canonical thermal_capacity_full_span_kwh."
                    )
            elif "capacity_kwh" in data and "thermal_capacity_full_span_kwh" not in data:
                data["thermal_capacity_full_span_kwh"] = data["capacity_kwh"]
            elif "thermal_capacity_full_span_kwh" in data and "capacity_kwh" not in data:
                data["capacity_kwh"] = data["thermal_capacity_full_span_kwh"]

            # Check for conflicting soc_min inputs
            if "soc_min_percent" in data and "optimizer_soc_min_fraction" in data:
                if abs(float(data["soc_min_percent"]) / 100.0 - float(data["optimizer_soc_min_fraction"])) > 1e-6:
                    raise ValueError(
                        f"Conflicting soc_min inputs: soc_min_percent={data['soc_min_percent']}% vs "
                        f"optimizer_soc_min_fraction={data['optimizer_soc_min_fraction']}. "
                        "Use canonical optimizer_soc_min_fraction."
                    )
            elif "soc_min_percent" in data and "optimizer_soc_min_fraction" not in data:
                data["optimizer_soc_min_fraction"] = float(data["soc_min_percent"]) / 100.0
            elif "optimizer_soc_min_fraction" in data and "soc_min_percent" not in data:
                data["soc_min_percent"] = float(data["optimizer_soc_min_fraction"]) * 100.0

            # Check for conflicting soc_max inputs
            if "soc_max_percent" in data and "optimizer_soc_max_fraction" in data:
                if abs(float(data["soc_max_percent"]) / 100.0 - float(data["optimizer_soc_max_fraction"])) > 1e-6:
                    raise ValueError(
                        f"Conflicting soc_max inputs: soc_max_percent={data['soc_max_percent']}% vs "
                        f"optimizer_soc_max_fraction={data['optimizer_soc_max_fraction']}. "
                        "Use canonical optimizer_soc_max_fraction."
                    )
            elif "soc_max_percent" in data and "optimizer_soc_max_fraction" not in data:
                data["optimizer_soc_max_fraction"] = float(data["soc_max_percent"]) / 100.0
            elif "optimizer_soc_max_fraction" in data and "soc_max_percent" not in data:
                data["soc_max_percent"] = float(data["optimizer_soc_max_fraction"]) * 100.0

        return data

    @model_validator(mode="after")
    def _validate_thermal_and_soc_bounds(self) -> "TESParameters":
        if self.physical_temperature_min_c >= self.physical_temperature_max_c:
            raise ValueError(
                f"physical_temperature_min_c ({self.physical_temperature_min_c}) must be < "
                f"physical_temperature_max_c ({self.physical_temperature_max_c})"
            )
        if self.optimizer_soc_min_fraction >= self.optimizer_soc_max_fraction:
            raise ValueError(
                f"optimizer_soc_min_fraction ({self.optimizer_soc_min_fraction}) must be < "
                f"optimizer_soc_max_fraction ({self.optimizer_soc_max_fraction})"
            )
        return self

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> "TESParameters":
        if update:
            up = dict(update)
            if "capacity_kwh" in up and "thermal_capacity_full_span_kwh" not in up:
                up["thermal_capacity_full_span_kwh"] = up["capacity_kwh"]
            elif "thermal_capacity_full_span_kwh" in up and "capacity_kwh" not in up:
                up["capacity_kwh"] = up["thermal_capacity_full_span_kwh"]
            if "soc_min_percent" in up and "optimizer_soc_min_fraction" not in up:
                up["optimizer_soc_min_fraction"] = up["soc_min_percent"] / 100.0
            elif "optimizer_soc_min_fraction" in up and "soc_min_percent" not in up:
                up["soc_min_percent"] = up["optimizer_soc_min_fraction"] * 100.0
            if "soc_max_percent" in up and "optimizer_soc_max_fraction" not in up:
                up["optimizer_soc_max_fraction"] = up["soc_max_percent"] / 100.0
            elif "optimizer_soc_max_fraction" in up and "soc_max_percent" not in up:
                up["soc_max_percent"] = up["optimizer_soc_max_fraction"] * 100.0
            return super().model_copy(update=up, deep=deep)
        return super().model_copy(update=update, deep=deep)

    @property
    def dispatchable_capacity_kwh(self) -> float:
        """Usable energy window accessible to optimizer: thermal_capacity_full_span_kwh * (optimizer_soc_max_fraction - optimizer_soc_min_fraction)."""
        return self.thermal_capacity_full_span_kwh * (self.optimizer_soc_max_fraction - self.optimizer_soc_min_fraction)

    @property
    def soc_min_energy_kwh(self) -> float:
        """Physical thermal energy stored above T_min at the operational SOC floor."""
        return self.thermal_capacity_full_span_kwh * self.optimizer_soc_min_fraction

    @property
    def soc_max_energy_kwh(self) -> float:
        """Physical thermal energy stored above T_min at the operational SOC ceiling."""
        return self.thermal_capacity_full_span_kwh * self.optimizer_soc_max_fraction

    @property
    def sand_mass_kg(self) -> float:
        """Equivalent quartz sand bed mass [kg] required for full-span thermal capacity."""
        from app.tes.thermal import ThermalStateMapper
        mapper = ThermalStateMapper(
            t_min_c=self.physical_temperature_min_c,
            t_max_c=self.physical_temperature_max_c,
        )
        return mapper.equivalent_sand_mass_for_capacity(self.thermal_capacity_full_span_kwh)

    @property
    def temperature_at_optimizer_min_soc_c(self) -> float:
        """Bulk sand temperature [°C] at the operational reserve floor (optimizer_soc_min_fraction)."""
        from app.tes.thermal import ThermalStateMapper
        mapper = ThermalStateMapper(
            t_min_c=self.physical_temperature_min_c,
            t_max_c=self.physical_temperature_max_c,
        )
        return mapper.temperature_from_soc_fraction(self.optimizer_soc_min_fraction)

    # Legacy aliases for backward compatibility
    @property
    def soc_min_kwh(self) -> float:
        return self.soc_min_energy_kwh

    @property
    def soc_max_kwh(self) -> float:
        return self.soc_max_energy_kwh

    @property
    def usable_capacity_kwh(self) -> float:
        """Deprecated legacy alias for dispatchable_capacity_kwh."""
        import warnings
        warnings.warn(
            "usable_capacity_kwh is deprecated. Use dispatchable_capacity_kwh or "
            "thermal_capacity_full_span_kwh explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dispatchable_capacity_kwh

    @property
    def initial_soc_kwh(self) -> float:
        return self.thermal_capacity_full_span_kwh * self.initial_soc_percent / 100.0

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency


class SiteParameters(BaseModel):
    """Site-level electrical and process parameters (not part of the TES itself)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grid_connection_limit_kw: float = Field(12.0, gt=0)
    other_loads_kw: float = Field(2.0, ge=0)
    process_heat_demand_kw: float = Field(1.5, ge=0)


class TariffParameters(BaseModel):
    """Variable (per-MWh) tariff components added on top of the spot price.

    Fixed monthly charges are deliberately NOT modelled here: they do not depend on
    when the TES charges and must not influence dispatch. Values are excl. VAT (A10).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    supplier_markup_eur_mwh: float = 0.0
    variable_grid_fee_eur_mwh: float = 0.0
    variable_tax_eur_mwh: float = 0.0


class Gen0PhysicalReferenceConfig(BaseModel):
    """Documented Gen0 physical reference configuration.

    All values tagged: PROVISIONAL ENGINEERING INPUT (not measured).
    Accurately represents the planned physical Gen0 hardware architecture:
      - Storage: dry quartz sand bed outside tubes/wells.
      - Heaters: top-entry cartridge heaters inserted into sealed drywells.
      - Heat Exchanger: closed helical AISI 304L tube coil carrying isolated process air.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status_tag: str = Field("PROVISIONAL ENGINEERING INPUT", description="Model calibration status")

    # Thermal storage bed
    sand_material: str = Field("quartz", description="Storage medium")
    sand_mass_kg: float = Field(310.0, description="Reference sand bed mass [kg] (300-320 kg)")
    storage_temperature_min_c: float = Field(80.0, description="Minimum operational bed temperature [°C]")
    storage_temperature_max_c: float = Field(300.0, description="Maximum operational bed temperature [°C]")

    # Electric heating stages (cartridge heaters in drywells)
    heater_count: int = Field(6, description="Number of cartridge heater elements")
    heater_power_each_kw: float = Field(1.5, description="Rated power per heater [kW]")
    heater_installed_power_kw: float = Field(9.0, description="Total installed electric heating [kW]")
    heater_architecture: str = Field(
        "top-entry cartridge heater in closed drywell",
        description="Physical heating method (isolated from direct sand contact)",
    )

    # Process air heat exchanger
    hx_material: str = Field("AISI 304L / EN 1.4307", description="Coil tube alloy")
    hx_tube_od_mm: float = Field(60.3, description="Tube outer diameter [mm]")
    hx_wall_mm: float = Field(2.0, description="Tube wall thickness [mm]")
    hx_mean_coil_diameter_mm: float = Field(420.0, description="Mean coil spiral diameter [mm]")
    hx_turns: int = Field(6, description="Number of active coil turns")
    hx_length_m: float = Field(8.2, description="Active tube length [m]")
    hx_area_m2: float = Field(1.55, description="External heat transfer area [m2]")

    # Process air operating conditions
    hx_reference_airflow_m3_h: float = Field(80.0, description="Reference process airflow [m3/h]")
    hx_reference_air_inlet_c: float = Field(40.0, description="Reference wood dryer return air inlet temp [°C]")
    hx_reference_u_w_m2k: float = Field(9.0, description="Reference overall U [W/(m2*K)] (conservative estimate)")

