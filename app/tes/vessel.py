"""Physical sand vessel and containment geometry engineering model (Phase 5.7).

Calculates internal volumes, sand fill heights, freeboard allowances, and mass-to-volume
relationships for the cylindrical sand TES containment vessel.

STATUS: PRELIMINARY GEN0 VESSEL -- NOT FOR FABRICATION
NOTE: Sand bulk density is an engineering estimate (1600 kg/m³) and MUST BE MEASURED
before vessel fabrication.
"""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

from app.tes.thermal import ThermalStateMapper


class VesselGeometryParameters(BaseModel):
    """Geometric configuration for cylindrical TES vessel."""

    internal_diameter_mm: float = Field(default=600.0, ge=100.0, description="Internal shell diameter [mm]")
    straight_shell_height_mm: float = Field(default=950.0, ge=100.0, description="Straight shell vertical height [mm]")
    sand_bulk_density_kg_m3: float = Field(default=1600.0, ge=500.0, le=3000.0, description="Provisional bulk sand density [kg/m3]")
    hx_displaced_volume_l: float = Field(default=24.0, ge=0.0, description="Displaced volume of helical heat exchanger coil [L]")
    internals_displaced_volume_l: float = Field(default=8.0, ge=0.0, description="Displaced volume of cartridge drywells and thermowells [L]")
    min_freeboard_mm: float = Field(default=80.0, ge=0.0, description="Minimum allowable expansion freeboard height [mm]")


class VesselCalculationResult(BaseModel):
    """Derived geometric and thermal results for vessel sizing."""

    # Geometry
    internal_diameter_mm: float
    straight_shell_height_mm: float
    sand_bulk_density_kg_m3: float
    gross_volume_l: float
    displaced_internals_l: float
    net_available_volume_l: float
    sand_mass_kg: float
    sand_volume_l: float
    sand_fill_height_mm: float
    freeboard_height_mm: float
    freeboard_volume_l: float
    freeboard_percent: float

    # Thermal capacity
    thermal_capacity_full_span_kwh: float
    dispatchable_capacity_kwh: float
    optimizer_soc_min_fraction: float
    optimizer_soc_max_fraction: float
    specific_storage_kwh_per_kg: float

    # Engineering diagnostics & caveats
    is_overflow: bool
    is_freeboard_insufficient: bool
    status_label: str = "PRELIMINARY GEN0 VESSEL -- NOT FOR FABRICATION"
    density_note: str = "MEASURE ACTUAL BULK DENSITY BEFORE FABRICATION"
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class VesselGeometryCalculator:
    """Engineering model for cylindrical sand TES vessel."""

    def __init__(
        self,
        params: VesselGeometryParameters | None = None,
        thermal_mapper: ThermalStateMapper | None = None,
        optimizer_soc_min_fraction: float = 0.10,
        optimizer_soc_max_fraction: float = 1.00,
    ) -> None:
        self.params = params or VesselGeometryParameters()
        self.mapper = thermal_mapper or ThermalStateMapper()
        self.optimizer_soc_min_fraction = float(optimizer_soc_min_fraction)
        self.optimizer_soc_max_fraction = float(optimizer_soc_max_fraction)

    @property
    def cross_section_area_m2(self) -> float:
        """Internal cross-sectional area of cylindrical shell [m2]."""
        d_m = self.params.internal_diameter_mm / 1000.0
        return (math.pi / 4.0) * (d_m ** 2)

    @property
    def gross_volume_liters(self) -> float:
        """Gross cylindrical volume [L]: V = pi * D^2 / 4 * H."""
        h_m = self.params.straight_shell_height_mm / 1000.0
        return self.cross_section_area_m2 * h_m * 1000.0

    @property
    def net_available_volume_liters(self) -> float:
        """Available volume after internal equipment displacement [L]."""
        disp = self.params.hx_displaced_volume_l + self.params.internals_displaced_volume_l
        return max(0.0, self.gross_volume_liters - disp)

    def calculate_for_sand_mass(self, sand_mass_kg: float) -> VesselCalculationResult:
        """Evaluate vessel fill parameters for a given quartz sand mass [kg]."""
        if sand_mass_kg <= 0:
            raise ValueError("sand_mass_kg must be positive")

        gross_v = self.gross_volume_liters
        net_v = self.net_available_volume_liters
        disp_v = self.params.hx_displaced_volume_l + self.params.internals_displaced_volume_l

        # Sand volume
        sand_vol_l = (sand_mass_kg / self.params.sand_bulk_density_kg_m3) * 1000.0

        # Fill height calculation based on net cross-section
        # Assuming internals are distributed across the vertical straight height
        net_v_per_mm = net_v / self.params.straight_shell_height_mm if self.params.straight_shell_height_mm > 0 else 1.0
        fill_height_mm = sand_vol_l / net_v_per_mm if net_v_per_mm > 0 else 0.0
        freeboard_mm = self.params.straight_shell_height_mm - fill_height_mm
        freeboard_vol_l = net_v - sand_vol_l
        freeboard_pct = (freeboard_mm / self.params.straight_shell_height_mm) * 100.0

        # Thermal capacity from canonical ThermalStateMapper
        full_span_kwh = self.mapper.capacity_for_sand_mass(sand_mass_kg)
        usable_fraction = self.optimizer_soc_max_fraction - self.optimizer_soc_min_fraction
        dispatchable_kwh = full_span_kwh * usable_fraction
        specific_kwh_kg = self.mapper.specific_stored_energy_kwh_per_kg()

        warnings_list: list[str] = []
        is_overflow = sand_vol_l > net_v
        if is_overflow:
            warnings_list.append(
                f"CRITICAL: Sand volume ({sand_vol_l:.1f} L) exceeds available net vessel volume ({net_v:.1f} L) by {sand_vol_l - net_v:.1f} L! Vessel overflow."
            )

        is_freeboard_insufficient = freeboard_mm < self.params.min_freeboard_mm
        if is_freeboard_insufficient and not is_overflow:
            warnings_list.append(
                f"WARNING: Freeboard height ({freeboard_mm:.1f} mm) is below design minimum ({self.params.min_freeboard_mm:.1f} mm)."
            )

        return VesselCalculationResult(
            internal_diameter_mm=self.params.internal_diameter_mm,
            straight_shell_height_mm=self.params.straight_shell_height_mm,
            sand_bulk_density_kg_m3=self.params.sand_bulk_density_kg_m3,
            gross_volume_l=gross_v,
            displaced_internals_l=disp_v,
            net_available_volume_l=net_v,
            sand_mass_kg=sand_mass_kg,
            sand_volume_l=sand_vol_l,
            sand_fill_height_mm=fill_height_mm,
            freeboard_height_mm=freeboard_mm,
            freeboard_volume_l=freeboard_vol_l,
            freeboard_percent=freeboard_pct,
            thermal_capacity_full_span_kwh=full_span_kwh,
            dispatchable_capacity_kwh=dispatchable_kwh,
            optimizer_soc_min_fraction=self.optimizer_soc_min_fraction,
            optimizer_soc_max_fraction=self.optimizer_soc_max_fraction,
            specific_storage_kwh_per_kg=specific_kwh_kg,
            is_overflow=is_overflow,
            is_freeboard_insufficient=is_freeboard_insufficient,
            warnings=warnings_list,
        )

    def calculate_capacity_driven(self, target_full_span_kwh: float) -> VesselCalculationResult:
        """Mode A: Capacity-driven sizing.

        Given target full-span thermal capacity [kWh], derives required sand mass
        and resulting vessel fill metrics.
        """
        if target_full_span_kwh <= 0:
            raise ValueError("target_full_span_kwh must be positive")
        required_mass_kg = self.mapper.equivalent_sand_mass_for_capacity(target_full_span_kwh)
        return self.calculate_for_sand_mass(required_mass_kg)

    def calculate_geometry_driven(self, fill_fraction: float) -> VesselCalculationResult:
        """Mode B: Geometry-driven sizing.

        Given sand fill fraction in [0, 1] relative to net available vessel volume,
        derives sand mass and resulting thermal capacity.
        """
        if not (0.0 < fill_fraction <= 1.0):
            raise ValueError("fill_fraction must be in (0, 1]")
        sand_vol_l = self.net_available_volume_liters * fill_fraction
        sand_mass_kg = (sand_vol_l / 1000.0) * self.params.sand_bulk_density_kg_m3
        return self.calculate_for_sand_mass(sand_mass_kg)

    def get_reference_scenarios(self) -> dict[str, VesselCalculationResult]:
        """Compute quick comparison scenarios for reference masses: 275.7 kg, 300 kg, 320 kg."""
        return {
            "ref_15kwh_275_7kg": self.calculate_for_sand_mass(275.7),
            "ref_300kg": self.calculate_for_sand_mass(300.0),
            "ref_320kg": self.calculate_for_sand_mass(320.0),
        }
