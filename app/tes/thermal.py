"""Physics-informed thermal models for Virtual TES Gen0.

Contains:
  1. ``ThermalStateMapper``: Enthalpy-based mapping between normalized SOC and
     bulk quartz-sand temperature using nonlinear heat capacity Cp(T) = T_K + 427.
  2. ``PhysicalStandingLossEstimator``: Diagnostic UA-based conduction/convection
     heat loss estimator Q_loss = UA * (T_sand - T_ambient).
  3. ``HelicalAirHXModel``: Lumped-bed Heat Exchanger model for the Gen0 helical
     air coil embedded in quartz sand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.config.parameters import TESParameters


class ThermalStateMapper:
    """Enthalpy-based mapping between normalized SOC and bulk quartz-sand temperature.

    Quartz sand specific heat capacity approximation:
        Cp(T) = T_K + 427  [J/(kg*K)]
    where T_K = T_C + 273.15.

    Sensible enthalpy relative to physical reference temperature T_min (80.0 °C):
        H_rel(T) = H(T) - H(T_min)  [J/kg]
        where H(T) = 0.5 * T_K^2 + 427 * T_K.
    Only enthalpy differences (relative enthalpy) have physical meaning:
        * H_rel(80 °C) = 0 J/kg
        * H_rel(300 °C) = delta_h_usable_j_kg ≈ 195,838 J/kg
        * H_rel(10% SOC) = 0.10 * delta_h_usable_j_kg ≈ 19,583.8 J/kg

    Specific usable sensible heat between T_min and T_max:
        q_usable = [H(T_max) - H(T_min)] / 3.6e6  [kWh/kg]  (≈ 0.0544 kWh/kg)

    Normalized physical SOC fraction:
        SOC_fraction = H_rel(T) / delta_h_usable_j_kg

    Analytical inverse mapping:
        T_K = -427 + sqrt(427^2 + 2 * (H_min + H_rel))
        T_C = T_K - 273.15
    """

    def __init__(
        self,
        t_min_c: float = 80.0,
        t_max_c: float = 300.0,
        sand_material: str = "quartz",
    ) -> None:
        if t_min_c >= t_max_c:
            raise ValueError(f"t_min_c ({t_min_c}) must be < t_max_c ({t_max_c})")
        self.t_min_c = float(t_min_c)
        self.t_max_c = float(t_max_c)
        self.t_min_k = self.t_min_c + 273.15
        self.t_max_k = self.t_max_c + 273.15
        self.sand_material = sand_material

        self.h_min_j_kg = self._enthalpy_at_k(self.t_min_k)
        self.h_max_j_kg = self._enthalpy_at_k(self.t_max_k)
        self.delta_h_usable_j_kg = self.h_max_j_kg - self.h_min_j_kg
        self.specific_usable_kwh_kg = self.delta_h_usable_j_kg / 3.6e6

    @staticmethod
    def _enthalpy_at_k(t_k: float) -> float:
        """Sensible enthalpy [J/kg] from Cp(T) = T_K + 427."""
        return 0.5 * (t_k**2) + 427.0 * t_k

    @staticmethod
    def _k_from_enthalpy(h_j_kg: float) -> float:
        """Inverse mapping: temperature [K] from sensible enthalpy [J/kg]."""
        disc = 427.0**2 + 2.0 * h_j_kg
        if disc < 0:
            raise ValueError(f"Invalid negative discriminant for enthalpy {h_j_kg}")
        return -427.0 + math.sqrt(disc)

    def cp_j_kgk(self, temperature_c: float) -> float:
        """Specific heat capacity [J/(kg*K)] at temperature_c."""
        t_k = temperature_c + 273.15
        return t_k + 427.0

    def specific_stored_energy_kwh_per_kg(self) -> float:
        """Usable sensible heat storage capacity [kWh/kg] between T_min and T_max."""
        return self.specific_usable_kwh_kg

    def equivalent_sand_mass_for_capacity(self, capacity_kwh: float) -> float:
        """Calculate required quartz sand mass [kg] to provide ``capacity_kwh``."""
        if capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        return capacity_kwh / self.specific_usable_kwh_kg

    @classmethod
    def capacity_kwh_to_sand_mass_kg(cls, capacity_kwh: float, t_min_c: float = 80.0, t_max_c: float = 300.0) -> float:
        """Calculate required quartz sand mass [kg] to provide ``capacity_kwh``."""
        return cls(t_min_c=t_min_c, t_max_c=t_max_c).equivalent_sand_mass_for_capacity(capacity_kwh)

    def capacity_for_sand_mass(self, sand_mass_kg: float) -> float:
        """Calculate usable storage capacity [kWh] for ``sand_mass_kg`` of quartz sand."""
        if sand_mass_kg <= 0:
            raise ValueError("sand_mass_kg must be positive")
        return sand_mass_kg * self.specific_usable_kwh_kg

    @classmethod
    def sand_mass_kg_to_capacity_kwh(cls, sand_mass_kg: float, t_min_c: float = 80.0, t_max_c: float = 300.0) -> float:
        """Calculate usable storage capacity [kWh] for ``sand_mass_kg`` of quartz sand."""
        return cls(t_min_c=t_min_c, t_max_c=t_max_c).capacity_for_sand_mass(sand_mass_kg)

    def temperature_from_soc_fraction(self, soc_fraction: float) -> float:
        """Calculate bulk sand temperature [°C] from normalized SOC fraction in [0, 1]."""
        # Clamp to [0, 1] for physical consistency
        soc_clamped = max(0.0, min(1.0, soc_fraction))
        h_target = self.h_min_j_kg + soc_clamped * self.delta_h_usable_j_kg
        t_k = self._k_from_enthalpy(h_target)
        return t_k - 273.15

    def soc_fraction_from_temperature(self, temperature_c: float) -> float:
        """Calculate normalized SOC fraction in [0, 1] from bulk sand temperature [°C]."""
        t_k = temperature_c + 273.15
        h = self._enthalpy_at_k(t_k)
        frac = (h - self.h_min_j_kg) / self.delta_h_usable_j_kg
        return max(0.0, min(1.0, frac))

    def relative_enthalpy_j_per_kg(self, temperature_c: float) -> float:
        """Sensible enthalpy relative to T_min [J/kg]: H_rel(T) = H(T) - H(T_min)."""
        t_k = float(temperature_c) + 273.15
        return self._enthalpy_at_k(t_k) - self.h_min_j_kg

    def relative_enthalpy_from_soc_fraction(self, soc_fraction: float) -> float:
        """Relative sensible enthalpy [J/kg] at normalized physical SOC fraction in [0, 1]."""
        soc_clamped = max(0.0, min(1.0, float(soc_fraction)))
        return soc_clamped * self.delta_h_usable_j_kg

    def temperature_from_relative_enthalpy(self, h_rel_j_kg: float) -> float:
        """Inverse mapping: bulk temperature [°C] from relative sensible enthalpy [J/kg]."""
        h_clamped = max(0.0, min(self.delta_h_usable_j_kg, float(h_rel_j_kg)))
        h_target = self.h_min_j_kg + h_clamped
        t_k = self._k_from_enthalpy(h_target)
        return t_k - 273.15


class PhysicalStandingLossEstimator:
    """Diagnostic physical heat loss estimator using conduction/convection UA model.

    Q_loss = UA_loss * (T_sand - T_ambient)

    Used for physical cross-checks against provisional fractional loss assumptions.
    """

    def __init__(
        self,
        ua_loss_w_per_k: float = 0.5,
        ambient_temperature_c: float = 20.0,
    ) -> None:
        self.ua_loss_w_per_k = float(ua_loss_w_per_k)
        self.ambient_temperature_c = float(ambient_temperature_c)

    def estimate_loss_power_kw(self, t_sand_c: float) -> float:
        """Instantaneous standing heat loss rate [kW]."""
        delta_t = max(0.0, t_sand_c - self.ambient_temperature_c)
        return (self.ua_loss_w_per_k * delta_t) / 1000.0

    def estimate_equivalent_daily_loss_percent(
        self,
        t_sand_c: float,
        stored_kwh: float,
    ) -> float:
        """Equivalent daily loss percentage relative to stored energy."""
        if stored_kwh <= 0:
            return 0.0
        daily_loss_kwh = self.estimate_loss_power_kw(t_sand_c) * 24.0
        return min(100.0, (daily_loss_kwh / stored_kwh) * 100.0)


class HelicalAirHXModel:
    """Physics-informed Lumped-Bed Heat Exchanger model for helical air coil in sand.

    LABEL: LUMPED-BED HX ENGINEERING MODEL (NOT CFD, NOT MEASURED)

    Planned Gen0 Reference Geometry:
      * Material: AISI 304L / EN 1.4307
      * Tube OD: 60.3 mm, Wall: 2.0 mm, Tube ID: ~56.3 mm
      * Coil: ~6 turns, Mean coil diameter: ~420 mm
      * Tube length: ~8.0-8.5 m
      * Heat-transfer external surface: A_HX = 1.55 m2 (reference default)

    Thermal Calculation (per interval / SOC):
      1. T_sand derived from enthalpy-based SOC mapping.
      2. Air density from ideal gas law: rho_air = P / (R_air * T_air_in_K)
         where R_air ≈ 287.05 J/(kg*K).
      3. Mass flow: m_dot = (airflow_m3_h / 3600) * rho_air [kg/s].
      4. Air heat capacity rate: C_air = m_dot * Cp_air [W/K].
      5. Conductance: UA = U * A [W/K].
      6. NTU: NTU = UA / C_air.
      7. Effectiveness: eps = 1 - exp(-NTU).
      8. Max thermal power: Q_hx_max = eps * C_air * max(0, T_sand - T_air_in) [W].
      9. Convert to kW: P_max_kw = Q_hx_max / 1000.
      10. Predicted outlet temperature: T_air_out = T_air_in + Q_hx / C_air [°C].
    """

    LABEL = "LUMPED-BED HX ENGINEERING MODEL (NOT CFD, NOT MEASURED)"

    def __init__(
        self,
        hx_area_m2: float = 1.55,
        overall_u_w_m2k: float = 9.0,
        airflow_m3_h: float = 80.0,
        air_inlet_temperature_c: float = 40.0,
        air_pressure_pa: float = 101325.0,
        air_cp_j_kgk: float = 1007.0,
        thermal_state_mapper: ThermalStateMapper | None = None,
        label_override: str | None = None,
    ) -> None:
        if hx_area_m2 <= 0:
            raise ValueError(f"hx_area_m2 must be > 0, got {hx_area_m2}")
        if overall_u_w_m2k <= 0:
            raise ValueError(f"overall_u_w_m2k must be > 0, got {overall_u_w_m2k}")
        if airflow_m3_h <= 0:
            raise ValueError(f"airflow_m3_h must be > 0, got {airflow_m3_h}")

        self.hx_area_m2 = float(hx_area_m2)
        self.overall_u_w_m2k = float(overall_u_w_m2k)
        self.airflow_m3_h = float(airflow_m3_h)
        self.air_inlet_temperature_c = float(air_inlet_temperature_c)
        self.air_pressure_pa = float(air_pressure_pa)
        self.air_cp_j_kgk = float(air_cp_j_kgk)
        self.mapper = thermal_state_mapper or ThermalStateMapper()
        self.label = label_override or self.LABEL

        # Precompute fluid flow parameters at inlet
        t_in_k = self.air_inlet_temperature_c + 273.15
        r_air = 287.05  # J/(kg*K)
        self.rho_air_kg_m3 = self.air_pressure_pa / (r_air * t_in_k)
        self.mass_flow_kg_s = (self.airflow_m3_h / 3600.0) * self.rho_air_kg_m3
        self.c_air_w_k = self.mass_flow_kg_s * self.air_cp_j_kgk
        self.ua_w_k = self.overall_u_w_m2k * self.hx_area_m2
        self.ntu = self.ua_w_k / self.c_air_w_k if self.c_air_w_k > 0 else 0.0
        self.effectiveness = 1.0 - math.exp(-self.ntu)

    def p_max_at_temperature_kw(self, t_sand_c: float) -> float:
        """Instantaneous maximum thermal discharge power [kW] at bulk sand temperature."""
        delta_t = max(0.0, t_sand_c - self.air_inlet_temperature_c)
        q_w = self.effectiveness * self.c_air_w_k * delta_t
        return q_w / 1000.0

    def p_max_at_soc_fraction_kw(self, soc_fraction: float) -> float:
        """Instantaneous maximum thermal discharge power [kW] at normalized SOC fraction."""
        t_sand_c = self.mapper.temperature_from_soc_fraction(soc_fraction)
        return self.p_max_at_temperature_kw(t_sand_c)

    def calculate_outlet_temperature_c(
        self,
        t_sand_c: float,
        actual_discharge_kw: float | None = None,
    ) -> float:
        """Calculate predicted process air outlet temperature [°C]."""
        p_max_w = self.p_max_at_temperature_kw(t_sand_c) * 1000.0
        if actual_discharge_kw is not None:
            q_actual_w = min(p_max_w, max(0.0, actual_discharge_kw * 1000.0))
        else:
            q_actual_w = p_max_w
        if self.c_air_w_k <= 0:
            return self.air_inlet_temperature_c
        return self.air_inlet_temperature_c + (q_actual_w / self.c_air_w_k)

    def get_thermal_snapshot(self, soc_fraction: float) -> dict[str, float]:
        """Return full physical diagnostics for reports and sensitivity tables."""
        t_sand_c = self.mapper.temperature_from_soc_fraction(soc_fraction)
        p_max_kw = self.p_max_at_temperature_kw(t_sand_c)
        t_out_c = self.calculate_outlet_temperature_c(t_sand_c)
        return {
            "soc_fraction": soc_fraction,
            "sand_temperature_c": t_sand_c,
            "airflow_m3_h": self.airflow_m3_h,
            "overall_u_w_m2k": self.overall_u_w_m2k,
            "hx_area_m2": self.hx_area_m2,
            "ua_w_k": self.ua_w_k,
            "c_air_w_k": self.c_air_w_k,
            "ntu": self.ntu,
            "effectiveness": self.effectiveness,
            "p_max_kw": p_max_kw,
            "air_outlet_temperature_c": t_out_c,
        }

    # --- DischargeLimitCurve Interface Implementation ---

    def max_discharge_power_kw(
        self,
        soc_kwh: float,
        params: TESParameters | None = None,
        capacity_kwh: float | None = None,
    ) -> float:
        """Maximum discharge power [kW] at current SOC kWh."""
        cap = params.capacity_kwh if params is not None else (capacity_kwh or 15.0)
        p_rated = params.max_discharge_power_kw if params is not None else float("inf")
        soc_frac = soc_kwh / cap if cap > 0 else 0.0
        p_hx = self.p_max_at_soc_fraction_kw(soc_frac)
        return min(p_rated, p_hx)

    def get_lp_upper_bounds(self, params: TESParameters) -> list[tuple[float, float]]:
        """Generate piecewise linear upper bounds (slope, intercept) for LP optimizer.

        P_discharge <= slope * SOC_kwh + intercept
        """
        # Sample across SOC fractions [0.0, 0.25, 0.50, 0.75, 1.0]
        soc_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
        points: list[tuple[float, float]] = []
        for sf in soc_fracs:
            soc_kwh = sf * params.capacity_kwh
            p_kw = self.max_discharge_power_kw(soc_kwh, params)
            points.append((soc_kwh, p_kw))

        bounds: list[tuple[float, float]] = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if abs(x1 - x0) < 1e-9:
                continue
            slope = (y1 - y0) / (x1 - x0)
            intercept = y0 - slope * x0
            bounds.append((slope, intercept))

        # Overall maximum upper bound
        max_p = max(pt[1] for pt in points)
        bounds.append((0.0, max_p))
        return bounds
