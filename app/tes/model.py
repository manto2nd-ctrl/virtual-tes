"""Energy-based virtual TES model (Gen0).

Energy balance per interval (explicit Euler, losses on start-of-interval SOC):

    r          = (1 - loss_pct_per_day/100) ** (dt_h / 24)
    SOC[t+1]   = r*SOC[t] + eta_c*P_charge*dt - P_discharge/eta_d*dt
    loss[t]    = (1 - r) * SOC[t]

    P_charge    : electrical heater power [kW]
    P_discharge : useful heat delivered to the process [kW]

The model is the single authority on physical feasibility: callers pass *requested*
setpoints, and the model clips them to:
  1. heater rating and external (grid) charge limit,
  2. discharge rating,
  3. SOC_max (no overfill),
  4. SOC_min (no discharge below the minimum).

It returns what *actually* happened, including which constraint bound. A future
hardware adapter will expose the same ``state``/``step``-like contract with measured data.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from app.config.parameters import TESParameters
from app.core.timegrid import ensure_utc
from app.tes.thermal import HelicalAirHXModel, PhysicalStandingLossEstimator, ThermalStateMapper

MODEL_VERSION = "tes-energy-v1"

#: Numerical tolerance [kWh] for SOC bound checks.
EPS_KWH = 1e-9


def retention_factor(standing_loss_percent_per_day: float, dt_h: float) -> float:
    """Fraction of stored energy retained after ``dt_h`` hours of standing losses."""
    return (1.0 - standing_loss_percent_per_day / 100.0) ** (dt_h / 24.0)


class DischargeLimitCurve(ABC):
    """Pluggable curve for SOC-dependent thermal discharge power limits."""

    @abstractmethod
    def max_discharge_power_kw(self, soc_kwh: float, params: TESParameters) -> float:
        """Maximum discharge power [kW thermal] achievable at current SOC."""

    @abstractmethod
    def get_lp_upper_bounds(self, params: TESParameters) -> list[tuple[float, float]]:
        """Return list of (slope, intercept) linear upper bounds for LP optimization.
        
        Each tuple represents: P_discharge <= slope * SOC_kwh + intercept.
        """


class ConstantDischargeLimit(DischargeLimitCurve):
    """Default limit: constant rated discharge power across the entire SOC range."""

    def max_discharge_power_kw(self, soc_kwh: float, params: TESParameters) -> float:
        return params.max_discharge_power_kw

    def get_lp_upper_bounds(self, params: TESParameters) -> list[tuple[float, float]]:
        return [(0.0, params.max_discharge_power_kw)]


class LinearDeratingDischargeLimit(DischargeLimitCurve):
    """Linear derating curve below a threshold SOC.

    Models physical heat exchanger behavior where heat transfer rate decreases
    at lower sand bed temperatures (lower SOC).

    Between `soc_min` and `derate_start_soc_percent`, max discharge power derates
    linearly down to `min_power_at_soc_min_kw`.
    """

    def __init__(
        self,
        derate_start_soc_percent: float = 30.0,
        min_power_at_soc_min_kw: float = 0.5,
    ) -> None:
        if derate_start_soc_percent <= 0:
            raise ValueError("derate_start_soc_percent must be > 0")
        if min_power_at_soc_min_kw < 0:
            raise ValueError("min_power_at_soc_min_kw must be >= 0")
        self.derate_start_soc_percent = derate_start_soc_percent
        self.min_power_at_soc_min_kw = min_power_at_soc_min_kw

    def max_discharge_power_kw(self, soc_kwh: float, params: TESParameters) -> float:
        soc_pct = 100.0 * soc_kwh / params.capacity_kwh if params.capacity_kwh > 0 else 0.0
        if soc_pct >= self.derate_start_soc_percent:
            return params.max_discharge_power_kw
        if soc_pct <= params.soc_min_percent:
            return min(params.max_discharge_power_kw, self.min_power_at_soc_min_kw)

        frac = (soc_pct - params.soc_min_percent) / (
            self.derate_start_soc_percent - params.soc_min_percent
        )
        p = self.min_power_at_soc_min_kw + frac * (
            params.max_discharge_power_kw - self.min_power_at_soc_min_kw
        )
        return max(0.0, min(params.max_discharge_power_kw, p))

    def get_lp_upper_bounds(self, params: TESParameters) -> list[tuple[float, float]]:
        soc_min_kwh = (params.soc_min_percent / 100.0) * params.capacity_kwh
        soc_knee_kwh = (self.derate_start_soc_percent / 100.0) * params.capacity_kwh
        p_min = self.min_power_at_soc_min_kw
        p_max = params.max_discharge_power_kw
        bounds: list[tuple[float, float]] = [(0.0, p_max)]
        if soc_knee_kwh > soc_min_kwh:
            slope = (p_max - p_min) / (soc_knee_kwh - soc_min_kwh)
            intercept = p_min - slope * soc_min_kwh
            bounds.append((slope, intercept))
        return bounds


class InvalidDischargeCurveError(ValueError):
    """Raised when a discharge capability curve violates physical or mathematical validity rules."""


class PiecewiseLinearDischargeLimit(DischargeLimitCurve):
    """Piecewise linear discharge power limit defined by configurable breakpoints.

    Breakpoints can be given as either (soc_percent, power_kw) or (soc_kwh, power_kw).
    For concave curves (like heat exchanger derating), each line segment provides
    a valid linear upper bound: P_discharge <= slope * SOC_kwh + intercept.

    Validation rules:
      1. Breakpoints must be strictly increasing in SOC.
      2. Discharge capability must be non-negative.
      3. Discharge capability must be non-decreasing with SOC.
      4. Segment slopes must be non-increasing (concavity requirement for LP bounding).
    """

    def __init__(
        self,
        breakpoints: list[tuple[float, float]],
        use_soc_percent: bool = True,
    ) -> None:
        if len(breakpoints) < 2:
            raise InvalidDischargeCurveError("PiecewiseLinearDischargeLimit requires at least 2 breakpoints")
        self.use_soc_percent = use_soc_percent
        self.breakpoints = list(breakpoints)
        self._validate()

    def _validate(self) -> None:
        for i in range(len(self.breakpoints) - 1):
            x0, y0 = self.breakpoints[i]
            x1, y1 = self.breakpoints[i + 1]

            if x1 <= x0:
                raise InvalidDischargeCurveError(
                    f"SOC breakpoints must be strictly increasing: {x0} >= {x1} at index {i}"
                )
            if y0 < 0 or y1 < 0:
                raise InvalidDischargeCurveError(
                    f"Discharge capability must be non-negative: ({x0}, {y0}), ({x1}, {y1})"
                )
            if y1 < y0 - 1e-9:
                raise InvalidDischargeCurveError(
                    f"Discharge capability must be non-decreasing with SOC: {y0} -> {y1} from {x0} to {x1}"
                )

        # Check concavity: slopes must be non-increasing
        slopes: list[float] = []
        for i in range(len(self.breakpoints) - 1):
            x0, y0 = self.breakpoints[i]
            x1, y1 = self.breakpoints[i + 1]
            slopes.append((y1 - y0) / (x1 - x0))

        for i in range(len(slopes) - 1):
            if slopes[i + 1] > slopes[i] + 1e-7:
                raise InvalidDischargeCurveError(
                    f"Slopes must be non-increasing (concave curve required for LP bounding): "
                    f"segment {i} slope={slopes[i]:.4f} < segment {i+1} slope={slopes[i+1]:.4f}"
                )

    def _soc_to_kwh(self, soc_val: float, params: TESParameters) -> float:
        return (soc_val / 100.0) * params.capacity_kwh if self.use_soc_percent else soc_val

    def max_discharge_power_kw(self, soc_kwh: float, params: TESParameters) -> float:
        soc_val = (soc_kwh / params.capacity_kwh * 100.0) if self.use_soc_percent else soc_kwh
        if soc_val <= self.breakpoints[0][0]:
            return max(0.0, self.breakpoints[0][1])
        if soc_val >= self.breakpoints[-1][0]:
            return max(0.0, self.breakpoints[-1][1])
        for i in range(len(self.breakpoints) - 1):
            x0, y0 = self.breakpoints[i]
            x1, y1 = self.breakpoints[i + 1]
            if x0 <= soc_val <= x1:
                if abs(x1 - x0) < 1e-9:
                    return max(0.0, y1)
                frac = (soc_val - x0) / (x1 - x0)
                return max(0.0, y0 + frac * (y1 - y0))
        return params.max_discharge_power_kw

    def get_lp_upper_bounds(self, params: TESParameters) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        for i in range(len(self.breakpoints) - 1):
            x0_raw, y0 = self.breakpoints[i]
            x1_raw, y1 = self.breakpoints[i + 1]
            x0 = self._soc_to_kwh(x0_raw, params)
            x1 = self._soc_to_kwh(x1_raw, params)
            if abs(x1 - x0) < 1e-9:
                continue
            slope = (y1 - y0) / (x1 - x0)
            intercept = y0 - slope * x0
            bounds.append((slope, intercept))
        max_y = max(pt[1] for pt in self.breakpoints)
        bounds.append((0.0, max_y))
        return bounds


#: Breakpoints for Gen0 physical heat exchanger engineering estimate:
#: Sand mass 300-320 kg, T_max ~300 C, AISI 304L helical coil (1.5-1.6 m2 area, ~8.0-8.5 m active length).
#: Marked explicitly as NOT MEASURED -- ENGINEERING ESTIMATE (A11).
GEN0_ENGINEERING_ESTIMATE_V1_BREAKPOINTS: list[tuple[float, float]] = [
    (10.0, 0.50),
    (20.0, 0.80),
    (30.0, 1.05),
    (40.0, 1.30),
    (50.0, 1.55),
    (60.0, 1.75),
    (70.0, 1.95),
    (80.0, 2.10),
    (90.0, 2.20),
    (100.0, 2.30),
]
GEN0_CURVE_NAME = "gen0_engineering_estimate_v1"
GEN0_CURVE_LABEL = "LUMPED-BED HX ENGINEERING MODEL (NOT CFD, NOT MEASURED)"


def get_gen0_discharge_curve(
    mode: Literal["fixed_gen0_hx", "scaled_hx_benchmark", "normalized_benchmark", "constant"] = "fixed_gen0_hx",
    capacity_kwh: float = 15.0,
    reference_capacity_kwh: float = 15.0,
    overall_u_w_m2k: float = 9.0,
    airflow_m3_h: float = 80.0,
    air_inlet_temperature_c: float = 40.0,
    hx_area_m2: float = 1.55,
    custom_breakpoints: list[tuple[float, float]] | None = None,
) -> DischargeLimitCurve:
    """Return the Gen0 engineering-estimate discharge capability curve.

    Status: LUMPED-BED HX ENGINEERING MODEL (NOT CFD, NOT MEASURED).

    Modes:
      - 'fixed_gen0_hx' (Mode A): Fixed physical Gen0 heat exchanger (1.55 m2 coil)
        operating in quartz sand bed. Same instantaneous Pmax at the same normalized SOC%
        regardless of whether bed capacity is 10, 15, 20, or 30 kWh.
      - 'scaled_hx_benchmark' (Mode B): Benchmark where HX area scales proportionally with
        storage capacity: A_HX = A_ref * (capacity / ref_capacity).
      - 'normalized_benchmark': Normalized piecewise linear derating curve.
      - 'constant': Constant rated discharge limit.
    """
    if mode == "fixed_gen0_hx":
        return HelicalAirHXModel(
            hx_area_m2=hx_area_m2,
            overall_u_w_m2k=overall_u_w_m2k,
            airflow_m3_h=airflow_m3_h,
            air_inlet_temperature_c=air_inlet_temperature_c,
        )
    elif mode == "scaled_hx_benchmark":
        scale = capacity_kwh / reference_capacity_kwh if reference_capacity_kwh > 0 else 1.0
        return HelicalAirHXModel(
            hx_area_m2=hx_area_m2 * scale,
            overall_u_w_m2k=overall_u_w_m2k,
            airflow_m3_h=airflow_m3_h,
            air_inlet_temperature_c=air_inlet_temperature_c,
            label_override="SCALED-HX BENCHMARK ASSUMPTION",
        )
    elif mode == "normalized_benchmark":
        pts = custom_breakpoints or list(GEN0_ENGINEERING_ESTIMATE_V1_BREAKPOINTS)
        return PiecewiseLinearDischargeLimit(breakpoints=pts, use_soc_percent=True)
    elif mode == "constant":
        return ConstantDischargeLimit()
    else:
        raise ValueError(f"Unknown discharge curve mode: {mode}")


@dataclass(frozen=True, slots=True)
class TESState:
    """Snapshot of the TES at a point in time (start of the next interval)."""

    timestamp_utc: datetime
    soc_kwh: float
    soc_percent: float
    available_charge_capacity_kwh: float  # room up to SOC_max
    available_discharge_energy_kwh: float  # stored energy above SOC_min


@dataclass(frozen=True, slots=True)
class TESStepResult:
    """Realised result of one TES interval."""

    interval_start_utc: datetime
    dt_h: float
    soc_start_kwh: float
    soc_end_kwh: float
    soc_end_percent: float
    requested_charge_kw: float
    requested_discharge_kw: float
    charge_power_kw: float  # electrical
    discharge_power_kw: float  # thermal, delivered to process
    energy_stored_kwh: float  # eta_c * P_c * dt
    energy_withdrawn_kwh: float  # P_d / eta_d * dt
    storage_loss_kwh: float
    charge_conversion_loss_kwh: float
    discharge_conversion_loss_kwh: float
    charge_limited_by: tuple[str, ...] = field(default_factory=tuple)
    discharge_limited_by: tuple[str, ...] = field(default_factory=tuple)
    soc_below_min_due_to_losses: bool = False

    @property
    def storage_loss_kw(self) -> float:
        """Average standing loss power over the interval."""
        return self.storage_loss_kwh / self.dt_h

    @property
    def electricity_kwh(self) -> float:
        return self.charge_power_kw * self.dt_h

    @property
    def heat_delivered_kwh(self) -> float:
        return self.discharge_power_kw * self.dt_h


def _validate_power(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def compute_step(
    params: TESParameters,
    soc_kwh: float,
    dt_h: float,
    requested_charge_kw: float,
    requested_discharge_kw: float,
    external_charge_limit_kw: float = math.inf,
    interval_start_utc: datetime | None = None,
    discharge_limit_curve: DischargeLimitCurve | None = None,
) -> TESStepResult:
    """Pure function: compute one feasible TES step. Does not mutate anything.

    Args:
        params: TES parameters.
        soc_kwh: stored energy at the start of the interval.
        dt_h: interval length in hours.
        requested_charge_kw: requested electrical heater power.
        requested_discharge_kw: requested useful heat to the process.
        external_charge_limit_kw: extra cap on charging from outside the TES
            (e.g. grid headroom = grid limit - other site loads).
        interval_start_utc: optional timestamp for bookkeeping.
        discharge_limit_curve: optional pluggable SOC-dependent discharge limit curve.
    """
    if not (dt_h > 0 and math.isfinite(dt_h)):
        raise ValueError(f"dt_h must be positive, got {dt_h}")
    _validate_power("requested_charge_kw", requested_charge_kw)
    _validate_power("requested_discharge_kw", requested_discharge_kw)
    if math.isnan(external_charge_limit_kw):
        raise ValueError("external_charge_limit_kw must not be NaN")
    if not (-EPS_KWH <= soc_kwh <= params.capacity_kwh + EPS_KWH):
        raise ValueError(f"soc_kwh {soc_kwh} outside physical range [0, {params.capacity_kwh}]")

    eta_c = params.charge_efficiency
    eta_d = params.discharge_efficiency
    s_min = params.soc_min_kwh
    s_max = params.soc_max_kwh

    # Standing loss calculation (explicitly provisional & configurable)
    if params.standing_loss_model == "provisional_fixed_rate_kw":
        loss_fixed_kwh = min(soc_kwh, params.standing_loss_fixed_kw * dt_h)
        soc_after_loss = soc_kwh - loss_fixed_kwh
        r = (soc_after_loss / soc_kwh) if soc_kwh > EPS_KWH else 1.0
    else:
        # Default: provisional_fractional (% per day)
        r = retention_factor(params.standing_loss_percent_per_day, dt_h)
        soc_after_loss = r * soc_kwh

    charge_limits: list[str] = []
    discharge_limits: list[str] = []

    # 1. Charge power limits (heater rating, grid headroom)
    p_c = requested_charge_kw
    if p_c > params.max_charge_power_kw:
        p_c = params.max_charge_power_kw
        charge_limits.append("max_charge_power")
    ext = max(0.0, external_charge_limit_kw)
    if p_c > ext:
        p_c = ext
        charge_limits.append("grid_limit")

    # 2. Discharge power limit (heat exchanger / blower rating + SOC-dependent curve)
    curve = discharge_limit_curve or ConstantDischargeLimit()
    curve_limit = curve.max_discharge_power_kw(soc_kwh, params)
    max_d = min(params.max_discharge_power_kw, curve_limit)

    p_d = requested_discharge_kw
    if p_d > max_d:
        p_d = max_d
        if max_d < params.max_discharge_power_kw:
            discharge_limits.append("soc_dependent_discharge_limit")
        else:
            discharge_limits.append("max_discharge_power")

    # 3. No overfill: soc_after_loss + eta_c*p_c*dt - p_d/eta_d*dt <= s_max
    p_c_soc_cap = (s_max - soc_after_loss + p_d / eta_d * dt_h) / (eta_c * dt_h)
    if p_c > p_c_soc_cap:
        p_c = max(0.0, p_c_soc_cap)
        charge_limits.append("soc_max")

    # 4. No discharge below minimum: soc_after_loss + eta_c*p_c*dt - p_d/eta_d*dt >= s_min
    p_d_soc_cap = (soc_after_loss + eta_c * p_c * dt_h - s_min) * eta_d / dt_h
    if p_d > p_d_soc_cap:
        p_d = max(0.0, p_d_soc_cap)
        discharge_limits.append("soc_min")

    stored = eta_c * p_c * dt_h
    withdrawn = p_d / eta_d * dt_h
    soc_end = soc_after_loss + stored - withdrawn

    # Clean floating-point noise at the bounds.
    if abs(soc_end - s_max) < EPS_KWH:
        soc_end = s_max
    if abs(soc_end - s_min) < EPS_KWH:
        soc_end = s_min
    soc_end = max(0.0, soc_end)

    below_min_by_losses = soc_end < s_min - EPS_KWH and p_d == 0.0

    return TESStepResult(
        interval_start_utc=ensure_utc(interval_start_utc) if interval_start_utc else None,  # type: ignore[arg-type]
        dt_h=dt_h,
        soc_start_kwh=soc_kwh,
        soc_end_kwh=soc_end,
        soc_end_percent=100.0 * soc_end / params.capacity_kwh,
        requested_charge_kw=requested_charge_kw,
        requested_discharge_kw=requested_discharge_kw,
        charge_power_kw=p_c,
        discharge_power_kw=p_d,
        energy_stored_kwh=stored,
        energy_withdrawn_kwh=withdrawn,
        storage_loss_kwh=soc_kwh - soc_after_loss,
        charge_conversion_loss_kwh=(1.0 - eta_c) * p_c * dt_h,
        discharge_conversion_loss_kwh=withdrawn - p_d * dt_h,
        charge_limited_by=tuple(charge_limits),
        discharge_limited_by=tuple(discharge_limits),
        soc_below_min_due_to_losses=below_min_by_losses,
    )


class VirtualTES:
    """Stateful virtual TES. Holds SOC and advances it interval by interval."""

    def __init__(
        self,
        params: TESParameters,
        initial_soc_kwh: float | None = None,
        timestamp_utc: datetime | None = None,
        discharge_limit_curve: DischargeLimitCurve | None = None,
    ) -> None:
        self.params = params
        self.discharge_limit_curve = discharge_limit_curve or ConstantDischargeLimit()
        soc = params.initial_soc_kwh if initial_soc_kwh is None else initial_soc_kwh
        if not (0.0 <= soc <= params.capacity_kwh):
            raise ValueError(f"initial SOC {soc} kWh outside [0, {params.capacity_kwh}]")
        if soc > params.soc_max_kwh + EPS_KWH:
            raise ValueError(f"initial SOC {soc} kWh above soc_max {params.soc_max_kwh}")
        self._soc_kwh = soc
        self._timestamp = ensure_utc(timestamp_utc) if timestamp_utc else None

    @property
    def soc_kwh(self) -> float:
        return self._soc_kwh

    @property
    def soc_percent(self) -> float:
        return 100.0 * self._soc_kwh / self.params.capacity_kwh

    def state(self) -> TESState:
        p = self.params
        return TESState(
            timestamp_utc=self._timestamp,  # type: ignore[arg-type]
            soc_kwh=self._soc_kwh,
            soc_percent=self.soc_percent,
            available_charge_capacity_kwh=max(0.0, p.soc_max_kwh - self._soc_kwh),
            available_discharge_energy_kwh=max(0.0, self._soc_kwh - p.soc_min_kwh),
        )

    def step(
        self,
        interval_start_utc: datetime,
        dt_h: float,
        requested_charge_kw: float,
        requested_discharge_kw: float,
        external_charge_limit_kw: float = math.inf,
    ) -> TESStepResult:
        """Apply setpoints for one interval and advance the internal SOC."""
        result = compute_step(
            self.params,
            self._soc_kwh,
            dt_h,
            requested_charge_kw,
            requested_discharge_kw,
            external_charge_limit_kw,
            interval_start_utc,
            discharge_limit_curve=self.discharge_limit_curve,
        )
        self._soc_kwh = result.soc_end_kwh
        self._timestamp = ensure_utc(interval_start_utc) + timedelta(hours=dt_h)
        return result

