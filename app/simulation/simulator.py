"""Interval-by-interval TES simulator.

The simulator is the loop that connects inputs (prices, heat demand, other loads),
a dispatch strategy (which proposes setpoints) and the TES model (which enforces
physics). It is deliberately strategy-agnostic: the Phase 4 optimizer will plug in as a
strategy that replays an optimized schedule, and the same loop is used for backtests.

Accounting identities checked in every summary:
    heat_delivered + unmet_heat = heat_demand
    E_elec = Q_delivered + dSOC + standing_losses + charge_conv_losses + discharge_conv_losses
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import Interval
from app.economics.tariff import effective_price_eur_mwh, energy_cost_eur
from app.tes.model import EPS_KWH, DischargeLimitCurve, VirtualTES

log = logging.getLogger(__name__)

#: Values below this [kW] are reported as exactly zero (float noise).
EPS_KW = 1e-9


# --------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class SimulationInputs:
    """Time-aligned input series for a simulation horizon."""

    intervals: list[Interval]
    spot_prices_eur_mwh: list[float]
    heat_demand_kw: list[float]
    other_loads_kw: list[float]
    bidding_zone: str = "LT"
    price_source: str = "unknown"
    target_terminal_soc_kwh: float | None = None

    def __post_init__(self) -> None:
        n = len(self.intervals)
        if n == 0:
            raise ValueError("simulation needs at least one interval")
        for name in ("spot_prices_eur_mwh", "heat_demand_kw", "other_loads_kw"):
            seq = getattr(self, name)
            if len(seq) != n:
                raise ValueError(f"{name} has {len(seq)} values, expected {n}")
            if any(not math.isfinite(v) for v in seq):
                raise ValueError(f"{name} contains non-finite values")
        if any(v < 0 for v in self.heat_demand_kw) or any(v < 0 for v in self.other_loads_kw):
            raise ValueError("heat demand and other loads must be >= 0")
        for a, b in zip(self.intervals, self.intervals[1:]):
            if a.end_utc != b.start_utc:
                raise ValueError(f"intervals not contiguous at {a.end_utc} / {b.start_utc}")

    @property
    def start_utc(self) -> datetime:
        return self.intervals[0].start_utc

    @property
    def end_utc(self) -> datetime:
        return self.intervals[-1].end_utc

    def sha256(self) -> str:
        """Stable fingerprint of all inputs (used to prove identical-input reruns)."""
        blob = json.dumps({
            "intervals": [[iv.start_utc.isoformat(), iv.end_utc.isoformat()] for iv in self.intervals],
            "prices": [repr(float(x)) for x in self.spot_prices_eur_mwh],
            "heat": [repr(float(x)) for x in self.heat_demand_kw],
            "other": [repr(float(x)) for x in self.other_loads_kw],
            "zone": self.bidding_zone, "source": self.price_source,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- strategy interface
@dataclass(frozen=True, slots=True)
class StepContext:
    index: int
    interval: Interval
    soc_kwh: float
    heat_demand_kw: float
    other_loads_kw: float
    grid_headroom_kw: float
    effective_price_eur_mwh: float


@dataclass(frozen=True, slots=True)
class Setpoint:
    charge_kw: float
    discharge_kw: float


class DispatchStrategy(ABC):
    """Proposes TES setpoints. It may look ahead over the horizon in ``prepare``."""

    name: str = "abstract"

    def prepare(
        self,
        inputs: SimulationInputs,
        effective_prices: list[float],
        tes: TESParameters,
        site: SiteParameters,
        target_terminal_soc_kwh: float | None = None,
    ) -> None:
        """Called once before the run (optional)."""

    @abstractmethod
    def decide(self, ctx: StepContext) -> Setpoint:
        """Return requested setpoints for interval ``ctx.index``."""

    def describe(self) -> dict:
        return {"name": self.name}


# --------------------------------------------------------------------------- outputs
@dataclass(frozen=True, slots=True)
class SimulationRow:
    interval_start_utc: datetime
    interval_end_utc: datetime
    dt_h: float
    spot_price_eur_mwh: float
    effective_price_eur_mwh: float
    heat_demand_kw: float
    heat_delivered_kw: float
    unmet_heat_kw: float
    charge_power_kw: float
    discharge_power_kw: float
    storage_loss_kw: float
    soc_start_kwh: float
    soc_end_kwh: float
    soc_end_percent: float
    other_loads_kw: float
    grid_power_kw: float  # total site import = TES heaters + other loads
    grid_headroom_kw: float
    electricity_kwh: float  # TES heater energy only
    cost_eur: float  # TES heater cost only (other loads are not dispatch-dependent)
    energy_stored_kwh: float
    energy_withdrawn_kwh: float
    storage_loss_kwh: float
    charge_conversion_loss_kwh: float
    discharge_conversion_loss_kwh: float
    charge_limited_by: str
    discharge_limited_by: str
    soc_below_min_due_to_losses: bool


@dataclass(frozen=True)
class SimulationSummary:
    n_intervals: int
    period_start_utc: str
    period_end_utc: str
    duration_h: float
    electricity_kwh: float
    cost_eur: float
    avg_price_paid_eur_mwh: float | None
    avg_spot_price_eur_mwh: float
    heat_demand_kwh: float
    heat_delivered_kwh: float
    unmet_heat_kwh: float
    cost_per_mwh_heat_eur: float | None
    energy_stored_kwh: float
    energy_withdrawn_kwh: float
    standing_losses_kwh: float
    charge_conversion_losses_kwh: float
    discharge_conversion_losses_kwh: float
    soc_initial_kwh: float
    soc_final_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    soc_avg_kwh: float
    soc_min_percent: float
    soc_max_percent: float
    soc_avg_percent: float
    charging_intervals: int
    charging_hours: float
    equivalent_cycles: float
    max_grid_power_kw: float
    grid_limit_kw: float
    grid_limit_violations: int
    intervals_soc_below_min_by_losses: int
    energy_balance_residual_kwh: float
    target_terminal_soc_kwh: float | None = None
    terminal_soc_deficit_kwh: float | None = None
    terminal_soc_surplus_kwh: float | None = None


@dataclass(frozen=True)
class SimulationResult:
    strategy: str
    strategy_config: dict
    initial_soc_kwh: float
    rows: list[SimulationRow]
    summary: SimulationSummary


# --------------------------------------------------------------------------- simulator
class Simulator:
    def __init__(
        self,
        tes: TESParameters,
        site: SiteParameters,
        tariff: TariffParameters,
        discharge_limit_curve: DischargeLimitCurve | None = None,
    ) -> None:
        self.tes = tes
        self.site = site
        self.tariff = tariff
        self.discharge_limit_curve = discharge_limit_curve

    def run(
        self,
        inputs: SimulationInputs,
        strategy: DispatchStrategy,
        initial_soc_kwh: float | None = None,
        target_terminal_soc_kwh: float | None = None,
    ) -> SimulationResult:
        soc0 = self.tes.initial_soc_kwh if initial_soc_kwh is None else initial_soc_kwh
        tes = VirtualTES(
            self.tes,
            soc0,
            inputs.start_utc,
            discharge_limit_curve=self.discharge_limit_curve,
        )
        eff_prices = [effective_price_eur_mwh(p, self.tariff) for p in inputs.spot_prices_eur_mwh]
        target_terminal = (
            target_terminal_soc_kwh
            if target_terminal_soc_kwh is not None
            else inputs.target_terminal_soc_kwh
        )
        strategy.prepare(
            inputs,
            eff_prices,
            self.tes,
            self.site,
            target_terminal_soc_kwh=target_terminal,
        )

        rows: list[SimulationRow] = []
        for i, iv in enumerate(inputs.intervals):
            demand = inputs.heat_demand_kw[i]
            other = inputs.other_loads_kw[i]
            headroom = max(0.0, self.site.grid_connection_limit_kw - other)
            if other > self.site.grid_connection_limit_kw:
                log.warning("other site loads exceed grid limit; TES charging blocked",
                            extra={"ctx": {"interval": iv.start_utc.isoformat(), "other_kw": other}})
            ctx = StepContext(i, iv, tes.soc_kwh, demand, other, headroom, eff_prices[i])
            sp = strategy.decide(ctx)
            # Heat priority: never discharge more heat than the process demands.
            req_dis = min(sp.discharge_kw, demand)
            step = tes.step(iv.start_utc, iv.duration_h, sp.charge_kw, req_dis, headroom)

            p_c = step.charge_power_kw if step.charge_power_kw > EPS_KW else 0.0
            p_d = step.discharge_power_kw
            unmet = demand - p_d
            unmet = unmet if unmet > EPS_KW else 0.0
            rows.append(SimulationRow(
                interval_start_utc=iv.start_utc, interval_end_utc=iv.end_utc, dt_h=iv.duration_h,
                spot_price_eur_mwh=inputs.spot_prices_eur_mwh[i], effective_price_eur_mwh=eff_prices[i],
                heat_demand_kw=demand, heat_delivered_kw=p_d, unmet_heat_kw=unmet,
                charge_power_kw=p_c, discharge_power_kw=p_d, storage_loss_kw=step.storage_loss_kw,
                soc_start_kwh=step.soc_start_kwh, soc_end_kwh=step.soc_end_kwh,
                soc_end_percent=step.soc_end_percent, other_loads_kw=other, grid_power_kw=p_c + other,
                grid_headroom_kw=headroom, electricity_kwh=p_c * iv.duration_h,
                cost_eur=energy_cost_eur(p_c, iv.duration_h, eff_prices[i]),
                energy_stored_kwh=step.energy_stored_kwh, energy_withdrawn_kwh=step.energy_withdrawn_kwh,
                storage_loss_kwh=step.storage_loss_kwh,
                charge_conversion_loss_kwh=step.charge_conversion_loss_kwh,
                discharge_conversion_loss_kwh=step.discharge_conversion_loss_kwh,
                charge_limited_by=",".join(step.charge_limited_by),
                discharge_limited_by=",".join(step.discharge_limited_by),
                soc_below_min_due_to_losses=step.soc_below_min_due_to_losses,
            ))

        summary = self._summarise(inputs, rows, soc0, target_terminal_soc_kwh=target_terminal)
        return SimulationResult(strategy.name, strategy.describe(), soc0, rows, summary)

    def _summarise(
        self,
        inputs: SimulationInputs,
        rows: list[SimulationRow],
        soc0: float,
        target_terminal_soc_kwh: float | None = None,
    ) -> SimulationSummary:
        cap = self.tes.capacity_kwh
        dur = sum(r.dt_h for r in rows)
        e_el = sum(r.electricity_kwh for r in rows)
        cost = sum(r.cost_eur for r in rows)
        q_dem = sum(r.heat_demand_kw * r.dt_h for r in rows)
        q_del = sum(r.heat_delivered_kw * r.dt_h for r in rows)
        stored = sum(r.energy_stored_kwh for r in rows)
        withdrawn = sum(r.energy_withdrawn_kwh for r in rows)
        standing = sum(r.storage_loss_kwh for r in rows)
        conv_c = sum(r.charge_conversion_loss_kwh for r in rows)
        conv_d = sum(r.discharge_conversion_loss_kwh for r in rows)
        soc_trace = [soc0] + [r.soc_end_kwh for r in rows]
        soc_avg = sum(r.soc_end_kwh * r.dt_h for r in rows) / dur
        soc_final = rows[-1].soc_end_kwh
        residual = e_el - (q_del + (soc_final - soc0) + standing + conv_c + conv_d)
        limit = self.site.grid_connection_limit_kw
        usable = self.tes.dispatchable_capacity_kwh

        deficit = None
        surplus = None
        if target_terminal_soc_kwh is not None:
            deficit = max(0.0, target_terminal_soc_kwh - soc_final)
            surplus = max(0.0, soc_final - target_terminal_soc_kwh)

        return SimulationSummary(
            n_intervals=len(rows),
            period_start_utc=inputs.start_utc.isoformat(), period_end_utc=inputs.end_utc.isoformat(),
            duration_h=dur, electricity_kwh=e_el, cost_eur=cost,
            avg_price_paid_eur_mwh=(cost / e_el * 1000.0) if e_el > EPS_KWH else None,
            avg_spot_price_eur_mwh=sum(r.spot_price_eur_mwh * r.dt_h for r in rows) / dur,
            heat_demand_kwh=q_dem, heat_delivered_kwh=q_del, unmet_heat_kwh=max(0.0, q_dem - q_del),
            cost_per_mwh_heat_eur=(cost / q_del * 1000.0) if q_del > EPS_KWH else None,
            energy_stored_kwh=stored, energy_withdrawn_kwh=withdrawn, standing_losses_kwh=standing,
            charge_conversion_losses_kwh=conv_c, discharge_conversion_losses_kwh=conv_d,
            soc_initial_kwh=soc0, soc_final_kwh=soc_final,
            soc_min_kwh=min(soc_trace), soc_max_kwh=max(soc_trace), soc_avg_kwh=soc_avg,
            soc_min_percent=100 * min(soc_trace) / cap, soc_max_percent=100 * max(soc_trace) / cap,
            soc_avg_percent=100 * soc_avg / cap,
            charging_intervals=sum(1 for r in rows if r.charge_power_kw > 0),
            charging_hours=sum(r.dt_h for r in rows if r.charge_power_kw > 0),
            equivalent_cycles=withdrawn / usable if usable > 0 else 0.0,
            max_grid_power_kw=max(r.grid_power_kw for r in rows), grid_limit_kw=limit,
            grid_limit_violations=sum(1 for r in rows if r.charge_power_kw > 0 and r.grid_power_kw > limit + EPS_KW),
            intervals_soc_below_min_by_losses=sum(1 for r in rows if r.soc_below_min_due_to_losses),
            energy_balance_residual_kwh=residual,
            target_terminal_soc_kwh=target_terminal_soc_kwh,
            terminal_soc_deficit_kwh=deficit,
            terminal_soc_surplus_kwh=surplus,
        )


def summary_to_dict(summary: SimulationSummary) -> dict:
    return asdict(summary)
