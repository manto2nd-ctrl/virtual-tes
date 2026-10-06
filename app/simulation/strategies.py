"""Simple (non-optimizing) dispatch strategies for Phase 2.

These are NOT the optimizer. They exist to exercise the simulator and will serve as
baselines in Phase 5:

* ``HeatFollowingStrategy`` - TES without arbitrage: always delivers the demand and
  charges only the minimum needed to keep SOC >= SOC_min ("just in time").
* ``CheapestIntervalsStrategy`` - charge at full available power in the N cheapest
  intervals of the horizon; otherwise charge only the minimum needed to keep SOC_min.
"""

from __future__ import annotations

import math

from app.config.parameters import SiteParameters, TESParameters
from app.simulation.simulator import DispatchStrategy, Setpoint, SimulationInputs, StepContext
from app.tes.model import retention_factor

#: Small safety margin [kWh] so float rounding never clips the heat delivery.
SAFETY_MARGIN_KWH = 1e-9


def min_charge_to_hold_soc_min(tes: TESParameters, soc_kwh: float, dt_h: float, discharge_kw: float) -> float:
    """Smallest electrical charge power [kW] such that SOC_end >= SOC_min after
    delivering ``discharge_kw`` during ``dt_h`` (including standing losses)."""
    r = retention_factor(tes.standing_loss_percent_per_day, dt_h)
    soc_end_no_charge = r * soc_kwh - discharge_kw / tes.discharge_efficiency * dt_h
    deficit = tes.soc_min_kwh + SAFETY_MARGIN_KWH - soc_end_no_charge
    return max(0.0, deficit) / (tes.charge_efficiency * dt_h)


class HeatFollowingStrategy(DispatchStrategy):
    name = "heat_following"

    def prepare(self, inputs, effective_prices, tes, site, target_terminal_soc_kwh: float | None = None) -> None:
        self._tes = tes

    def decide(self, ctx: StepContext) -> Setpoint:
        dis = ctx.heat_demand_kw
        chg = min_charge_to_hold_soc_min(self._tes, ctx.soc_kwh, ctx.interval.duration_h, dis)
        return Setpoint(charge_kw=chg, discharge_kw=dis)


class CheapestIntervalsStrategy(DispatchStrategy):
    """Charge at max power during the N cheapest intervals (by effective price).

    If ``n_intervals`` is None it is derived from the energy need of the horizon:
        N = ceil( sum(demand*dt) / (eta_c*eta_d) / (P_charge_max * mean_dt) )
    If ``target_terminal_soc_kwh`` is supplied and greater than initial SOC,
    the deficit is added to the required electrical charging budget.
    Ties are broken by time (earlier first) to keep results deterministic.
    """

    name = "cheapest_n"

    def __init__(self, n_intervals: int | None = None) -> None:
        if n_intervals is not None and n_intervals < 0:
            raise ValueError("n_intervals must be >= 0")
        self.n_requested = n_intervals
        self.n_used: int | None = None
        self._cheap: set[int] = set()

    def prepare(
        self,
        inputs: SimulationInputs,
        effective_prices: list[float],
        tes: TESParameters,
        site: SiteParameters,
        target_terminal_soc_kwh: float | None = None,
    ) -> None:
        self._tes = tes
        n = self.n_requested
        if n is None:
            heat_kwh = sum(d * iv.duration_h for d, iv in zip(inputs.heat_demand_kw, inputs.intervals))
            elec_kwh = heat_kwh / tes.round_trip_efficiency
            if target_terminal_soc_kwh is not None:
                soc0 = tes.initial_soc_kwh
                if target_terminal_soc_kwh > soc0:
                    elec_kwh += (target_terminal_soc_kwh - soc0) / tes.charge_efficiency
            mean_dt = sum(iv.duration_h for iv in inputs.intervals) / len(inputs.intervals)
            n = math.ceil(elec_kwh / (tes.max_charge_power_kw * mean_dt)) if tes.max_charge_power_kw > 0 else 0
        n = min(n, len(inputs.intervals))
        order = sorted(range(len(effective_prices)), key=lambda i: (effective_prices[i], i))
        self._cheap = set(order[:n])
        self.n_used = n

    def is_cheap(self, index: int) -> bool:
        return index in self._cheap

    def decide(self, ctx: StepContext) -> Setpoint:
        dis = ctx.heat_demand_kw
        if ctx.index in self._cheap:
            chg = self._tes.max_charge_power_kw  # TES model clips to grid headroom / SOC_max
        else:
            chg = min_charge_to_hold_soc_min(self._tes, ctx.soc_kwh, ctx.interval.duration_h, dis)
        return Setpoint(charge_kw=chg, discharge_kw=dis)

    def describe(self) -> dict:
        return {"name": self.name, "n_requested": self.n_requested, "n_used": self.n_used}
