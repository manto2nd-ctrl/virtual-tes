"""Application service: build inputs, run a simulation, persist everything.

This is the orchestration layer used by CLIs now and by the API/scheduler later.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app import __version__
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import Interval, local_day_bounds_utc, local_day_intervals
from app.database.models import SimulationResultRecord, SimulationRun
from app.database.repositories import (
    PriceInsertReport,
    PriceRepository,
    ProfileRepository,
    RunStatusRepository,
    SimulationRepository,
)
from app.models.domain import PriceFetchResult
from app.process.profiles import PowerProfile
from app.providers.price_provider import PriceProvider, PriceProviderError
from app.simulation.simulator import (
    DispatchStrategy,
    SimulationInputs,
    SimulationResult,
    Simulator,
    summary_to_dict,
)
from app.tes.model import MODEL_VERSION, DischargeLimitCurve

log = logging.getLogger(__name__)


def align_prices(fetch: PriceFetchResult, intervals: list[Interval]) -> list[float]:
    """Map fetched price points onto the simulation grid. Missing intervals raise.

    (Phase 3 will add expansion of 60-min prices onto a 15-min grid.)
    """
    by_start = {p.delivery_start_utc: p for p in fetch.points}
    out: list[float] = []
    for iv in intervals:
        p = by_start.get(iv.start_utc)
        if p is None or p.delivery_end_utc != iv.end_utc:
            raise PriceProviderError(f"no {iv.duration_minutes}-min price for interval starting {iv.start_utc}")
        out.append(p.price_eur_mwh)
    return out


def build_day_inputs(day: date, tz: ZoneInfo, resolution_minutes: int, bidding_zone: str,
                     provider: PriceProvider, heat_profile: PowerProfile, site_profile: PowerProfile,
                     ) -> tuple[SimulationInputs, PriceFetchResult]:
    intervals = local_day_intervals(day, tz, resolution_minutes)
    start, end = local_day_bounds_utc(day, tz)
    fetch = provider.fetch_day_ahead(bidding_zone, start, end)
    inputs = SimulationInputs(
        intervals=intervals,
        spot_prices_eur_mwh=align_prices(fetch, intervals),
        heat_demand_kw=heat_profile.series(intervals),
        other_loads_kw=site_profile.series(intervals),
        bidding_zone=bidding_zone,
        price_source=provider.source,
    )
    return inputs, fetch


def config_snapshot(tes: TESParameters, site: SiteParameters, tariff: TariffParameters,
                    strategy: DispatchStrategy | dict, heat_profile: PowerProfile,
                    site_profile: PowerProfile, resolution_minutes: int, timezone: str) -> dict:
    return {
        "tes": tes.model_dump(), "site": site.model_dump(), "tariff": tariff.model_dump(),
        "strategy": strategy.describe() if isinstance(strategy, DispatchStrategy) else strategy,
        "heat_demand_profile": heat_profile.describe(), "site_load_profile": site_profile.describe(),
        "resolution_minutes": resolution_minutes, "timezone": timezone,
    }


def result_records(result: SimulationResult) -> list[SimulationResultRecord]:
    return [
        SimulationResultRecord(
            interval_start_utc=r.interval_start_utc, interval_end_utc=r.interval_end_utc,
            spot_price_eur_mwh=r.spot_price_eur_mwh, effective_price_eur_mwh=r.effective_price_eur_mwh,
            heat_demand_kw=r.heat_demand_kw, heat_delivered_kw=r.heat_delivered_kw, unmet_heat_kw=r.unmet_heat_kw,
            charge_power_kw=r.charge_power_kw, discharge_power_kw=r.discharge_power_kw,
            storage_loss_kw=r.storage_loss_kw, soc_start_kwh=r.soc_start_kwh, soc_end_kwh=r.soc_end_kwh,
            soc_end_percent=r.soc_end_percent, other_loads_kw=r.other_loads_kw, grid_power_kw=r.grid_power_kw,
            electricity_kwh=r.electricity_kwh, cost_eur=r.cost_eur,
            charge_limited_by=r.charge_limited_by, discharge_limited_by=r.discharge_limited_by,
            soc_below_min_due_to_losses=r.soc_below_min_due_to_losses,
        )
        for r in result.rows
    ]


@dataclass
class DaySimulationOutcome:
    run_id: str | None
    inputs: SimulationInputs
    result: SimulationResult
    price_report: PriceInsertReport | None


def simulate_day(
    *, day: date, tz: ZoneInfo, resolution_minutes: int, bidding_zone: str,
    tes: TESParameters, site: SiteParameters, tariff: TariffParameters,
    provider: PriceProvider, heat_profile: PowerProfile, site_profile: PowerProfile,
    strategy: DispatchStrategy, initial_soc_kwh: float | None = None,
    target_terminal_soc_kwh: float | None = None,
    discharge_limit_curve: DischargeLimitCurve | None = None,
    session: Session | None = None, note: str | None = None,
) -> DaySimulationOutcome:
    """Run a one-day simulation; if ``session`` is given, persist prices, profiles, run and status event."""
    inputs, fetch = build_day_inputs(day, tz, resolution_minutes, bidding_zone, provider,
                                     heat_profile, site_profile)
    result = Simulator(tes, site, tariff, discharge_limit_curve=discharge_limit_curve).run(
        inputs,
        strategy,
        initial_soc_kwh=initial_soc_kwh,
        target_terminal_soc_kwh=target_terminal_soc_kwh,
    )

    run_id: str | None = None
    report: PriceInsertReport | None = None
    if session is not None:
        report = PriceRepository(session, tz).store_fetch(fetch)
        profiles = ProfileRepository(session)
        profiles.store_heat_demand(heat_profile.name, inputs.intervals, inputs.heat_demand_kw)
        profiles.store_site_load(site_profile.name, inputs.intervals, inputs.other_loads_kw)
        run_id = str(uuid.uuid4())
        run = SimulationRun(
            id=run_id, run_type="single_day", strategy=result.strategy, bidding_zone=bidding_zone,
            price_source=provider.source, period_start_utc=inputs.start_utc, period_end_utc=inputs.end_utc,
            config_snapshot={**config_snapshot(tes, site, tariff, result.strategy_config, heat_profile,
                                               site_profile, resolution_minutes, str(tz)),
                             "initial_soc_kwh": result.initial_soc_kwh,
                             "target_terminal_soc_kwh": target_terminal_soc_kwh},
            inputs_sha256=inputs.sha256(), model_version=MODEL_VERSION, app_version=__version__,
            summary=summary_to_dict(result.summary), note=note,
        )
        SimulationRepository(session).add_run(run, result_records(result))
        RunStatusRepository(session).record_status(
            run_id=run_id,
            run_type="simulation",
            status="COMPLETED",
            message=f"Simulation {run_id} completed successfully",
            details={"strategy": result.strategy, "duration_h": result.summary.duration_h},
        )
        session.commit()
        log.info("simulation stored", extra={"ctx": {"run_id": run_id, "strategy": result.strategy}})
    return DaySimulationOutcome(run_id, inputs, result, report)

