"""Optimization service: orchestrates problem construction, LP solver, and persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app import __version__
from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import Interval, local_day_bounds_utc, local_day_intervals
from app.database.models import OptimizationRun, OptimizationSchedule
from app.database.repositories import (
    OptimizationRepository,
    PriceRepository,
    ProfileRepository,
    RunStatusRepository,
)
from app.economics.tariff import effective_price_eur_mwh
from app.models.domain import PricePoint
from app.optimization.domain import (
    OptimizationIntervalInput,
    OptimizationProblemInput,
    OptimizationResult,
)
from app.optimization.lp_optimizer import LPOptimizer
from app.process.profiles import PowerProfile
from app.providers.price_provider import PriceProvider, PriceProviderError
from app.tes.model import ConstantDischargeLimit, DischargeLimitCurve

log = logging.getLogger(__name__)

OPTIMIZATION_MODEL_VERSION = "tes-lp-v1"


@dataclass(frozen=True)
class OptimizationRunOutcome:
    run_id: str | None
    result: OptimizationResult
    run_record: OptimizationRun | None
    schedule_records: list[OptimizationSchedule]


def align_or_fetch_prices(
    day: date,
    tz: ZoneInfo,
    resolution_minutes: int,
    bidding_zone: str,
    intervals: list[Interval],
    provider: PriceProvider | None = None,
    session: Session | None = None,
    prices: list[PricePoint] | None = None,
) -> list[PricePoint]:
    """Retrieve prices from provided list, database, or provider for the day's intervals."""
    if prices is not None:
        points = prices
    elif session is not None:
        start_utc, end_utc = local_day_bounds_utc(day, tz)
        repo = PriceRepository(session, tz)
        db_prices = repo.get_prices(bidding_zone, start_utc, end_utc, latest_only=True)
        points = [
            PricePoint(
                bidding_zone=p.bidding_zone,
                delivery_start_utc=p.delivery_start_utc,
                delivery_end_utc=p.delivery_end_utc,
                price_eur_mwh=p.price_eur_mwh,
                resolution_minutes=p.resolution_minutes,
                source=p.source,
                published_at=p.published_at,
                currency=p.currency,
            )
            for p in db_prices
        ]
    elif provider is not None:
        start_utc, end_utc = local_day_bounds_utc(day, tz)
        fetch = provider.fetch_day_ahead(bidding_zone, start_utc, end_utc)
        points = fetch.points
    else:
        raise ValueError("Must provide either prices, session, or provider.")

    by_start = {p.delivery_start_utc: p for p in points}
    aligned: list[PricePoint] = []
    for iv in intervals:
        p = by_start.get(iv.start_utc)
        if p is None:
            # Check for 60-min price covering this 15-min interval
            # Start of the enclosing hour:
            hour_start = iv.start_utc.replace(minute=0, second=0, microsecond=0)
            p_hour = by_start.get(hour_start)
            if p_hour is not None and p_hour.resolution_minutes == 60:
                p = PricePoint(
                    bidding_zone=p_hour.bidding_zone,
                    delivery_start_utc=iv.start_utc,
                    delivery_end_utc=iv.end_utc,
                    price_eur_mwh=p_hour.price_eur_mwh,
                    resolution_minutes=resolution_minutes,
                    source=p_hour.source,
                    currency=p_hour.currency,
                )
            else:
                raise PriceProviderError(f"Missing price for interval starting {iv.start_utc.isoformat()}")
        aligned.append(p)
    return aligned


def optimize_day(
    day: date,
    tz: ZoneInfo,
    resolution_minutes: int,
    bidding_zone: str,
    tes: TESParameters,
    site: SiteParameters,
    tariff: TariffParameters,
    heat_profile: PowerProfile,
    site_profile: PowerProfile,
    initial_soc_kwh: float,
    target_terminal_soc_kwh: float | None = None,
    terminal_soc_condition: str = "exact",
    discharge_limit_curve: DischargeLimitCurve | None = None,
    provider: PriceProvider | None = None,
    prices: list[PricePoint] | None = None,
    session: Session | None = None,
    note: str | None = None,
    unmet_heat_penalty_eur_per_kwh: float = 10000.0,
    throughput_penalty_eur_per_kwh: float = 1e-5,
) -> OptimizationRunOutcome:
    """End-to-end day-ahead dispatch optimization service."""
    intervals = local_day_intervals(day, tz, resolution_minutes)
    horizon_start_utc, horizon_end_utc = local_day_bounds_utc(day, tz)

    aligned_points = align_or_fetch_prices(
        day=day,
        tz=tz,
        resolution_minutes=resolution_minutes,
        bidding_zone=bidding_zone,
        intervals=intervals,
        provider=provider,
        session=session,
        prices=prices,
    )

    heat_demands = heat_profile.series(intervals)
    site_loads = site_profile.series(intervals)

    opt_inputs: list[OptimizationIntervalInput] = []
    for iv, p, hd, sl in zip(intervals, aligned_points, heat_demands, site_loads, strict=True):
        eff_price = effective_price_eur_mwh(p.price_eur_mwh, tariff)
        opt_inputs.append(
            OptimizationIntervalInput(
                start_utc=iv.start_utc,
                end_utc=iv.end_utc,
                spot_price_eur_mwh=p.price_eur_mwh,
                effective_price_eur_mwh=eff_price,
                heat_demand_kw=hd,
                other_site_load_kw=sl,
                auxiliary_load_kw=0.0,
            )
        )

    curve = discharge_limit_curve or ConstantDischargeLimit()

    problem = OptimizationProblemInput(
        intervals=opt_inputs,
        tes_params=tes,
        grid_connection_limit_kw=site.grid_connection_limit_kw,
        initial_soc_kwh=initial_soc_kwh,
        target_terminal_soc_kwh=target_terminal_soc_kwh,
        terminal_soc_condition=terminal_soc_condition,  # type: ignore
        discharge_limit_curve=curve,
        unmet_heat_penalty_eur_per_kwh=unmet_heat_penalty_eur_per_kwh,
        throughput_penalty_eur_per_kwh=throughput_penalty_eur_per_kwh,
    )

    optimizer = LPOptimizer()
    result = optimizer.optimize(problem)

    run_id: str | None = None
    run_record: OptimizationRun | None = None
    schedule_records: list[OptimizationSchedule] = []

    if session is not None:
        run_id = str(uuid.uuid4())
        status_repo = RunStatusRepository(session)
        status_repo.record_status(
            run_id=run_id,
            run_type="optimization",
            status="STARTED",
            message=f"Starting optimization for {day.isoformat()}",
        )

        cfg_dict = {
            "tes": tes.model_dump(),
            "site": site.model_dump(),
            "tariff": tariff.model_dump(),
            "heat_profile": heat_profile.describe(),
            "site_profile": site_profile.describe(),
            "resolution_minutes": resolution_minutes,
            "timezone": str(tz),
            "target_terminal_soc_kwh": problem.resolved_target_terminal_soc_kwh(),
            "terminal_soc_condition": terminal_soc_condition,
        }

        inputs_hash = hashlib.sha256(
            json.dumps(
                {
                    "prices": [p.price_eur_mwh for p in aligned_points],
                    "heat": heat_demands,
                    "site": site_loads,
                    "initial_soc": initial_soc_kwh,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        run_record = OptimizationRun(
            id=run_id,
            horizon_start_utc=horizon_start_utc,
            horizon_end_utc=horizon_end_utc,
            bidding_zone=bidding_zone,
            config_snapshot=cfg_dict,
            inputs_sha256=inputs_hash,
            model_version=OPTIMIZATION_MODEL_VERSION,
            solver=result.solver,
            status=result.status,
            objective_eur=result.objective_eur,
            initial_soc_kwh=initial_soc_kwh,
            summary=result.metrics.to_dict(),
        )

        for iv_res in result.intervals:
            schedule_records.append(
                OptimizationSchedule(
                    run_id=run_id,
                    interval_start_utc=iv_res.start_utc,
                    interval_end_utc=iv_res.end_utc,
                    spot_price_eur_mwh=iv_res.spot_price_eur_mwh,
                    effective_price_eur_mwh=iv_res.effective_price_eur_mwh,
                    charge_power_kw=iv_res.charge_power_kw,
                    discharge_power_kw=iv_res.discharge_power_kw,
                    unmet_heat_kw=iv_res.unmet_heat_kw,
                    predicted_soc_kwh=iv_res.soc_kwh,
                    predicted_soc_percent=iv_res.soc_percent,
                    heat_demand_kw=iv_res.heat_demand_kw,
                    other_site_load_kw=iv_res.other_site_load_kw,
                    auxiliary_load_kw=iv_res.auxiliary_load_kw,
                    grid_power_kw=iv_res.grid_power_kw,
                    cost_eur=iv_res.cost_eur,
                )
            )

        opt_repo = OptimizationRepository(session)
        opt_repo.add_run(run_record, schedule_records)

        # Profile persistence
        prof_repo = ProfileRepository(session)
        prof_repo.store_heat_demand(heat_profile.name, intervals, heat_demands)
        prof_repo.store_site_load(site_profile.name, intervals, site_loads)

        status_repo.record_status(
            run_id=run_id,
            run_type="optimization",
            status="COMPLETED",
            message=f"Optimal schedule generated. Electricity cost: {result.metrics.electricity_cost_eur:.2f} EUR",
            details=result.metrics.to_dict(),
        )

        session.commit()

    return OptimizationRunOutcome(
        run_id=run_id,
        result=result,
        run_record=run_record,
        schedule_records=schedule_records,
    )
