"""Market Data Service: provider hierarchy, cross-validation, and shadow mode.

Phase 5.8 Architecture:
1. PRIMARY: LitgridPriceProvider (Nord Pool Lietuva)
2. SECONDARY / VALIDATION: EleringPriceProvider (LT series)
3. TERTIARY FALLBACK: VoltonPriceProvider (Lithuania day-ahead spot)
4. QUATERNARY: EntsoePriceProvider

Features:
- Cross-source price validation (<= 0.01 EUR/MWh tolerance; flags MARKET_DATA_SOURCE_MISMATCH)
- Strict no-silent-mock-fallback rule (raises / flags LIVE MARKET DATA UNAVAILABLE)
- Stale data retention and flagging
- Source provenance preservation for optimizer
- Real data shadow TES operation (zero hardware commands)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import DayCoverageResult, ensure_utc, local_day_bounds_utc, validate_local_market_day_coverage
from app.database.repositories import PriceRepository
from app.economics.tariff import effective_price_eur_mwh
from app.models.domain import MarketPriceInterval, PriceFetchResult, PricePoint
from app.optimization.domain import (
    OptimizationIntervalInput,
    OptimizationProblemInput,
    OptimizationResult,
)
from app.optimization.lp_optimizer import LPOptimizer
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.elering import EleringPriceProvider
from app.providers.entsoe import EntsoePriceProvider
from app.providers.litgrid import LitgridPriceProvider, expand_intervals_to_resolution
from app.providers.price_provider import PriceProvider, PriceProviderError
from app.providers.volton import VoltonPriceProvider
from app.tes.heater_staging import IdealizedContinuousHeaterModel
from app.tes.model import PiecewiseLinearDischargeLimit, VirtualTES, get_gen0_discharge_curve
from app.tes.thermal import ThermalStateMapper

log = logging.getLogger(__name__)

CROSS_SOURCE_TOLERANCE_EUR_MWH = 0.01


class LiveMarketDataUnavailableError(RuntimeError):
    """Raised when all live market data sources fail and no mock fallback is allowed."""


@dataclass(frozen=True, slots=True)
class CrossSourceValidationReport:
    """Outcome of comparing Litgrid vs Elering day-ahead price observations."""

    status: Literal["MATCH", "MISMATCH", "NOT CHECKED"]
    intervals_checked: int
    matching_intervals: int
    mismatched_intervals: int
    max_absolute_diff_eur_mwh: float
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    flag: str | None = None  # Set to "MARKET_DATA_SOURCE_MISMATCH" when status == "MISMATCH"


@dataclass(slots=True)
class MarketStatusSnapshot:
    """Dashboard-ready status summary of the market data subsystem."""

    primary_source: str = "LITGRID"
    primary_health: str = "OK"  # "OK" | "DEGRADED" | "ERROR"
    primary_error: str | None = None
    primary_api_reachable: bool = True
    primary_current_available: bool = True
    primary_tomorrow_coverage: str = "NOT_YET_PUBLISHED"  # "COMPLETE" | "PARTIAL" | "NOT_YET_PUBLISHED"
    primary_overall_status: str = "HEALTHY"  # "HEALTHY" | "DEGRADED" | "UNAVAILABLE"

    cross_check_source: str = "ELERING"
    cross_check_health: str = "OK"  # "OK" | "ACTIVE FALLBACK" | "DEGRADED" | "ERROR"
    cross_check_error: str | None = None
    cross_check_api_reachable: bool = True
    cross_check_current_available: bool = True
    cross_check_tomorrow_coverage: str = "NOT_YET_PUBLISHED"  # "COMPLETE" | "PARTIAL" | "NOT_YET_PUBLISHED"
    cross_check_overall_status: str = "HEALTHY"  # "HEALTHY" | "DEGRADED" | "ACTIVE_FALLBACK" | "UNAVAILABLE"

    fallback_used: str = "NO"  # "YES" | "NO"
    fallback_reason: str | None = None
    optimization_price_eur_mwh: float | None = None
    optimization_price_basis: str = "EFFECTIVE_VARIABLE_PRICE"

    latest_interval_utc: datetime | None = None
    latest_interval_str: str = ""
    tomorrow_prices: str = "WAITING"  # "AVAILABLE" | "WAITING"
    resolution: str = "15 min"
    validation: str = "NOT CHECKED"  # "MATCH" | "MISMATCH" | "NOT CHECKED"
    data_freshness: str = "LIVE"  # "LIVE" | "STALE DATA" | "LIVE MARKET DATA UNAVAILABLE"
    active_source: str = "LITGRID"
    intervals_count: int = 0
    raw_payload_hash: str | None = None
    validation_report: CrossSourceValidationReport | None = None


@dataclass(frozen=True)
class HierarchicalFetchResult:
    """Result of multi-provider hierarchical fetch."""

    intervals: list[MarketPriceInterval]
    active_source: str
    is_stale: bool
    status_snapshot: MarketStatusSnapshot
    validation_report: CrossSourceValidationReport
    litgrid_result: PriceFetchResult | None = None
    elering_result: PriceFetchResult | None = None
    fallback_result: PriceFetchResult | None = None


@dataclass(frozen=True)
class ShadowOperationResult:
    """Result of virtual TES shadow operation driven by real Lithuanian market data."""

    status_label: str  # "REAL LT MARKET DATA + VIRTUAL TES OPERATION"
    market_status: MarketStatusSnapshot
    active_source: str
    optimization_result: OptimizationResult
    timeline_rows: list[dict[str, Any]]
    chart_data: list[dict[str, Any]]
    total_cost_eur: float
    total_charge_kwh: float
    total_heat_kwh: float
    max_grid_power_kw: float


class MarketDataService:
    """Orchestrates Lithuanian provider hierarchy, cross-validation, and shadow dispatch."""

    def __init__(
        self,
        litgrid_provider: PriceProvider | None = None,
        elering_provider: PriceProvider | None = None,
        volton_provider: PriceProvider | None = None,
        entsoe_provider: PriceProvider | None = None,
        tz: ZoneInfo = ZoneInfo("Europe/Vilnius"),
    ) -> None:
        self.litgrid = litgrid_provider or LitgridPriceProvider()
        self.elering = elering_provider or EleringPriceProvider()
        self.volton = volton_provider or VoltonPriceProvider()
        self.entsoe = entsoe_provider or EntsoePriceProvider()
        self.tz = tz

    def validate_cross_source(
        self,
        litgrid_intervals: list[MarketPriceInterval],
        elering_intervals: list[MarketPriceInterval],
        tolerance_eur_mwh: float = CROSS_SOURCE_TOLERANCE_EUR_MWH,
    ) -> CrossSourceValidationReport:
        """Compare Litgrid and Elering prices for overlapping intervals."""
        if not litgrid_intervals or not elering_intervals:
            return CrossSourceValidationReport(
                status="NOT CHECKED",
                intervals_checked=0,
                matching_intervals=0,
                mismatched_intervals=0,
                max_absolute_diff_eur_mwh=0.0,
            )

        ele_by_start = {iv.start_utc: iv for iv in elering_intervals}
        checked = 0
        matching = 0
        mismatched = 0
        max_diff = 0.0
        mismatches: list[dict[str, Any]] = []

        for lg in litgrid_intervals:
            el = ele_by_start.get(lg.start_utc)
            if el is None:
                continue

            checked += 1
            diff = abs(lg.price_eur_mwh - el.price_eur_mwh)
            if diff > max_diff:
                max_diff = diff

            if diff <= tolerance_eur_mwh:
                matching += 1
            else:
                mismatched += 1
                mismatches.append({
                    "start_utc": lg.start_utc.isoformat(),
                    "litgrid_price_eur_mwh": lg.price_eur_mwh,
                    "elering_price_eur_mwh": el.price_eur_mwh,
                    "abs_diff_eur_mwh": round(diff, 4),
                })

        if checked == 0:
            return CrossSourceValidationReport(
                status="NOT CHECKED",
                intervals_checked=0,
                matching_intervals=0,
                mismatched_intervals=0,
                max_absolute_diff_eur_mwh=0.0,
            )

        if mismatched > 0:
            log.warning(
                "MARKET_DATA_SOURCE_MISMATCH: %d intervals differ beyond tolerance %.2f EUR/MWh (max diff %.4f)",
                mismatched,
                tolerance_eur_mwh,
                max_diff,
            )
            return CrossSourceValidationReport(
                status="MISMATCH",
                intervals_checked=checked,
                matching_intervals=matching,
                mismatched_intervals=mismatched,
                max_absolute_diff_eur_mwh=round(max_diff, 4),
                mismatches=mismatches,
                flag="MARKET_DATA_SOURCE_MISMATCH",
            )

        return CrossSourceValidationReport(
            status="MATCH",
            intervals_checked=checked,
            matching_intervals=matching,
            mismatched_intervals=0,
            max_absolute_diff_eur_mwh=round(max_diff, 4),
        )

    def fetch_market_data(
        self,
        bidding_zone: str = "LT",
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        target_resolution_minutes: int = 15,
        session: Session | None = None,
    ) -> HierarchicalFetchResult:
        """Fetch market data following the strict provider hierarchy.

        Hierarchy:
          1. LITGRID (Primary)
          2. ELERING (Secondary / Fallback & Cross-check)
          3. VOLTON (Tertiary fallback)
          4. ENTSOE (Quaternary fallback)

        Enforces:
        - Cross-source validation if both Litgrid and Elering succeed
        - Strict no silent mock fallback
        - Stale data retention and flagging if live providers fail
        """
        now_utc = datetime.now(timezone.utc)
        if start_utc is None:
            # Default to last 24h through tomorrow
            s_date = datetime.now(self.tz).date()
            start_utc, _ = local_day_bounds_utc(s_date, self.tz)
        if end_utc is None:
            e_date = datetime.now(self.tz).date() + timedelta(days=2)
            _, end_utc = local_day_bounds_utc(e_date, self.tz)

        start_utc = ensure_utc(start_utc)
        end_utc = ensure_utc(end_utc)

        litgrid_res: PriceFetchResult | None = None
        elering_res: PriceFetchResult | None = None
        volton_res: PriceFetchResult | None = None
        entsoe_res: PriceFetchResult | None = None

        primary_health = "OK"
        primary_err: str | None = None
        cross_health = "OK"
        cross_err: str | None = None

        # 1. Primary: Litgrid
        try:
            litgrid_res = self.litgrid.fetch_day_ahead(bidding_zone, start_utc, end_utc)
            if session is not None:
                repo = PriceRepository(session, self.tz)
                repo.store_fetch(litgrid_res)
                session.commit()
        except Exception as exc:
            primary_health = "ERROR"
            primary_err = str(exc)
            log.warning("Litgrid fetch failed: %s", exc)

        # 2. Secondary / Cross-check: Elering
        try:
            elering_res = self.elering.fetch_day_ahead(bidding_zone, start_utc, end_utc)
            if session is not None:
                repo = PriceRepository(session, self.tz)
                repo.store_fetch(elering_res)
                session.commit()
        except Exception as exc:
            cross_health = "ERROR"
            cross_err = str(exc)
            log.warning("Elering fetch failed: %s", exc)

        # Cross-source validation if both succeeded
        lg_intervals = litgrid_res.market_intervals if litgrid_res else []
        el_intervals = elering_res.market_intervals if elering_res else []
        lg_comp = expand_intervals_to_resolution(lg_intervals, target_resolution_minutes)
        el_comp = expand_intervals_to_resolution(el_intervals, target_resolution_minutes)
        val_report = self.validate_cross_source(lg_comp, el_comp)

        # Determine active data based on hierarchy
        active_intervals: list[MarketPriceInterval] = []
        active_source = "LITGRID"
        fallback_res: PriceFetchResult | None = None
        is_stale = False
        freshness = "LIVE"

        if litgrid_res and litgrid_res.market_intervals:
            active_intervals = litgrid_res.market_intervals
            active_source = "LITGRID"
        elif elering_res and elering_res.market_intervals:
            active_intervals = elering_res.market_intervals
            active_source = "ELERING"
            fallback_res = elering_res
            log.info("Litgrid unavailable; using Elering fallback prices")
        else:
            # Try Volton
            try:
                volton_res = self.volton.fetch_day_ahead(bidding_zone, start_utc, end_utc)
                if session is not None:
                    repo = PriceRepository(session, self.tz)
                    repo.store_fetch(volton_res)
                    session.commit()
                active_intervals = volton_res.market_intervals
                active_source = "VOLTON"
                fallback_res = volton_res
                log.info("Litgrid and Elering unavailable; using Volton fallback prices")
            except Exception as v_exc:
                log.warning("Volton fetch failed: %s", v_exc)
                # Try ENTSO-E
                try:
                    entsoe_res = self.entsoe.fetch_day_ahead(bidding_zone, start_utc, end_utc)
                    if session is not None:
                        repo = PriceRepository(session, self.tz)
                        repo.store_fetch(entsoe_res)
                        session.commit()
                    active_intervals = [
                        MarketPriceInterval.from_price_point(pt, raw_payload_hash=entsoe_res.raw_payload)
                        for pt in entsoe_res.points
                    ]
                    active_source = "ENTSOE"
                    fallback_res = entsoe_res
                    log.info("Litgrid, Elering, Volton unavailable; using ENTSO-E fallback prices")
                except Exception as ent_exc:
                    log.warning("ENTSO-E fetch failed: %s", ent_exc)

        # If ALL live providers failed:
        if not active_intervals:
            # Check DB for stale cached data
            if session is not None:
                repo = PriceRepository(session, self.tz)
                db_prices = repo.get_prices(bidding_zone, start_utc, end_utc, latest_only=True)
                if db_prices:
                    active_intervals = [
                        MarketPriceInterval(
                            start_utc=p.delivery_start_utc,
                            end_utc=p.delivery_end_utc,
                            price_eur_mwh=p.price_eur_mwh,
                            bidding_zone=p.bidding_zone,
                            source=p.source,
                            original_resolution_minutes=p.resolution_minutes,
                            fetched_at_utc=p.fetched_at,
                            source_record_id=f"db-stale-{p.id}",
                            is_derived=False,
                        )
                        for p in db_prices
                    ]
                    active_source = db_prices[0].source
                    is_stale = True
                    freshness = "STALE DATA"
                    log.warning("ALL live providers failed. Retaining last valid dataset with STALE DATA status.")

            if not active_intervals:
                # Strictly reject mock fallback
                freshness = "LIVE MARKET DATA UNAVAILABLE"
                raise LiveMarketDataUnavailableError(
                    "LIVE MARKET DATA UNAVAILABLE: All live market data providers failed (no mock fallback allowed)"
                )

        # Standardize resolution (e.g. expand 60m to 15m; preserve 15m intact)
        expanded_intervals = expand_intervals_to_resolution(active_intervals, target_resolution_minutes)

        # Analyze latest interval and tomorrow availability
        latest_interval = max((iv.end_utc for iv in expanded_intervals), default=None)
        latest_str = latest_interval.strftime("%Y-%m-%d %H:%M UTC") if latest_interval else "None"

        # Check current interval coverage and tomorrow coverage across providers (Phase 5.8.3)
        cur_now_utc = datetime.now(timezone.utc)
        tomorrow_date = datetime.now(self.tz).date() + timedelta(days=1)
        is_live_query = (start_utc <= cur_now_utc < end_utc)

        # 1. Litgrid detailed status
        lg_api_reachable = (litgrid_res is not None and primary_err is None)
        lg_has_curr = False
        lg_tom_status = "NOT_YET_PUBLISHED"
        if litgrid_res and litgrid_res.market_intervals:
            if is_live_query:
                lg_has_curr = select_current_market_interval(litgrid_res.market_intervals, cur_now_utc) is not None
            else:
                lg_has_curr = len(litgrid_res.market_intervals) > 0
            lg_tom_cov = validate_local_market_day_coverage(tomorrow_date, self.tz, litgrid_res.market_intervals, target_resolution_minutes)
            lg_tom_status = "COMPLETE" if lg_tom_cov.is_complete else ("PARTIAL" if lg_tom_cov.received_intervals > 0 else "NOT_YET_PUBLISHED")

        if not lg_api_reachable:
            lg_overall = "UNAVAILABLE"
            primary_health = "ERROR"
        elif not lg_has_curr:
            lg_overall = "DEGRADED"
            primary_health = "DEGRADED"
        else:
            lg_overall = "HEALTHY"
            primary_health = "OK"

        # 2. Elering detailed status
        el_api_reachable = (elering_res is not None and cross_err is None)
        el_has_curr = False
        el_tom_status = "NOT_YET_PUBLISHED"
        if elering_res and elering_res.market_intervals:
            if is_live_query:
                el_has_curr = select_current_market_interval(elering_res.market_intervals, cur_now_utc) is not None
            else:
                el_has_curr = len(elering_res.market_intervals) > 0
            el_tom_cov = validate_local_market_day_coverage(tomorrow_date, self.tz, elering_res.market_intervals, target_resolution_minutes)
            el_tom_status = "COMPLETE" if el_tom_cov.is_complete else ("PARTIAL" if el_tom_cov.received_intervals > 0 else "NOT_YET_PUBLISHED")

        if not el_api_reachable:
            el_overall = "UNAVAILABLE"
            cross_health = "ERROR"
        elif active_source == "ELERING":
            el_overall = "ACTIVE_FALLBACK"
            cross_health = "OK"
        elif not el_has_curr:
            el_overall = "DEGRADED"
            cross_health = "DEGRADED"
        else:
            el_overall = "HEALTHY"
            cross_health = "OK"

        fallback_used = "YES" if active_source != "LITGRID" else "NO"
        if active_source != "LITGRID":
            if not lg_api_reachable:
                fallback_reason = f"Primary provider LITGRID API is unreachable ({primary_err or 'Network error'}); active fallback is {active_source}."
            elif not lg_has_curr:
                fallback_reason = f"Litgrid API returned no current data for requested period; active fallback provider is {active_source}."
            else:
                fallback_reason = f"Primary provider LITGRID degraded; active fallback provider is {active_source}."
        else:
            fallback_reason = None

        # Check tomorrow coverage on active expanded intervals
        tom_cov = validate_local_market_day_coverage(tomorrow_date, self.tz, expanded_intervals, target_resolution_minutes)
        tomorrow_status = tom_cov.status  # "TOMORROW COMPLETE", "PARTIAL TOMORROW DATA", or "TOMORROW NOT YET PUBLISHED"

        res_label = f"{target_resolution_minutes} min"

        snapshot = MarketStatusSnapshot(
            primary_source="LITGRID",
            primary_health=primary_health,
            primary_error=primary_err,
            primary_api_reachable=lg_api_reachable,
            primary_current_available=lg_has_curr,
            primary_tomorrow_coverage=lg_tom_status,
            primary_overall_status=lg_overall,
            cross_check_source="ELERING",
            cross_check_health=cross_health,
            cross_check_error=cross_err,
            cross_check_api_reachable=el_api_reachable,
            cross_check_current_available=el_has_curr,
            cross_check_tomorrow_coverage=el_tom_status,
            cross_check_overall_status=el_overall,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            latest_interval_utc=latest_interval,
            latest_interval_str=latest_str,
            tomorrow_prices=tomorrow_status,
            resolution=res_label,
            validation=val_report.status,
            data_freshness=freshness,
            active_source=active_source,
            intervals_count=len(expanded_intervals),
            raw_payload_hash=expanded_intervals[0].raw_payload_hash if expanded_intervals else None,
            validation_report=val_report,
        )

        return HierarchicalFetchResult(
            intervals=expanded_intervals,
            active_source=active_source,
            is_stale=is_stale,
            status_snapshot=snapshot,
            validation_report=val_report,
            litgrid_result=litgrid_res,
            elering_result=elering_res,
            fallback_result=fallback_res,
        )

    def run_shadow_operation(
        self,
        bidding_zone: str = "LT",
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        tes_params: TESParameters | None = None,
        site_params: SiteParameters | None = None,
        tariff_params: TariffParameters | None = None,
        session: Session | None = None,
    ) -> ShadowOperationResult:
        """Execute receding-horizon LP optimization driven by real Lithuanian market data in Shadow Mode.

        STRICT SAFETY CONSTRAINT:
        Virtual TES only. ZERO commands to hardware, PLC, Modbus, or relays.
        """
        fetch_res = self.fetch_market_data(
            bidding_zone=bidding_zone,
            start_utc=start_utc,
            end_utc=end_utc,
            target_resolution_minutes=15,
            session=session,
        )

        intervals = fetch_res.intervals
        if not intervals:
            raise LiveMarketDataUnavailableError("No market intervals available for shadow operation")

        tes = tes_params or TESParameters(
            thermal_capacity_full_span_kwh=15.0,
            optimizer_soc_min_fraction=0.10,
            optimizer_soc_max_fraction=1.00,
            max_charge_power_kw=9.0,
            max_discharge_power_kw=3.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.90,
            standing_loss_percent_per_day=2.0,
            auxiliary_power_kw=0.05,
        )
        site = site_params or SiteParameters(
            grid_connection_limit_kw=12.0,
            process_heat_demand_kw=1.5,
            other_loads_kw=2.0,
        )
        tariff = tariff_params or TariffParameters(
            supplier_markup_eur_mwh=1.5,
            variable_grid_fee_eur_mwh=25.0,
            variable_tax_eur_mwh=5.0,
        )
        curve = get_gen0_discharge_curve("fixed_gen0_hx", capacity_kwh=15.0)

        heat_profile = ConstantHeatDemand(value_kw=site.process_heat_demand_kw)
        site_profile = ConstantSiteLoad(value_kw=site.other_loads_kw)

        # Build optimization problem
        opt_inputs: list[OptimizationIntervalInput] = []
        for iv in intervals:
            eff = effective_price_eur_mwh(iv.price_eur_mwh, tariff)
            opt_inputs.append(
                OptimizationIntervalInput(
                    start_utc=iv.start_utc,
                    end_utc=iv.end_utc,
                    spot_price_eur_mwh=iv.price_eur_mwh,
                    effective_price_eur_mwh=eff,
                    heat_demand_kw=site.process_heat_demand_kw,
                    other_site_load_kw=site.other_loads_kw,
                    auxiliary_load_kw=tes.auxiliary_power_kw,
                )
            )

        problem = OptimizationProblemInput(
            intervals=opt_inputs,
            tes_params=tes,
            grid_connection_limit_kw=site.grid_connection_limit_kw,
            initial_soc_kwh=7.5,
            target_terminal_soc_kwh=7.5,
            terminal_soc_condition="exact",
            discharge_limit_curve=curve,
            lexicographic=True,
        )

        optimizer = LPOptimizer()
        opt_result = optimizer.optimize(problem)

        mapper = ThermalStateMapper()
        timeline_rows: list[dict[str, Any]] = []
        chart_series: list[dict[str, Any]] = []

        total_cost = 0.0
        total_charge_kwh = 0.0
        total_heat_kwh = 0.0
        max_grid_kw = 0.0

        for item in opt_result.intervals:
            dt_local = item.start_utc.astimezone(self.tz)
            local_time_str = dt_local.strftime("%m-%d %H:%M")
            sand_temp = mapper.temperature_from_soc_fraction(item.soc_percent / 100.0)
            p_max_hx = curve.max_discharge_power_kw(item.soc_kwh, tes)
            p_grid_total = item.charge_power_kw + item.other_site_load_kw + item.auxiliary_load_kw

            if p_grid_total > max_grid_kw:
                max_grid_kw = p_grid_total

            dt_h = (item.end_utc - item.start_utc).total_seconds() / 3600.0
            total_charge_kwh += item.charge_power_kw * dt_h
            total_heat_kwh += item.discharge_power_kw * dt_h
            total_cost += item.cost_eur

            row = {
                "local_time": local_time_str,
                "spot_price_eur_mwh": round(item.spot_price_eur_mwh, 2),
                "effective_price_eur_mwh": round(item.effective_price_eur_mwh, 2),
                "requested_charge_kw": round(item.charge_power_kw, 2),
                "actual_charge_kw": round(item.charge_power_kw, 2),
                "process_demand_kw": round(item.heat_demand_kw, 2),
                "hx_max_available_kw": round(p_max_hx, 2),
                "actual_discharge_kw": round(item.discharge_power_kw, 2),
                "unmet_heat_kw": round(item.unmet_heat_kw, 2),
                "soc_kwh": round(item.soc_kwh, 2),
                "soc_percent": round(item.soc_percent, 1),
                "sand_temperature_c": round(sand_temp, 1),
                "grid_total_kw": round(p_grid_total, 2),
                "interval_cost_eur": round(item.cost_eur, 3),
                "source": fetch_res.active_source,
            }
            timeline_rows.append(row)

            chart_series.append({
                "time": local_time_str,
                "spot_price": round(item.spot_price_eur_mwh, 2),
                "charge_power": round(item.charge_power_kw, 2),
                "discharge_power": round(item.discharge_power_kw, 2),
                "soc_percent": round(item.soc_percent, 1),
                "sand_temperature_c": round(sand_temp, 1),
                "total_grid_power": round(p_grid_total, 2),
                "site_load": round(item.other_site_load_kw, 2),
                "aux_load": round(item.auxiliary_load_kw, 3),
                "grid_limit": site.grid_connection_limit_kw,
                "configured_charge_limit": tes.max_charge_power_kw,
            })

        return ShadowOperationResult(
            status_label="REAL LT MARKET DATA + VIRTUAL TES OPERATION",
            market_status=fetch_res.status_snapshot,
            active_source=fetch_res.active_source,
            optimization_result=opt_result,
            timeline_rows=timeline_rows,
            chart_data=chart_series,
            total_cost_eur=round(total_cost, 2),
            total_charge_kwh=round(total_charge_kwh, 2),
            total_heat_kwh=round(total_heat_kwh, 2),
            max_grid_power_kw=round(max_grid_kw, 2),
        )

    def get_market_view_context(
        self,
        session: Session | None = None,
        now_utc: datetime | None = None,
        tariff_params: TariffParameters | None = None,
        allow_mock: bool = False,
    ) -> dict[str, Any]:
        """Build full, ready-to-render view model for Overview and Market pages (Phase 5.8.1)."""
        now_utc = ensure_utc(now_utc or datetime.now(timezone.utc))
        now_local = now_utc.astimezone(self.tz)
        current_time_local_str = now_local.strftime("%H:%M:%S Europe/Vilnius")

        tariff = tariff_params or TariffParameters(
            supplier_markup_eur_mwh=1.5,
            variable_grid_fee_eur_mwh=25.0,
            variable_tax_eur_mwh=5.0,
        )
        total_adders = (
            tariff.supplier_markup_eur_mwh
            + tariff.variable_grid_fee_eur_mwh
            + tariff.variable_tax_eur_mwh
        )

        try:
            fetch_res = self.fetch_market_data(session=session)
            intervals = fetch_res.intervals
            active_source = fetch_res.active_source
            fallback_used = (active_source != "LITGRID")
            fallback_reason = (
                f"Litgrid API returned no current data for requested period; active fallback provider is {active_source}."
                if fallback_used else None
            )
            is_stale = fetch_res.is_stale
            snap = fetch_res.status_snapshot
            error_message = snap.primary_error
        except Exception as exc:
            log.warning("MarketDataService view fetch failed: %s", exc)
            intervals = []
            active_source = "NONE"
            fallback_used = False
            fallback_reason = None
            is_stale = False
            error_message = str(exc)
            snap = MarketStatusSnapshot(
                primary_source="LITGRID",
                primary_health="ERROR",
                primary_error=error_message,
                cross_check_source="ELERING",
                cross_check_health="ERROR",
                cross_check_error=error_message,
                latest_interval_utc=None,
                latest_interval_str="None",
                tomorrow_prices="WAITING",
                resolution="15 min",
                validation="NOT CHECKED",
                data_freshness="SOURCE ERROR",
                active_source="NONE",
                intervals_count=0,
            )

        cur_iv = select_current_market_interval(intervals, now_utc)
        next_iv = select_next_market_interval(intervals, cur_iv, now_utc)
        today_stats = compute_today_market_stats(intervals, now_local.date(), self.tz)
        tomorrow_stats = compute_tomorrow_market_stats(intervals, now_local.date(), self.tz)
        debug_records = get_market_debug_records(intervals, count=8, tz=self.tz)

        status_badge = determine_data_status(
            intervals=intervals,
            cur_iv=cur_iv,
            now_utc=now_utc,
            tomorrow_available=tomorrow_stats["available"],
            is_stale_flag=is_stale,
            has_error=(error_message is not None and not intervals),
            is_mock=False,
        )

        if cur_iv is not None:
            s_loc = cur_iv.start_utc.astimezone(self.tz).strftime("%H:%M")
            e_loc = cur_iv.end_utc.astimezone(self.tz).strftime("%H:%M")
            interval_str = f"{s_loc} -> {e_loc}"
            cur_price = cur_iv.price_eur_mwh
            res_label = f"{cur_iv.resolution_minutes} min"
            orig_res_label = f"{cur_iv.original_resolution_minutes} min"
            is_derived = cur_iv.is_derived
            if cur_iv.fetched_at_utc:
                last_fetched_str = cur_iv.fetched_at_utc.astimezone(self.tz).strftime("%H:%M:%S")
                age_min = max(0, int((now_utc - cur_iv.fetched_at_utc).total_seconds() / 60))
            else:
                last_fetched_str = now_local.strftime("%H:%M:%S")
                age_min = 0
            cur_source = cur_iv.source
        else:
            interval_str = "NO CURRENT INTERVAL"
            cur_price = None
            res_label = "15 min"
            orig_res_label = "15 min"
            is_derived = False
            last_fetched_str = now_local.strftime("%H:%M:%S")
            age_min = 0
            cur_source = active_source

        if next_iv is not None:
            ns_loc = next_iv.start_utc.astimezone(self.tz).strftime("%H:%M")
            ne_loc = next_iv.end_utc.astimezone(self.tz).strftime("%H:%M")
            next_interval_str = f"{ns_loc} -> {ne_loc}"
            next_price = next_iv.price_eur_mwh
            price_diff = (next_price - cur_price) if cur_price is not None else None
        else:
            next_interval_str = "--:-- -> --:--"
            next_price = None
            price_diff = None

        eff_price = (cur_price + total_adders) if cur_price is not None else None

        # Build TES Action Now
        tes_action = None
        if intervals and cur_iv is not None:
            try:
                shadow_res = self.run_shadow_operation(session=session)
                matched_item = None
                for item in shadow_res.optimization_result.intervals:
                    if item.start_utc <= now_utc < item.end_utc:
                        matched_item = item
                        break

                if matched_item is not None:
                    chg = matched_item.charge_power_kw
                    dis = matched_item.discharge_power_kw
                    if chg > 0.05:
                        cmd = "CHARGE"
                        req_kw = chg
                        act_kw = chg
                    elif dis > 0.05:
                        cmd = "DISCHARGE"
                        req_kw = dis
                        act_kw = dis
                    else:
                        cmd = "HOLD"
                        req_kw = 0.0
                        act_kw = 0.0

                    mapper = ThermalStateMapper()
                    sand_temp = mapper.temperature_from_soc_fraction(matched_item.soc_percent / 100.0)
                    curve = get_gen0_discharge_curve("fixed_gen0_hx", capacity_kwh=15.0)
                    tes_dummy = TESParameters(thermal_capacity_full_span_kwh=15.0)
                    p_max_hx = curve.max_discharge_power_kw(matched_item.soc_kwh, tes_dummy)
                    grid_tot = chg + matched_item.other_site_load_kw + matched_item.auxiliary_load_kw

                    future_prices = [
                        it.spot_price_eur_mwh
                        for it in shadow_res.optimization_result.intervals
                        if it.start_utc > now_utc
                    ]
                    future_avg = (
                        sum(future_prices) / len(future_prices)
                        if future_prices
                        else matched_item.spot_price_eur_mwh
                    )

                    reason_code, reason_text = resolve_tes_action_reason(
                        command=cmd,
                        requested_kw=req_kw,
                        actual_kw=act_kw,
                        soc_percent=matched_item.soc_percent,
                        sand_temp_c=sand_temp,
                        hx_max_available_kw=p_max_hx,
                        process_demand_kw=matched_item.heat_demand_kw,
                        grid_total_kw=grid_tot,
                        grid_connection_limit_kw=12.0,
                        current_price_eur_mwh=matched_item.spot_price_eur_mwh,
                        future_avg_price_eur_mwh=future_avg,
                        has_market_data=True,
                    )

                    act_s = matched_item.start_utc.astimezone(self.tz).strftime("%H:%M")
                    act_e = matched_item.end_utc.astimezone(self.tz).strftime("%H:%M")
                    tes_action = {
                        "status": "ACTIVE",
                        "interval_str": f"{act_s} -> {act_e}",
                        "price_eur_mwh": round(matched_item.spot_price_eur_mwh, 2),
                        "soc_percent": round(matched_item.soc_percent, 1),
                        "soc_kwh": round(matched_item.soc_kwh, 2),
                        "sand_temperature_c": round(sand_temp, 1),
                        "command": cmd,
                        "requested_kw": round(req_kw, 2),
                        "actual_kw": round(act_kw, 2),
                        "reason_code": reason_code,
                        "reason_text": reason_text,
                    }
            except Exception as shadow_err:
                log.warning("Could not calculate active TES shadow action: %s", shadow_err)

        if tes_action is None:
            tes_action = {
                "status": "NO ACTIVE SHADOW SCHEDULE",
                "interval_str": interval_str,
                "price_eur_mwh": cur_price or 0.0,
                "soc_percent": 50.0,
                "soc_kwh": 7.5,
                "sand_temperature_c": 190.0,
                "command": "NO ACTIVE SHADOW SCHEDULE",
                "requested_kw": 0.0,
                "actual_kw": 0.0,
                "reason_code": "WAITING_FOR_MARKET_DATA",
                "reason_text": "No active shadow schedule generated; waiting for live market data.",
            }

        return {
            "current_time_local_str": current_time_local_str,
            "current_time_utc_str": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "now_utc": now_utc,
            "interval_str": interval_str,
            "current_interval": cur_iv,
            "price_eur_mwh": round(cur_price, 2) if cur_price is not None else None,
            "price_eur_mwh_str": f"{cur_price:.2f}" if cur_price is not None else "--.--",
            "source": cur_source,
            "active_source": active_source,
            "primary_source": snap.primary_source,
            "data_status": status_badge,
            "last_fetched_str": last_fetched_str,
            "age_minutes": age_min,
            "resolution": res_label,
            "original_resolution": orig_res_label,
            "is_derived": is_derived,
            "next_interval_str": next_interval_str,
            "next_price_eur_mwh": round(next_price, 2) if next_price is not None else None,
            "next_price_eur_mwh_str": f"{next_price:.2f}" if next_price is not None else "--.--",
            "price_diff_eur_mwh": round(price_diff, 2) if price_diff is not None else None,
            "price_diff_eur_mwh_str": f"{price_diff:+.2f}" if price_diff is not None else "--.--",
            "fallback_used": "YES" if fallback_used else "NO",
            "fallback_reason": fallback_reason,
            "market_price_eur_mwh": round(cur_price, 2) if cur_price is not None else None,
            "supplier_markup_eur_mwh": round(tariff.supplier_markup_eur_mwh, 2),
            "variable_grid_fee_eur_mwh": round(tariff.variable_grid_fee_eur_mwh, 2),
            "variable_tax_eur_mwh": round(tariff.variable_tax_eur_mwh, 2),
            "total_adders_eur_mwh": round(total_adders, 2),
            "effective_price_eur_mwh": round(eff_price, 2) if eff_price is not None else None,
            "effective_price_eur_mwh_str": f"{eff_price:.2f}" if eff_price is not None else "--.--",
            "optimization_price_eur_mwh": round(eff_price, 2) if eff_price is not None else None,
            "optimization_price_eur_mwh_str": f"{eff_price:.2f}" if eff_price is not None else "--.--",
            "optimization_price_basis": "EFFECTIVE_VARIABLE_PRICE",
            "primary_status": {
                "api_reachable": "YES" if snap.primary_api_reachable else "NO",
                "current_available": "YES" if snap.primary_current_available else "NO",
                "tomorrow_coverage": snap.primary_tomorrow_coverage,
                "overall_status": snap.primary_overall_status,
            },
            "cross_check_status": {
                "api_reachable": "YES" if snap.cross_check_api_reachable else "NO",
                "current_available": "YES" if snap.cross_check_current_available else "NO",
                "tomorrow_coverage": snap.cross_check_tomorrow_coverage,
                "overall_status": snap.cross_check_overall_status,
            },
            "actuation_model": "CONTINUOUS IDEALIZED",
            "hardware_staging_implemented": False,
            "staging_warning": "Continuous optimizer request — physical heater staging not yet modeled.",
            "available_tes_charging_headroom_kw": 9.95,
            "grid_margin_at_full_charge_kw": 0.95,
            "today_stats": today_stats,
            "tomorrow_stats": tomorrow_stats,
            "debug_records": debug_records,
            "tes_action": tes_action,
            "market_status": snap,
            "validation": snap.validation,
            "error_message": error_message,
        }


# =========================================================================
# Phase 5.8.1 Helper Functions
# =========================================================================


def select_current_market_interval(
    intervals: list[MarketPriceInterval],
    now_utc: datetime,
) -> MarketPriceInterval | None:
    """Select the unique market interval covering now_utc: interval.start_utc <= now_utc < interval.end_utc.

    Enforces:
    - Timezone-aware UTC timestamp comparison
    - Half-open interval [start, end)
    - Exact start boundary belongs to the interval
    - Exact end boundary belongs to the next interval
    """
    now_utc = ensure_utc(now_utc)
    for iv in intervals:
        start = ensure_utc(iv.start_utc)
        end = ensure_utc(iv.end_utc)
        if start <= now_utc < end:
            return iv
    return None


def select_next_market_interval(
    intervals: list[MarketPriceInterval],
    current_interval: MarketPriceInterval | None = None,
    now_utc: datetime | None = None,
) -> MarketPriceInterval | None:
    """Select the interval immediately following current_interval."""
    if current_interval is not None:
        target_start = ensure_utc(current_interval.end_utc)
        for iv in intervals:
            if ensure_utc(iv.start_utc) == target_start:
                return iv
    if now_utc is not None:
        now_utc = ensure_utc(now_utc)
        future = [iv for iv in intervals if ensure_utc(iv.start_utc) > now_utc]
        if future:
            future.sort(key=lambda x: x.start_utc)
            return future[0]
    return None


def get_today_intervals(
    intervals: list[MarketPriceInterval],
    target_date: date | None = None,
    tz: ZoneInfo = ZoneInfo("Europe/Vilnius"),
) -> list[MarketPriceInterval]:
    """Get all intervals for target_date in Europe/Vilnius local time (preserving 15-minute resolution)."""
    if target_date is None:
        target_date = datetime.now(tz).date()
    cov = validate_local_market_day_coverage(target_date, tz, intervals, 15)
    return cov.intervals


def get_tomorrow_intervals(
    intervals: list[MarketPriceInterval],
    reference_today: date | None = None,
    tz: ZoneInfo = ZoneInfo("Europe/Vilnius"),
) -> list[MarketPriceInterval]:
    """Get all intervals for tomorrow relative to reference_today in Europe/Vilnius local time."""
    if reference_today is None:
        reference_today = datetime.now(tz).date()
    tomorrow_date = reference_today + timedelta(days=1)
    cov = validate_local_market_day_coverage(tomorrow_date, tz, intervals, 15)
    return cov.intervals


def compute_today_market_stats(
    intervals: list[MarketPriceInterval],
    target_date: date | None = None,
    tz: ZoneInfo = ZoneInfo("Europe/Vilnius"),
) -> dict[str, Any]:
    """Compute min, max, average and formatted series for today's market intervals."""
    if target_date is None:
        target_date = datetime.now(tz).date()
    cov = validate_local_market_day_coverage(target_date, tz, intervals, 15)
    today_ivs = cov.intervals

    if not today_ivs:
        return {
            "intervals_count": 0,
            "expected_intervals": cov.expected_intervals,
            "coverage_fraction": 0.0,
            "coverage_percent_str": "0.0%",
            "is_complete": False,
            "min_price_eur_mwh": 0.0,
            "min_price_time_str": "--:--",
            "max_price_eur_mwh": 0.0,
            "max_price_time_str": "--:--",
            "avg_price_eur_mwh": 0.0,
            "intervals": [],
        }

    min_iv = min(today_ivs, key=lambda x: x.price_eur_mwh)
    max_iv = max(today_ivs, key=lambda x: x.price_eur_mwh)
    avg_price = sum(x.price_eur_mwh for x in today_ivs) / len(today_ivs)

    formatted_ivs = []
    for iv in today_ivs:
        s_local = iv.start_utc.astimezone(tz).strftime("%H:%M")
        e_local = iv.end_utc.astimezone(tz).strftime("%H:%M")
        formatted_ivs.append({
            "start_local": s_local,
            "end_local": e_local,
            "time_label": f"{s_local}–{e_local}",
            "price_eur_mwh": round(iv.price_eur_mwh, 2),
            "start_utc": iv.start_utc.isoformat(),
            "end_utc": iv.end_utc.isoformat(),
            "is_derived": iv.is_derived,
            "original_resolution_minutes": iv.original_resolution_minutes,
            "source": iv.source,
        })

    return {
        "intervals_count": len(today_ivs),
        "expected_intervals": cov.expected_intervals,
        "coverage_fraction": cov.coverage_fraction,
        "coverage_percent_str": f"{cov.coverage_fraction * 100.0:.1f}%",
        "is_complete": cov.is_complete,
        "min_price_eur_mwh": round(min_iv.price_eur_mwh, 2),
        "min_price_time_str": min_iv.start_utc.astimezone(tz).strftime("%H:%M"),
        "max_price_eur_mwh": round(max_iv.price_eur_mwh, 2),
        "max_price_time_str": max_iv.start_utc.astimezone(tz).strftime("%H:%M"),
        "avg_price_eur_mwh": round(avg_price, 2),
        "intervals": formatted_ivs,
    }


def compute_tomorrow_market_stats(
    intervals: list[MarketPriceInterval],
    reference_today: date | None = None,
    target_date: date | None = None,
    tz: ZoneInfo = ZoneInfo("Europe/Vilnius"),
) -> dict[str, Any]:
    """Compute statistics for tomorrow's market data (if cleared and published)."""
    ref_d = reference_today if reference_today is not None else target_date
    if ref_d is None:
        ref_d = datetime.now(tz).date()
    tomorrow_date = ref_d + timedelta(days=1)
    cov = validate_local_market_day_coverage(tomorrow_date, tz, intervals, 15)
    tomorrow_ivs = cov.intervals

    if not tomorrow_ivs:
        return {
            "status": "TOMORROW PRICES NOT YET AVAILABLE",
            "available": False,
            "is_partial": False,
            "intervals_count": 0,
            "expected_intervals": cov.expected_intervals,
            "coverage_fraction": 0.0,
            "coverage_percent_str": "0.0%",
            "first_interval_str": "--:--",
            "last_interval_str": "--:--",
            "min_price_eur_mwh": None,
            "max_price_eur_mwh": None,
            "avg_price_eur_mwh": None,
            "intervals": [],
        }

    min_val = min(x.price_eur_mwh for x in tomorrow_ivs)
    max_val = max(x.price_eur_mwh for x in tomorrow_ivs)
    avg_val = sum(x.price_eur_mwh for x in tomorrow_ivs) / len(tomorrow_ivs)
    return {
        "status": cov.status,
        "available": cov.is_complete,
        "is_partial": not cov.is_complete,
        "intervals_count": cov.received_intervals,
        "expected_intervals": cov.expected_intervals,
        "coverage_fraction": cov.coverage_fraction,
        "coverage_percent_str": f"{cov.coverage_fraction * 100.0:.1f}%",
        "first_interval_str": cov.first_interval_local or "--:--",
        "last_interval_str": cov.last_interval_local or "--:--",
        "min_price_eur_mwh": round(min_val, 2),
        "max_price_eur_mwh": round(max_val, 2),
        "avg_price_eur_mwh": round(avg_val, 2),
        "intervals": [
            {
                "start_local": iv.start_utc.astimezone(tz).strftime("%H:%M"),
                "end_local": iv.end_utc.astimezone(tz).strftime("%H:%M"),
                "price_eur_mwh": round(iv.price_eur_mwh, 2),
            }
            for iv in tomorrow_ivs
        ],
    }


def get_market_debug_records(
    intervals: list[MarketPriceInterval],
    count: int = 8,
    tz: ZoneInfo = ZoneInfo("Europe/Vilnius"),
) -> list[dict[str, Any]]:
    """Return the last `count` normalized market intervals formatted for debugging."""
    sorted_ivs = sorted(intervals, key=lambda x: x.start_utc)
    slice_ivs = sorted_ivs[-count:] if len(sorted_ivs) >= count else sorted_ivs
    records = []
    for iv in slice_ivs:
        records.append({
            "start_local": iv.start_utc.astimezone(tz).strftime("%H:%M"),
            "end_local": iv.end_utc.astimezone(tz).strftime("%H:%M"),
            "start_utc": iv.start_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "price_eur_mwh": round(iv.price_eur_mwh, 2),
            "source": iv.source,
            "fetched_at": iv.fetched_at_utc.astimezone(tz).strftime("%H:%M:%S") if iv.fetched_at_utc else "--",
            "original_resolution": f"{iv.original_resolution_minutes} min",
            "derived_flag": "YES" if iv.is_derived else "NO",
            "source_record_id": iv.source_record_id or "",
            "raw_payload_hash": (iv.raw_payload_hash[:10] + "...") if iv.raw_payload_hash else "--",
        })
    return records


def determine_data_status(
    intervals: list[MarketPriceInterval],
    cur_iv: MarketPriceInterval | None,
    now_utc: datetime,
    tomorrow_available: bool,
    is_stale_flag: bool = False,
    has_error: bool = False,
    is_mock: bool = False,
) -> str:
    """Classify market dataset status deterministically (Phase 5.8.1)."""
    if is_mock:
        return "MOCK DATA"
    if has_error and not intervals:
        return "SOURCE ERROR"
    if not intervals:
        return "NO CURRENT INTERVAL"

    newest_end = max(iv.end_utc for iv in intervals)
    if is_stale_flag or now_utc >= newest_end:
        return "STALE DATA"

    if cur_iv is None:
        return "NO CURRENT INTERVAL"

    if not tomorrow_available:
        return "WAITING FOR TOMORROW PRICES"

    return "LIVE DATA"


def resolve_tes_action_reason(
    command: str,
    requested_kw: float,
    actual_kw: float,
    soc_percent: float,
    sand_temp_c: float,
    hx_max_available_kw: float,
    process_demand_kw: float,
    grid_total_kw: float,
    grid_connection_limit_kw: float,
    current_price_eur_mwh: float,
    future_avg_price_eur_mwh: float,
    has_market_data: bool = True,
) -> tuple[str, str]:
    """Map operational state to deterministic reason codes and human-readable explanations."""
    if not has_market_data:
        return "WAITING_FOR_MARKET_DATA", "Waiting for live market price horizon to optimize dispatch."

    if command == "CHARGE":
        if grid_total_kw >= grid_connection_limit_kw - 0.05:
            return "GRID_LIMITED", f"Charging power constrained by site grid connection headroom limit ({grid_connection_limit_kw:.1f} kW)."
        if soc_percent >= 99.9:
            return "STORAGE_FULL", "Thermal storage is at maximum allowable SOC (100.0%); charging paused."
        return "CHEAP_RELATIVE_TO_FUTURE", "Current spot price is below upcoming high-price intervals; accumulating heat in sand."

    elif command == "DISCHARGE":
        if actual_kw < process_demand_kw - 0.05 and actual_kw >= hx_max_available_kw - 0.05:
            return "HX_LIMITED", f"Discharge power constrained by heat exchanger thermodynamic limit ({hx_max_available_kw:.2f} kW at {sand_temp_c:.1f} °C)."
        if soc_percent <= 10.1:
            return "SOC_RESERVE", "Thermal storage is at 10.0% operational reserve floor; discharge halted to protect temperature floor."
        return "EXPENSIVE_PRICE_DISCHARGE", "Discharging stored thermal energy to supply heat demand during elevated electricity price interval."

    else:  # HOLD / IDLE
        if soc_percent >= 99.9:
            return "STORAGE_FULL", "Thermal storage is fully charged (100.0% SOC); charging paused."
        if soc_percent <= 10.1:
            return "SOC_RESERVE", "Thermal storage at minimum reserve floor (10.0% SOC); discharge halted."
        if current_price_eur_mwh < future_avg_price_eur_mwh:
            return "IDLE_OPTIMAL", "Holding charge; awaiting optimal low-price charge or high-price discharge window."
        return "IDLE_OPTIMAL", "Holding stored energy; economic dispatch indicates idle state."

