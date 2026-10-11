"""Authoritative Market Price and Time Synchronization Service (Phase 5.11 - Stage 2).

Centralizes all market-price lookups across the application based on UTC settlement intervals.
Enforces:
1. UTC as immutable database, cache, and interval lookup key.
2. Europe/Vilnius as the display timezone for Lithuanian market operations.
3. Explicit storage and distinction of:
   - Spot market price (EUR/MWh and EUR/kWh)
   - Effective variable electricity price (EUR/MWh and EUR/kWh)
   - Configured variable tariff components (markup, grid distribution fee, taxes)
   - Market provider (LITGRID, ELERING, VOLTON, ENTSOE, SYNTHETIC)
   - Actual market interval timestamps (UTC bounds and local strings)
   - Strict Data Quality classification: LIVE, FALLBACK, HISTORICAL, or SYNTHETIC
4. Strictly forbidden: labeling synthetic, mock, or stale cached data as "LIVE".
5. Historical price integrity: older executed intervals always receive their historical interval price,
   never the latest current spot price.
6. Explicit declared price basis for all economics (EFFECTIVE_VARIABLE_PRICE).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.parameters import TariffParameters
from app.core.timegrid import ensure_utc, local_day_bounds_utc
from app.database.models import DayAheadPrice, TariffConfig
from app.database.repositories import PriceRepository
from app.economics.tariff import effective_price_eur_mwh

log = logging.getLogger(__name__)

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")
UTC = timezone.utc


class DataQuality(str, Enum):
    """Strict data quality classification for market prices."""
    LIVE = "LIVE"
    FALLBACK = "FALLBACK"
    HISTORICAL = "HISTORICAL"
    SYNTHETIC = "SYNTHETIC"


@dataclass(frozen=True, slots=True)
class TariffBreakdown:
    """Explicit variable tariff components in EUR/MWh and EUR/kWh."""
    supplier_markup_eur_mwh: float
    variable_grid_fee_eur_mwh: float
    variable_tax_eur_mwh: float
    total_variable_adders_eur_mwh: float
    supplier_markup_eur_kwh: float
    variable_grid_fee_eur_kwh: float
    variable_tax_eur_kwh: float
    total_variable_adders_eur_kwh: float


@dataclass(frozen=True, slots=True)
class AuthoritativeMarketPrice:
    """Canonical market price snapshot for a specific UTC settlement interval."""
    # 1. UTC Interval Boundaries (Database Keys)
    interval_start_utc: datetime
    interval_end_utc: datetime
    duration_minutes: int

    # 2. Local Europe/Vilnius Representation (Display Only)
    interval_start_local: datetime
    interval_end_local: datetime
    interval_local_str: str
    interval_full_local_str: str

    # 3. Market Pricing (EUR/MWh and EUR/kWh)
    spot_price_eur_mwh: float
    spot_price_eur_kwh: float
    effective_price_eur_mwh: float
    effective_price_eur_kwh: float

    # 4. Tariff Breakdown
    tariff: TariffBreakdown

    # 5. Provenance & Data Quality
    provider: str
    data_quality: DataQuality
    price_basis: str  # Declared economic basis, e.g. "EFFECTIVE_VARIABLE_PRICE"
    is_stale: bool
    is_synthetic: bool
    fetched_at_utc: datetime | None = None
    source_record_id: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict with formatted numbers and ISO timestamps."""
        d = asdict(self)
        d["data_quality"] = self.data_quality.value
        d["interval_start_utc_iso"] = self.interval_start_utc.isoformat()
        d["interval_end_utc_iso"] = self.interval_end_utc.isoformat()
        d["interval_start_local_iso"] = self.interval_start_local.isoformat()
        d["interval_end_local_iso"] = self.interval_end_local.isoformat()
        return d


def snap_to_settlement_interval(
    dt_utc: datetime,
    resolution_minutes: int = 15,
) -> tuple[datetime, datetime]:
    """Snap any UTC datetime to its enclosing settlement interval [start_utc, end_utc)."""
    dt = ensure_utc(dt_utc)
    snapped_minute = (dt.minute // resolution_minutes) * resolution_minutes
    start_utc = dt.replace(minute=snapped_minute, second=0, microsecond=0)
    end_utc = start_utc + timedelta(minutes=resolution_minutes)
    return start_utc, end_utc


def resolve_active_tariff(
    db: Session | None = None,
    override: TariffParameters | None = None,
) -> TariffBreakdown:
    """Resolve active variable tariff parameters, preferring DB config if present."""
    if override is not None:
        markup = override.supplier_markup_eur_mwh
        grid = override.variable_grid_fee_eur_mwh
        tax = override.variable_tax_eur_mwh
    elif db is not None:
        try:
            cfg = db.execute(
                select(TariffConfig).order_by(TariffConfig.created_at.desc()).limit(1)
            ).scalar_one_or_none()
            if cfg is not None:
                markup = cfg.supplier_markup_eur_mwh
                grid = cfg.variable_grid_fee_eur_mwh
                tax = cfg.variable_tax_eur_mwh
            else:
                markup = 1.50
                grid = 25.00
                tax = 5.00
        except Exception as e:
            log.warning("Tariff resolution fallback: %s", e)
            markup = 1.50
            grid = 25.00
            tax = 5.00
    else:
        markup = 1.50
        grid = 25.00
        tax = 5.00

    total_adders = markup + grid + tax
    return TariffBreakdown(
        supplier_markup_eur_mwh=round(markup, 2),
        variable_grid_fee_eur_mwh=round(grid, 2),
        variable_tax_eur_mwh=round(tax, 2),
        total_variable_adders_eur_mwh=round(total_adders, 2),
        supplier_markup_eur_kwh=round(markup / 1000.0, 5),
        variable_grid_fee_eur_kwh=round(grid / 1000.0, 5),
        variable_tax_eur_kwh=round(tax / 1000.0, 5),
        total_variable_adders_eur_kwh=round(total_adders / 1000.0, 5),
    )


SOURCE_PRIORITY = {"LITGRID": 1, "ELERING": 2, "VOLTON": 3, "ENTSOE": 4}


class MarketPriceService:
    """Authoritative market price retrieval and provenance resolver."""

    def __init__(self, tz: ZoneInfo = VILNIUS_TZ) -> None:
        self.tz = tz

    def get_authoritative_price_for_interval(
        self,
        interval_start_utc: datetime,
        interval_end_utc: datetime | None = None,
        db: Session | None = None,
        tariff_params: TariffParameters | None = None,
        now_utc: datetime | None = None,
        source: str | None = None,
        resolution_minutes: int = 15,
    ) -> AuthoritativeMarketPrice:
        """Retrieve the authoritative market price for a specific UTC settlement interval.

        Guarantees:
        - If the interval is older than the current interval, returns the matching HISTORICAL price.
          Older executed intervals NEVER receive the latest current price.
        - If live data is missing or synthetic, data quality is strictly SYNTHETIC (never LIVE).
        - If secondary/tertiary source was used, data quality is strictly FALLBACK (never LIVE).
        """
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(timezone.utc)
        start_utc, end_utc = snap_to_settlement_interval(interval_start_utc, resolution_minutes)
        if interval_end_utc is not None:
            end_utc = ensure_utc(interval_end_utc)

        duration_min = int((end_utc - start_utc).total_seconds() / 60.0)
        start_local = start_utc.astimezone(self.tz)
        end_local = end_utc.astimezone(self.tz)
        int_str = f"{start_local.strftime('%H:%M')} -> {end_local.strftime('%H:%M')}"
        int_full_str = f"{start_local.strftime('%Y-%m-%d %H:%M')} -> {end_local.strftime('%H:%M')} Europe/Vilnius"

        # Timing context relative to now
        is_past = (end_utc <= now)
        is_current = (start_utc <= now < end_utc)
        is_future = (start_utc >= now)

        tariff_bd = resolve_active_tariff(db=db, override=tariff_params)

        # 1. Attempt lookup in persistent database (DayAheadPrice table)
        db_price_row: DayAheadPrice | None = None
        if db is not None:
            try:
                repo = PriceRepository(db, self.tz)
                prices = repo.get_prices("LT", start_utc, end_utc, source=source, latest_only=True)
                if prices:
                    prices.sort(key=lambda p: (SOURCE_PRIORITY.get(p.source.upper(), 99), -p.version))
                    db_price_row = prices[0]
            except Exception as dbe:
                log.warning("Authoritative price DB query failed: %s", dbe)

        if db_price_row is not None:
            spot_mwh = db_price_row.price_eur_mwh
            provider = db_price_row.source
            fetched_at = db_price_row.fetched_at
            rec_id = f"db-price-{db_price_row.id}"

            # Determine strict data quality
            if is_past:
                # Older interval: historical by definition
                quality = DataQuality.HISTORICAL
                is_stale = False
            elif is_current:
                # Age check: fetched within last 45 minutes?
                age_sec = (now - ensure_utc(fetched_at)).total_seconds() if fetched_at else 0
                if age_sec > 2700:
                    quality = DataQuality.HISTORICAL
                    is_stale = True
                elif provider.upper() == "LITGRID":
                    quality = DataQuality.LIVE
                    is_stale = False
                elif provider.upper() in ("ELERING", "VOLTON", "ENTSOE"):
                    quality = DataQuality.FALLBACK
                    is_stale = False
                else:
                    quality = DataQuality.SYNTHETIC
                    is_stale = False
            else:
                # Future interval
                if provider.upper() == "LITGRID":
                    quality = DataQuality.LIVE
                elif provider.upper() in ("ELERING", "VOLTON", "ENTSOE"):
                    quality = DataQuality.FALLBACK
                else:
                    quality = DataQuality.SYNTHETIC
                is_stale = False

            eff_mwh = spot_mwh + tariff_bd.total_variable_adders_eur_mwh
            return AuthoritativeMarketPrice(
                interval_start_utc=start_utc,
                interval_end_utc=end_utc,
                duration_minutes=duration_min,
                interval_start_local=start_local,
                interval_end_local=end_local,
                interval_local_str=int_str,
                interval_full_local_str=int_full_str,
                spot_price_eur_mwh=round(spot_mwh, 2),
                spot_price_eur_kwh=round(spot_mwh / 1000.0, 5),
                effective_price_eur_mwh=round(eff_mwh, 2),
                effective_price_eur_kwh=round(eff_mwh / 1000.0, 5),
                tariff=tariff_bd,
                provider=provider,
                data_quality=quality,
                price_basis="EFFECTIVE_VARIABLE_PRICE",
                is_stale=is_stale,
                is_synthetic=(quality == DataQuality.SYNTHETIC),
                fetched_at_utc=fetched_at,
                source_record_id=rec_id,
            )

        # 2. If not found in DB and current/future, query live market data providers
        if not is_past:
            try:
                from app.services.market_data_service import MarketDataService
                market_svc = MarketDataService(tz=self.tz)
                fetch_res = market_svc.fetch_market_data(
                    start_utc=start_utc,
                    end_utc=end_utc,
                    session=db,
                )
                if fetch_res.intervals:
                    # Match exact interval
                    matched = None
                    for iv in fetch_res.intervals:
                        if iv.start_utc <= start_utc < iv.end_utc:
                            matched = iv
                            break
                    if matched is None and fetch_res.intervals:
                        matched = fetch_res.intervals[0]

                    if matched is not None:
                        spot_mwh = matched.price_eur_mwh
                        provider = fetch_res.active_source
                        if fetch_res.is_stale:
                            quality = DataQuality.HISTORICAL
                            is_stale = True
                        elif provider.upper() == "LITGRID":
                            quality = DataQuality.LIVE
                            is_stale = False
                        elif provider.upper() in ("ELERING", "VOLTON", "ENTSOE"):
                            quality = DataQuality.FALLBACK
                            is_stale = False
                        else:
                            quality = DataQuality.SYNTHETIC
                            is_stale = False

                        eff_mwh = spot_mwh + tariff_bd.total_variable_adders_eur_mwh
                        return AuthoritativeMarketPrice(
                            interval_start_utc=start_utc,
                            interval_end_utc=end_utc,
                            duration_minutes=duration_min,
                            interval_start_local=start_local,
                            interval_end_local=end_local,
                            interval_local_str=int_str,
                            interval_full_local_str=int_full_str,
                            spot_price_eur_mwh=round(spot_mwh, 2),
                            spot_price_eur_kwh=round(spot_mwh / 1000.0, 5),
                            effective_price_eur_mwh=round(eff_mwh, 2),
                            effective_price_eur_kwh=round(eff_mwh / 1000.0, 5),
                            tariff=tariff_bd,
                            provider=provider,
                            data_quality=quality,
                            price_basis="EFFECTIVE_VARIABLE_PRICE",
                            is_stale=is_stale,
                            is_synthetic=(quality == DataQuality.SYNTHETIC),
                            fetched_at_utc=matched.fetched_at_utc,
                            source_record_id=matched.source_record_id,
                        )
            except Exception as live_err:
                log.warning("Live market fetch failed during authoritative resolution: %s", live_err)

        # 3. Fallback: synthetic baseline with STRICT SYNTHETIC data quality flag
        # Synthetic data must NEVER be labeled LIVE!
        spot_mwh = 50.0
        eff_mwh = spot_mwh + tariff_bd.total_variable_adders_eur_mwh
        return AuthoritativeMarketPrice(
            interval_start_utc=start_utc,
            interval_end_utc=end_utc,
            duration_minutes=duration_min,
            interval_start_local=start_local,
            interval_end_local=end_local,
            interval_local_str=int_str,
            interval_full_local_str=int_full_str,
            spot_price_eur_mwh=round(spot_mwh, 2),
            spot_price_eur_kwh=round(spot_mwh / 1000.0, 5),
            effective_price_eur_mwh=round(eff_mwh, 2),
            effective_price_eur_kwh=round(eff_mwh / 1000.0, 5),
            tariff=tariff_bd,
            provider="SYNTHETIC",
            data_quality=DataQuality.SYNTHETIC,
            price_basis="EFFECTIVE_VARIABLE_PRICE",
            is_stale=True,
            is_synthetic=True,
            note="Synthetic baseline used; strictly classified as SYNTHETIC data quality.",
        )

    def get_current_authoritative_price(
        self,
        db: Session | None = None,
        tariff_params: TariffParameters | None = None,
        now_utc: datetime | None = None,
    ) -> AuthoritativeMarketPrice:
        """Convenience method to retrieve authoritative price for the current 15-minute interval."""
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(timezone.utc)
        start_utc, end_utc = snap_to_settlement_interval(now, resolution_minutes=15)
        return self.get_authoritative_price_for_interval(
            interval_start_utc=start_utc,
            interval_end_utc=end_utc,
            db=db,
            tariff_params=tariff_params,
            now_utc=now,
        )


# Global singleton helper
_market_price_service: MarketPriceService | None = None


def get_market_price_service(tz: ZoneInfo = VILNIUS_TZ) -> MarketPriceService:
    global _market_price_service
    if _market_price_service is None:
        _market_price_service = MarketPriceService(tz=tz)
    return _market_price_service


def get_authoritative_market_price(
    interval_start_utc: datetime,
    interval_end_utc: datetime | None = None,
    db: Session | None = None,
    tariff_params: TariffParameters | None = None,
    now_utc: datetime | None = None,
    source: str | None = None,
    tz: ZoneInfo = VILNIUS_TZ,
) -> AuthoritativeMarketPrice:
    """Global functional entrypoint for authoritative UTC-based market price lookup."""
    svc = get_market_price_service(tz=tz)
    return svc.get_authoritative_price_for_interval(
        interval_start_utc=interval_start_utc,
        interval_end_utc=interval_end_utc,
        db=db,
        tariff_params=tariff_params,
        now_utc=now_utc,
        source=source,
    )
