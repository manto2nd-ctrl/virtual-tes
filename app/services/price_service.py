"""Price collection and persistence service.

Orchestrates price fetching from any PriceProvider (ENTSO-E, CSV, Mock),
stores raw audit payloads, versions/inserts price points via PriceRepository,
and returns rich summary statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import mean
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.core.timegrid import ensure_utc, local_day_bounds_utc
from app.database.repositories import PriceInsertReport, PriceRepository
from app.models.domain import PricePoint
from app.providers.price_provider import PriceProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PriceCollectionResult:
    bidding_zone: str
    source: str
    period_start_utc: datetime
    period_end_utc: datetime
    total_points_fetched: int
    inserted: int
    duplicates: int
    corrected_versions: list[dict]
    raw_market_data_id: int | None
    min_price_eur_mwh: float | None
    max_price_eur_mwh: float | None
    mean_price_eur_mwh: float | None
    points: list[PricePoint] = field(default_factory=list)


def collect_day_ahead_prices(
    provider: PriceProvider,
    bidding_zone: str,
    session: Session,
    tz: ZoneInfo,
    target_date: date | None = None,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> PriceCollectionResult:
    """Fetch day-ahead prices for a given calendar day (or explicit UTC range) and persist them.

    Either `target_date` OR (`start_utc` and `end_utc`) must be provided.
    """
    if target_date is not None:
        p_start_utc, p_end_utc = local_day_bounds_utc(target_date, tz)
    elif start_utc is not None and end_utc is not None:
        p_start_utc = ensure_utc(start_utc)
        p_end_utc = ensure_utc(end_utc)
    else:
        raise ValueError("Must provide either target_date or both start_utc and end_utc.")

    log.info(
        "collecting day-ahead prices",
        extra={
            "ctx": {
                "provider": provider.source,
                "bidding_zone": bidding_zone,
                "start_utc": p_start_utc.isoformat(),
                "end_utc": p_end_utc.isoformat(),
            }
        },
    )

    fetch_result = provider.fetch_day_ahead(
        bidding_zone=bidding_zone,
        start_utc=p_start_utc,
        end_utc=p_end_utc,
    )

    repo = PriceRepository(session=session, tz=tz)
    report: PriceInsertReport = repo.store_fetch(fetch_result)
    session.commit()

    prices = [p.price_eur_mwh for p in fetch_result.points]
    min_price = min(prices) if prices else None
    max_price = max(prices) if prices else None
    avg_price = round(mean(prices), 2) if prices else None

    return PriceCollectionResult(
        bidding_zone=bidding_zone,
        source=provider.source,
        period_start_utc=p_start_utc,
        period_end_utc=p_end_utc,
        total_points_fetched=len(fetch_result.points),
        inserted=report.inserted,
        duplicates=report.duplicates,
        corrected_versions=report.corrected_versions,
        raw_market_data_id=report.raw_market_data_id,
        min_price_eur_mwh=min_price,
        max_price_eur_mwh=max_price,
        mean_price_eur_mwh=avg_price,
        points=fetch_result.points,
    )
