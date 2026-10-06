"""Unit tests for price collection service and persistence."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import select

from app.database.models import DayAheadPrice, RawMarketData
from app.database.session import init_db, make_engine, make_session_factory
from app.models.domain import PricePoint
from app.providers.csv_provider import CsvPriceProvider
from app.providers.mock import MockPriceProvider
from app.services.price_service import collect_day_ahead_prices

SAMPLE_CSV_DATA = """bidding_zone,delivery_start_utc,delivery_end_utc,price_eur_mwh,resolution_minutes
LT,2026-10-06T00:00:00Z,2026-10-06T00:15:00Z,75.00,15
LT,2026-10-06T00:15:00Z,2026-10-06T00:30:00Z,80.00,15
LT,2026-10-06T00:30:00Z,2026-10-06T00:45:00Z,65.00,15
LT,2026-10-06T00:45:00Z,2026-10-06T01:00:00Z,60.00,15
"""

SAMPLE_CSV_CORRECTED_DATA = """bidding_zone,delivery_start_utc,delivery_end_utc,price_eur_mwh,resolution_minutes
LT,2026-10-06T00:00:00Z,2026-10-06T00:15:00Z,95.00,15
LT,2026-10-06T00:15:00Z,2026-10-06T00:30:00Z,80.00,15
LT,2026-10-06T00:30:00Z,2026-10-06T00:45:00Z,65.00,15
LT,2026-10-06T00:45:00Z,2026-10-06T01:00:00Z,60.00,15
"""


@pytest.fixture
def memory_db():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        yield session


def test_collect_day_ahead_prices_mock(memory_db):
    tz = ZoneInfo("Europe/Vilnius")
    provider = MockPriceProvider(tz=tz, resolution_minutes=15, seed=42)
    target_date = date(2026, 10, 6)

    res = collect_day_ahead_prices(
        provider=provider,
        bidding_zone="LT",
        session=memory_db,
        tz=tz,
        target_date=target_date,
    )

    # 24 hours * 4 quarters = 96 intervals
    assert res.total_points_fetched == 96
    assert res.inserted == 96
    assert res.duplicates == 0
    assert len(res.corrected_versions) == 0
    assert res.raw_market_data_id is not None
    assert res.min_price_eur_mwh is not None
    assert res.max_price_eur_mwh is not None
    assert res.min_price_eur_mwh < res.max_price_eur_mwh

    # Verify rows in database
    prices_in_db = list(memory_db.execute(select(DayAheadPrice)).scalars())
    assert len(prices_in_db) == 96

    # Verify raw market payload audit
    raw_in_db = list(memory_db.execute(select(RawMarketData)).scalars())
    assert len(raw_in_db) == 1
    assert raw_in_db[0].source == "mock"
    assert raw_in_db[0].bidding_zone == "LT"
    assert "provider" in raw_in_db[0].payload


def test_collect_day_ahead_prices_idempotency_and_versioning(memory_db):
    tz = ZoneInfo("Europe/Vilnius")
    provider1 = CsvPriceProvider(csv_content=SAMPLE_CSV_DATA)

    # First collect
    res1 = collect_day_ahead_prices(
        provider=provider1,
        bidding_zone="LT",
        session=memory_db,
        tz=tz,
        start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
    )
    assert res1.inserted == 4
    assert res1.duplicates == 0

    # Collect same again -> duplicates detected, no inserts
    res2 = collect_day_ahead_prices(
        provider=provider1,
        bidding_zone="LT",
        session=memory_db,
        tz=tz,
        start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
    )
    assert res2.inserted == 0
    assert res2.duplicates == 4

    # Collect corrected data (first interval changed from 75 to 95)
    provider2 = CsvPriceProvider(csv_content=SAMPLE_CSV_CORRECTED_DATA)
    res3 = collect_day_ahead_prices(
        provider=provider2,
        bidding_zone="LT",
        session=memory_db,
        tz=tz,
        start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
    )
    assert res3.inserted == 1
    assert res3.duplicates == 3
    assert len(res3.corrected_versions) == 1
    assert res3.corrected_versions[0]["new_version"] == 2
    assert res3.corrected_versions[0]["new_price"] == 95.00
