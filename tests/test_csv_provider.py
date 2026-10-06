"""Unit tests for CsvPriceProvider."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

from app.providers.csv_provider import CsvPriceProvider
from app.providers.price_provider import PriceProviderError

SAMPLE_CSV_STANDARD = """bidding_zone,delivery_start_utc,delivery_end_utc,price_eur_mwh,resolution_minutes
LT,2026-10-06T00:00:00Z,2026-10-06T00:15:00Z,78.50,15
LT,2026-10-06T00:15:00Z,2026-10-06T00:30:00Z,65.20,15
LT,2026-10-06T00:30:00Z,2026-10-06T00:45:00Z,50.10,15
LT,2026-10-06T00:45:00Z,2026-10-06T01:00:00Z,42.00,15
"""

SAMPLE_CSV_MINIMAL = """timestamp,price
2026-10-06 00:00:00,82.4
2026-10-06 00:15:00,74.1
2026-10-06 00:30:00,61.9
"""

SAMPLE_CSV_EUROPEAN = """timestamp;price
2026-10-06T00:00:00Z;85,50
2026-10-06T01:00:00Z;90,25
"""


def test_csv_parse_standard():
    provider = CsvPriceProvider(csv_content=SAMPLE_CSV_STANDARD)
    points = provider.parse_csv(SAMPLE_CSV_STANDARD, default_bidding_zone="LT")

    assert len(points) == 4
    assert points[0].bidding_zone == "LT"
    assert points[0].delivery_start_utc == datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)
    assert points[0].delivery_end_utc == datetime(2026, 10, 6, 0, 15, tzinfo=timezone.utc)
    assert points[0].price_eur_mwh == 78.50
    assert points[0].resolution_minutes == 15

    assert points[3].price_eur_mwh == 42.00
    assert points[3].delivery_start_utc == datetime(2026, 10, 6, 0, 45, tzinfo=timezone.utc)


def test_csv_parse_minimal():
    tz = ZoneInfo("Europe/Vilnius")
    provider = CsvPriceProvider(csv_content=SAMPLE_CSV_MINIMAL, default_tz=tz)
    points = provider.parse_csv(SAMPLE_CSV_MINIMAL, default_bidding_zone="LT")

    assert len(points) == 3
    # Timestamp without offset parsed with default_tz and converted to UTC
    # Vilnius is UTC+3 in October (EEST)
    assert points[0].delivery_start_utc == datetime(2026, 10, 5, 21, 0, tzinfo=timezone.utc)
    assert points[0].resolution_minutes == 15
    assert points[0].price_eur_mwh == 82.40


def test_csv_parse_european_semicolon_and_comma_decimal():
    provider = CsvPriceProvider(csv_content=SAMPLE_CSV_EUROPEAN)
    points = provider.parse_csv(SAMPLE_CSV_EUROPEAN, default_bidding_zone="LT")

    assert len(points) == 2
    assert points[0].price_eur_mwh == 85.50
    assert points[1].price_eur_mwh == 90.25
    assert points[0].resolution_minutes == 60


def test_csv_fetch_day_ahead_filtering():
    provider = CsvPriceProvider(csv_content=SAMPLE_CSV_STANDARD)
    res = provider.fetch_day_ahead(
        bidding_zone="LT",
        start_utc=datetime(2026, 10, 6, 0, 15, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 6, 0, 45, tzinfo=timezone.utc),
    )

    assert len(res.points) == 2
    assert res.points[0].delivery_start_utc == datetime(2026, 10, 6, 0, 15, tzinfo=timezone.utc)
    assert res.points[1].delivery_start_utc == datetime(2026, 10, 6, 0, 30, tzinfo=timezone.utc)
    assert res.raw_content_type == "text/csv"
    assert res.raw_payload is not None


def test_csv_from_file_path(tmp_path: Path):
    csv_file = tmp_path / "prices.csv"
    csv_file.write_text(SAMPLE_CSV_STANDARD, encoding="utf-8")

    provider = CsvPriceProvider(file_path=csv_file)
    res = provider.fetch_day_ahead(
        bidding_zone="LT",
        start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
    )
    assert len(res.points) == 4


def test_csv_missing_file_raises(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.csv"
    provider = CsvPriceProvider(file_path=non_existent)
    with pytest.raises(PriceProviderError, match="CSV file not found"):
        provider.fetch_day_ahead(
            bidding_zone="LT",
            start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
            end_utc=datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
        )


def test_csv_missing_columns_raises():
    invalid_csv = "col_a,col_b\n1,2"
    provider = CsvPriceProvider(csv_content=invalid_csv)
    with pytest.raises(PriceProviderError, match="must contain a timestamp column"):
        provider.parse_csv(invalid_csv)


def test_csv_utf8_bom_support():
    bom_csv = "\ufefftimestamp,price\n2026-10-06T00:00:00Z,65.0\n"
    provider = CsvPriceProvider(csv_content=bom_csv)
    points = provider.parse_csv(bom_csv)
    assert len(points) == 1
    assert points[0].price_eur_mwh == 65.0
