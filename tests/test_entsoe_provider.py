"""Unit tests for ENTSO-E price provider."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
import httpx

from app.models.domain import PricePoint
from app.providers.entsoe import EntsoePriceProvider
from app.providers.price_provider import PriceProviderError

SAMPLE_ENTSOE_15M_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
    <mRID>doc123</mRID>
    <type>A44</type>
    <sender_MarketParticipant.mRID codingScheme="A01">10X1001A1001A450</sender_MarketParticipant.mRID>
    <createdDateTime>2026-10-05T12:00:00Z</createdDateTime>
    <period.timeInterval>
        <start>2026-10-06T00:00Z</start>
        <end>2026-10-06T01:00Z</end>
    </period.timeInterval>
    <TimeSeries>
        <mRID>ts1</mRID>
        <businessType>A62</businessType>
        <currency_Unit.name>EUR</currency_Unit.name>
        <price_Measure_Unit.name>MWH</price_Measure_Unit.name>
        <Period>
            <timeInterval>
                <start>2026-10-06T00:00Z</start>
                <end>2026-10-06T01:00Z</end>
            </timeInterval>
            <resolution>PT15M</resolution>
            <Point>
                <position>1</position>
                <price.amount>75.40</price.amount>
            </Point>
            <Point>
                <position>2</position>
                <price.amount>68.20</price.amount>
            </Point>
            <Point>
                <position>3</position>
                <price.amount>60.00</price.amount>
            </Point>
            <Point>
                <position>4</position>
                <price.amount>54.10</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""

SAMPLE_ENTSOE_60M_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
    <createdDateTime>2026-10-05T12:00:00Z</createdDateTime>
    <TimeSeries>
        <currency_Unit.name>EUR</currency_Unit.name>
        <Period>
            <timeInterval>
                <start>2026-10-06T00:00Z</start>
                <end>2026-10-06T02:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <price.amount>85.50</price.amount>
            </Point>
            <Point>
                <position>2</position>
                <price.amount>92.30</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""

SAMPLE_ENTSOE_ERROR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
    <createdDateTime>2026-10-05T12:00:00Z</createdDateTime>
    <Reason>
        <code>999</code>
        <text>No matching data found for Data item Day-ahead Prices [12.1.D] and interval 2026-10-06T00:00:00Z/2026-10-07T00:00:00Z.</text>
    </Reason>
</Acknowledgement_MarketDocument>
"""


def test_entsoe_parse_15m_xml():
    provider = EntsoePriceProvider(api_token="test_token")
    points = provider.parse_xml_response(SAMPLE_ENTSOE_15M_XML, bidding_zone="LT")

    assert len(points) == 4
    assert points[0].price_eur_mwh == 75.40
    assert points[0].resolution_minutes == 15
    assert points[0].delivery_start_utc == datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)
    assert points[0].delivery_end_utc == datetime(2026, 10, 6, 0, 15, tzinfo=timezone.utc)
    assert points[0].bidding_zone == "LT"

    assert points[1].price_eur_mwh == 68.20
    assert points[1].delivery_start_utc == datetime(2026, 10, 6, 0, 15, tzinfo=timezone.utc)
    assert points[1].delivery_end_utc == datetime(2026, 10, 6, 0, 30, tzinfo=timezone.utc)

    assert points[3].price_eur_mwh == 54.10
    assert points[3].delivery_start_utc == datetime(2026, 10, 6, 0, 45, tzinfo=timezone.utc)
    assert points[3].delivery_end_utc == datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)


def test_entsoe_parse_60m_xml():
    provider = EntsoePriceProvider(api_token="test_token")
    points = provider.parse_xml_response(SAMPLE_ENTSOE_60M_XML, bidding_zone="LT")

    assert len(points) == 2
    assert points[0].resolution_minutes == 60
    assert points[0].price_eur_mwh == 85.50
    assert points[0].delivery_start_utc == datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)
    assert points[0].delivery_end_utc == datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)

    assert points[1].resolution_minutes == 60
    assert points[1].price_eur_mwh == 92.30
    assert points[1].delivery_start_utc == datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    assert points[1].delivery_end_utc == datetime(2026, 10, 6, 2, 0, tzinfo=timezone.utc)


def test_entsoe_parse_acknowledgement_error():
    provider = EntsoePriceProvider(api_token="test_token")
    with pytest.raises(PriceProviderError) as exc_info:
        provider.parse_xml_response(SAMPLE_ENTSOE_ERROR_XML, bidding_zone="LT")
    assert "999" in str(exc_info.value)
    assert "No matching data found" in str(exc_info.value)


def test_entsoe_parse_malformed_xml():
    provider = EntsoePriceProvider(api_token="test_token")
    with pytest.raises(PriceProviderError, match="Failed to parse ENTSO-E XML"):
        provider.parse_xml_response("<invalid><unclosed>", bidding_zone="LT")

    with pytest.raises(PriceProviderError, match="Empty response"):
        provider.parse_xml_response("", bidding_zone="LT")


def test_entsoe_missing_token_raises():
    provider = EntsoePriceProvider(api_token=None)
    with pytest.raises(PriceProviderError, match="ENTSO-E API token is not configured"):
        provider.fetch_day_ahead(
            bidding_zone="LT",
            start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
            end_utc=datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc),
        )


def test_entsoe_unknown_bidding_zone():
    provider = EntsoePriceProvider(api_token="test_token")
    with pytest.raises(PriceProviderError, match="Unknown bidding zone"):
        provider.fetch_day_ahead(
            bidding_zone="INVALID_ZONE",
            start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
            end_utc=datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc),
        )


def test_entsoe_http_retry_and_success():
    """Verify that transient 500 errors trigger backoff retry and succeed on subsequent attempt."""
    responses = [
        httpx.Response(500, text="Internal Server Error"),
        httpx.Response(200, text=SAMPLE_ENTSOE_15M_XML),
    ]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)

    provider = EntsoePriceProvider(
        api_token="test_token",
        http_client=client,
        initial_backoff_sec=0.01,
        max_retries=3,
    )

    result = provider.fetch_day_ahead(
        bidding_zone="LT",
        start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
    )

    assert len(result.points) == 4
    assert result.http_status == 200
    assert result.raw_payload == SAMPLE_ENTSOE_15M_XML
    assert result.points[0].price_eur_mwh == 75.40


def test_entsoe_http_401_fails_fast():
    """Verify that non-transient 4xx errors (e.g. 401 Unauthorized) fail fast without retrying."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, text="Unauthorized: Invalid security token")

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)

    provider = EntsoePriceProvider(
        api_token="bad_token",
        http_client=client,
        initial_backoff_sec=0.01,
        max_retries=3,
    )

    with pytest.raises(PriceProviderError, match="HTTP 401"):
        provider.fetch_day_ahead(
            bidding_zone="LT",
            start_utc=datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc),
            end_utc=datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
        )

    # Must fail on the first attempt without looping through retries
    assert call_count == 1
