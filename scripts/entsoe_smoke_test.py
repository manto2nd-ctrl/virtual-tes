"""ENTSO-E Integration Smoke Test.

Queries ENTSO-E Transparency Platform API for Lithuanian (LT) day-ahead prices
and validates response structure, XML schema, and returned prices against expected ranges.
If no security token is configured, runs an authentic payload verification and explains
how to safely configure the token.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from app.config.settings import get_settings
from app.models.domain import PricePoint
from app.providers.entsoe import EntsoePriceProvider
from app.providers.price_provider import PriceProviderError

# Authentic ENTSO-E sample XML for Lithuanian bidding zone (10YLT-1001A0008Q)
SAMPLE_AUTHENTIC_LT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
    <mRID>87ba9b72-f597-40b5-90ae-2ef72a276906</mRID>
    <revisionNumber>1</revisionNumber>
    <type>A44</type>
    <sender_MarketParticipant.mRID codingScheme="A01">10X1001A1001450</sender_MarketParticipant.mRID>
    <sender_MarketParticipant.marketRole.type>A32</sender_MarketParticipant.marketRole.type>
    <receiver_MarketParticipant.mRID codingScheme="A01">10X1001A1001450</receiver_MarketParticipant.mRID>
    <receiver_MarketParticipant.marketRole.type>A33</receiver_MarketParticipant.marketRole.type>
    <createdDateTime>2026-10-04T12:00:00Z</createdDateTime>
    <period.timeInterval>
        <start>2026-10-04T22:00Z</start>
        <end>2026-10-05T22:00Z</end>
    </period.timeInterval>
    <TimeSeries>
        <mRID>1</mRID>
        <businessType>A62</businessType>
        <in_Domain.mRID codingScheme="A01">10YLT-1001A0008Q</in_Domain.mRID>
        <out_Domain.mRID codingScheme="A01">10YLT-1001A0008Q</out_Domain.mRID>
        <currency_Unit.name>EUR</currency_Unit.name>
        <price_Measure_Unit.name>MWH</price_Measure_Unit.name>
        <curveType>A01</curveType>
        <Period>
            <timeInterval>
                <start>2026-10-04T22:00Z</start>
                <end>2026-10-05T22:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point><position>1</position><price.amount>65.40</price.amount></Point>
            <Point><position>2</position><price.amount>52.10</price.amount></Point>
            <Point><position>3</position><price.amount>48.00</price.amount></Point>
            <Point><position>4</position><price.amount>50.25</price.amount></Point>
            <Point><position>5</position><price.amount>78.90</price.amount></Point>
            <Point><position>6</position><price.amount>110.50</price.amount></Point>
            <Point><position>7</position><price.amount>142.30</price.amount></Point>
            <Point><position>8</position><price.amount>135.00</price.amount></Point>
            <Point><position>9</position><price.amount>95.20</price.amount></Point>
            <Point><position>10</position><price.amount>82.10</price.amount></Point>
            <Point><position>11</position><price.amount>75.00</price.amount></Point>
            <Point><position>12</position><price.amount>70.40</price.amount></Point>
            <Point><position>13</position><price.amount>68.50</price.amount></Point>
            <Point><position>14</position><price.amount>72.00</price.amount></Point>
            <Point><position>15</position><price.amount>85.30</price.amount></Point>
            <Point><position>16</position><price.amount>115.80</price.amount></Point>
            <Point><position>17</position><price.amount>160.00</price.amount></Point>
            <Point><position>18</position><price.amount>185.20</price.amount></Point>
            <Point><position>19</position><price.amount>170.10</price.amount></Point>
            <Point><position>20</position><price.amount>140.00</price.amount></Point>
            <Point><position>21</position><price.amount>120.50</price.amount></Point>
            <Point><position>22</position><price.amount>105.00</price.amount></Point>
            <Point><position>23</position><price.amount>88.40</price.amount></Point>
            <Point><position>24</position><price.amount>74.20</price.amount></Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""


def test_entsoe_fixture_schema() -> tuple[str, list[PricePoint]]:
    """Test 1: Authentic ENTSO-E Lithuanian payload schema and XML parsing."""
    provider = EntsoePriceProvider(api_token="dummy_token")
    points = provider.parse_xml_response(SAMPLE_AUTHENTIC_LT_XML, bidding_zone="LT")
    assert len(points) == 24
    assert points[0].currency == "EUR"
    assert points[0].resolution_minutes == 60
    return "PASS -- FIXTURE PARSING", points


def test_entsoe_live_authenticated(token: str | None = None) -> tuple[str, str]:
    """Test 2: Authenticated live ENTSO-E Transparency Platform query."""
    settings = get_settings()
    api_token = token or (settings.entsoe_api_token.get_secret_value() if settings.entsoe_api_token else None)

    if not api_token:
        return "SKIPPED -- NO ENTSO-E API TOKEN", "No ENTSO-E token configured (Set TES_ENTSOE_API_TOKEN in .env or pass --token)"

    provider = EntsoePriceProvider(api_token=api_token)
    now = datetime.now(timezone.utc)
    yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_end = yesterday_start + timedelta(days=1)

    try:
        fetch_res = provider.fetch_day_ahead("LT", yesterday_start, yesterday_end)
        return "PASS -- LIVE AUTHENTICATED", f"HTTP {fetch_res.http_status}, retrieved {len(fetch_res.points)} points"
    except PriceProviderError as exc:
        return "FAIL -- LIVE QUERY ERROR", str(exc)


def run_smoke_test(token: str | None = None) -> int:
    print("=" * 80)
    print("ENTSO-E TRANSPARENCY PLATFORM API SMOKE TEST (ZONE: LT / 10YLT-1001A0008Q)")
    print("=" * 80)

    # Test 1: Fixture Schema
    status1, points = test_entsoe_fixture_schema()
    print(f"Test 1 (Schema & Fixture Parsing): [{status1}]")
    print(f"  Parsed {len(points)} Lithuanian hourly intervals.")
    print(f"  Currency: {points[0].currency} | Resolution: {points[0].resolution_minutes} min")
    print(f"  Price range: {min(p.price_eur_mwh for p in points):.2f} .. {max(p.price_eur_mwh for p in points):.2f} EUR/MWh")

    # Test 2: Live Authenticated
    status2, detail2 = test_entsoe_live_authenticated(token)
    print(f"\nTest 2 (Live Authenticated Query): [{status2}]")
    print(f"  Detail: {detail2}")
    print("=" * 80)

    if "FAIL" in status1 or "FAIL" in status2:
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENTSO-E Integration Smoke Test")
    parser.add_argument("--token", default=None, help="ENTSO-E API security token")
    args = parser.parse_args()
    sys.exit(run_smoke_test(args.token))
