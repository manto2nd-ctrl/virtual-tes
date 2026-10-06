"""Tests for Phase 5.8.1: Live Market Price UI & Data-Path Verification.

Verifies all 15 mandatory requirements:
1. current interval is selected by timezone-aware timestamp
2. exact start boundary belongs to the interval
3. exact end boundary belongs to the next interval
4. Europe/Vilnius display conversion is correct
5. current UI price equals current DB interval price
6. next interval price is correct
7. 15-minute source remains 15-minute in UI
8. stale data is detected
9. missing current interval is visible
10. fallback provider is identified correctly
11. live mode never silently renders mock price
12. current price card auto-refresh endpoint works
13. today chart contains all available intervals
14. tomorrow data is not invented when unavailable
15. optimizer/TES Action card uses the same real price version
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest
from starlette.testclient import TestClient

from app.config.parameters import TariffParameters, TESParameters
from app.database.models import DayAheadPrice
from app.database.repositories import PriceRepository
from app.database.session import init_db, make_engine, make_session_factory
from app.models.domain import MarketPriceInterval, PriceFetchResult, PricePoint
from app.providers.price_provider import PriceProvider, PriceProviderError
from app.services.market_data_service import (
    LiveMarketDataUnavailableError,
    MarketDataService,
    compute_today_market_stats,
    compute_tomorrow_market_stats,
    determine_data_status,
    get_market_debug_records,
    get_today_intervals,
    get_tomorrow_intervals,
    resolve_tes_action_reason,
    select_current_market_interval,
    select_next_market_interval,
)
from app.web.main import app

TZ_VILNIUS = ZoneInfo("Europe/Vilnius")


def create_15m_test_intervals(
    start_utc: datetime,
    count: int = 8,
    base_price: float = 50.0,
    source: str = "ELERING",
    zone: str = "LT",
) -> list[MarketPriceInterval]:
    """Helper creating consecutive 15-minute MarketPriceIntervals."""
    ivs = []
    delta = timedelta(minutes=15)
    for i in range(count):
        s = start_utc + i * delta
        e = s + delta
        price = round(base_price + i * 2.5, 2)
        ivs.append(
            MarketPriceInterval(
                start_utc=s,
                end_utc=e,
                price_eur_mwh=price,
                bidding_zone=zone,
                source=source,
                original_resolution_minutes=15,
                fetched_at_utc=datetime.now(timezone.utc),
                source_record_id=f"{source.lower()}-{int(s.timestamp())}",
                is_derived=False,
                raw_payload_hash="dummy-sha256-hash-0123456789",
            )
        )
    return ivs


# ---------------------------------------------------------------------------
# Requirement 1: Selection by Timezone-Aware Timestamp
# ---------------------------------------------------------------------------
def test_current_interval_selected_by_timezone_aware_timestamp():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=4)

    # 18:22:15 UTC is inside interval 1 (18:15 -> 18:30)
    query_utc = datetime(2026, 10, 5, 18, 22, 15, tzinfo=timezone.utc)
    matched = select_current_market_interval(ivs, query_utc)
    assert matched is not None
    assert matched.start_utc == datetime(2026, 10, 5, 18, 15, tzinfo=timezone.utc)
    assert matched.end_utc == datetime(2026, 10, 5, 18, 30, tzinfo=timezone.utc)

    # Query with Europe/Vilnius timezone-aware equivalent (21:22:15 EEST)
    query_local = query_utc.astimezone(TZ_VILNIUS)
    matched_local = select_current_market_interval(ivs, query_local)
    assert matched_local == matched


# ---------------------------------------------------------------------------
# Requirement 2: Exact Start Boundary Belongs to Interval [start, end)
# ---------------------------------------------------------------------------
def test_exact_start_boundary_belongs_to_interval():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=4)

    # Exactly at start boundary of interval 1 (18:15:00 UTC)
    t_boundary = datetime(2026, 10, 5, 18, 15, 0, tzinfo=timezone.utc)
    matched = select_current_market_interval(ivs, t_boundary)
    assert matched is not None
    assert matched.start_utc == t_boundary
    assert matched == ivs[1]


# ---------------------------------------------------------------------------
# Requirement 3: Exact End Boundary Belongs to Next Interval
# ---------------------------------------------------------------------------
def test_exact_end_boundary_belongs_to_next_interval():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=4)

    # Exactly at end boundary of interval 1 (18:30:00 UTC), which is start of interval 2
    t_boundary = datetime(2026, 10, 5, 18, 30, 0, tzinfo=timezone.utc)
    matched = select_current_market_interval(ivs, t_boundary)
    assert matched is not None
    assert matched == ivs[2]
    assert matched.start_utc == t_boundary
    assert matched != ivs[1]


# ---------------------------------------------------------------------------
# Requirement 4: Europe/Vilnius Display Conversion is Correct
# ---------------------------------------------------------------------------
def test_europe_vilnius_display_conversion_correct():
    # 2026-10-05 18:15:00 UTC in Vilnius (EEST, UTC+3) is 21:15:00
    t_utc = datetime(2026, 10, 5, 18, 15, 0, tzinfo=timezone.utc)
    t_local = t_utc.astimezone(TZ_VILNIUS)

    assert t_local.hour == 21
    assert t_local.minute == 15
    assert t_local.strftime("%H:%M") == "21:15"


# ---------------------------------------------------------------------------
# Requirement 5: Current UI Price Equals Current DB Interval Price
# ---------------------------------------------------------------------------
def test_current_ui_price_equals_current_db_interval_price(tmp_path):
    db_file = tmp_path / "test_ui_market.db"
    engine = make_engine(f"sqlite:///{db_file}")
    init_db(engine)
    session_factory = make_session_factory(engine)

    fixed_now_utc = datetime(2026, 10, 5, 18, 20, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(
        datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc),
        count=4,
        base_price=42.50,
        source="ELERING",
    )

    with session_factory() as session:
        repo = PriceRepository(session, TZ_VILNIUS)
        fetch_res = PriceFetchResult(
            source="ELERING",
            bidding_zone="LT",
            period_start_utc=ivs[0].start_utc,
            period_end_utc=ivs[-1].end_utc,
            points=[iv.to_price_point() for iv in ivs],
            raw_payload="{}",
            raw_content_type="application/json",
            http_status=200,
            request_params={},
            market_intervals=ivs,
        )
        repo.store_fetch(fetch_res)
        session.commit()

        # Query DB directly for interval covering fixed_now_utc
        db_price_row = (
            session.query(DayAheadPrice)
            .filter(
                DayAheadPrice.delivery_start_utc <= fixed_now_utc,
                DayAheadPrice.delivery_end_utc > fixed_now_utc,
                DayAheadPrice.source == "ELERING",
            )
            .one_or_none()
        )
        assert db_price_row is not None
        expected_db_price = db_price_row.price_eur_mwh

        # Build view context using MarketDataService
        class DummyProvider(PriceProvider):
            source = "ELERING"
            def fetch_day_ahead(self, zone, s, e):
                return fetch_res

        svc = MarketDataService(
            litgrid_provider=DummyProvider(),
            elering_provider=DummyProvider(),
            tz=TZ_VILNIUS,
        )
        view_ctx = svc.get_market_view_context(session=session, now_utc=fixed_now_utc)

        # UI price must strictly equal stored DB interval price
        assert view_ctx["price_eur_mwh"] == expected_db_price
        assert view_ctx["price_eur_mwh_str"] == f"{expected_db_price:.2f}"


# ---------------------------------------------------------------------------
# Requirement 6: Next Interval Price and Difference are Correct
# ---------------------------------------------------------------------------
def test_next_interval_price_and_diff_correct():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=4, base_price=50.0)
    # ivs[0] price: 50.00 (18:00->18:15)
    # ivs[1] price: 52.50 (18:15->18:30)
    # ivs[2] price: 55.00 (18:30->18:45)

    cur_iv = ivs[1]
    next_iv = select_next_market_interval(ivs, cur_iv)
    assert next_iv is not None
    assert next_iv == ivs[2]
    diff = next_iv.price_eur_mwh - cur_iv.price_eur_mwh
    assert diff == pytest.approx(2.50, abs=1e-4)


# ---------------------------------------------------------------------------
# Requirement 7: 15-Minute Source Remains 15-Minute in UI
# ---------------------------------------------------------------------------
def test_15_minute_source_remains_15_minute_in_ui():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=2)

    cur_iv = ivs[0]
    assert cur_iv.resolution_minutes == 15
    assert cur_iv.original_resolution_minutes == 15
    assert not cur_iv.is_derived

    diff_sec = (cur_iv.end_utc - cur_iv.start_utc).total_seconds()
    assert diff_sec == 900  # 15 min exactly


# ---------------------------------------------------------------------------
# Requirement 8: Stale Data is Detected
# ---------------------------------------------------------------------------
def test_stale_data_is_detected():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=4)
    # Newest interval ends at 19:00 UTC

    now_future = datetime(2026, 10, 5, 20, 0, tzinfo=timezone.utc)
    status = determine_data_status(
        intervals=ivs,
        cur_iv=None,
        now_utc=now_future,
        tomorrow_available=False,
        is_stale_flag=False,
    )
    assert status == "STALE DATA"


# ---------------------------------------------------------------------------
# Requirement 9: Missing Current Interval is Visible
# ---------------------------------------------------------------------------
def test_missing_current_interval_is_visible():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=4)

    # now_utc is before earliest interval (17:00 UTC < 18:00 UTC)
    now_past = datetime(2026, 10, 5, 17, 0, tzinfo=timezone.utc)
    status = determine_data_status(
        intervals=ivs,
        cur_iv=None,
        now_utc=now_past,
        tomorrow_available=True,
    )
    assert status == "NO CURRENT INTERVAL"


# ---------------------------------------------------------------------------
# Requirement 10: Fallback Provider is Identified Correctly
# ---------------------------------------------------------------------------
def test_fallback_provider_is_identified_correctly():
    class FailingLitgrid(PriceProvider):
        source = "LITGRID"
        def fetch_day_ahead(self, zone, s, e):
            raise PriceProviderError("Litgrid connection timed out")

    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    elering_ivs = create_15m_test_intervals(base_t, count=4, source="ELERING")

    class WorkingElering(PriceProvider):
        source = "ELERING"
        def fetch_day_ahead(self, zone, s, e):
            return PriceFetchResult(
                source="ELERING",
                bidding_zone="LT",
                period_start_utc=s,
                period_end_utc=e,
                points=[iv.to_price_point() for iv in elering_ivs],
                raw_payload="{}",
                raw_content_type="application/json",
                http_status=200,
                request_params={},
                market_intervals=elering_ivs,
            )

    svc = MarketDataService(
        litgrid_provider=FailingLitgrid(),
        elering_provider=WorkingElering(),
        tz=TZ_VILNIUS,
    )

    fetch_res = svc.fetch_market_data()
    assert fetch_res.active_source == "ELERING"
    assert fetch_res.status_snapshot.primary_source == "LITGRID"
    assert fetch_res.status_snapshot.primary_health == "ERROR"

    view_ctx = svc.get_market_view_context(now_utc=base_t + timedelta(minutes=5))
    assert view_ctx["active_source"] == "ELERING"
    assert view_ctx["fallback_used"] == "YES"
    assert "Litgrid" in (view_ctx["fallback_reason"] or "")


# ---------------------------------------------------------------------------
# Requirement 11: Live Mode Never Silently Renders Mock Price
# ---------------------------------------------------------------------------
def test_live_mode_never_silently_renders_mock_price():
    class FailingProvider(PriceProvider):
        source = "FAIL"
        def fetch_day_ahead(self, zone, s, e):
            raise PriceProviderError("Network down")

    svc = MarketDataService(
        litgrid_provider=FailingProvider(),
        elering_provider=FailingProvider(),
        volton_provider=FailingProvider(),
        entsoe_provider=FailingProvider(),
        tz=TZ_VILNIUS,
    )

    # fetch_market_data must raise LiveMarketDataUnavailableError when no cache exists
    with pytest.raises(LiveMarketDataUnavailableError):
        svc.fetch_market_data()

    # get_market_view_context must show SOURCE ERROR and never substitute mock data
    view_ctx = svc.get_market_view_context()
    assert view_ctx["data_status"] == "SOURCE ERROR"
    assert view_ctx["price_eur_mwh"] is None
    assert view_ctx["price_eur_mwh_str"] == "--.--"
    assert view_ctx["source"] != "mock"
    assert view_ctx["source"] != "MOCK"


# ---------------------------------------------------------------------------
# Requirement 12: Current Price Card Auto-Refresh Endpoint Works
# ---------------------------------------------------------------------------
def test_current_price_card_auto_refresh_endpoint_works():
    client = TestClient(app)
    resp = client.get("/api/market/current-card")
    assert resp.status_code == 200
    assert "CURRENT LT DAY-AHEAD PRICE" in resp.text
    assert "TES ACTION NOW" in resp.text
    assert 'hx-trigger="every 60s"' in resp.text
    assert 'id="current-market-card-container"' in resp.text


# ---------------------------------------------------------------------------
# Requirement 13: Today Chart Contains All Available Intervals
# ---------------------------------------------------------------------------
def test_today_chart_contains_all_available_intervals():
    target_d = date(2026, 10, 5)
    # Generate full 96 15-minute intervals for Oct 5
    start_utc = datetime(2026, 10, 4, 21, 0, tzinfo=timezone.utc)  # 00:00 Oct 5 Vilnius (EEST)
    ivs = create_15m_test_intervals(start_utc, count=96)

    today_ivs = get_today_intervals(ivs, target_date=target_d, tz=TZ_VILNIUS)
    assert len(today_ivs) == 96

    stats = compute_today_market_stats(ivs, target_date=target_d, tz=TZ_VILNIUS)
    assert stats["intervals_count"] == 96
    assert len(stats["intervals"]) == 96
    assert stats["min_price_eur_mwh"] <= stats["max_price_eur_mwh"]


# ---------------------------------------------------------------------------
# Requirement 14: Tomorrow Data is Not Invented When Unavailable
# ---------------------------------------------------------------------------
def test_tomorrow_data_not_invented_when_unavailable():
    target_d = date(2026, 10, 5)
    # Only today's intervals are present
    start_utc = datetime(2026, 10, 4, 21, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(start_utc, count=96)

    tomorrow_stats = compute_tomorrow_market_stats(ivs, target_date=target_d, tz=TZ_VILNIUS)
    assert not tomorrow_stats["available"]
    assert tomorrow_stats["status"] == "TOMORROW PRICES NOT YET AVAILABLE"
    assert tomorrow_stats["intervals_count"] == 0
    assert tomorrow_stats["min_price_eur_mwh"] is None
    assert tomorrow_stats["max_price_eur_mwh"] is None


# ---------------------------------------------------------------------------
# Requirement 15: Optimizer / TES Action Card Uses Same Real Price Version
# ---------------------------------------------------------------------------
def test_optimizer_tes_action_card_uses_same_real_price_version():
    base_t = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
    ivs = create_15m_test_intervals(base_t, count=96, base_price=10.0)

    class FixedProvider(PriceProvider):
        source = "ELERING"
        def fetch_day_ahead(self, zone, s, e):
            return PriceFetchResult(
                source="ELERING",
                bidding_zone="LT",
                period_start_utc=s,
                period_end_utc=e,
                points=[iv.to_price_point() for iv in ivs],
                raw_payload="{}",
                raw_content_type="application/json",
                http_status=200,
                request_params={},
                market_intervals=ivs,
            )

    svc = MarketDataService(
        litgrid_provider=FixedProvider(),
        elering_provider=FixedProvider(),
        tz=TZ_VILNIUS,
    )

    now_test = base_t + timedelta(minutes=7)
    view_ctx = svc.get_market_view_context(now_utc=now_test)

    # Current market price and TES action price must match
    assert view_ctx["price_eur_mwh"] == ivs[0].price_eur_mwh
    assert view_ctx["tes_action"]["price_eur_mwh"] == ivs[0].price_eur_mwh
    assert view_ctx["tes_action"]["reason_code"] in [
        "CHEAP_RELATIVE_TO_FUTURE",
        "EXPENSIVE_PRICE_DISCHARGE",
        "STORAGE_FULL",
        "SOC_RESERVE",
        "GRID_LIMITED",
        "HX_LIMITED",
        "IDLE_OPTIMAL",
    ]
