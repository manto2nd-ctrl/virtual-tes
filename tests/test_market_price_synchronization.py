"""Tests for Market Price and Time Synchronization (Phase 5.11 - Stage 2).

Validates:
1. Authoritative market-price lookup based on UTC settlement intervals.
2. Lithuanian display times (Europe/Vilnius) while database keys remain UTC.
3. Explicit storage and distinction of:
   - Spot price (EUR/MWh and EUR/kWh)
   - Effective variable electricity price (EUR/MWh and EUR/kWh)
   - Tariff components (markup, grid fee, taxes)
   - Market provider
   - Interval timestamps
   - Strict Data Quality: LIVE, FALLBACK, HISTORICAL, or SYNTHETIC
4. Daylight-saving transitions (spring 23h, autumn 25h, normal 24h).
5. Historical price integrity: older executed intervals never receive the latest price.
6. Acceptance criteria: synthetic or stale data is never labeled live.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import pytest
from zoneinfo import ZoneInfo

from app.config.parameters import TariffParameters
from app.config.settings import get_settings
from app.core.timegrid import ensure_utc, local_day_bounds_utc, local_day_intervals
from app.database.models import DayAheadPrice
from app.database.repositories import PriceRepository
from app.database.session import init_db, make_engine, make_session_factory
from app.models.domain import PricePoint
from app.services.energy_system_state import get_energy_system_state
from app.services.market_price_service import (
    DataQuality,
    MarketPriceService,
    get_authoritative_market_price,
    snap_to_settlement_interval,
)

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")
UTC = timezone.utc


@pytest.fixture
def test_db():
    """Create an isolated in-memory or test sqlite session."""
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    session = session_factory()
    yield session
    session.close()


def test_utc_interval_snapping_and_bounds():
    """Any timestamp snaps to exact 15-minute UTC interval boundaries."""
    # 2026-10-11 20:47:33 UTC should snap to [20:45, 21:00)
    dt = datetime(2026, 10, 11, 20, 47, 33, tzinfo=UTC)
    start_utc, end_utc = snap_to_settlement_interval(dt, resolution_minutes=15)
    assert start_utc == datetime(2026, 10, 11, 20, 45, 0, tzinfo=UTC)
    assert end_utc == datetime(2026, 10, 11, 21, 0, 0, tzinfo=UTC)

    # Duration is exactly 15 minutes
    assert (end_utc - start_utc) == timedelta(minutes=15)


def test_lithuanian_dst_transitions_in_utc():
    """UTC arithmetic cleanly spans Lithuanian daylight saving transitions."""
    # Spring Forward: 2026-03-29 (23 hours = 92 intervals)
    spring_day = date(2026, 3, 29)
    spring_intervals = local_day_intervals(spring_day, VILNIUS_TZ, resolution_minutes=15)
    assert len(spring_intervals) == 92
    spring_hours = sum(iv.duration_h for iv in spring_intervals)
    assert spring_hours == 23.0

    # Autumn Fall Back: 2026-10-25 (25 hours = 100 intervals)
    autumn_day = date(2026, 10, 25)
    autumn_intervals = local_day_intervals(autumn_day, VILNIUS_TZ, resolution_minutes=15)
    assert len(autumn_intervals) == 100
    autumn_hours = sum(iv.duration_h for iv in autumn_intervals)
    assert autumn_hours == 25.0


def test_authoritative_price_fields_and_tariff_breakdown(test_db):
    """Authoritative lookup returns all required distinct fields, units, and declared price basis."""
    now_utc = datetime(2026, 10, 11, 14, 0, 0, tzinfo=UTC)
    tariff = TariffParameters(
        supplier_markup_eur_mwh=1.50,
        variable_grid_fee_eur_mwh=25.00,
        variable_tax_eur_mwh=5.00,
    )

    auth = get_authoritative_market_price(
        interval_start_utc=now_utc,
        db=test_db,
        tariff_params=tariff,
        now_utc=now_utc,
    )

    # 1. Timestamps
    assert auth.interval_start_utc == now_utc
    assert auth.interval_end_utc == now_utc + timedelta(minutes=15)
    assert auth.duration_minutes == 15
    assert auth.interval_start_local.tzinfo is not None

    # 2. Prices & Conversion (€/kWh = €/MWh / 1000)
    assert auth.spot_price_eur_kwh == pytest.approx(auth.spot_price_eur_mwh / 1000.0, rel=1e-5)
    assert auth.effective_price_eur_kwh == pytest.approx(auth.effective_price_eur_mwh / 1000.0, rel=1e-5)

    # 3. Tariff Components
    assert auth.tariff.supplier_markup_eur_mwh == 1.50
    assert auth.tariff.variable_grid_fee_eur_mwh == 25.00
    assert auth.tariff.variable_tax_eur_mwh == 5.00
    assert auth.tariff.total_variable_adders_eur_mwh == 31.50
    assert auth.tariff.total_variable_adders_eur_kwh == pytest.approx(0.0315, rel=1e-5)

    # Effective price = Spot + 31.50
    assert auth.effective_price_eur_mwh == pytest.approx(auth.spot_price_eur_mwh + 31.50, abs=1e-3)

    # 4. Economic Basis
    assert auth.price_basis == "EFFECTIVE_VARIABLE_PRICE"


def test_historical_executed_interval_never_receives_latest_price(test_db):
    """Older executed interval receives its own historical interval price, not the latest current price."""
    repo = PriceRepository(test_db, VILNIUS_TZ)

    # Store a historical price: 2026-06-15 10:00 UTC -> €40.00/MWh
    hist_time = datetime(2026, 6, 15, 10, 0, 0, tzinfo=UTC)
    repo.insert_prices([
        PricePoint(
            bidding_zone="LT",
            source="LITGRID",
            delivery_start_utc=hist_time,
            delivery_end_utc=hist_time + timedelta(minutes=15),
            resolution_minutes=15,
            price_eur_mwh=40.00,
            currency="EUR",
        )
    ])

    # Store a current price: 2026-06-15 14:00 UTC -> €150.00/MWh
    curr_time = datetime(2026, 6, 15, 14, 0, 0, tzinfo=UTC)
    repo.insert_prices([
        PricePoint(
            bidding_zone="LT",
            source="LITGRID",
            delivery_start_utc=curr_time,
            delivery_end_utc=curr_time + timedelta(minutes=15),
            resolution_minutes=15,
            price_eur_mwh=150.00,
            currency="EUR",
        )
    ])
    test_db.commit()

    # Query older interval (10:00) when "now" is 14:10
    now_utc = datetime(2026, 6, 15, 14, 10, 0, tzinfo=UTC)
    auth_hist = get_authoritative_market_price(
        interval_start_utc=hist_time,
        db=test_db,
        now_utc=now_utc,
    )

    # Must return historical price (€40.00), NEVER the latest price (€150.00)
    assert auth_hist.spot_price_eur_mwh == 40.00
    assert auth_hist.spot_price_eur_kwh == 0.0400
    # Must be classified as HISTORICAL
    assert auth_hist.data_quality == DataQuality.HISTORICAL


def test_synthetic_and_stale_data_never_labeled_live(test_db):
    """Synthetic or stale data must NEVER have DataQuality.LIVE."""
    svc = MarketPriceService(tz=VILNIUS_TZ)
    future_unknown = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)

    # No data exists in DB or providers for 2030
    auth = svc.get_authoritative_price_for_interval(
        interval_start_utc=future_unknown,
        db=test_db,
        now_utc=datetime(2026, 10, 11, 12, 0, 0, tzinfo=UTC),
    )

    # Synthetic fallback must have SYNTHETIC data quality, NEVER LIVE!
    assert auth.data_quality == DataQuality.SYNTHETIC
    assert auth.data_quality != DataQuality.LIVE
    assert auth.is_synthetic is True


def test_fallback_provider_classified_as_fallback(test_db):
    """Data fetched from secondary/tertiary providers (e.g. ELERING) is strictly FALLBACK, not LIVE."""
    repo = PriceRepository(test_db, VILNIUS_TZ)
    test_time = datetime(2026, 6, 16, 14, 0, 0, tzinfo=UTC)

    # Insert an Elering fallback observation where no Litgrid row exists
    repo.insert_prices([
        PricePoint(
            bidding_zone="LT",
            source="ELERING",
            delivery_start_utc=test_time,
            delivery_end_utc=test_time + timedelta(minutes=15),
            resolution_minutes=15,
            price_eur_mwh=120.00,
            currency="EUR",
        )
    ])
    test_db.commit()

    auth = get_authoritative_market_price(
        interval_start_utc=test_time,
        db=test_db,
        now_utc=test_time + timedelta(minutes=5),
    )

    assert auth.provider == "ELERING"
    assert auth.data_quality == DataQuality.FALLBACK
    assert auth.data_quality != DataQuality.LIVE


def test_energy_system_state_parity_with_market_price_service(test_db):
    """EnergySystemState uses canonical market price service for identical spot and effective prices."""
    now_utc = datetime.now(timezone.utc)
    auth_price = get_authoritative_market_price(interval_start_utc=now_utc, db=test_db, now_utc=now_utc)
    system_state = get_energy_system_state(db=test_db, now_utc=now_utc)

    # Exact parity across the application
    assert system_state.spot_price_eur_mwh == auth_price.spot_price_eur_mwh
    assert system_state.effective_price_eur_mwh == auth_price.effective_price_eur_mwh
    assert system_state.spot_price_eur_kwh == auth_price.spot_price_eur_kwh
    assert system_state.effective_price_eur_kwh == auth_price.effective_price_eur_kwh
    assert system_state.active_market_provider == auth_price.provider
    assert system_state.data_quality == auth_price.data_quality.value
