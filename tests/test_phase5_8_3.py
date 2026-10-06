"""20 Comprehensive Regression Unit Tests for Phase 5.8.3.

Covers:
1. Incomplete tomorrow day cannot be marked complete
2. Normal local day expects 96 intervals
3. Spring DST day expects 92 intervals
4. Autumn DST day expects 100 intervals
5. UTC boundaries map correctly to Europe/Vilnius local day
6. Today chart timestamps are monotonic
7. All current-day intervals appear in chart
8. PAUSED runtime never reports an action as EXECUTING
9. PAUSED runtime does not evolve TES state
10. RUNNING runtime can execute optimizer command
11. Optimization price used is persisted
12. Optimizer price basis is visible in UI / context
13. Current grid margin is calculated from actual heater power
14. 9.95 kW charging headroom remains distinct from current grid margin
15. Primary provider can be reachable but degraded
16. Fallback provider state is explicit
17. Cross-source validation distinguishes MATCH / MISMATCH / NOT CHECKED
18. Continuous optimizer power is labeled idealized
19. Heater staging is not falsely claimed implemented
20. End-to-end dashboard context and data-path consistency
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy.orm import Session

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import (
    DayCoverageResult,
    expected_intervals_in_local_day,
    local_day_bounds_utc,
    validate_local_market_day_coverage,
)
from app.database.models import ShadowIntervalRecord, ShadowTESSession
from app.models.domain import MarketPriceInterval
from app.database.repositories import PriceRepository, ShadowRepository
from app.database.session import init_db, make_engine, make_session_factory
from app.services.market_data_service import (
    MarketDataService,
    compute_today_market_stats,
    compute_tomorrow_market_stats,
)
from app.services.shadow_runtime import ShadowRuntimeService, get_market_interval_bounds
from app.tes.heater_staging import FutureDiscreteHeaterStagingModel, IdealizedContinuousHeaterModel

TZ_VILNIUS = ZoneInfo("Europe/Vilnius")


@pytest.fixture
def db_session(tmp_path) -> Session:
    """Create a temporary SQLite database for Phase 5.8.3 tests."""
    db_file = tmp_path / "test_phase5_8_3.db"
    engine = make_engine(f"sqlite:///{db_file}")
    init_db(engine)
    factory = make_session_factory(engine)
    session = factory()
    yield session
    session.close()


def _make_dummy_intervals(
    start_utc: datetime, count: int, step_minutes: int = 15, base_price: float = 50.0, source: str = "LITGRID"
) -> list[MarketPriceInterval]:
    res = []
    for i in range(count):
        s = start_utc + timedelta(minutes=i * step_minutes)
        e = s + timedelta(minutes=step_minutes)
        res.append(
            MarketPriceInterval(
                start_utc=s,
                end_utc=e,
                price_eur_mwh=base_price + (i % 5),
                bidding_zone="LT",
                source=source,
                original_resolution_minutes=step_minutes,
            )
        )
    return res


# ---------------------------------------------------------------------------
# Tests 1 - 5: Time Boundaries & Completeness Semantics
# ---------------------------------------------------------------------------

def test_01_incomplete_tomorrow_day_cannot_be_marked_complete():
    """Requirement 1: 4 intervals must NOT be marked complete as full day coverage."""
    ref_today = date(2026, 6, 1)
    tomorrow = date(2026, 6, 2)
    s_utc, _ = local_day_bounds_utc(tomorrow, TZ_VILNIUS)

    # 4 intervals (e.g. 00:00 -> 01:00 Europe/Vilnius)
    intervals = _make_dummy_intervals(s_utc, count=4)

    cov = validate_local_market_day_coverage(tomorrow, TZ_VILNIUS, intervals, 15)
    assert cov.is_complete is False
    assert cov.status == "PARTIAL TOMORROW DATA"
    assert cov.received_intervals == 4
    assert cov.expected_intervals == 96
    assert abs(cov.coverage_fraction - (4.0 / 96.0)) < 1e-4

    stats = compute_tomorrow_market_stats(intervals, reference_today=ref_today, tz=TZ_VILNIUS)
    assert stats["available"] is False
    assert stats["is_partial"] is True
    assert stats["status"] == "PARTIAL TOMORROW DATA"
    assert stats["intervals_count"] == 4
    assert stats["expected_intervals"] == 96


def test_02_normal_local_day_expects_96_intervals():
    """Requirement 1 & 2: A normal 24h local calendar day expects 96 15-minute intervals."""
    d = date(2026, 5, 20)
    expected = expected_intervals_in_local_day(d, TZ_VILNIUS, resolution_minutes=15)
    assert expected == 96


def test_03_spring_dst_day_expects_92_intervals():
    """Requirement 1 & 2: Spring DST transition day (23h) in Europe/Vilnius expects 92 intervals."""
    # Last Sunday in March 2026 is March 29
    d = date(2026, 3, 29)
    expected = expected_intervals_in_local_day(d, TZ_VILNIUS, resolution_minutes=15)
    assert expected == 92


def test_04_autumn_dst_day_expects_100_intervals():
    """Requirement 1 & 2: Autumn DST transition day (25h) in Europe/Vilnius expects 100 intervals."""
    # Last Sunday in October 2026 is October 25
    d = date(2026, 10, 25)
    expected = expected_intervals_in_local_day(d, TZ_VILNIUS, resolution_minutes=15)
    assert expected == 100


def test_05_utc_boundaries_map_correctly_to_vilnius_local_day():
    """Requirement 2: UTC boundaries correctly reflect Europe/Vilnius offset (UTC+3 in summer)."""
    d = date(2026, 7, 10)
    s_utc, e_utc = local_day_bounds_utc(d, TZ_VILNIUS)

    # 2026-07-10 00:00 EEST (UTC+3) is 2026-07-09 21:00 UTC
    assert s_utc == datetime(2026, 7, 9, 21, 0, tzinfo=timezone.utc)
    assert e_utc == datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc)
    assert (e_utc - s_utc).total_seconds() == 24 * 3600


# ---------------------------------------------------------------------------
# Tests 6 - 7: Today Price Chart Time Axis & Monotonicity
# ---------------------------------------------------------------------------

def test_06_today_chart_timestamps_are_monotonic():
    """Requirement 3: Today price chart timestamps must be strictly monotonic."""
    d = date(2026, 6, 10)
    s_utc, _ = local_day_bounds_utc(d, TZ_VILNIUS)
    intervals = _make_dummy_intervals(s_utc, count=96)

    stats = compute_today_market_stats(intervals, d, TZ_VILNIUS)
    chart_ivs = stats["intervals"]
    assert len(chart_ivs) == 96

    for i in range(len(chart_ivs) - 1):
        dt_curr = datetime.fromisoformat(chart_ivs[i]["start_utc"])
        dt_next = datetime.fromisoformat(chart_ivs[i + 1]["start_utc"])
        assert dt_next > dt_curr, f"Non-monotonic timestamp at index {i}"


def test_07_all_current_day_intervals_appear_in_chart():
    """Requirement 3: A complete day has all 96 intervals present in chart data."""
    d = date(2026, 6, 10)
    s_utc, _ = local_day_bounds_utc(d, TZ_VILNIUS)
    intervals = _make_dummy_intervals(s_utc, count=96)

    stats = compute_today_market_stats(intervals, d, TZ_VILNIUS)
    assert stats["is_complete"] is True
    assert stats["intervals_count"] == 96
    assert stats["expected_intervals"] == 96
    assert stats["coverage_fraction"] == 1.0
    assert stats["intervals"][0]["start_local"] == "00:00"
    assert stats["intervals"][-1]["end_local"] == "00:00" or stats["intervals"][-1]["end_local"] == "24:00"


# ---------------------------------------------------------------------------
# Tests 8 - 10: Shadow Runtime State & Paused / Running Action Semantics
# ---------------------------------------------------------------------------

def test_08_paused_runtime_never_reports_action_as_executing(db_session: Session):
    """Requirement 4: When PAUSED, action label must NEVER say EXECUTING."""
    svc = ShadowRuntimeService()
    sess = svc.get_or_create_session(db_session)
    svc.start_session(db_session, initial_soc_percent=50.0, session_id=sess.id)
    svc.pause_session(db_session, session_id=sess.id)

    state = svc.get_live_dashboard_state(db_session)
    assert state["execution_state"] == "PAUSED"
    assert "EXECUTING" not in state["action_now"]
    assert state["action_now"].startswith("PAUSED")
    assert state["heater_actual_kw"] == 0.0
    assert state["state_evolution"] == "PAUSED"


def test_09_paused_runtime_does_not_evolve_tes_state(db_session: Session):
    """Requirement 4 & 10: When PAUSED, physical interval step does NOT alter stored energy or SOC."""
    svc = ShadowRuntimeService()
    sess = svc.get_or_create_session(db_session)
    svc.start_session(db_session, initial_soc_percent=50.0, session_id=sess.id)
    svc.pause_session(db_session, session_id=sess.id)

    initial_energy = sess.current_stored_energy_kwh
    initial_soc = sess.current_soc_fraction
    initial_temp = sess.current_sand_temperature_c

    s_utc = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    e_utc = datetime(2026, 6, 1, 10, 15, tzinfo=timezone.utc)

    # Attempt to execute an interval step while session is paused
    rec = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=s_utc,
        interval_end_utc=e_utc,
        spot_price_eur_mwh=40.0,
        requested_charge_kw=5.0,
    )

    assert rec.actual_charge_kw == 0.0
    assert rec.action_type == "PAUSED"
    assert sess.current_stored_energy_kwh == initial_energy
    assert sess.current_soc_fraction == initial_soc
    assert sess.current_sand_temperature_c == initial_temp


def test_10_running_runtime_can_execute_optimizer_command(db_session: Session):
    """Requirement 10: When RUNNING, interval step actuates heater and evolves state."""
    svc = ShadowRuntimeService()
    sess = svc.get_or_create_session(db_session)
    svc.start_session(db_session, initial_soc_percent=20.0, process_enabled=False, session_id=sess.id)

    s_utc = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    e_utc = datetime(2026, 6, 1, 12, 15, tzinfo=timezone.utc)

    rec = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=s_utc,
        interval_end_utc=e_utc,
        spot_price_eur_mwh=10.0,
        requested_charge_kw=4.0,
    )

    assert rec.actual_charge_kw == 4.0
    assert rec.action_type == "CHARGE"
    assert sess.current_stored_energy_kwh > 3.0  # Increased from 20% (3.0 kWh)
    assert sess.current_soc_fraction > 0.20


# ---------------------------------------------------------------------------
# Tests 11 - 12: Optimization Price & Basis Persistence
# ---------------------------------------------------------------------------

def test_11_optimization_price_used_is_persisted(db_session: Session):
    """Requirement 5: Optimization price is persisted to DB in ShadowIntervalRecord and session."""
    svc = ShadowRuntimeService()
    sess = svc.get_or_create_session(db_session)
    svc.start_session(db_session, session_id=sess.id)

    s_utc = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
    e_utc = datetime(2026, 6, 1, 14, 15, tzinfo=timezone.utc)

    # Spot price = 50, Tariff adders = 1.5 + 25.0 + 5.0 = 31.5 -> Effective = 81.5 EUR/MWh
    rec = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=s_utc,
        interval_end_utc=e_utc,
        spot_price_eur_mwh=50.0,
        requested_charge_kw=2.0,
    )

    assert rec.effective_price_eur_mwh == 81.50
    assert rec.optimization_price_eur_mwh == 81.50
    assert rec.optimization_price_basis == "EFFECTIVE_VARIABLE_PRICE"
    assert sess.optimization_price_eur_mwh == 81.50
    assert sess.optimization_price_basis == "EFFECTIVE_VARIABLE_PRICE"


def test_12_optimizer_price_basis_is_visible_in_ui(db_session: Session):
    """Requirement 5: Optimizer price basis is exposed in live dashboard state context."""
    svc = ShadowRuntimeService()
    sess = svc.get_or_create_session(db_session)
    state = svc.get_live_dashboard_state(db_session)

    assert "optimization_price_eur_mwh" in state
    assert "optimization_price_basis" in state
    assert state["optimization_price_basis"] == "EFFECTIVE_VARIABLE_PRICE"


# ---------------------------------------------------------------------------
# Tests 13 - 14: Grid Margin & Headroom Terminology Cleanup
# ---------------------------------------------------------------------------

def test_13_current_grid_margin_calculated_from_actual_heater_power(db_session: Session):
    """Requirement 7: Current unused grid margin is dynamically 12 - (P_heater + 2.05)."""
    svc = ShadowRuntimeService()
    sess = svc.get_or_create_session(db_session)
    svc.start_session(db_session, session_id=sess.id)

    # Case A: When heater is charging at 2.67 kW:
    # Total grid = 2.67 + 2.0 (site) + 0.05 (aux) = 4.72 kW
    # Current unused margin = 12.0 - 4.72 = 7.28 kW
    curr_start_utc, _ = get_market_interval_bounds(datetime.now(timezone.utc))
    sess.planned_schedule = [{
        "start_utc": curr_start_utc.isoformat(),
        "charge_kw": 2.67,
        "discharge_kw": 0.0,
    }]
    ShadowRepository(db_session).update_session(sess)
    db_session.commit()

    state = svc.get_live_dashboard_state(db_session)
    assert abs(state["heater_actual_kw"] - 2.67) < 1e-2
    assert abs(state["current_unused_grid_margin_kw"] - 7.28) < 1e-2

    # Case B: When PAUSED (0.0 kW heater):
    # Total grid = 2.05 kW -> Current unused margin = 9.95 kW
    svc.pause_session(db_session, session_id=sess.id)
    state_paused = svc.get_live_dashboard_state(db_session)
    assert state_paused["heater_actual_kw"] == 0.0
    assert abs(state_paused["current_unused_grid_margin_kw"] - 9.95) < 1e-2


def test_14_charging_headroom_remains_distinct_from_unused_margin(db_session: Session):
    """Requirement 7: 9.95 kW available headroom stays distinct from instantaneous unused margin."""
    svc = ShadowRuntimeService()
    sess = svc.get_or_create_session(db_session)
    svc.start_session(db_session, session_id=sess.id)

    curr_start_utc, _ = get_market_interval_bounds(datetime.now(timezone.utc))
    sess.planned_schedule = [{
        "start_utc": curr_start_utc.isoformat(),
        "charge_kw": 2.67,
        "discharge_kw": 0.0,
    }]
    ShadowRepository(db_session).update_session(sess)
    db_session.commit()

    state = svc.get_live_dashboard_state(db_session)

    # Available headroom is the physical envelope: 12.0 - 2.0 - 0.05 = 9.95 kW
    assert state["available_tes_charging_headroom_kw"] == 9.95
    # Current unused margin reflects the active operating point: 7.28 kW
    assert state["current_unused_grid_margin_kw"] == 7.28
    # Margin at full 9 kW charge is safety reserve: 0.95 kW
    assert state["grid_margin_at_full_charge_kw"] == 0.95
    assert state["available_tes_charging_headroom_kw"] != state["current_unused_grid_margin_kw"]


# ---------------------------------------------------------------------------
# Tests 15 - 17: Provider Health & Hierarchy Semantics
# ---------------------------------------------------------------------------

def test_15_primary_provider_can_be_reachable_but_degraded(monkeypatch, db_session: Session):
    """Requirement 8: Litgrid reachable but returning empty data results in DEGRADED status."""
    from app.models.domain import PriceFetchResult
    from app.services.market_data_service import MarketDataService

    svc = MarketDataService()

    # Mock litgrid provider returning empty result without error
    class MockLitgridReachableEmpty:
        source = "LITGRID"
        def fetch_day_ahead(self, *args, **kwargs):
            now_u = datetime.now(timezone.utc)
            return PriceFetchResult(
                source="LITGRID",
                bidding_zone="LT",
                period_start_utc=now_u,
                period_end_utc=now_u + timedelta(hours=24),
                points=[],
                market_intervals=[],
                raw_payload="[]",
            )

    monkeypatch.setattr(svc, "litgrid", MockLitgridReachableEmpty())
    # Fallback elering provides data
    res = svc.fetch_market_data("LT", session=db_session)
    snap = res.status_snapshot

    assert snap.primary_api_reachable is True
    assert snap.primary_current_available is False
    assert snap.primary_overall_status == "DEGRADED"
    assert snap.active_source == "ELERING"


def test_16_fallback_provider_state_is_explicit(monkeypatch, db_session: Session):
    """Requirement 8: When fallback is active, status snapshot clearly states ACTIVE_FALLBACK."""
    from app.services.market_data_service import MarketDataService

    svc = MarketDataService()

    class MockLitgridError:
        source = "LITGRID"
        def fetch_day_ahead(self, *args, **kwargs):
            raise RuntimeError("Connection timed out to Litgrid API")

    monkeypatch.setattr(svc, "litgrid", MockLitgridError())
    res = svc.fetch_market_data("LT", session=db_session)
    snap = res.status_snapshot

    assert snap.fallback_used == "YES"
    assert snap.active_source == "ELERING"
    assert snap.cross_check_overall_status == "ACTIVE_FALLBACK"
    assert "unreachable" in snap.fallback_reason.lower() or "connection" in snap.fallback_reason.lower()


def test_17_cross_source_validation_distinguishes_match_mismatch_not_checked():
    """Requirement 8 & Phase 5.8: Validation correctly outputs MATCH, MISMATCH, or NOT CHECKED."""
    from app.services.market_data_service import MarketDataService

    svc = MarketDataService()
    t0 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)

    # 1. MATCH: absolute delta <= 0.01 EUR/MWh
    iv_p1 = [MarketPriceInterval(start_utc=t0, end_utc=t1, price_eur_mwh=45.10, bidding_zone="LT", source="LITGRID", original_resolution_minutes=15)]
    iv_c1 = [MarketPriceInterval(start_utc=t0, end_utc=t1, price_eur_mwh=45.105, bidding_zone="LT", source="ELERING", original_resolution_minutes=15)]
    res_match = svc.validate_cross_source(iv_p1, iv_c1)
    assert res_match.status == "MATCH"

    # 2. MISMATCH: absolute delta > 0.01 EUR/MWh
    iv_c2 = [MarketPriceInterval(start_utc=t0, end_utc=t1, price_eur_mwh=46.50, bidding_zone="LT", source="ELERING", original_resolution_minutes=15)]
    res_mismatch = svc.validate_cross_source(iv_p1, iv_c2)
    assert res_mismatch.status == "MISMATCH"

    # 3. NOT CHECKED: missing cross-check
    res_not_checked = svc.validate_cross_source(iv_p1, [])
    assert res_not_checked.status == "NOT CHECKED"


# ---------------------------------------------------------------------------
# Tests 18 - 19: Actuation Semantics & Discrete Staging Interface
# ---------------------------------------------------------------------------

def test_18_continuous_optimizer_power_is_labeled_idealized():
    """Requirement 6: Continuous model outputs CONTINUOUS IDEALIZED with explicit warning."""
    model = IdealizedContinuousHeaterModel(p_max_kw=9.0)
    res = model.dispatch(requested_power_kw=2.67)

    assert res.model_type == "CONTINUOUS IDEALIZED"
    assert res.actual_power_kw == 2.67
    assert res.active_stages == []
    assert "physical heater staging not yet modeled" in res.warning_note.lower()


def test_19_heater_staging_is_not_falsely_claimed_implemented():
    """Requirement 6: Discrete staging model raises NotImplementedError and states future work."""
    model = FutureDiscreteHeaterStagingModel()
    with pytest.raises(NotImplementedError) as exc_info:
        model.dispatch(2.67)
    assert "future work" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 20: Full Context & End-to-End Data Path Verification
# ---------------------------------------------------------------------------

def test_20_all_existing_data_paths_and_dashboard_context_consistent(db_session: Session):
    """Requirement 20: Overview view model integrates all Phase 5.8.3 fields consistently."""
    m_svc = MarketDataService()
    s_svc = ShadowRuntimeService(market_service=m_svc)

    sess = s_svc.get_or_create_session(db_session)
    s_svc.start_session(db_session, session_id=sess.id)

    market_ctx = m_svc.get_market_view_context(session=db_session)
    shadow_ctx = s_svc.get_live_dashboard_state(db_session)

    # Core assertions confirming Phase 5.8.3 completeness
    assert "available_tes_charging_headroom_kw" in shadow_ctx
    assert shadow_ctx["available_tes_charging_headroom_kw"] == 9.95
    assert shadow_ctx["grid_margin_at_full_charge_kw"] == 0.95
    assert shadow_ctx["actuation_model"] == "CONTINUOUS IDEALIZED"
    assert shadow_ctx["hardware_staging"] == "NOT YET IMPLEMENTED"
    assert shadow_ctx["optimization_price_basis"] == "EFFECTIVE_VARIABLE_PRICE"
    assert market_ctx["optimization_price_basis"] == "EFFECTIVE_VARIABLE_PRICE"
    assert "status" in market_ctx["tomorrow_stats"]
