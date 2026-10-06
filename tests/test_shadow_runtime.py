"""20 Regression unit tests for Phase 5.8.2 Live Virtual TES Shadow Runtime.

Verifies:
1. shadow start persists a session
2. initial SOC maps correctly to energy and temperature
3. live current market price is used
4. starting mid-interval uses fractional dt
5. state progresses with real elapsed time
6. same interval cannot execute twice
7. charge power respects 9 kW heater limit
8. total grid power respects 12 kW limit
9. charge and discharge can occur simultaneously
10. process demand is independent from market price
11. HX limit constrains process discharge
12. SOC never exceeds 100%
13. operational discharge does not go below the optimizer reserve
14. standing losses are applied
15. state persists after app restart
16. missed intervals catch up correctly
17. new market prices trigger reoptimization
18. shadow history remains immutable
19. energy balance residual is within tolerance
20. no hardware control command is sent
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.database.models import (
    DayAheadPrice,
    ShadowEventAudit,
    ShadowIntervalRecord,
    ShadowTESSession,
)
from app.database.repositories import PriceRepository, ShadowRepository
from app.database.session import init_db, make_engine, make_session_factory
from app.services.shadow_runtime import (
    EPS_ENERGY,
    ShadowRuntimeService,
    get_market_interval_bounds,
)
from app.tes.thermal import ThermalStateMapper

TZ_VILNIUS = ZoneInfo("Europe/Vilnius")


@pytest.fixture
def db_session(tmp_path) -> Session:
    """Create a temporary SQLite database with triggers for testing."""
    db_file = tmp_path / "test_shadow.db"
    db_url = f"sqlite:///{db_file}"
    engine = make_engine(db_url)
    init_db(engine)
    factory = make_session_factory(engine)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def seed_prices(db_session: Session) -> list[DayAheadPrice]:
    """Seed sample 15-minute day-ahead prices for testing."""
    now_utc = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)
    prices = []
    for i in range(16):
        start = now_utc + timedelta(minutes=15 * i)
        end = start + timedelta(minutes=15)
        # Price curve: starts cheap (10-30 €/MWh), rises later
        price_val = 15.0 + 5.0 * (i % 4)
        p = DayAheadPrice(
            bidding_zone="LT",
            source="ELERING",
            delivery_start_utc=start,
            delivery_end_utc=end,
            delivery_start_local=start.astimezone(TZ_VILNIUS).isoformat(),
            resolution_minutes=15,
            price_eur_mwh=price_val,
            currency="EUR",
            version=1,
            fetched_at=now_utc,
        )
        prices.append(p)
        db_session.add(p)
    db_session.commit()
    return prices


# ===========================================================================
# 1. Shadow start persists a session
# ===========================================================================
def test_01_shadow_start_persists_session(db_session: Session, seed_prices):
    svc = ShadowRuntimeService()
    now_utc = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)

    session_obj = svc.start_session(
        db=db_session,
        initial_soc_percent=50.0,
        process_demand_kw=1.5,
        process_enabled=True,
        now_utc=now_utc,
    )

    repo = ShadowRepository(db_session)
    fetched = repo.get_session(session_obj.id)
    assert fetched is not None
    assert fetched.status == "RUNNING"
    assert fetched.initial_soc_fraction == pytest.approx(0.50)
    assert fetched.current_soc_fraction == pytest.approx(0.50)
    assert fetched.process_heat_demand_kw == 1.50
    assert fetched.process_enabled is True
    assert fetched.active_schedule_version >= 1

    # Audit was created
    audits = repo.get_audits(session_obj.id)
    assert len(audits) >= 1
    assert audits[0].event_type == "START"


# ===========================================================================
# 2. Initial SOC maps correctly to energy and temperature
# ===========================================================================
def test_02_initial_soc_maps_correctly():
    svc = ShadowRuntimeService()
    mapper = ThermalStateMapper(80.0, 300.0)

    # 50% SOC -> 7.5 kWh -> ~197.8 °C
    frac50, kwh50, temp50 = svc.convert_initial_conditions(initial_soc_percent=50.0)
    assert frac50 == pytest.approx(0.50)
    assert kwh50 == pytest.approx(7.50)
    assert temp50 == pytest.approx(mapper.temperature_from_soc_fraction(0.50), abs=0.1)
    assert 195.0 <= temp50 <= 200.0

    # 10% SOC (optimizer reserve) -> 1.5 kWh -> ~104.71 °C
    frac10, kwh10, temp10 = svc.convert_initial_conditions(initial_soc_percent=10.0)
    assert frac10 == pytest.approx(0.10)
    assert kwh10 == pytest.approx(1.50)
    assert temp10 == pytest.approx(104.71, abs=0.2)

    # 100% SOC -> 15.0 kWh -> 300.0 °C
    frac100, kwh100, temp100 = svc.convert_initial_conditions(initial_soc_percent=100.0)
    assert frac100 == pytest.approx(1.00)
    assert kwh100 == pytest.approx(15.00)
    assert temp100 == pytest.approx(300.0, abs=0.1)

    # From energy input: 7.5 kWh
    frac_e, kwh_e, temp_e = svc.convert_initial_conditions(initial_energy_kwh=7.5)
    assert frac_e == pytest.approx(0.50)
    assert kwh_e == pytest.approx(7.50)

    # From temperature input: 300 °C
    frac_t, kwh_t, temp_t = svc.convert_initial_conditions(initial_temp_c=300.0)
    assert frac_t == pytest.approx(1.00)
    assert kwh_t == pytest.approx(15.00)


# ===========================================================================
# 3. Live current market price is used
# ===========================================================================
def test_03_live_current_market_price_is_used(db_session: Session, seed_prices):
    svc = ShadowRuntimeService()
    now_utc = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=now_utc)

    # Step with specific live market price
    start_utc = now_utc
    end_utc = start_utc + timedelta(minutes=15)
    live_spot = 42.85  # EUR/MWh

    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=start_utc,
        interval_end_utc=end_utc,
        spot_price_eur_mwh=live_spot,
        requested_charge_kw=9.0,
    )

    assert record.spot_price_eur_mwh == pytest.approx(42.85)
    # Effective price includes adders: 42.85 + 1.5 + 25.0 + 5.0 = 74.35
    assert record.effective_price_eur_mwh == pytest.approx(74.35)
    assert record.interval_cost_eur > 0


# ===========================================================================
# 4. Starting mid-interval uses fractional dt
# ===========================================================================
def test_04_starting_mid_interval_uses_fractional_dt(db_session: Session):
    svc = ShadowRuntimeService()
    # Starting at 00:46 for 00:45 -> 01:00 interval
    start_utc = datetime(2026, 10, 6, 0, 46, tzinfo=timezone.utc)
    end_utc = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)

    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=start_utc)

    # Remaining duration: exactly 14 minutes = 14/60 hours
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=start_utc,
        interval_end_utc=end_utc,
        spot_price_eur_mwh=20.0,
        requested_charge_kw=9.0,
    )

    expected_dt = 14.0 / 60.0
    assert record.duration_hours == pytest.approx(expected_dt, abs=1e-4)

    # Consumed electricity is 9.0 kW * 14/60 h = 2.1 kWh (not 2.25 kWh for 15 min)
    assert record.electricity_consumed_kwh == pytest.approx(9.0 * expected_dt, abs=1e-3)


# ===========================================================================
# 5. State progresses with real elapsed time
# ===========================================================================
def test_05_state_progresses_with_real_elapsed_time(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    # Set planned charging at 9 kW
    sess.planned_schedule = [
        {"start_utc": t0.isoformat(), "charge_kw": 9.0, "discharge_kw": 0.0}
    ]
    db_session.commit()

    # Continuous tick after 10 seconds
    t1 = t0 + timedelta(seconds=10)
    svc.step_continuous_tick(db=db_session, now_utc=t1)

    db_session.refresh(sess)
    assert sess.last_state_update_utc == t1
    # Energy should have increased from 7.5 kWh:
    # dE ~ 9 kW * 0.95 * (10/3600) h ~ 0.02375 kWh
    assert sess.current_stored_energy_kwh > 7.50


# ===========================================================================
# 6. Same interval cannot execute twice (Idempotency)
# ===========================================================================
def test_06_same_interval_cannot_execute_twice(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    rec1 = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t1,
        spot_price_eur_mwh=30.0,
        requested_charge_kw=9.0,
    )
    e1 = sess.current_stored_energy_kwh

    # Attempt to execute again for the same interval start
    rec2 = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t1,
        spot_price_eur_mwh=30.0,
        requested_charge_kw=9.0,
    )

    assert rec1.id == rec2.id
    # Stored energy was NOT charged a second time
    assert sess.current_stored_energy_kwh == pytest.approx(e1)

    # Total interval records in DB is strictly 1
    repo = ShadowRepository(db_session)
    records = repo.get_intervals(sess.id)
    assert len(records) == 1


# ===========================================================================
# 7. Charge power respects 9 kW heater limit
# ===========================================================================
def test_07_charge_power_respects_9kw_heater_limit(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=30.0, now_utc=t0)

    # Request excessive charge power: 20 kW
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=10.0,
        requested_charge_kw=20.0,
    )

    assert record.requested_charge_kw == pytest.approx(20.0)
    assert record.actual_charge_kw == pytest.approx(9.0)


# ===========================================================================
# 8. Total grid power respects 12 kW limit
# ===========================================================================
def test_08_total_grid_power_respects_12kw_limit(db_session: Session):
    # Set other site load = 5.0 kW (headroom = 12.0 - 5.0 - 0.05 = 6.95 kW)
    custom_site = SiteParameters(grid_connection_limit_kw=12.0, other_loads_kw=5.0)
    svc = ShadowRuntimeService(site_params=custom_site)

    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=30.0, now_utc=t0)

    # Request full 9 kW charging
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=10.0,
        requested_charge_kw=9.0,
    )

    # Charge must be clipped to headroom = 6.95 kW
    assert record.actual_charge_kw == pytest.approx(6.95, abs=1e-4)
    # Total grid power must equal 12.0 kW exactly
    assert record.grid_power_total_kw == pytest.approx(12.0, abs=1e-4)


# ===========================================================================
# 9. Charge and discharge can occur simultaneously
# ===========================================================================
def test_09_charge_and_discharge_simultaneous(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    # Charge at 9.0 kW while discharging 1.5 kW to process
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=15.0,
        requested_charge_kw=9.0,
        requested_discharge_kw=1.5,
    )

    assert record.actual_charge_kw == pytest.approx(9.0)
    assert record.actual_discharge_kw == pytest.approx(1.5)
    assert "CHARGE" in record.action_type and "SUPPLY" in record.action_type


# ===========================================================================
# 10. Process demand is independent from market price
# ===========================================================================
def test_10_process_demand_independent_from_market_price(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(
        db=db_session,
        initial_soc_percent=50.0,
        process_demand_kw=1.5,
        process_enabled=True,
        now_utc=t0,
    )

    # Extremely high market price: 500 EUR/MWh
    rec_high = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=500.0,
    )
    # Process demand was still fully delivered
    assert rec_high.process_demand_kw == pytest.approx(1.5)
    assert rec_high.actual_discharge_kw == pytest.approx(1.5)


# ===========================================================================
# 11. HX limit constrains process discharge
# ===========================================================================
def test_11_hx_limit_constrains_discharge(db_session: Session):
    svc = ShadowRuntimeService()
    # At 82 °C sand temperature (inlet 40 °C), HX max output is derated to < 0.6 kW
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_temp_c=82.0, now_utc=t0)

    # Process requests 2.5 kW
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=50.0,
        requested_discharge_kw=2.5,
    )

    # Actual discharge must be constrained by HX limit
    assert record.actual_discharge_kw <= record.hx_power_limit_kw
    assert record.actual_discharge_kw < 1.0


# ===========================================================================
# 12. SOC never exceeds 100%
# ===========================================================================
def test_12_soc_never_exceeds_100_percent(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    # Start near full capacity: 98% SOC (14.7 kWh)
    sess = svc.start_session(db=db_session, initial_soc_percent=98.0, now_utc=t0)

    # Request maximum 9.0 kW charging
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=10.0,
        requested_charge_kw=9.0,
        requested_discharge_kw=0.0,
    )

    # Must be clamped at 100% (15.0 kWh)
    assert record.soc_end_fraction <= 1.000001
    assert record.stored_energy_end_kwh <= 15.000001


# ===========================================================================
# 13. Operational discharge does not go below the optimizer reserve
# ===========================================================================
def test_13_operational_discharge_does_not_go_below_optimizer_reserve(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    # Start close to 10% reserve: 11% SOC (1.65 kWh)
    sess = svc.start_session(db=db_session, initial_soc_percent=11.0, now_utc=t0)

    # Request 3.0 kW discharge for 15 min
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=50.0,
        requested_discharge_kw=3.0,
    )

    # Must stop discharge at 10% SOC reserve (1.50 kWh)
    assert record.stored_energy_end_kwh >= 1.50 - 1e-5
    assert record.soc_end_fraction >= 0.10 - 1e-5


# ===========================================================================
# 14. Standing losses are applied
# ===========================================================================
def test_14_standing_losses_are_applied(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=80.0, now_utc=t0)

    # Zero charge, zero discharge for 15 min
    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=50.0,
        requested_charge_kw=0.0,
        requested_discharge_kw=0.0,
    )

    assert record.standing_loss_kwh > 0
    assert record.stored_energy_end_kwh < record.stored_energy_start_kwh


# ===========================================================================
# 15. State persists after app restart
# ===========================================================================
def test_15_state_persists_after_app_restart(db_session: Session):
    svc1 = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess1 = svc1.start_session(db=db_session, initial_soc_percent=65.0, now_utc=t0)

    svc1.execute_interval_step(
        db=db_session,
        shadow_session=sess1,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=25.0,
        requested_charge_kw=9.0,
    )

    end_energy = sess1.current_stored_energy_kwh

    # Simulate fresh app startup
    svc2 = ShadowRuntimeService()
    repo2 = ShadowRepository(db_session)
    sess2 = repo2.get_session(sess1.id)

    assert sess2 is not None
    assert sess2.status == "RUNNING"
    assert sess2.current_stored_energy_kwh == pytest.approx(end_energy)
    assert sess2.current_soc_fraction == pytest.approx(end_energy / 15.0)


# ===========================================================================
# 16. Missed intervals catch up correctly
# ===========================================================================
def test_16_missed_intervals_catch_up_correctly(db_session: Session, seed_prices):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    # Advance clock by 3 intervals (45 minutes)
    t_after_reboot = t0 + timedelta(minutes=45)
    caught_up = svc.catch_up_missed_intervals(db=db_session, shadow_session=sess, now_utc=t_after_reboot)

    assert caught_up == 3
    repo = ShadowRepository(db_session)
    intervals = repo.get_intervals(sess.id, ascending=True)
    assert len(intervals) == 3
    assert intervals[0].interval_start_utc == t0
    assert intervals[1].interval_start_utc == t0 + timedelta(minutes=15)
    assert intervals[2].interval_start_utc == t0 + timedelta(minutes=30)


# ===========================================================================
# 17. New market prices trigger reoptimization
# ===========================================================================
def test_17_new_market_prices_trigger_reoptimization(db_session: Session, seed_prices):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    v1 = sess.active_schedule_version
    # Trigger reoptimization
    svc.reoptimize(db=db_session, shadow_session=sess, trigger_reason="NEW_DAY_AHEAD_DATA", now_utc=t0)

    assert sess.active_schedule_version > v1
    assert len(sess.planned_schedule) > 0


# ===========================================================================
# 18. Shadow history remains immutable
# ===========================================================================
def test_18_shadow_history_remains_immutable(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    rec = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=30.0,
    )

    # Attempt direct SQLite UPDATE on shadow_interval_records
    with pytest.raises(Exception, match=r"(?i)(append-only|forbidden|abort)"):
        db_session.execute(
            text(f"UPDATE shadow_interval_records SET actual_charge_kw = 99.0 WHERE id = {rec.id}")
        )
        db_session.commit()

    db_session.rollback()

    # Attempt direct SQLite DELETE
    with pytest.raises(Exception, match=r"(?i)(append-only|forbidden|abort)"):
        db_session.execute(
            text(f"DELETE FROM shadow_interval_records WHERE id = {rec.id}")
        )
        db_session.commit()


# ===========================================================================
# 19. Energy balance residual is within tolerance
# ===========================================================================
def test_19_energy_balance_residual_is_within_tolerance(db_session: Session):
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    record = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=25.0,
        requested_charge_kw=7.5,
        requested_discharge_kw=1.5,
    )

    # Numerical residual must be within tolerance
    assert record.energy_balance_residual_kwh < EPS_ENERGY
    assert record.energy_balance_residual_kwh >= 0.0


# ===========================================================================
# 20. No hardware control command is sent
# ===========================================================================
def test_20_no_hardware_control_command_is_sent(db_session: Session):
    """Verify that the shadow service does not bind to hardware, PLC, or Modbus ports."""
    svc = ShadowRuntimeService()
    t0 = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
    sess = svc.start_session(db=db_session, initial_soc_percent=50.0, now_utc=t0)

    # Step and tick execute entirely in pure Python / SQLite
    rec = svc.execute_interval_step(
        db=db_session,
        shadow_session=sess,
        interval_start_utc=t0,
        interval_end_utc=t0 + timedelta(minutes=15),
        spot_price_eur_mwh=10.0,
        requested_charge_kw=9.0,
    )

    # Status remains purely digital twin
    assert rec.actual_charge_kw > 0
    assert not hasattr(svc, "modbus_client")
    assert not hasattr(svc, "plc_client")
    assert sess.model_version == "gen0-shadow-v1"
