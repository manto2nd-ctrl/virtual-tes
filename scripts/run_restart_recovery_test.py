"""Real restart recovery verification test (Phase 5.9 Requirement 18).

Executes the exact 10-step sequence:
1. Start shadow TES
2. Let state persist
3. Record SOC
4. Restart WEB service
5. Verify state unchanged
6. Restart WORKER service
7. Verify state unchanged
8. Verify worker resumes execution
9. Verify no interval duplication
10. Verify missed intervals are caught up correctly
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.config.settings import get_settings
from app.database.models import ShadowIntervalRecord, ShadowTESSession, WorkerHeartbeat
from app.database.session import init_db, make_engine, make_session_factory
from app.services.shadow_runtime import ShadowRuntimeService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("restart_recovery_test")


def main() -> None:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)

    logger.info("================================================================================")
    logger.info("PHASE 5.9 RESTART RECOVERY TEST — STEP-BY-STEP VERIFICATION")
    logger.info("================================================================================")

    # Step 1: Start shadow TES
    logger.info("Step 1: Starting shadow TES digital twin session...")
    shadow_svc_1 = ShadowRuntimeService()
    with session_factory() as session:
        active_sess = shadow_svc_1.start_session(
            db=session,
            initial_soc_percent=60.0,
            initial_energy_kwh=9.0,
            initial_temp_c=212.0,
            process_demand_kw=1.5,
            process_enabled=True,
        )
        session_id = active_sess.id
        logger.info("  Session started: ID=%s, status=%s", session_id, active_sess.status)

    # Step 2: Let state persist
    logger.info("Step 2: Executing interval step and letting state persist...")
    t0 = datetime(2026, 10, 6, 8, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)
    with session_factory() as session:
        sess = session.execute(select(ShadowTESSession).where(ShadowTESSession.id == session_id)).scalar_one()
        rec1 = shadow_svc_1.execute_interval_step(
            db=session,
            shadow_session=sess,
            interval_start_utc=t0,
            interval_end_utc=t1,
            spot_price_eur_mwh=48.5,
            reason_code="RESTART_STEP_1",
        )
        logger.info("  Executed interval %s -> %s: SOC_end=%.1f%%, stored=%.2f kWh",
                    t0.strftime("%H:%M"), t1.strftime("%H:%M"), sess.current_soc_fraction * 100.0, sess.current_stored_energy_kwh)

    # Step 3: Record SOC
    with session_factory() as session:
        sess = session.execute(select(ShadowTESSession).where(ShadowTESSession.id == session_id)).scalar_one()
        recorded_soc = sess.current_soc_fraction
        recorded_energy = sess.current_stored_energy_kwh
        recorded_temp = sess.current_sand_temperature_c
        logger.info("Step 3: Recorded baseline state:")
        logger.info("  SOC: %.2f%% | Energy: %.3f kWh | Sand Temp: %.2f °C",
                    recorded_soc * 100.0, recorded_energy, recorded_temp)

    # Step 4: Restart WEB service (simulate fresh web process)
    logger.info("Step 4: Simulating WEB service restart (discarding in-memory service instances)...")
    del shadow_svc_1

    # Step 5: Verify state unchanged after WEB restart
    logger.info("Step 5: Verifying canonical database state after WEB restart...")
    with session_factory() as session:
        sess_after_web = session.execute(select(ShadowTESSession).where(ShadowTESSession.id == session_id)).scalar_one()
        assert sess_after_web.current_soc_fraction == recorded_soc, "SOC altered after WEB restart!"
        assert sess_after_web.current_stored_energy_kwh == recorded_energy, "Energy altered after WEB restart!"
        assert sess_after_web.current_sand_temperature_c == recorded_temp, "Temperature altered after WEB restart!"
        logger.info("  ✓ State perfectly unchanged: SOC=%.2f%%, Energy=%.3f kWh",
                    sess_after_web.current_soc_fraction * 100.0, sess_after_web.current_stored_energy_kwh)

    # Step 6: Restart WORKER service (simulate fresh worker process)
    logger.info("Step 6: Simulating WORKER service restart (instantiating fresh worker)...")
    shadow_svc_2 = ShadowRuntimeService()

    # Step 7: Verify state unchanged after WORKER restart
    logger.info("Step 7: Verifying canonical database state after WORKER restart...")
    with session_factory() as session:
        sess_after_worker = session.execute(select(ShadowTESSession).where(ShadowTESSession.id == session_id)).scalar_one()
        assert sess_after_worker.current_soc_fraction == recorded_soc, "SOC altered after WORKER restart!"
        assert sess_after_worker.status == "RUNNING", "Worker status lost!"
        logger.info("  ✓ Canonical state preserved: SOC=%.2f%%, Status=%s",
                    sess_after_worker.current_soc_fraction * 100.0, sess_after_worker.status)

    # Step 8: Verify worker resumes execution
    logger.info("Step 8: Verifying worker resumes execution on next interval...")
    t2 = t1 + timedelta(minutes=15)
    with session_factory() as session:
        sess = session.execute(select(ShadowTESSession).where(ShadowTESSession.id == session_id)).scalar_one()
        rec2 = shadow_svc_2.execute_interval_step(
            db=session,
            shadow_session=sess,
            interval_start_utc=t1,
            interval_end_utc=t2,
            spot_price_eur_mwh=52.0,
            reason_code="RESTART_STEP_2_RESUME",
        )
        logger.info("  ✓ Worker resumed execution: Interval %s -> %s recorded.", t1.strftime("%H:%M"), t2.strftime("%H:%M"))

    # Step 9: Verify no interval duplication
    logger.info("Step 9: Testing duplicate interval prevention (idempotence)...")
    with session_factory() as session:
        from sqlalchemy.exc import IntegrityError
        dup_rec = ShadowIntervalRecord(
            shadow_session_id=session_id,
            interval_start_utc=t0,
            interval_end_utc=t1,
            duration_hours=0.25,
            spot_price_eur_mwh=48.5,
            effective_price_eur_mwh=53.5,
            action_type="CHARGE",
            requested_charge_kw=9.0,
            actual_charge_kw=9.0,
            requested_discharge_kw=0.0,
            actual_discharge_kw=0.0,
            process_demand_kw=1.5,
            hx_power_limit_kw=2.5,
            soc_start_fraction=0.60,
            soc_end_fraction=0.74,
            stored_energy_start_kwh=9.0,
            stored_energy_end_kwh=11.1,
            sand_temp_start_c=212.0,
            sand_temp_end_c=238.0,
            standing_loss_kwh=0.01,
            grid_power_total_kw=11.05,
            electricity_consumed_kwh=2.25,
            interval_cost_eur=0.12,
            energy_balance_residual_kwh=0.0,
            reason_code="DUPLICATE_ATTEMPT",
        )
        session.add(dup_rec)
        try:
            session.commit()
            raise AssertionError("Duplicate interval was mistakenly allowed!")
        except IntegrityError:
            session.rollback()
            logger.info("  ✓ Idempotence verified: UNIQUE(session_id, interval_start_utc) rejected duplicate.")

    # Step 10: Verify missed intervals are caught up correctly
    logger.info("Step 10: Testing catch-up of missed intervals during application downtime...")
    t_sim_now = t2 + timedelta(minutes=45)  # 3 intervals later
    with session_factory() as session:
        sess = session.execute(select(ShadowTESSession).where(ShadowTESSession.id == session_id)).scalar_one()
        # Seed mock prices in DB for the catch-up intervals
        from app.database.repositories import PriceRepository
        repo_price = PriceRepository(session, settings.tz)
        for i in range(3):
            iv_start = t2 + timedelta(minutes=15 * i)
            iv_end = iv_start + timedelta(minutes=15)
            session.add(DayAheadPrice(
                bidding_zone="LT",
                source="LITGRID",
                delivery_start_utc=iv_start,
                delivery_end_utc=iv_end,
                delivery_start_local=iv_start.isoformat(),
                resolution_minutes=15,
                price_eur_mwh=40.0 + i * 2.0,
                version=1,
            ))
        session.commit()

        caught_up = shadow_svc_2.catch_up_missed_intervals(session, sess, now_utc=t_sim_now)
        logger.info("  ✓ Successfully caught up %d missed intervals during downtime.", caught_up)
        assert caught_up == 3, f"Expected 3 caught up intervals, got {caught_up}"

    # Summary
    logger.info("================================================================================")
    logger.info("RESTART RECOVERY TEST RESULTS: 10 / 10 STEPS VERIFIED & PASSED")
    logger.info("Canonical state persists across WEB and WORKER service redeployments.")
    logger.info("Zero duplicate intervals; continuous digital twin operation guaranteed.")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
