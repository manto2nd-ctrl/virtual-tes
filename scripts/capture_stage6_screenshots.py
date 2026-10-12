"""Capture Stage 6 screenshots for History & Savings and Overview."""
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from playwright.sync_api import sync_playwright

from app.config.settings import get_settings
from app.database.session import make_engine, make_session_factory
from app.services.shadow_runtime import ShadowRuntimeService

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")

def seed_demo_intervals():
    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    with session_factory() as db:
        runtime = ShadowRuntimeService(tz=VILNIUS_TZ)
        sess = runtime.get_or_create_session(db)
        sess.status = "RUNNING"
        sess.process_enabled = True
        sess.process_heat_demand_kw = 1.5
        db.commit()

        # Seed intervals within today's local Vilnius day
        now_local = datetime.now(VILNIUS_TZ)
        t_base = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

        # Interval 1: 00:00-00:15 - Off-peak TES charging (8 kW charge)
        t1_start = t_base.astimezone(timezone.utc)
        t1_end = (t_base + timedelta(minutes=15)).astimezone(timezone.utc)
        runtime.execute_interval_step(
            db=db,
            shadow_session=sess,
            interval_start_utc=t1_start,
            interval_end_utc=t1_end,
            spot_price_eur_mwh=42.50,
            requested_charge_kw=8.0,
            requested_discharge_kw=0.0,
            reason_code="OFF_PEAK_CHARGE",
        )

        # Interval 2: 00:15-00:30 - Process heat delivery from TES
        t2_start = (t_base + timedelta(minutes=15)).astimezone(timezone.utc)
        t2_end = (t_base + timedelta(minutes=30)).astimezone(timezone.utc)
        runtime.execute_interval_step(
            db=db,
            shadow_session=sess,
            interval_start_utc=t2_start,
            interval_end_utc=t2_end,
            spot_price_eur_mwh=88.00,
            requested_charge_kw=0.0,
            requested_discharge_kw=1.5,
            reason_code="PROCESS_HEAT_DELIVERY",
        )

        # Interval 3: 00:30-00:45 - Process heat delivery from TES
        t3_start = (t_base + timedelta(minutes=30)).astimezone(timezone.utc)
        t3_end = (t_base + timedelta(minutes=45)).astimezone(timezone.utc)
        runtime.execute_interval_step(
            db=db,
            shadow_session=sess,
            interval_start_utc=t3_start,
            interval_end_utc=t3_end,
            spot_price_eur_mwh=105.00,
            requested_charge_kw=0.0,
            requested_discharge_kw=1.5,
            reason_code="PROCESS_HEAT_DELIVERY",
        )

        # Interval 4: 00:45-01:00 - Process heat delivery
        t4_start = (t_base + timedelta(minutes=45)).astimezone(timezone.utc)
        t4_end = (t_base + timedelta(minutes=60)).astimezone(timezone.utc)
        runtime.execute_interval_step(
            db=db,
            shadow_session=sess,
            interval_start_utc=t4_start,
            interval_end_utc=t4_end,
            spot_price_eur_mwh=120.00,
            requested_charge_kw=0.0,
            requested_discharge_kw=1.5,
            reason_code="PROCESS_HEAT_DELIVERY",
        )

        db.commit()
        print("Seeded demo intervals for Stage 6 screenshots across today.")

def main():
    seed_demo_intervals()

    # Launch uvicorn
    proc = subprocess.Popen(
        [r".\.venv\Scripts\python.exe", "-m", "uvicorn", "app.web.main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)

    out_dir = Path(r"C:\Users\kaim\.gemini\antigravity\brain\89c32645-5b66-4c0d-ace3-ea31d74b5c89")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 2200})
            page = context.new_page()

            # 1. Capture History & Savings
            print("Capturing History & Savings page...")
            page.goto("http://127.0.0.1:8000/dashboard/history?range=all", wait_until="networkidle")
            time.sleep(2)
            p1 = out_dir / "stage6_economic_history.png"
            page.screenshot(path=str(p1), full_page=True)
            print(f"Captured: {p1}")

            # 2. Capture Overview
            print("Capturing Overview page...")
            page.goto("http://127.0.0.1:8000/dashboard/overview", wait_until="networkidle")
            time.sleep(2)
            p2 = out_dir / "stage6_overview.png"
            page.screenshot(path=str(p2), full_page=True)
            print(f"Captured: {p2}")

            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        print("Uvicorn stopped.")

if __name__ == "__main__":
    main()
