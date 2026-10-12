"""Capture Stage 5 screenshots for History & Savings and Overview."""
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
        db.commit()

        # Execute 3 intervals so that history has live audit records
        t_now = datetime.now(timezone.utc)
        for i in range(3, 0, -1):
            t_start = t_now - timedelta(minutes=15 * i)
            t_end = t_start + timedelta(minutes=15)
            runtime.execute_interval_step(
                db=db,
                shadow_session=sess,
                interval_start_utc=t_start,
                interval_end_utc=t_end,
                spot_price_eur_mwh=85.50 + (i * 2.5),
                reason_code="DEMO_SETTLEMENT",
            )
        db.commit()
        print("Seeded demo intervals for Stage 5 screenshots.")

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
            context = browser.new_context(viewport={"width": 1440, "height": 1800})
            page = context.new_page()

            # 1. Capture History & Savings
            print("Capturing History & Savings page...")
            page.goto("http://127.0.0.1:8000/dashboard/history?range=all", wait_until="networkidle")
            time.sleep(2)
            p1 = out_dir / "stage5_history_ledger.png"
            page.screenshot(path=str(p1), full_page=True)
            print(f"Captured: {p1}")

            # 2. Capture Overview
            print("Capturing Overview page...")
            page.goto("http://127.0.0.1:8000/dashboard/overview", wait_until="networkidle")
            time.sleep(2)
            p2 = out_dir / "stage5_overview.png"
            page.screenshot(path=str(p2), full_page=True)
            print(f"Captured: {p2}")

            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        print("Uvicorn stopped.")

if __name__ == "__main__":
    main()
