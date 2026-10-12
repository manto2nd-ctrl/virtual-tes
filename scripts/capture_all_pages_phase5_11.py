"""Comprehensive script to capture high-resolution screenshots of all 15 pages of GEN0 Energy Manager.
Phase 5.11 - Unified Simulation, Data Integrity & Dryer Validation.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from playwright.sync_api import sync_playwright

from app.config.settings import get_settings
from app.database.models import WorkerHeartbeat
from app.database.session import make_engine, make_session_factory
from app.services.shadow_runtime import ShadowRuntimeService

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")
OUTPUT_DIR = Path(r"C:\Users\kaim\.gemini\antigravity\brain\89c32645-5b66-4c0d-ace3-ea31d74b5c89")

PAGES = [
    ("page_01_login.png", "http://127.0.0.1:8000/login", "Login Page (Industrial Authentication)"),
    ("page_02_overview.png", "http://127.0.0.1:8000/dashboard/overview", "Overview (Top 5 KPIs, Energy Flow Diagram, 24h Dispatch, 8h Thermal Trends, Settlement Table)"),
    ("page_03_market_dispatch.png", "http://127.0.0.1:8000/dashboard/market", "Market Price & Dispatch (Litgrid/Elering Live Day-Ahead Prices, Effective Tariffs)"),
    ("page_04_history_savings.png", "http://127.0.0.1:8000/dashboard/history?range=all", "History & Savings Ledger (Multi-Horizon Virtual Estimated Savings, Inventory Valuation, Dryer Unit Economics)"),
    ("page_05_tes_thermal.png", "http://127.0.0.1:8000/dashboard/physical", "TES Physical Model (Quartz Sand Non-Linear Enthalpy, 80-300C, Physical SOC)"),
    ("page_06_dryer_v1.png", "http://127.0.0.1:8000/dashboard/dryer", "Virtual Wood Dryer V1 (Moisture Kinetics, Water Removal Rate, Power Deficit Analysis)"),
    ("page_07_bess_proposed.png", "http://127.0.0.1:8000/dashboard/bess", "Proposed BESS Architecture (Proposed / Inactive Baseline, Zero Premature Savings)"),
    ("page_08_heat_exchanger.png", "http://127.0.0.1:8000/dashboard/hx", "Helical Air Heat Exchanger (Effectiveness-NTU Model, 1.55 m2 Coil, U=9 W/m2K)"),
    ("page_09_vessel_geometry.png", "http://127.0.0.1:8000/dashboard/vessel", "Vessel Geometry & Quartz Sand Sizing (275.7 kg Quartz Sand, Insulation Losses)"),
    ("page_10_economic_models.png", "http://127.0.0.1:8000/dashboard/economics", "Economic Models & Optimization Heuristics (Linear Programming vs Direct Electric)"),
    ("page_11_sizing_matrix.png", "http://127.0.0.1:8000/dashboard/sizing", "Sizing Matrix (Storage Capacity vs Heater Sizing Parametric Analysis)"),
    ("page_12_sensitivity.png", "http://127.0.0.1:8000/dashboard/sensitivity", "Sensitivity Analysis (Electricity Price & Process Heat Demand Sweeps)"),
    ("page_13_saved_scenarios.png", "http://127.0.0.1:8000/dashboard/scenarios", "Saved Engineering Scenarios (Scenario Storage and Model Comparison)"),
    ("page_14_system_devices.png", "http://127.0.0.1:8000/dashboard/assumptions", "System Devices & Assumptions (Worker Heartbeats, Hardware Parameters, Safety Interlocks)"),
    ("page_15_dispatch_schedule.png", "http://127.0.0.1:8000/dashboard/dispatch", "Dispatch Schedule & Shadow Audit (Raw 15-Minute Interval Execution Log)"),
]


def seed_rich_telemetry():
    """Seed comprehensive intervals, worker heartbeat, and batch state for pristine screenshot rendering."""
    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    now_utc = datetime.now(timezone.utc)

    with session_factory() as db:
        runtime = ShadowRuntimeService(tz=VILNIUS_TZ)
        sess = runtime.get_or_create_session(db)
        sess.status = "RUNNING"
        sess.process_enabled = True
        sess.process_heat_demand_kw = 1.5
        sess.current_soc_percent = 48.2
        sess.current_energy_kwh = 7.23
        sess.current_temp_c = 193.4
        db.commit()

        # Seed authoritative worker heartbeat
        hb = WorkerHeartbeat(
            worker_id="gen0-worker-lead",
            timestamp_utc=now_utc,
            status="WAITING FOR NEXT INTERVAL",
            app_env="production",
            market_status="LIVE",
            shadow_status="RUNNING",
            last_market_fetch_utc=now_utc - timedelta(minutes=4),
            last_optimization_utc=now_utc - timedelta(minutes=1),
            last_executed_interval_utc=now_utc - timedelta(minutes=5),
            next_interval_utc=now_utc + timedelta(minutes=10),
            details_json={
                "is_leader": True,
                "db_size_mb": 2.45,
                "model_version": "Phase 5.11 Unified System State",
                "bidding_zone": "LT",
                "status_code": "WAITING FOR NEXT INTERVAL",
            },
        )
        db.add(hb)

        # Seed intervals within today's local Vilnius day
        now_local = datetime.now(VILNIUS_TZ)
        t_base = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

        intervals = [
            (0, 15, 38.50, 8.5, 0.0, "OFF_PEAK_CHARGE"),
            (15, 30, 42.00, 6.0, 0.0, "OFF_PEAK_CHARGE"),
            (30, 45, 84.50, 0.0, 1.5, "PROCESS_HEAT_DELIVERY"),
            (45, 60, 96.20, 0.0, 1.5, "PROCESS_HEAT_DELIVERY"),
            (60, 75, 112.00, 0.0, 1.5, "PROCESS_HEAT_DELIVERY"),
            (75, 90, 125.40, 0.0, 1.5, "PROCESS_HEAT_DELIVERY"),
        ]

        for m_start, m_end, spot, chg, disch, reason in intervals:
            t_s = (t_base + timedelta(minutes=m_start)).astimezone(timezone.utc)
            t_e = (t_base + timedelta(minutes=m_end)).astimezone(timezone.utc)
            runtime.execute_interval_step(
                db=db,
                shadow_session=sess,
                interval_start_utc=t_s,
                interval_end_utc=t_e,
                spot_price_eur_mwh=spot,
                requested_charge_kw=chg,
                requested_discharge_kw=disch,
                reason_code=reason,
            )

        db.commit()
        print("Successfully seeded telemetry, heartbeat, and 15-minute execution history.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_rich_telemetry()

    print("Starting local Uvicorn server...")
    proc = subprocess.Popen(
        [r".\.venv\Scripts\python.exe", "-m", "uvicorn", "app.web.main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        with sync_playwright() as p:
            print("Launching Microsoft Edge / Chromium headless browser...")
            browser = p.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 1000},
                device_scale_factor=1.25,
            )
            page = context.new_page()

            for filename, url, label in PAGES:
                print(f"Capturing {filename} - {label}...")
                target_path = OUTPUT_DIR / filename
                page.goto(url, wait_until="networkidle", timeout=30000)
                # Wait 1.5 seconds for Chart.js, SVGs, and animations to settle
                time.sleep(1.5)
                page.screenshot(path=str(target_path), full_page=True)
                print(f"  -> Successfully saved: {target_path} ({target_path.stat().st_size // 1024} KB)")

            browser.close()
            print("\nALL 15 PAGES CAPTURED SUCCESSFULLY!")
    finally:
        print("Shutting down local server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server shutdown complete.")


if __name__ == "__main__":
    main()
