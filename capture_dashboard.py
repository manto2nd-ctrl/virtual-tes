"""Capture high-resolution screenshots of all Virtual TES Gen0 Dashboard pages.

Uses Playwright with Microsoft Edge in headless mode.
Saves screenshots directly to the artifact directory.
"""

from __future__ import annotations

import os
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\kaim\.gemini\antigravity\brain\89c32645-5b66-4c0d-ace3-ea31d74b5c89"
BASE_URL = "http://127.0.0.1:8000"

PAGES = [
    ("01_overview.png", "/dashboard/overview", "Overview"),
    ("02_physical.png", "/dashboard/physical", "TES Physical Model"),
    ("03_hx.png", "/dashboard/hx", "Heat Exchanger"),
    ("04_vessel.png", "/dashboard/vessel", "Vessel / Sand Sizing"),
    ("05_dispatch.png", "/dashboard/dispatch", "Market & Dispatch"),
    ("06_economics.png", "/dashboard/economics", "Economic Comparison"),
    ("07_sizing.png", "/dashboard/sizing", "Sizing Study"),
    ("08_sensitivity.png", "/dashboard/sensitivity", "Sensitivity Analysis"),
    ("09_assumptions.png", "/dashboard/assumptions", "Engineering Assumptions"),
    ("10_scenarios.png", "/dashboard/scenarios", "Scenario Explorer"),
]


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"Artifact directory: {ARTIFACT_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        # Desktop viewport
        context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page = context.new_page()

        print("Testing pages and capturing screenshots...")
        for filename, path, name in PAGES:
            url = f"{BASE_URL}{path}"
            print(f"Loading {name} ({url})...")
            page.goto(url, wait_until="networkidle")
            # Extra wait for Chart.js rendering and animations
            time.sleep(1.0)

            out_path = os.path.join(ARTIFACT_DIR, filename)
            page.screenshot(path=out_path, full_page=True)
            print(f"  Saved screenshot: {out_path} ({os.path.getsize(out_path)} bytes)")

        # Test interactive manipulation on /dashboard/physical
        print("\nTesting interactive manipulation on /dashboard/physical...")
        page.goto(f"{BASE_URL}/dashboard/physical", wait_until="networkidle")
        time.sleep(0.5)
        # Change slider to 75%
        page.evaluate("() => { onSocSliderChange(75); }")
        time.sleep(0.5)
        shot_phys_interactive = os.path.join(ARTIFACT_DIR, "02_physical_interactive_75soc.png")
        page.screenshot(path=shot_phys_interactive, full_page=False)
        print(f"  Saved interactive physical screenshot: {shot_phys_interactive}")

        # Test interactive manipulation on /dashboard/hx
        print("\nTesting interactive manipulation on /dashboard/hx...")
        page.goto(f"{BASE_URL}/dashboard/hx", wait_until="networkidle")
        time.sleep(0.5)
        # Click Optimistic Preset
        page.evaluate("() => { applyPreset(12.0, 100.0, 'Optimistic'); }")
        time.sleep(0.5)
        shot_hx_interactive = os.path.join(ARTIFACT_DIR, "03_hx_interactive_u12.png")
        page.screenshot(path=shot_hx_interactive, full_page=False)
        print(f"  Saved interactive HX screenshot: {shot_hx_interactive}")

        # Test interactive manipulation on /dashboard/vessel
        print("\nTesting interactive manipulation on /dashboard/vessel...")
        page.goto(f"{BASE_URL}/dashboard/vessel", wait_until="networkidle")
        time.sleep(0.5)
        # Load 300 kg preset
        page.evaluate("() => { loadVesselPreset(300.0, 16.32); }")
        time.sleep(0.5)
        shot_vessel_interactive = os.path.join(ARTIFACT_DIR, "04_vessel_interactive_300kg.png")
        page.screenshot(path=shot_vessel_interactive, full_page=False)
        print(f"  Saved interactive vessel screenshot: {shot_vessel_interactive}")

        context.close()
        browser.close()
        print("\nAll screenshots captured successfully!")


if __name__ == "__main__":
    main()
