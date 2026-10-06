"""Capture screenshots of the Phase 5.8.2 Live Virtual TES Shadow Runtime."""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    out_dir = Path(r"C:\Users\kaim\.gemini\antigravity\brain\89c32645-5b66-4c0d-ace3-ea31d74b5c89")
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 2600})
        page = context.new_page()

        page.on("console", lambda msg: print(f"[Browser Console] {msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: print(f"[Browser Error] {err}"))

        print("Navigating to overview...")
        page.goto("http://127.0.0.1:8000/dashboard/overview", wait_until="networkidle")
        time.sleep(3)

        # Screenshot 1: Full overview running
        s1 = out_dir / "01_overview_shadow_running.png"
        page.screenshot(path=str(s1), full_page=True)
        print(f"Captured: {s1}")

        # Test interaction: Click "Pause"
        print("Pausing session...")
        pause_btn = page.locator("button:has-text('Pause')")
        if pause_btn.count() > 0:
            pause_btn.first.click()
            time.sleep(2)
            s2 = out_dir / "02_shadow_paused.png"
            page.screenshot(path=str(s2), full_page=True)
            print(f"Captured: {s2}")

            # Test interaction: Click "Resume"
            print("Resuming session...")
            resume_btn = page.locator("button:has-text('Resume')")
            if resume_btn.count() > 0:
                resume_btn.first.click()
                time.sleep(2)
                s3 = out_dir / "03_shadow_resumed.png"
                page.screenshot(path=str(s3), full_page=True)
                print(f"Captured: {s3}")

        browser.close()
    print("Screenshot capture complete!")

if __name__ == "__main__":
    main()
