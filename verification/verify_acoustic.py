
import os
import sys
from playwright.sync_api import sync_playwright

def verify_experiments_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        try:
            # 1. Visit main experiments page
            print("Visiting experiments.html...")
            page.goto("http://localhost:5173/pages/experiments.html")
            page.wait_for_load_state("networkidle")

            # 2. Check for the new experiment card
            print("Checking for Acoustic Levitation card...")
            card = page.locator("text=Acoustic Levitation").first
            if card.is_visible():
                print("SUCCESS: Acoustic Levitation card found.")
                page.screenshot(path="verification/experiments_dashboard.png")
            else:
                print("ERROR: Acoustic Levitation card not found on dashboard.")

            # 3. Visit the new experiment page
            print("Visiting acoustic-levitation.html...")
            page.goto("http://localhost:5173/pages/acoustic-levitation.html")

            # Wait for canvas to be present
            page.wait_for_selector("#canvas-container canvas")

            # Wait a bit for init
            page.wait_for_timeout(2000)

            # 4. Check for WebGPU/WebGL errors in console
            msgs = []
            page.on("console", lambda msg: msgs.append(msg.text))

            print("Taking screenshot of Acoustic Levitation experiment...")
            page.screenshot(path="verification/acoustic_levitation_page.png")

            print("Console logs:")
            for m in msgs:
                print(f"  {m}")

        except Exception as e:
            print(f"Verification failed: {e}")
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    verify_experiments_page()
