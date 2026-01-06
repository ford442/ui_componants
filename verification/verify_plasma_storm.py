
from playwright.sync_api import sync_playwright
import time

def verify_plasma_storm():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        context = browser.new_context()
        page = context.new_page()

        # 1. Verify entry in experiments.html
        print("Navigating to experiments.html...")
        page.goto("http://localhost:5173/pages/experiments.html")
        page.wait_for_load_state("networkidle")

        # Scroll to bottom to see the new entry
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        page.screenshot(path="verification/experiments_list.png")
        print("Screenshot of experiments list saved.")

        # 2. Verify Plasma Storm page
        print("Navigating to plasma-storm.html...")
        page.goto("http://localhost:5173/pages/plasma-storm.html")
        page.wait_for_load_state("networkidle")

        # Wait for initialization
        time.sleep(3)

        # Capture logs to see if WebGPU init
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        page.screenshot(path="verification/plasma_storm.png")
        print("Screenshot of Plasma Storm saved.")

        browser.close()

if __name__ == "__main__":
    verify_plasma_storm()
