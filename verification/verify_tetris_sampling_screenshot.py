
import os
import sys
import time
from playwright.sync_api import sync_playwright, expect

def verify_tetris_sampling():
    with sync_playwright() as p:
        # Launch with WebGPU enabled to give it a fighting chance, though headless often fails context creation
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        print("Verifying src/pages/tetris.html...")

        try:
            page.goto("http://localhost:5173/pages/tetris.html")

            # Verify Title
            expect(page.get_by_text("Block Sampling Test")).to_be_visible()
            expect(page.get_by_text("Screen-Space Sampling from /block.png")).to_be_visible()

            # Verify Canvas exists
            canvas = page.locator("#canvas-container canvas")
            expect(canvas).to_be_visible()

            # Wait a bit for the animation to start (even if it renders black)
            page.wait_for_timeout(2000)

            # Take Screenshot
            screenshot_path = os.path.abspath("verification/tetris_sampling_screenshot.png")
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            print("SUCCESS: tetris.html loaded with correct title and canvas.")

        except Exception as e:
            print(f"FAILURE: {e}")
            sys.exit(1)

        browser.close()

if __name__ == "__main__":
    verify_tetris_sampling()
