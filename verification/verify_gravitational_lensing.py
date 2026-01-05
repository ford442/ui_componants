from playwright.sync_api import sync_playwright, expect
import time
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu', '--enable-features=Vulkan']
        )
        page = browser.new_page()

        # Navigate to the experiment
        url = "http://localhost:5173/pages/gravitational-lensing.html"
        print(f"Navigating to {url}...")

        # Check console logs for errors
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        try:
            response = page.goto(url, wait_until="networkidle")
            if not response.ok:
                print(f"Failed to load page: {response.status}")
                return

            print("Page loaded. Waiting for initialization...")
            time.sleep(2) # Give it time to initialize WebGL/WebGPU

            # Check if canvas exists
            canvas = page.locator("canvas").first
            expect(canvas).to_be_visible()
            print("Canvas found and visible.")

            # Take screenshot
            screenshot_path = "verification/gravitational-lensing.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            browser.close()

if __name__ == "__main__":
    run()
