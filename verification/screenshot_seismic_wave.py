
import sys
from playwright.sync_api import sync_playwright

def verify_seismic_wave_screenshot():
    page_url = "http://localhost:5173/pages/seismic-wave.html"
    screenshot_path = "verification/seismic_wave.png"

    with sync_playwright() as p:
        # Enable unsafe webgpu for headless chrome to attempt to render WebGPU content
        browser = p.chromium.launch(args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        print(f"Navigating to {page_url}...")
        try:
            page.goto(page_url)
            # Wait longer for init and animation to start
            page.wait_for_timeout(3000)
        except Exception as e:
            print(f"Failed to load {page_url}: {e}")
            return False

        # Check if canvas exists
        try:
            page.wait_for_selector("canvas", timeout=5000)
        except Exception as e:
             print(f"No canvas found in {page_url}")
             return False

        # Take screenshot
        page.screenshot(path=screenshot_path)
        print(f"📸 Screenshot saved to {screenshot_path}")
        browser.close()
        return True

if __name__ == "__main__":
    if not verify_seismic_wave_screenshot():
        sys.exit(1)
