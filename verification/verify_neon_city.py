import os
from playwright.sync_api import sync_playwright

def verify_neon_city():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        url = "http://localhost:5173/pages/neon_city.html"
        print(f"Checking {url}...")

        try:
            page.goto(url)
            # Wait for shader compilation and render
            page.wait_for_timeout(3000)

            # Take screenshot
            screenshot_path = "/app/verification/neon_city_upgrade.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"FAILED: {e}")
            exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neon_city()
