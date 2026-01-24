from playwright.sync_api import sync_playwright
import time

def verify_primordial_soup():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        url = "http://localhost:5173/pages/primordial-soup.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for canvas to be present
        page.wait_for_selector("#primordial-soup-container canvas")

        # Wait a bit for simulation to start (WebGPU init might take a moment, or fallback)
        time.sleep(2)

        # Take screenshot
        screenshot_path = "verification/primordial_soup.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_primordial_soup()
