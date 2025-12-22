from playwright.sync_api import sync_playwright
import time

def verify_temporal_core():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Navigate to the experiments page
            page.goto("http://localhost:5173/pages/experiments.html")

            # Wait for the page to load
            page.wait_for_load_state("networkidle")

            # Scroll to the Temporal Data Core section
            element = page.locator("#temporal-core-container")
            element.scroll_into_view_if_needed()

            # Wait for a bit to let animations run
            time.sleep(2)

            # Take a screenshot of the specific container
            element.screenshot(path="verification/temporal_core_screenshot.png")
            print("Screenshot taken successfully.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_temporal_core()
