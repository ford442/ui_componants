from playwright.sync_api import sync_playwright
import time

def verify_terraforming():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 800})

        try:
            print("Navigating to experiments.html...")
            page.goto("http://localhost:5173/pages/experiments.html")
            page.wait_for_load_state("networkidle")

            # Scroll to the element
            locator = page.locator("#planetary-terraforming-container")
            if locator.count() > 0:
                locator.scroll_into_view_if_needed()
                time.sleep(2) # Wait for animation/particles to init

                # Take screenshot
                screenshot_path = "verification/planetary_terraforming.png"
                page.screenshot(path=screenshot_path)
                print(f"Screenshot saved to {screenshot_path}")
            else:
                print("Element #planetary-terraforming-container not found!")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_terraforming()
