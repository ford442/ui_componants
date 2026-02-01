from playwright.sync_api import sync_playwright

def verify_crystal_growth():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to experiments.html...")
            response = page.goto("http://localhost:5173/pages/experiments.html")
            print(f"Response status: {response.status}")

            page.wait_for_load_state("networkidle")

            # Locate the crystal growth container
            container = page.locator("#crystal-growth-container")

            # Scroll into view
            container.scroll_into_view_if_needed()

            # Wait a bit for initialization
            page.wait_for_timeout(2000)

            if container.count() > 0:
                print("SUCCESS: Crystal Growth container found.")
                # Take screenshot
                page.screenshot(path="verification/crystal_growth.png")
                print("Screenshot saved to verification/crystal_growth.png")
            else:
                print("FAILURE: Crystal Growth container NOT found.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_crystal_growth()
