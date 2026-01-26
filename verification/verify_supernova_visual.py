from playwright.sync_api import sync_playwright

def verify_supernova():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to experiments.html...")
            page.goto("http://localhost:5173/pages/experiments.html")
            page.wait_for_load_state("networkidle")

            # Locate the Supernova container
            locator = page.locator("#supernova-remnant-container")
            locator.scroll_into_view_if_needed()

            # Wait a bit for the animation to start/render
            page.wait_for_timeout(2000)

            print("Taking screenshot...")
            page.screenshot(path="verification/supernova_remnant.png")
            print("Screenshot saved to verification/supernova_remnant.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_supernova()
