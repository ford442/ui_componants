from playwright.sync_api import sync_playwright

def verify_silicon_life():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to experiments dashboard...")
            page.goto("http://localhost:5173/pages/experiments.html")
            page.wait_for_load_state("networkidle")

            # 1. Verify Card Exists
            print("Verifying card...")
            card = page.locator("text=Silicon Life")
            if card.count() > 0:
                print("Silicon Life card found.")

            # Scroll to it
            card.scroll_into_view_if_needed()
            page.screenshot(path="verification/dashboard_screenshot.png")

            # 2. Verify Full Page Load
            print("Navigating to standalone page...")
            page.goto("http://localhost:5173/pages/silicon-life.html")
            page.wait_for_load_state("networkidle")

            # Wait for canvas
            page.wait_for_selector("#silicon-life-container canvas")
            print("Canvas found.")

            # Wait a bit for simulation to run
            page.wait_for_timeout(1000)

            page.screenshot(path="verification/silicon_life_screenshot.png")
            print("Screenshots taken.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_silicon_life()
