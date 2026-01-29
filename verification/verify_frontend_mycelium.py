from playwright.sync_api import sync_playwright

def verify_mycelium():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to Galactic Mycelium...")
            page.goto("http://localhost:5173/pages/galactic-mycelium.html")

            # Wait for canvas to be present
            page.wait_for_selector("#canvas-container canvas")

            # Wait a bit for animation
            page.wait_for_timeout(2000)

            # Take screenshot
            print("Taking screenshot...")
            page.screenshot(path="verification/galactic-mycelium.png")
            print("Screenshot saved to verification/galactic-mycelium.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_mycelium()
