from playwright.sync_api import sync_playwright

def verify_experiment():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the page - corrected URL
        url = "http://localhost:5173/pages/tetris-experiments.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for the hologram container
        print("Waiting for container...")
        try:
            page.wait_for_selector("#hologram-container", timeout=5000)
        except Exception as e:
            print(f"Error finding selector: {e}")
            page.screenshot(path="verification/error_state.png")
            browser.close()
            return

        # Wait a bit for the canvas to render
        print("Waiting for render...")
        page.wait_for_timeout(3000)

        # Verify canvas exists inside
        canvas = page.query_selector("#hologram-container canvas")
        if not canvas:
            print("ERROR: Canvas not found inside #hologram-container")
            browser.close()
            return

        print("Canvas found. Taking screenshot...")
        page.screenshot(path="verification/hologram_glass.png")
        print("Screenshot saved.")

        browser.close()

if __name__ == "__main__":
    verify_experiment()
