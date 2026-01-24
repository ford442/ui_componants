from playwright.sync_api import sync_playwright

def snapshot_experiment():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to atmospheric-entry.html...")
            response = page.goto("http://localhost:5173/pages/atmospheric-entry.html")
            print(f"Response status: {response.status}")

            # Wait for canvas to be present
            page.wait_for_selector("#atmospheric-entry-container canvas", timeout=10000)

            # Wait a bit for the animation to start and particles to spawn
            page.wait_for_timeout(2000)

            # Take screenshot
            path = "verification/atmospheric_entry_snapshot.png"
            page.screenshot(path=path)
            print(f"Screenshot saved to {path}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    snapshot_experiment()
