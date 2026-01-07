from playwright.sync_api import sync_playwright


def verify_experiments(page):
    """Run verification steps for dashboard and experiment pages.

    This script checks both the Holographic Data Stream and Neural Data Core
    experiments (if present) and captures screenshots for the dashboard and
    individual experiment pages.
    """

    # Dashboard / experiments page
    print("Navigating to experiments.html...")
    page.goto("http://localhost:5173/pages/experiments.html")
    page.wait_for_load_state("networkidle")

    # Screenshot dashboard
    page.screenshot(path="verification/dashboard.png")
    print("Dashboard screenshot saved.")

    # Holographic experiment verification
    print("Checking for Holographic Data Stream section...")
    holographic_section = page.locator("text=Holographic Data Stream")
    if holographic_section.count() > 0:
        print("Found Holographic Data Stream section.")
        print("Navigating to holographic-stream.html...")
        page.goto("http://localhost:5173/pages/holographic-stream.html")
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/holographic_stream.png")
        print("Holographic experiment screenshot saved.")
    else:
        print("Holographic Data Stream section not found; skipping holographic verification.")

    # Neural Data Core verification
    print("Checking for Neural Data Core section...")
    neural_section = page.locator("text=Neural Data Core")
    if neural_section.count() > 0:
        print("Found Neural Data Core section.")
        print("Navigating to neural-data-core.html...")
        page.goto("http://localhost:5173/pages/neural-data-core.html")
        page.wait_for_selector("canvas")
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/neural_data_core.png")
        print("Neural Data Core experiment screenshot saved.")
    else:
        print("Neural Data Core section not found; skipping neural verification.")


if __name__ == "__main__":
    with sync_playwright() as p:
        # Enable unsafe webgpu where available
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        # Capture console logs
        page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

        try:
            verify_experiments(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
