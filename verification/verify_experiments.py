from playwright.sync_api import sync_playwright

def verify_experiments(page):
    print("Navigating to experiments.html...")
    page.goto("http://localhost:5173/pages/experiments.html")
    page.wait_for_load_state("networkidle")

    print("Checking for console errors...")
    # This is handled by the event listener in main

    # Check for the new experiment section
    print("Checking for Holographic Data Stream section...")
    section = page.locator("text=Holographic Data Stream")
    if section.count() > 0:
        print("Found Holographic Data Stream section.")
    else:
        print("ERROR: Holographic Data Stream section not found.")

    # Screenshot the dashboard
    page.screenshot(path="verification/dashboard.png")
    print("Dashboard screenshot saved.")

    # Navigate to the new experiment page
    print("Navigating to holographic-stream.html...")
    page.goto("http://localhost:5173/pages/holographic-stream.html")
    page.wait_for_timeout(2000) # Wait for animation/initialization

    # Screenshot the experiment
    page.screenshot(path="verification/holographic_stream.png")
    print("Experiment screenshot saved.")

if __name__ == "__main__":
    with sync_playwright() as p:
        # Enable unsafe webgpu
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
