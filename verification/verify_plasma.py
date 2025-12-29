from playwright.sync_api import sync_playwright

def verify_experiments(page):
    print("Navigating to experiments page...")
    page.goto("http://localhost:5173/pages/experiments.html")
    print("Page loaded.")

    # Wait for the plasma confinement container to be visible
    print("Waiting for plasma-confinement-container...")
    page.wait_for_selector("#plasma-confinement-container")

    # Wait a bit for the animation to start and particles to appear
    page.wait_for_timeout(3000)

    # Take a screenshot
    print("Taking screenshot...")
    page.screenshot(path="verification/experiments_page.png")

    # Also verify that we can navigate to the standalone page
    print("Navigating to standalone plasma confinement page...")
    page.goto("http://localhost:5173/pages/plasma-confinement.html")

    # Wait for canvas container
    page.wait_for_selector("#canvas-container")

    # Wait for WebGPU/WebGL to initialize
    page.wait_for_timeout(3000)

    # Take another screenshot
    print("Taking standalone screenshot...")
    page.screenshot(path="verification/plasma_confinement_standalone.png")

    # Capture console logs to check for errors
    print("Checking for console errors...")
    # Note: Console logs are captured via event listener in the main block if needed,
    # but here we just rely on the script completing without error.

if __name__ == "__main__":
    with sync_playwright() as p:
        # Launch with WebGPU enabled (unsafe-webgpu flag)
        browser = p.chromium.launch(
            headless=True,
            args=["--enable-unsafe-webgpu"]
        )
        page = browser.new_page()

        # Listen for console messages
        page.on("console", lambda msg: print(f"Browser console: {msg.text}"))

        try:
            verify_experiments(page)
            print("Verification script completed successfully.")
        except Exception as e:
            print(f"Verification failed: {e}")
        finally:
            browser.close()
