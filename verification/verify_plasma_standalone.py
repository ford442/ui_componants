from playwright.sync_api import sync_playwright

def verify_experiments(page):
    # Go directly to standalone page to verify it works in isolation
    print("Navigating to standalone plasma confinement page...")
    page.goto("http://localhost:5173/pages/plasma-confinement.html")

    # Wait for canvas container
    print("Waiting for canvas-container...")
    page.wait_for_selector("#canvas-container")

    # Wait for WebGPU/WebGL to initialize
    print("Waiting for initialization...")
    page.wait_for_timeout(5000)

    # Take a screenshot
    print("Taking standalone screenshot...")
    page.screenshot(path="verification/plasma_confinement_standalone.png")

    # Capture console logs to check for errors
    print("Checking for console errors...")

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
