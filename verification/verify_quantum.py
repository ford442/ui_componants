from playwright.sync_api import sync_playwright

def verify_experiments(page):
    print("Navigating to experiments page...")
    # Using 5173 as it's the default Vite port, but I'll check the log if it fails.
    try:
        page.goto("http://localhost:5173/pages/experiments.html")
        page.wait_for_load_state("networkidle")
    except Exception as e:
        print(f"Failed to load page: {e}")
        return

    print("Checking for Quantum Data Stream section...")
    # Check if the new section exists
    try:
        page.wait_for_selector("#quantum-data-stream-container", timeout=5000)
        print("Quantum Data Stream container found.")
    except:
        print("Quantum Data Stream container NOT found.")

    # Scroll to the element to make sure it's rendered
    element = page.locator("#quantum-data-stream-container")
    element.scroll_into_view_if_needed()

    # Take a screenshot of the main experiments page showing the new section
    print("Taking screenshot of experiments page...")
    page.screenshot(path="verification/experiments_page_with_quantum.png")

    print("Navigating to the full experiment page...")
    # Navigate directly to the page to be sure, or click the link
    # NOTE: Updated to new filename
    page.goto("http://localhost:5173/pages/quantum-data-stream.html")

    # Wait for the full page to load
    page.wait_for_load_state("networkidle")

    # Check for canvas elements
    print("Checking for canvases...")
    # We expect 2 canvases: one for WebGL2, one for WebGPU
    try:
        page.wait_for_selector("canvas", timeout=5000)
        canvases = page.locator("canvas").count()
        print(f"Found {canvases} canvas elements.")
    except:
         print("No canvas elements found.")

    # Wait a bit for animation
    page.wait_for_timeout(2000)

    # Take a screenshot of the full experiment
    print("Taking screenshot of full experiment...")
    page.screenshot(path="verification/quantum_data_stream_full.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        # Pass args to launch, not new_context
        browser = p.chromium.launch(
            headless=True,
            args=["--enable-unsafe-webgpu", "--use-gl=egl"]
        )
        context = browser.new_context()
        page = context.new_page()

        # Subscribe to console logs to see JS errors
        page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

        try:
            verify_experiments(page)
        finally:
            browser.close()
