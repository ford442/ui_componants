from playwright.sync_api import sync_playwright

def verify_experiments(page):
    page.goto("http://localhost:5173/pages/experiments.html")

    # Check if the "Cosmic String Instability" section exists
    page.wait_for_selector("text=Cosmic String Instability")

    # Take a screenshot of the experiment card
    element = page.locator("#cosmic-string-container").locator("xpath=..")
    element.screenshot(path="verification/experiments_card.png")

    print("Experiments page verified.")

def verify_cosmic_string_page(page):
    page.goto("http://localhost:5173/pages/cosmic-string.html")

    # Wait for canvas
    page.wait_for_selector("canvas")

    # Wait a bit for simulation to start
    page.wait_for_timeout(2000)

    # Take a screenshot
    page.screenshot(path="verification/cosmic_string_page.png")

    print("Cosmic String page verified.")

    # Check for WebGPU error message or success logs (console logs are tricky in headless, but we can check for the fallback message if WebGPU fails)
    # Note: Headless chromium usually needs specific flags for WebGPU.

    # Check if we have the WebGL2 canvas (id/class might not be set, but it's a canvas)
    # The code adds a canvas.
    canvases = page.locator("canvas").count()
    print(f"Found {canvases} canvases.")

if __name__ == "__main__":
    with sync_playwright() as p:
        # Enable unsafe webgpu for headless
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()
        try:
            verify_experiments(page)
            verify_cosmic_string_page(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
