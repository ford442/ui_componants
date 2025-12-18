from playwright.sync_api import sync_playwright

def verify_neon_city():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Navigate to the Neon City page
            page.goto("http://localhost:5173/pages/neon_city.html")

            # Wait for the container
            page.wait_for_selector("#neon-city-container")

            # Verify that the canvas was created inside the container
            # The JS code appends a canvas for WebGL2
            # `this.container.appendChild(this.glCanvas);`
            page.wait_for_selector("#neon-city-container canvas")

            print("Canvas found inside container - Experiment Initialized Successfully")

            # Give it some time to render/animate
            page.wait_for_timeout(2000)

            # Take a screenshot
            page.screenshot(path="verification/neon_city_fixed.png")
            print("Screenshot saved to verification/neon_city_fixed.png")

        except Exception as e:
            print(f"Error: {e}")
            exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neon_city()
