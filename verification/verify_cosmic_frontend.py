from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=['--enable-unsafe-webgpu'],
            headless=True
        )
        page = browser.new_page()

        try:
            print("Navigating to Cosmic Radiation page...")
            page.goto("http://localhost:5173/pages/cosmic-radiation.html")

            # Expect title to be correct
            expect(page).to_have_title("Cosmic Radiation - Hybrid Render")

            # Wait for canvas (it might take a moment to init WebGPU/WebGL)
            canvas = page.locator("canvas").first
            expect(canvas).to_be_visible(timeout=5000)

            # Wait a bit for simulation to run
            page.wait_for_timeout(2000)

            # Take screenshot
            screenshot_path = "verification/cosmic_radiation.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
