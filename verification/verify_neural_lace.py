from playwright.sync_api import sync_playwright
import time

def verify_neural_lace():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        try:
            print("Navigating to Neural Lace...")
            page.goto("http://localhost:5173/pages/neural-lace.html")

            # Wait for canvas
            page.wait_for_selector("#canvas-container canvas")

            # Wait a bit for simulation to run
            print("Waiting for simulation...")
            time.sleep(2)

            # Take screenshot
            screenshot_path = "verification/neural_lace.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neural_lace()
