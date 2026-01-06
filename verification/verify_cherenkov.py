from playwright.sync_api import sync_playwright
import time

def verify_cherenkov():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        print("Navigating to Cherenkov Radiation experiment...")
        page.goto("http://localhost:5173/pages/cherenkov-radiation.html")

        # Wait for initialization
        time.sleep(2)

        # Check for canvas
        canvas = page.locator("canvas").first
        if canvas.count() > 0:
            print("Canvas found.")
        else:
            print("ERROR: Canvas not found.")

        # Check for WebGPU error message (expected in this env)
        error_msg = page.locator(".webgpu-error")
        if error_msg.count() > 0:
            print("WebGPU error message present (Expected in headless).")

        # Take screenshot
        screenshot_path = "verification/cherenkov_screenshot.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_cherenkov()
