from playwright.sync_api import sync_playwright
import time

def verify_chaos_attractor():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the experiments page
        url = "http://localhost:5173/pages/experiments.html"
        print(f"Navigating to {url}...")
        try:
            page.goto(url, timeout=60000)
        except Exception as e:
            print(f"Navigation failed: {e}")
            return

        # Find the Chaos Attractor card
        card_selector = "#chaos-attractor-container"
        print(f"Looking for {card_selector}...")

        try:
            page.wait_for_selector(card_selector, timeout=30000)
            element = page.locator(card_selector)
            element.scroll_into_view_if_needed()
            print("Found and scrolled to Chaos Attractor container.")
        except Exception as e:
            print(f"Error finding container: {e}")
            return

        # Wait for WebGPU/WebGL init
        print("Waiting for experiment to initialize...")
        time.sleep(5)

        # Take screenshot
        screenshot_path = "verification/chaos_attractor_upgrade.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        # Check for error messages in the container
        # Note: The .webgpu-error class is added by the script if it fails
        error_msg = page.locator(f"{card_selector} .webgpu-error").count()
        if error_msg > 0:
            print("ERROR: WebGPU Error displayed in container.")
        else:
            print("SUCCESS: No visible error messages.")

        browser.close()

if __name__ == "__main__":
    verify_chaos_attractor()
