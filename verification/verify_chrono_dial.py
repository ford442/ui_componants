from playwright.sync_api import sync_playwright
import time

def verify_chrono_dial():
    with sync_playwright() as p:
        # Launch browser with WebGPU enabled (though headless might not fully support it, we check for no crash)
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu', '--use-gl=swiftshader']
        )
        page = browser.new_page()

        # Navigate to the Chrono Dial page
        url = "http://localhost:5173/pages/chrono-dial.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for canvas to be present
        page.wait_for_selector("canvas", timeout=5000)

        # Wait a bit for initialization
        time.sleep(2)

        # Check for console logs
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        # Take a screenshot
        screenshot_path = "verification/chrono_dial_verification.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_chrono_dial()
