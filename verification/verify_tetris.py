
from playwright.sync_api import sync_playwright, expect
import time

def run():
    with sync_playwright() as p:
        # Launch with WebGPU enabled (though headless might still fall back to software or have issues, but let's try)
        # We need to use args to enable unsafe webgpu if needed, though recent Chrome versions might support it better.
        # But here we mainly want to see if the page loads without the specific JS import error.
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        # Capture console logs to check for errors
        console_logs = []
        page.on("console", lambda msg: console_logs.append(msg.text))

        try:
            # Navigate to the page
            # Assuming dev server runs on port 5173 (default for Vite)
            # Note: The root is 'src', so the URL should be http://localhost:5173/pages/tetris.html
            # (Based on memory: "browser URLs must omit the src/ prefix")
            # Wait, memory said: "Due to the root: 'src' configuration in Vite, browser URLs must omit the src/ prefix (e.g., access http://localhost:5173/pages/file.html for src/pages/file.html)."
            url = "http://localhost:5173/pages/tetris.html"
            print(f"Navigating to {url}")
            page.goto(url)

            # Wait for some time to allow scripts to load and execute
            page.wait_for_timeout(2000)

            # Check for the canvas container content or just existence
            # The canvas is added by the JS class.
            # NeonTetrisRain adds a canvas.
            canvas_count = page.locator("canvas").count()
            print(f"Canvas count: {canvas_count}")

            # Check for the critical error "TetrisExperiment is not exported" in logs
            error_found = False
            for log in console_logs:
                print(f"Console: {log}")
                if "not exported" in log or "Failed to fetch" in log or "Uncaught SyntaxError" in log:
                    error_found = True

            if error_found:
                 print("Critical error found in logs!")
            else:
                 print("No critical import errors found.")

            # Take a screenshot
            screenshot_path = "verification/tetris_verification.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
