
import os
import time
from playwright.sync_api import sync_playwright, expect

def verify_tetris_experiments():
    with sync_playwright() as p:
        # Launch browser with WebGPU enabled (though headless chrome might struggle with GPU in some envs)
        # Using --enable-unsafe-webgpu to try and get some GPU features if possible,
        # but primarily checking for page load and lack of JS errors.
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        # Capture console logs to check for errors
        console_logs = []
        page.on("console", lambda msg: console_logs.append(msg.text))

        page.on("pageerror", lambda exc: print(f"Page Error: {exc}"))

        try:
            # Navigate to the new page
            url = "http://localhost:5173/pages/tetris-experiments.html"
            print(f"Navigating to {url}...")
            page.goto(url)

            # Wait for content to load
            page.wait_for_selector(".experiment-grid", timeout=10000)
            print("Experiment grid loaded.")

            # Check for the 3 headers
            expect(page.get_by_role("heading", name="Neon Tetris Rain")).to_be_visible()
            expect(page.get_by_role("heading", name="Voxel Destruct")).to_be_visible()
            expect(page.get_by_role("heading", name="Holographic Glass")).to_be_visible()
            print("All 3 experiment headers found.")

            # Wait a bit for canvases to initialize
            time.sleep(3)

            # Check for canvases
            rain_canvas = page.locator("#neon-rain-container canvas")
            voxel_canvas = page.locator("#voxel-destruct-container canvas")
            holo_canvas = page.locator("#hologram-container canvas")

            print(f"Rain canvases found: {rain_canvas.count()}")
            print(f"Voxel canvases found: {voxel_canvas.count()}")
            print(f"Holo canvases found: {holo_canvas.count()}")

            # Take screenshot
            screenshot_path = "verification/tetris_experiments.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved to {screenshot_path}")

            # Check console for errors
            error_logs = [log for log in console_logs if "error" in log.lower()]
            if error_logs:
                print("Console Errors found:")
                for log in error_logs:
                    print(f" - {log}")
            else:
                print("No console errors detected.")

        except Exception as e:
            print(f"Verification failed: {e}")
            # Take emergency screenshot
            page.screenshot(path="verification/error_state.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_tetris_experiments()
