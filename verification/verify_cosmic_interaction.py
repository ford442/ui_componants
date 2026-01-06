
from playwright.sync_api import sync_playwright
import time
import os

def verify_cosmic_interaction():
    with sync_playwright() as p:
        # Launch browser with WebGPU enabled (though unsafe-webgpu might be needed for headless)
        # Note: Headless WebGPU support varies.
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        url = "http://localhost:5173/pages/cosmic-string.html"
        print(f"Navigating to {url}...")

        try:
            page.goto(url)
            page.wait_for_load_state("networkidle")

            # Allow time for initialization
            time.sleep(2)

            # Check for console errors
            page.on("console", lambda msg: print(f"Console: {msg.text}"))

            # Screenshot 1: Initial state (Mouse at 0,0 default or off screen)
            screenshot_path_1 = "verification/cosmic_initial.png"
            page.screenshot(path=screenshot_path_1)
            print(f"Captured initial state: {screenshot_path_1}")

            # Interact: Move mouse to center
            # Center of viewport
            viewport_size = page.viewport_size
            center_x = viewport_size['width'] / 2
            center_y = viewport_size['height'] / 2

            print(f"Moving mouse to center: {center_x}, {center_y}")
            page.mouse.move(center_x, center_y)

            # Wait for reaction (particles moving, string bending)
            time.sleep(2)

            # Screenshot 2: Interaction state
            screenshot_path_2 = "verification/cosmic_interaction.png"
            page.screenshot(path=screenshot_path_2)
            print(f"Captured interaction state: {screenshot_path_2}")

            # Move mouse to top left
            print("Moving mouse to 100, 100")
            page.mouse.move(100, 100)
            time.sleep(1)

            screenshot_path_3 = "verification/cosmic_interaction_2.png"
            page.screenshot(path=screenshot_path_3)
            print(f"Captured interaction state 2: {screenshot_path_3}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_cosmic_interaction()
