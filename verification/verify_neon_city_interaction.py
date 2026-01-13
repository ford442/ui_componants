import time
from playwright.sync_api import sync_playwright

def verify_neon_city_interaction():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        # Capture console messages
        hovered_building_detected = False

        def on_console(msg):
            nonlocal hovered_building_detected
            text = msg.text
            print(f"Console: {text}")
            if "Hovered building:" in text:
                hovered_building_detected = True

        page.on("console", on_console)

        try:
            print("Navigating to neon_city.html...")
            page.goto("http://localhost:5173/pages/neon_city.html")
            page.wait_for_timeout(2000) # Wait for init

            # Move mouse around to trigger raycasting
            print("Simulating mouse movement...")

            # Center and move around
            viewport = page.viewport_size
            w, h = viewport['width'], viewport['height']

            # Sweep the mouse
            steps = 10
            for i in range(steps):
                x = w * (i / steps)
                y = h * 0.5 + (h * 0.2 * (i % 2 - 0.5)) # Zigzag
                page.mouse.move(x, y)
                page.wait_for_timeout(100)

            if hovered_building_detected:
                print("SUCCESS: Hovered building detected.")
            else:
                print("FAILURE: No hovered building detected.")
                # We do not exit here to ensure screenshot is taken

            # Take screenshot
            page.screenshot(path="verification/neon_city_interaction.png")
            print("Screenshot saved to verification/neon_city_interaction.png")

        except Exception as e:
            print(f"Error: {e}")
            exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neon_city_interaction()
