import time
from playwright.sync_api import sync_playwright

def verify_neon_city_hack():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        # Capture console messages
        hacked_building_detected = False

        def on_console(msg):
            nonlocal hacked_building_detected
            text = msg.text
            # print(f"Console: {text}")
            if "Hacked building:" in text:
                print(f"SUCCESS LOG: {text}")
                hacked_building_detected = True

        page.on("console", on_console)

        try:
            print("Navigating to neon_city.html...")
            page.goto("http://localhost:5173/pages/neon_city.html")
            page.wait_for_timeout(2000) # Wait for init

            # Move mouse around and click to trigger hack
            print("Simulating mouse movement and clicks...")

            viewport = page.viewport_size
            w, h = viewport['width'], viewport['height']

            # Sweep and click
            steps = 20
            for i in range(steps):
                x = w * (i / steps)
                # Zigzag vertically to cover ground
                y = h * 0.6 + (h * 0.2 * (i % 2 - 0.5))

                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.up()
                page.wait_for_timeout(100)

                if hacked_building_detected:
                    break

            if hacked_building_detected:
                print("SUCCESS: Hacked building detected.")
            else:
                print("FAILURE: No hacked building detected.")
                exit(1)

            # Take screenshot
            page.screenshot(path="verification/neon_city_hack_final.png")
            print("Screenshot saved to verification/neon_city_hack_final.png")

        except Exception as e:
            print(f"Error: {e}")
            exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neon_city_hack()
