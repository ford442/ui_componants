from playwright.sync_api import sync_playwright

def verify_fireworks_interaction():
    with sync_playwright() as p:
        # Launch with WebGPU support
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Page Error: {err}"))

        try:
            print("Navigating to fireworks.html...")
            page.goto("http://localhost:5173/pages/fireworks.html")

            # Wait for initialization
            page.wait_for_timeout(3000)

            # Simulate mouse interaction
            # Click and drag across the screen
            width = page.viewport_size['width']
            height = page.viewport_size['height']

            print("Simulating mouse drag...")
            page.mouse.move(width / 2, height / 2)
            page.mouse.down()
            page.mouse.move(width / 2 + 200, height / 2) # Drag right
            page.wait_for_timeout(500) # Wait for simulation to react
            page.mouse.move(width / 2 + 200, height / 2 + 200) # Drag down
            page.wait_for_timeout(500)
            page.mouse.up()

            # Wait a bit for particles to move
            page.wait_for_timeout(1000)

            # Take screenshot
            screenshot_path = "verification/fireworks_interaction.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_fireworks_interaction()
