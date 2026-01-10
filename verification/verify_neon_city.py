from playwright.sync_api import sync_playwright

def verify_neon_city():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        # Capture console messages
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Page Error: {err}"))

        try:
            print("Navigating to neon_city.html...")
            response = page.goto("http://localhost:5173/pages/neon_city.html")
            print(f"Response status: {response.status}")

            # Wait for container
            page.wait_for_selector("#neon-city-container", timeout=5000)
            print("Container found.")

            # Wait for initialization
            page.wait_for_timeout(2000)

            # Simulate mouse move
            print("Simulating mouse interaction...")
            page.mouse.move(100, 100)
            page.wait_for_timeout(500)
            page.mouse.move(500, 500)
            page.wait_for_timeout(2000) # Wait for animation to settle/update

            print("Interaction completed without crash.")

            # Take screenshot
            page.screenshot(path="verification/neon_city.png")
            print("Screenshot saved to verification/neon_city.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neon_city()
