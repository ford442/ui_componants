from playwright.sync_api import sync_playwright

def verify_neon_city():
    with sync_playwright() as p:
        # Enable WebGPU
        args = ["--enable-unsafe-webgpu"]
        browser = p.chromium.launch(headless=True, args=args)
        page = browser.new_page()

        # Log console
        page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

        try:
            page.goto("http://localhost:5173/pages/neon_city.html")
            page.wait_for_selector("#neon-city-container")
            page.wait_for_selector("#neon-city-container canvas")
            print("Canvas found - Init Success")

            # Interact
            print("Interacting...")
            for i in range(20):
                page.mouse.move(100 + i*10, 100 + i*5)
                page.wait_for_timeout(50)

            page.wait_for_timeout(1500)
            page.screenshot(path="verification/neon_city_fixed.png")
            print("Screenshot saved")
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neon_city()
