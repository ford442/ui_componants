from playwright.sync_api import sync_playwright

def verify_experiment():
    with sync_playwright() as p:
        # Launch with WebGPU enabled
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        # Listen for console logs
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        print("Navigating to experiment page...")
        # Note: Port might vary, assuming 5173 default for Vite
        try:
            page.goto("http://localhost:5173/pages/hybrid-magnetic-field.html")

            # Wait for canvas to be present
            page.wait_for_selector("#canvas-container canvas")

            # Wait a bit for initialization and animation
            page.wait_for_timeout(2000)

            # Screenshot
            print("Taking screenshot...")
            page.screenshot(path="verification/hybrid_magnetic_field.png")
            print("Screenshot saved.")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            browser.close()

if __name__ == "__main__":
    verify_experiment()
