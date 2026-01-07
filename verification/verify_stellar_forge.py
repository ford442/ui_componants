from playwright.sync_api import sync_playwright

def verify_stellar_forge():
    with sync_playwright() as p:
        # Launch browser with WebGPU enabled (though software emulation might be tricky)
        # Note: --enable-unsafe-webgpu is needed for headless chrome usually,
        # but software rasterization for WebGL/WebGPU in headless is often black/empty.
        # We will try best effort.
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu', '--use-gl=swiftshader']
        )
        page = browser.new_page()

        # Navigate to the experiment page
        try:
            page.goto('http://localhost:5173/pages/stellar-forge.html')

            # Wait for canvas to be present
            page.wait_for_selector('canvas', timeout=5000)

            # Wait a bit for animation
            page.wait_for_timeout(2000)

            # Check for console errors
            page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

            # Take screenshot
            page.screenshot(path='verification/stellar_forge_screenshot.png')
            print("Screenshot taken.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_stellar_forge()
