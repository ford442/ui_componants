from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        # Launch with WebGPU support enabled
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        context = browser.new_context()
        page = context.new_page()

        # Capture console logs to debug WebGPU/WebGL issues
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        url = "http://localhost:5173/pages/nano-plex.html"
        print(f"Navigating to {url}")
        page.goto(url)

        # Wait for initialization
        time.sleep(2)

        # Check for canvas
        canvas = page.locator("canvas")
        count = canvas.count()
        print(f"Found {count} canvas elements")

        if count > 0:
            page.screenshot(path="verification/nano_plex.png")
            print("Screenshot saved to verification/nano_plex.png")
        else:
            print("No canvas found!")

        browser.close()

if __name__ == "__main__":
    run()
