from playwright.sync_api import sync_playwright
import time

def verify_spectral_loom():
    print("Starting verification for Spectral Loom...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Navigate to the page
        url = "http://localhost:5173/pages/spectral-loom.html"
        print(f"Navigating to {url}...")
        try:
            page.goto(url)
        except Exception as e:
            print(f"Error navigating: {e}")
            return

        # 2. Check for container
        print("Checking for container #canvas-container...")
        try:
            page.wait_for_selector("#canvas-container", timeout=5000)
            print("SUCCESS: Container found.")
        except Exception as e:
            print(f"FAILURE: Container not found. {e}")
            browser.close()
            return

        # 3. Setup console listener
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        # 4. Wait for initialization (WebGPU might take a moment)
        time.sleep(2)

        # 5. Simulate Interaction (Weaving Mode)
        print("Simulating interaction (mousedown)...")
        # Get center of screen
        viewport = page.viewport_size
        cw, ch = viewport['width'], viewport['height']

        # Mouse move to center
        page.mouse.move(cw / 2, ch / 2)

        # Mouse down (trigger tension)
        page.mouse.down()
        time.sleep(1) # Hold for 1 second

        # Mouse move while holding
        page.mouse.move(cw / 2 + 100, ch / 2)
        time.sleep(0.5)

        # Mouse up
        page.mouse.up()
        print("Interaction sequence complete.")

        # 6. Take Screenshot
        print("Taking screenshot...")
        page.screenshot(path="verification/spectral_loom_verified.png")

        browser.close()
    print("Verification finished.")

if __name__ == "__main__":
    verify_spectral_loom()
