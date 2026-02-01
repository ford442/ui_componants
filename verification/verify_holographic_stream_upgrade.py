from playwright.sync_api import sync_playwright
import time

def verify_holographic_stream():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Navigating to Holographic Stream...")
        # Note: In Vite, root is src, so URL is http://localhost:5173/pages/holographic-stream.html
        url = "http://localhost:5173/pages/holographic-stream.html"
        try:
            page.goto(url)
        except Exception as e:
            print(f"Error navigating to {url}: {e}")
            exit(1)

        # Wait for canvas
        try:
            page.wait_for_selector("canvas", timeout=5000)
            print("Canvas found.")
        except:
            print("Error: Canvas not found.")
            exit(1)

        # Monitor console
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        # Wait for initialization
        time.sleep(2)

        # Simulate mouse movement
        print("Simulating mouse interaction...")
        page.mouse.move(100, 100)
        time.sleep(0.5)
        page.mouse.move(200, 200)
        time.sleep(0.5)
        page.mouse.move(400, 400)
        time.sleep(1)

        # Take screenshot
        print("Taking screenshot...")
        page.screenshot(path="verification/holographic_stream_upgrade.png")

        if errors:
            print("Verification FAILED: Console errors detected.")
            for e in errors:
                print(f"  - {e}")
            exit(1)

        print("Verification PASSED.")
        browser.close()

if __name__ == "__main__":
    verify_holographic_stream()
