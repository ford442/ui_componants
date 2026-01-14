
import sys
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"Page Error: {exc}"))

        url = "http://localhost:5173/pages/photon-containment.html"
        print(f"Navigating to {url}...")
        try:
            page.goto(url, wait_until="load", timeout=10000)

            # Wait for canvas elements
            page.wait_for_selector("canvas", state="attached", timeout=5000)
            print("Canvases found.")

            # Wait a bit for initialization
            page.wait_for_timeout(2000)

            print("Page loaded successfully without crash.")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    run()
