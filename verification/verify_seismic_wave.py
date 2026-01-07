
import sys
from playwright.sync_api import sync_playwright

def verify_experiment_load(page_url):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        print(f"Navigating to {page_url}...")
        try:
            page.goto(page_url)
            page.wait_for_timeout(2000) # Wait for init
        except Exception as e:
            print(f"Failed to load {page_url}: {e}")
            return False

        if console_errors:
            print(f"Console errors found in {page_url}:")
            for err in console_errors:
                print(f" - {err}")

        # Check if canvas exists
        # Wait for canvas to appear
        try:
            page.wait_for_selector("canvas", timeout=5000)
        except Exception as e:
             print(f"No canvas found in {page_url}")
             # Dump page content for debugging
             print("Page content:")
             print(page.content())
             return False

        print(f"✅ Verified {page_url}")
        browser.close()
        return True

if __name__ == "__main__":
    url = "http://localhost:5173/pages/seismic-wave.html"
    if not verify_experiment_load(url):
        sys.exit(1)
