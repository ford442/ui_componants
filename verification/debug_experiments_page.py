from playwright.sync_api import sync_playwright
import sys

def test_experiments_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))

        print("Navigating to experiments.html...")
        page.goto("http://localhost:5173/pages/experiments.html")
        page.wait_for_timeout(5000)

        if errors:
            print("Errors found on experiments.html:")
            for e in errors:
                print(f" - {e}")
            sys.exit(1)
        else:
            print("No errors found on experiments.html")
            sys.exit(0)

if __name__ == "__main__":
    test_experiments_page()
