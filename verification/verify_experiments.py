from playwright.sync_api import sync_playwright

def verify_experiments(page):
    print("Navigating to experiments page...")
    page.goto("http://localhost:5173/pages/experiments.html")

    print("Checking if Cyber Crystal card exists...")
    card = page.wait_for_selector('h2:has-text("Cyber Crystal")')
    if card:
        print("Cyber Crystal card found!")
    else:
        print("Cyber Crystal card NOT found!")

    print("Taking screenshot of experiments page...")
    page.screenshot(path="verification/experiments_page.png")

    print("Navigating to Cyber Crystal experiment...")
    page.goto("http://localhost:5173/pages/cyber-crystal.html")

    # Wait a bit for initialization
    page.wait_for_timeout(2000)

    # Check for WebGL canvas
    print("Checking for canvas...")
    canvas = page.query_selector('canvas')
    if canvas:
        print("Canvas found!")
    else:
        print("Canvas NOT found!")

    print("Taking screenshot of Cyber Crystal page...")
    page.screenshot(path="verification/cyber_crystal.png")

    # Check console for errors
    print("Console messages:")
    # (Note: console capturing is set up in the main block)

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Capture console logs
        page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

        try:
            verify_experiments(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
