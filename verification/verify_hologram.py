from playwright.sync_api import sync_playwright

def verify_experiments(page):
    # Go to main page
    print("Navigating to index.html...")
    page.goto("http://localhost:5173/index.html")
    page.wait_for_load_state("networkidle")

    # Check for the new Hologram link
    print("Checking for Hologram link...")
    hologram_link = page.locator('a[href="pages/hologram.html"]')
    if hologram_link.count() > 0:
        print("Hologram link found.")
    else:
        print("Error: Hologram link not found.")

    page.screenshot(path="verification/index_page.png")

    # Navigate to Hologram page
    print("Navigating to Hologram page...")
    hologram_link.click()
    page.wait_for_load_state("networkidle")

    # Take screenshot of hologram page
    print("Taking screenshot of hologram page...")
    page.screenshot(path="verification/hologram_page.png")

    # Check for canvas elements
    print("Checking for canvas elements...")
    canvases = page.locator('canvas')
    count = canvases.count()
    print(f"Found {count} canvas elements.")

    # Verify existing experiment still loads (e.g., buttons)
    print("Navigating to Buttons page...")
    page.goto("http://localhost:5173/pages/buttons.html")
    page.wait_for_load_state("networkidle")
    print("Taking screenshot of buttons page...")
    page.screenshot(path="verification/buttons_page.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_experiments(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
