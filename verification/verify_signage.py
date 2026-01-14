
from playwright.sync_api import sync_playwright, expect

def test_signage_lab():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the experiment page
        # Note: In Vite, src/ is the root, so we go to /pages/signage_lab.html
        print("Navigating to page...")
        page.goto("http://localhost:5173/pages/signage_lab.html")

        # Verify title
        expect(page).to_have_title("Multilingual Signage Lab")
        print("Title verified.")

        # Take initial screenshot of default state (Vertical Scroll)
        page.wait_for_timeout(1000) # Wait for animation to settle
        page.screenshot(path="verification/signage_scroll.png")
        print("Scroll screenshot taken.")

        # Switch to Chromatic Overlap tab
        page.get_by_role("button", name="Chromatic Overlap").click()
        page.wait_for_timeout(500)

        # Verify Stack container is visible
        container = page.locator(".stack-mode-container")
        expect(container).to_be_visible()
        print("Stack mode verified.")

        # Take screenshot of Stack state
        page.screenshot(path="verification/signage_stack.png")
        print("Stack screenshot taken.")

        browser.close()

if __name__ == "__main__":
    test_signage_lab()
