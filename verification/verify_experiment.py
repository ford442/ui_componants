from playwright.sync_api import sync_playwright

def verify_experiments_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the experiments page
        # Note: In Vite, root is src, so URL is http://localhost:5173/pages/experiments.html
        page.goto("http://localhost:5173/pages/experiments.html")

        # Wait for content to load
        page.wait_for_selector("text=Biomechanical Growth")

        # Take a screenshot of the new experiment card
        # We need to scroll to it first. It should be near the bottom or middle.
        # But wait, I added it near the middle/bottom of the HTML.

        element = page.locator("#biomechanical-growth-container")
        element.scroll_into_view_if_needed()

        # Give it a moment for the canvas to render (WebGL/WebGPU)
        page.wait_for_timeout(2000)

        # Take screenshot of the card
        page.screenshot(path="verification/experiments_page_new_card.png")

        # Also verify the standalone page
        page.goto("http://localhost:5173/pages/biomechanical-growth.html")
        page.wait_for_selector("#canvas-container")
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/biomechanical_growth_page.png")

        # Check console logs for errors
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        browser.close()

if __name__ == "__main__":
    verify_experiments_page()
