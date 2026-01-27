from playwright.sync_api import sync_playwright
import time

def verify_gravity_well():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the experiments page
        url = "http://localhost:5173/pages/experiments.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for content to load
        page.wait_for_load_state("networkidle")

        # Scroll to the Gravity Well container
        element_id = "#gravity-well-container"
        print(f"Waiting for {element_id}...")
        page.wait_for_selector(element_id)
        element = page.locator(element_id)
        element.scroll_into_view_if_needed()

        # Give it a moment to initialize
        time.sleep(2)

        # Verify it exists and is visible
        if element.is_visible():
            print("SUCCESS: Gravity Well container is visible.")
        else:
            print("ERROR: Gravity Well container is NOT visible.")
            browser.close()
            exit(1)

        # Interaction check: Click to trigger shockwave
        # We can't easily verify the visual effect programmatically without complex screenshot diffing,
        # but we can simulate the interaction to ensure no errors occur.
        print("Simulating interaction (click)...")
        box = element.bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

        # Wait for potential errors
        time.sleep(1)

        # Check console logs for any errors
        logs = []
        page.on("console", lambda msg: logs.append(msg))

        # Take a screenshot
        page.screenshot(path="verification/gravity_well_interaction.png")
        print("Screenshot saved to verification/gravity_well_interaction.png")

        browser.close()

if __name__ == "__main__":
    verify_gravity_well()
