from playwright.sync_api import sync_playwright

def verify_subatomic_collider():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Check console logs for errors - Setup BEFORE navigation
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        # Navigate to the experiments page
        url = "http://localhost:5173/pages/experiments.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for content to load
        page.wait_for_selector("text=Subatomic Collider")

        # Scroll to the container
        element = page.locator("#subatomic-collider-container")
        element.scroll_into_view_if_needed()
        print("Scrolled to Subatomic Collider container.")

        # Give it a moment for the canvas to render
        page.wait_for_timeout(2000)

        # Check if canvas exists inside
        canvas_count = element.locator("canvas").count()
        print(f"Found {canvas_count} canvas elements in container.")

        if canvas_count < 1:
            print("ERROR: No canvas found in Subatomic Collider container.")
            exit(1)

        # Take screenshot
        page.screenshot(path="verification/subatomic_collider_before.png")
        print("Screenshot saved to verification/subatomic_collider_before.png")

        # Get element bounding box
        box = element.bounding_box()
        if box:
            # Simulate click in the center
            cx = box['x'] + box['width'] / 2
            cy = box['y'] + box['height'] / 2
            print(f"Clicking at {cx}, {cy}")
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.wait_for_timeout(500) # Hold for 500ms

            # Take screenshot during hold
            page.screenshot(path="verification/subatomic_collider_click.png")
            print("Screenshot saved to verification/subatomic_collider_click.png")

            page.mouse.up()
            print("Click simulation complete.")

        browser.close()

if __name__ == "__main__":
    verify_subatomic_collider()
