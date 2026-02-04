from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 800})

        # 1. Screenshot the standalone page
        print("Navigating to dark-matter-web.html...")
        try:
            page.goto("http://localhost:5173/pages/dark-matter-web.html")
            page.wait_for_timeout(2000) # Wait for animation/webgl init
            page.screenshot(path="verification/dark-matter-web.png")
            print("Captured dark-matter-web.png")
        except Exception as e:
            print(f"Error capturing standalone: {e}")

        # 2. Screenshot the experiments page section
        print("Navigating to experiments.html...")
        try:
            page.goto("http://localhost:5173/pages/experiments.html")
            page.wait_for_load_state("networkidle")

            # Scroll to the new container
            loc = page.locator("#dark-matter-web-container")
            if loc.count() > 0:
                loc.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)
                page.screenshot(path="verification/experiments_page_entry.png")
                print("Captured experiments_page_entry.png")
            else:
                print("Container #dark-matter-web-container not found")
        except Exception as e:
             print(f"Error capturing experiments page: {e}")

        browser.close()

if __name__ == "__main__":
    run()
