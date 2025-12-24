from playwright.sync_api import sync_playwright

def verify_experiments():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Visit Experiments Page
        print("Visiting experiments.html...")
        page.goto("http://localhost:5173/pages/experiments.html")
        page.wait_for_load_state("networkidle")

        # Scroll to Crystal Cavern section
        crystal_section = page.locator("text=Crystal Cavern").first
        if crystal_section.is_visible():
            print("Crystal Cavern section found.")
            crystal_section.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            page.screenshot(path="verification/experiments_page_scrolled.png")
            print("Captured experiments_page_scrolled.png")
        else:
            print("Crystal Cavern section NOT found.")

        browser.close()

if __name__ == "__main__":
    verify_experiments()
