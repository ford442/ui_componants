from playwright.sync_api import sync_playwright

def verify_experiments(page):
    # Go to the experiments page
    # Note: Using http://localhost:5173/pages/experiments.html as per memory instruction about omitting src/
    page.goto("http://localhost:5173/pages/experiments.html")

    # Wait for the Force Field container to be visible
    page.wait_for_selector("#force-field-container")

    # Scroll to the bottom to see the new experiment
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

    # Wait a bit for the canvas to initialize and render
    page.wait_for_timeout(2000)

    # Take a screenshot of the Force Field section
    # We can select the section containing the force field
    # It is the last .experiment-section
    elements = page.query_selector_all(".experiment-section")
    if elements:
        last_element = elements[-1]
        last_element.scroll_into_view_if_needed()
        page.screenshot(path="verification/force_field.png")
        print("Screenshot taken: verification/force_field.png")
    else:
        print("Error: Could not find experiment section")

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
