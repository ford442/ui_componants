from playwright.sync_api import sync_playwright, expect
import time

def verify_experiments(page):
    # Go to the experiments list
    page.goto("http://localhost:5173/pages/experiments.html")
    time.sleep(1) # Wait for load

    # Check if the new experiment is listed
    new_exp_link = page.get_by_role("link", name="View Full Page Experiment").nth(-1) # Assuming it's the last one added

    # Actually, let's verify by text content or specific ID
    # The new section has h2 "Gravitational Nebula"
    header = page.get_by_role("heading", name="Gravitational Nebula")
    expect(header).to_be_visible()

    # Take a screenshot of the experiments list
    page.screenshot(path="verification/experiments_list.png")

    # Navigate to the new experiment page
    # Find the link inside the gravitational nebula section
    # Use a more specific locator strategy
    container = page.locator(".component-card", has_text="Gravitational Nebula")
    link = container.get_by_role("link", name="View Full Page Experiment")
    link.click()

    # Wait for the new page to load
    time.sleep(2)

    # Check if canvas exists
    canvas_container = page.locator("#canvas-container")
    expect(canvas_container).to_be_visible()

    # Take a screenshot of the new experiment
    page.screenshot(path="verification/gravitational_nebula.png")

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
