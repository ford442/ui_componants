from playwright.sync_api import sync_playwright

def verify_experiments(page):
    # Navigate to the experiments page
    page.goto("http://localhost:5173/pages/experiments.html")

    # Check if the new experiment card exists
    page.wait_for_selector("text=Portal Vortex")

    # Take a screenshot of the experiments page with the new card
    page.screenshot(path="verification/experiments_page.png")

    # Click the "Enter Portal" button
    page.click("text=Enter Portal")

    # Wait for the portal page to load
    page.wait_for_selector("text=PORTAL VORTEX")

    # Wait a bit for the canvas to initialize (though webgpu/webgl might fail in headless, we verify structure)
    page.wait_for_timeout(2000)

    # Take a screenshot of the portal experiment
    page.screenshot(path="verification/portal_vortex.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_experiments(page)
            print("Verification successful")
        except Exception as e:
            print(f"Verification failed: {e}")
        finally:
            browser.close()
