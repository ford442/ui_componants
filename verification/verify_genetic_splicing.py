from playwright.sync_api import sync_playwright

def verify_experiment():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        url = "http://localhost:5173/pages/genetic-splicing.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for the container to be visible
        page.wait_for_selector("#genetic-splicing-container", state="visible")

        # Wait a bit for the WebGL/WebGPU to initialize and render
        page.wait_for_timeout(2000)

        # Take a screenshot
        screenshot_path = "verification/genetic_splicing.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_experiment()
