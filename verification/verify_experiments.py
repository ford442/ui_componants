from playwright.sync_api import sync_playwright

def verify_experiments(page):
    page.goto("http://localhost:5173/pages/experiments.html")
    page.wait_for_selector("text=Neural Data Core")
    page.screenshot(path="verification/experiments_page.png")
    print("Experiments page verified.")

    page.goto("http://localhost:5173/pages/neural-data-core.html")
    # Wait for canvas to be present
    page.wait_for_selector("canvas")
    # Give it a moment to render
    page.wait_for_timeout(2000)
    page.screenshot(path="verification/neural_data_core.png")
    print("Neural Data Core page verified.")

if __name__ == "__main__":
    with sync_playwright() as p:
        # Enable unsafe-webgpu for potential WebGPU support in headless (though software rasterizer usually kicks in for WebGL)
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()
        try:
            verify_experiments(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
