from playwright.sync_api import sync_playwright, Page, expect

def verify_pattern_tests(page: Page):
    # Capture console errors
    console_errors = []
    def on_console(msg):
        if msg.type == "error":
            print(f"Console error: {msg.text}")
            console_errors.append(msg.text)
        else:
            print(f"Console log: {msg.text}")

    page.on("console", on_console)

    # Capture network failures
    failed_requests = []
    def on_request_failed(request):
        print(f"Request failed: {request.url} - {request.failure}")
        failed_requests.append(request.url)

    page.on("requestfailed", on_request_failed)

    # Navigate
    url = "http://localhost:5173/pages/pattern_tests.html"
    print(f"Navigating to {url}")
    response = page.goto(url)
    assert response.ok, f"Failed to load page: {response.status}"

    # Check for specific asset loading
    # We want to ensure unlit-button.png is loaded (or attempted and succeeded)
    # Since we can't easily wait for a specific network request after the fact,
    # we rely on 'requestfailed' not catching it.

    # Wait for page content
    page.wait_for_load_state("networkidle")

    # Check for WebGPU error message (expected in headless)
    # OR canvas if supported.
    webgpu_error = page.locator("#webgpu-error")
    if webgpu_error.is_visible():
        print("WebGPU not supported (expected in headless). Checking for clean failure.")
        expect(webgpu_error).to_be_visible()
    else:
        print("WebGPU might be supported or error hidden.")

    # Verify no WGSL errors in console
    for err in console_errors:
        if "WGSL" in err or "struct member" in err or "unresolved value" in err:
            raise Exception(f"Found WGSL error in console: {err}")

    # Verify no 404 for the button image
    for req in failed_requests:
        if "unlit-button.png" in req or "buttons.png" in req:
             raise Exception(f"Asset failed to load: {req}")

    # Take screenshot
    page.screenshot(path="verification/pattern_tests.png")
    print("Verification passed, screenshot saved.")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_pattern_tests(page)
        finally:
            browser.close()
