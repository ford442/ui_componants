from playwright.sync_api import sync_playwright

def verify_seismic_wave():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--enable-unsafe-webgpu"])
        page = browser.new_page()
        page.goto("http://localhost:5173/pages/seismic-wave.html")
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/seismic_wave_screenshot.png")
        print("Screenshot saved to verification/seismic_wave_screenshot.png")
        browser.close()

if __name__ == "__main__":
    verify_seismic_wave()
