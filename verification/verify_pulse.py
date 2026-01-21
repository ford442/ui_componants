
from playwright.sync_api import sync_playwright

def verify_pulse():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-unsafe-swiftshader"])
        page = browser.new_page()
        page.set_viewport_size({"width": 800, "height": 600})

        url = "http://localhost:5173/pages/bioluminescent-abyss.html"
        print(f"Navigating to {url}...")
        page.goto(url)
        page.wait_for_timeout(2000) # Wait for initialization

        # Click center
        print("Clicking center...")
        page.mouse.click(400, 300)

        # Wait for pulse to expand
        page.wait_for_timeout(300)

        print("Taking screenshot...")
        page.screenshot(path="verification/pulse_verification.png")
        browser.close()

if __name__ == "__main__":
    verify_pulse()
