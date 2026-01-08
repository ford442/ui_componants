from playwright.sync_api import sync_playwright

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--enable-unsafe-webgpu"])
        page = browser.new_page()
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        try:
            page.goto("http://localhost:5173/pages/experiments.html")
            page.wait_for_timeout(2000)
            page.screenshot(path="verification/experiments_dashboard_check.png")
            print("Dashboard loaded successfully.")
        except Exception as e:
            print(f"FAILED: experiments.html exception: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_dashboard()
