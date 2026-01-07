from playwright.sync_api import sync_playwright

def verify_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        # Listen for console logs
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"PageError: {exc}"))

        print("Navigating to experiments dashboard...")
        try:
            page.goto("http://localhost:5173/pages/experiments.html")

            # Wait for content
            page.wait_for_selector(".component-card")

            # Wait a bit for iframes to load
            print("Waiting for iframes to load...")
            page.wait_for_timeout(5000)

            # Scroll down to trigger lazy loads if any (though iframes usually load)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

            # Screenshot
            print("Taking dashboard screenshot...")
            page.screenshot(path="verification/experiments_dashboard.png", full_page=True)
            print("Screenshot saved.")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            browser.close()

if __name__ == "__main__":
    verify_all()
