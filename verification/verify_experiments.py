from playwright.sync_api import sync_playwright

def verify_experiments(page):
    page.goto("http://localhost:5173/pages/experiments.html")
    page.wait_for_selector("text=Bioluminescent Abyss")
    page.screenshot(path="verification/experiments_list.png")
    print("Experiments list screenshot saved.")

    page.goto("http://localhost:5173/pages/bioluminescent-abyss.html")
    # Wait a bit for the canvas to render
    page.wait_for_timeout(2000)
    page.screenshot(path="verification/bioluminescent_abyss.png")
    print("Bioluminescent Abyss screenshot saved.")

    # Collect console logs to check for WebGPU/WebGL errors
    page.on("console", lambda msg: print(f"Console: {msg.text}"))

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()
        try:
            verify_experiments(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
