from playwright.sync_api import sync_playwright

def verify_experiments(page):
    print("Navigating to experiments page...")
    # Using the local dev server port
    page.goto("http://localhost:5173/pages/experiments.html")
    page.wait_for_load_state("networkidle")

    print("Checking if Singularity Reactor card exists...")
    # Check if the new experiment card exists
    reactor_card = page.locator("h2", has_text="Singularity Reactor")
    if reactor_card.is_visible():
        print("Singularity Reactor card found.")
    else:
        print("Singularity Reactor card NOT found.")

    print("Taking screenshot of experiments list...")
    page.screenshot(path="verification/experiments_list.png", full_page=True)

    print("Navigating to Singularity Reactor page...")
    page.goto("http://localhost:5173/pages/singularity-reactor.html")

    # Wait for canvas to be present
    page.wait_for_selector("canvas")

    # Wait a bit for animation to start and shaders to compile
    page.wait_for_timeout(2000)

    # Check for console errors
    page.on("console", lambda msg: print(f"Console: {msg.text}"))

    print("Taking screenshot of Singularity Reactor...")
    page.screenshot(path="verification/singularity_reactor.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        # Enable unsafe webgpu for headless chrome if possible, though software rasterizer might be used
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()
        try:
            verify_experiments(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
