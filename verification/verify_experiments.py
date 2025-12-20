from playwright.sync_api import sync_playwright

def verify_experiments(page):
    print("Navigating to experiments page...")
    page.goto("http://localhost:5173/pages/experiments.html")

    print("Checking title...")
    title = page.title()
    print(f"Page title: {title}")

    print("Waiting for Cyber-Biology section...")
    page.wait_for_selector("#cyber-biology-container")

    # Wait a bit for the canvas to initialize and render
    page.wait_for_timeout(2000)

    print("Taking screenshot...")
    page.screenshot(path="verification/experiments_page.png", full_page=True)

    print("Checking if Cyber-Biology canvas exists inside container...")
    canvas_count = page.evaluate("""() => {
        const container = document.getElementById('cyber-biology-container');
        return container.querySelectorAll('canvas').length;
    }""")
    print(f"Canvas count in cyber-biology-container: {canvas_count}")

    if canvas_count < 1:
        raise Exception("No canvas found in Cyber-Biology container!")

    print("Verification complete.")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_experiments(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
