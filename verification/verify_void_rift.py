from playwright.sync_api import sync_playwright
import time

def test_void_rift():
    with sync_playwright() as p:
        # Launch browser with WebGPU enabled (unsafe-webgpu required for headless)
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        # 1. Test standalone page
        print("Navigating to Void Rift standalone page...")
        page.goto("http://localhost:5173/pages/void-rift.html")

        # Wait for initialization (WebGPU init can take a moment)
        time.sleep(3)

        # Check for canvas elements
        canvases = page.locator('canvas')
        count = canvases.count()
        print(f"Found {count} canvases on standalone page")

        if count >= 1:
            page.screenshot(path="verification/void_rift_standalone.png")
            print("Screenshot saved: verification/void_rift_standalone.png")
        else:
            print("ERROR: No canvases found on standalone page")

        # 2. Test Experiments Dashboard card
        print("Navigating to Experiments Dashboard...")
        page.goto("http://localhost:5173/pages/experiments.html")
        time.sleep(2)

        # Scroll to the new card
        # The card id is 'void-rift-container'
        card = page.locator('#void-rift-container')
        if card.count() > 0:
            card.scroll_into_view_if_needed()
            time.sleep(2) # Wait for it to render
            page.screenshot(path="verification/void_rift_card.png")
            print("Screenshot saved: verification/void_rift_card.png")
        else:
            print("ERROR: Void Rift card not found on experiments page")

        browser.close()

if __name__ == "__main__":
    test_void_rift()
