
import os
from playwright.sync_api import sync_playwright, expect

def verify_void_singularity():
    with sync_playwright() as p:
        # Launch with WebGPU enabled flags
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu', '--enable-features=Vulkan']
        )
        page = browser.new_page()

        # Listen for console logs
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        url = "http://localhost:5173/pages/void-singularity.html"
        print(f"Navigating to {url}...")

        try:
            page.goto(url, wait_until="networkidle")

            # Wait for canvas to exist
            page.wait_for_selector("canvas", timeout=5000)
            print("SUCCESS: Canvas element found.")

            # Wait a bit for initialization and rendering
            page.wait_for_timeout(3000)

            # Check for error message
            if page.locator(".webgpu-error").is_visible():
                print("NOTE: WebGPU fallback message visible (Expected in headless).")

            # Take screenshot
            output_path = "verification/void_singularity_verify.png"
            page.screenshot(path=output_path)
            print(f"Screenshot saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_void_singularity()
