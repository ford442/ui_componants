from playwright.sync_api import sync_playwright
import time

def verify_quantum_tunneling():
    print("Starting verification for Quantum Tunneling...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Capture console messages
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Page Error: {err}"))

        try:
            # Navigate to the experiments page
            url = "http://localhost:5173/pages/experiments.html"
            print(f"Navigating to {url}...")
            page.goto(url)

            # Wait for the container to be present
            print("Waiting for #quantum-tunneling-container...")
            locator = page.locator("#quantum-tunneling-container")
            locator.wait_for(state="attached", timeout=10000)

            # Scroll into view
            print("Scrolling into view...")
            locator.scroll_into_view_if_needed()

            # Wait for initialization (give it some time for WebGPU/WebGL to start)
            print("Waiting for initialization...")
            time.sleep(3)

            # Check if there are canvas elements inside
            canvas_count = locator.locator("canvas").count()
            print(f"Found {canvas_count} canvas elements in container.")
            if canvas_count < 1:
                raise Exception("No canvas found in Quantum Tunneling container!")

            # Simulate mouse interaction
            print("Simulating mouse interaction...")
            box = locator.bounding_box()
            if box:
                # Move mouse across the container to trigger updates
                page.mouse.move(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.5)
                time.sleep(0.5)
                page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.2)
                time.sleep(0.5)

            # Take screenshot
            output_path = "verification/quantum_tunneling_upgrade.png"
            print(f"Taking screenshot to {output_path}...")
            page.screenshot(path=output_path)

            print("Verification Successful!")

        except Exception as e:
            print(f"Verification Failed: {e}")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_quantum_tunneling()
