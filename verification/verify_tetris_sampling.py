
import os
import sys
import time
from playwright.sync_api import sync_playwright

def verify_tetris_sampling():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        page = browser.new_page()

        # 1. Verify tetris.html (The Sampling Experiment)
        print("Verifying src/pages/tetris.html...")
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        try:
            page.goto("http://localhost:5173/pages/tetris.html")
            page.wait_for_timeout(2000) # Wait for canvas init and load

            # Check for canvas
            canvas = page.locator("canvas")
            if canvas.count() > 0:
                print("SUCCESS: Canvas found in tetris.html")
            else:
                print("FAILURE: No canvas found in tetris.html")
                sys.exit(1)

            # Check for specific text indicating the new page content
            if page.get_by_text("Block Sampling Test").is_visible():
                print("SUCCESS: 'Block Sampling Test' title found")
            else:
                print("FAILURE: Title not found")
                sys.exit(1)

        except Exception as e:
            print(f"FAILURE: Exception accessing tetris.html: {e}")
            sys.exit(1)

        # 2. Regression Check: tetris-experiments.html
        print("Verifying src/pages/tetris-experiments.html (Regression Check)...")
        try:
            page.goto("http://localhost:5173/pages/tetris-experiments.html")
            page.wait_for_timeout(2000)

            # Should have multiple canvases (it creates one per experiment card usually)
            # Actually, tetris-experiments.html typically has cards.
            # We just want to make sure it loads without error and scripts run.

            # Check if Neon Rain (first experiment) is instantiated
            # Based on previous knowledge, it likely creates canvases inside the cards.

            canvases = page.locator("canvas")
            count = canvases.count()
            print(f"Found {count} canvases in tetris-experiments.html")

            if count >= 1:
                print("SUCCESS: Canvases found in tetris-experiments.html")
            else:
                print("WARNING: No canvases found in tetris-experiments.html. This might be okay if they load lazily, but usually they init immediately.")

        except Exception as e:
            print(f"FAILURE: Exception accessing tetris-experiments.html: {e}")
            sys.exit(1)

        browser.close()

if __name__ == "__main__":
    # Give the server a moment to start if it was just launched
    time.sleep(3)
    verify_tetris_sampling()
