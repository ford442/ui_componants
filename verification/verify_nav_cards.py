from playwright.sync_api import sync_playwright, expect
import time

def verify_nav_cards():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Wait for server to be ready
        time.sleep(3)

        try:
            page.goto("http://localhost:5173/index.html")

            # Wait for content to load
            expect(page.locator("h1")).to_contain_text("UI Components Library")

            # Check for new cards
            experiments_card = page.locator("a[href='pages/experiments.html']")
            tetris_card = page.locator("a[href='pages/tetris-experiments.html']")
            fireworks_card = page.locator("a[href='pages/firework-experiments.html']")

            expect(experiments_card).to_be_visible()
            expect(tetris_card).to_be_visible()
            expect(fireworks_card).to_be_visible()

            # Verify text content
            expect(experiments_card.locator("h2")).to_have_text("Experiments")
            expect(tetris_card.locator("h2")).to_have_text("Tetris Labs")
            expect(fireworks_card.locator("h2")).to_have_text("Fireworks")

            # Check order by getting all nav cards
            cards = page.locator(".nav-card").all()
            hrefs = [card.get_attribute("href") for card in cards]

            print("Found cards:", hrefs)

            # Expected sequence segment
            expected_sequence = [
                "pages/indicators.html",
                "pages/experiments.html",
                "pages/tetris-experiments.html",
                "pages/firework-experiments.html",
                "pages/composite_blending.html"
            ]

            # Find the index of indicators.html
            try:
                start_idx = hrefs.index("pages/indicators.html")
                # Check if the next 4 match expected
                actual_segment = hrefs[start_idx:start_idx+5]
                assert actual_segment == expected_sequence, f"Order mismatch! Expected {expected_sequence}, got {actual_segment}"
                print("Card order verified successfully.")
            except ValueError:
                print("Could not find indicators.html to verify order.")
                exit(1)

            # Take screenshot of the relevant section
            # We can scroll to the experiments card to ensure it's in view
            experiments_card.scroll_into_view_if_needed()
            page.screenshot(path="verification/nav_cards.png")

        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="verification/error.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_nav_cards()
