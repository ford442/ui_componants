from playwright.sync_api import sync_playwright

def verify_experiments():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to experiments page
        # Note: root is src, so experiments.html is directly at /pages/experiments.html (since src is root)
        # Wait, if root: 'src', then localhost:5173/pages/experiments.html should work
        page.goto("http://localhost:5173/pages/experiments.html")

        # Wait for experiments to load
        page.wait_for_selector("#neural-network-container")
        page.wait_for_selector("#fluid-sim-container")

        # Scroll to neural network
        neural_net = page.locator("#neural-network-container")
        neural_net.scroll_into_view_if_needed()
        page.wait_for_timeout(1000) # Wait for animation/render
        page.screenshot(path="verification/neural_net.png")

        # Scroll to fluid sim
        fluid_sim = page.locator("#fluid-sim-container")
        fluid_sim.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        page.screenshot(path="verification/fluid_sim.png")

        browser.close()

if __name__ == "__main__":
    verify_experiments()
