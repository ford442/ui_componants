import os
import time
from playwright.sync_api import sync_playwright

def verify_experiments():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Start a local server in the background (we assume it's running or we can start one)
        # But wait, usually I should rely on the user to run the server or I should run it.
        # Since I am in the sandbox, I should assume `npm run dev` needs to be running.
        # However, `run_in_bash_session` shares the session.
        # Actually, I'll just build it and serve the dist or just serve src.
        # Let's try to access the files via a simple python http server to avoid vite complexity if possible,
        # but vite config is set up for 'src' root.

        # It's better to rely on `npm run preview` after build or `npm run dev`.
        # I'll assume the user wants me to verify using `npm run dev` running in background.
        # But I need to start it.

        # Actually, let's just check the files exist and contain expected content for now,
        # and maybe try to load them if I can start a server.

        # Since I can run long running processes, I will start `npm run dev` in background
        # and then test against localhost.

        base_url = "http://localhost:5173"

        experiments = [
            "/pages/experiments.html",
            "/pages/cyber-rain.html",
            "/pages/gravitational-nebula.html",
            "/pages/crystal-cavern.html"
        ]

        for exp in experiments:
            url = f"{base_url}{exp}"
            print(f"Checking {url}...")
            try:
                page.goto(url)
                # Wait a bit for JS to init
                time.sleep(2)

                # Check for console errors?
                # We can attach a listener, but page.goto might have already triggered them.
                # Ideally we listen before goto.

                # Take screenshot
                page.screenshot(path=f"verification/{exp.split('/')[-1]}.png")
                print(f"  - Screenshot saved: verification/{exp.split('/')[-1]}.png")

                # Check title
                title = page.title()
                print(f"  - Title: {title}")

            except Exception as e:
                print(f"  - Error loading {exp}: {e}")

        browser.close()

if __name__ == "__main__":
    verify_experiments()
