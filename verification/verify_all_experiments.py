from playwright.sync_api import sync_playwright

def verify_all_experiments():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Capture console messages
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Page Error: {err}"))

        try:
            print("Navigating to experiments.html...")
            response = page.goto("http://localhost:5173/pages/experiments.html")
            print(f"Response status: {response.status}")

            page.wait_for_load_state("networkidle")

            # Check for Temporal Core container
            if page.locator("#temporal-core-container").count() > 0:
                print("SUCCESS: Temporal Core container found.")
            else:
                print("FAILURE: Temporal Core container NOT found.")

            # Check for Hybrid Engine container
            if page.locator("#hybrid-engine-container").count() > 0:
                print("SUCCESS: Hybrid Engine container found.")
            else:
                print("FAILURE: Hybrid Engine container NOT found.")

            # Check for Gravitational Nebula container
            if page.locator("#gravitational-nebula-container").count() > 0:
                print("SUCCESS: Gravitational Nebula container found.")
            else:
                print("FAILURE: Gravitational Nebula container NOT found.")

            # Check for Spectral Loom container
            if page.locator("#spectral-loom-container").count() > 0:
                print("SUCCESS: Spectral Loom container found.")
            else:
                print("FAILURE: Spectral Loom container NOT found.")

            # Check for Chaos Attractor container
            if page.locator("#chaos-attractor-container").count() > 0:
                print("SUCCESS: Chaos Attractor container found.")
            else:
                print("FAILURE: Chaos Attractor container NOT found.")
            # Check for Energy Vortex container
            if page.locator("#energy-vortex-container").count() > 0:
                print("SUCCESS: Energy Vortex container found.")
            else:
                print("FAILURE: Energy Vortex container NOT found.")

            # Check for Subatomic Collider container
            if page.locator("#subatomic-collider-container").count() > 0:
                print("SUCCESS: Subatomic Collider container found.")
            else:
                print("FAILURE: Subatomic Collider container NOT found.")

            # Check for Dark Energy Prism container
            if page.locator("#dark-energy-prism-container").count() > 0:
                print("SUCCESS: Dark Energy Prism container found.")
            else:
                print("FAILURE: Dark Energy Prism container NOT found.")
            # Check for Dyson Swarm container
            if page.locator("#dyson-swarm-container").count() > 0:
                print("SUCCESS: Dyson Swarm container found.")
            else:
                print("FAILURE: Dyson Swarm container NOT found.")

            # Check for Neuro-Morphic Crystal container
            if page.locator("#neuro-morphic-crystal-container").count() > 0:
                print("SUCCESS: Neuro-Morphic Crystal container found.")
            else:
                print("FAILURE: Neuro-Morphic Crystal container NOT found.")

            # Check for Neutrino Storm container
            if page.locator("#neutrino-storm-container").count() > 0:
                print("SUCCESS: Neutrino Storm container found.")
            else:
                print("FAILURE: Neutrino Storm container NOT found.")

            print("Verification finished.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_all_experiments()
