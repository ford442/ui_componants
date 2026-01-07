import os
import time
from playwright.sync_api import sync_playwright

experiments = [
    "bioluminescent-abyss.html",
    "biomechanical-growth.html",
    "buttons.html",
    "cherenkov-radiation.html",
    "composite_blending.html",
    "cosmic-string.html",
    "crystal-cavern.html",
    "cyber-crystal.html",
    "cyber-rain.html",
    "gravitational-lensing.html",
    "gravitational-nebula.html",
    "hologram.html",
    "hybrid-magnetic-field.html",
    "indicators.html",
    "knobs.html",
    "neon_city.html",
    "pattern_tests.html",
    "plasma-confinement.html",
    "plasma-storm.html",
    "portal_vortex.html",
    "quantum-data-stream.html",
    "singularity-reactor.html",
    "stellar-forge.html",
    "surfaces.html",
    "switches.html",
    "temporal-crystal.html",
    "void-singularity.html"
]

def verify_experiments():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        context = browser.new_context()
        page = context.new_page()

        # Capture console errors
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        passed = 0
        failed = 0
        failed_experiments = []

        print(f"Starting verification of {len(experiments)} experiments...")

        for exp in experiments:
            url = f"http://localhost:5173/pages/{exp}"
            print(f"Checking {exp}...")
            console_errors.clear()

            try:
                page.goto(url)
                # Wait for a bit to let scripts run and potential errors to appear
                page.wait_for_timeout(1000)

                if console_errors:
                    print(f"  FAILED: Console errors found in {exp}:")
                    for err in console_errors:
                        print(f"    - {err}")
                    failed += 1
                    failed_experiments.append(exp)
                else:
                    print(f"  PASSED")
                    passed += 1

            except Exception as e:
                print(f"  FAILED: Exception checking {exp}: {e}")
                failed += 1
                failed_experiments.append(exp)

        print("-" * 30)
        print(f"Verification Complete.")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print(f"Failed experiments: {failed_experiments}")
            exit(1)
        else:
            exit(0)

if __name__ == "__main__":
    verify_experiments()
