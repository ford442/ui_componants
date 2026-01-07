
from playwright.sync_api import sync_playwright
import time
import sys

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--enable-unsafe-webgpu'])
        page = browser.new_page()

        console_logs = []
        page.on("console", lambda msg: console_logs.append(msg.text))
        page.on("pageerror", lambda exc: console_logs.append(f"PageError: {exc}"))

        print("Navigating to buttons.html...")
        try:
            page.goto("http://localhost:5173/pages/buttons.html")
            page.wait_for_load_state("networkidle")
            time.sleep(2) # Wait for JS to execute

            # Check if UIComponents is defined
            ui_components_defined = page.evaluate("typeof window.UIComponents !== 'undefined'")
            print(f"UIComponents defined: {ui_components_defined}")

            if not ui_components_defined:
                print("FAILURE: UIComponents is not defined.")
            else:
                print("SUCCESS: UIComponents is defined.")

            # Check for specific errors in logs
            syntax_errors = [log for log in console_logs if "SyntaxError" in log]
            reference_errors = [log for log in console_logs if "ReferenceError" in log]

            if syntax_errors:
                print("FAILURE: SyntaxErrors found:")
                for err in syntax_errors:
                    print(f"  - {err}")

            if reference_errors:
                print("FAILURE: ReferenceErrors found:")
                for err in reference_errors:
                    print(f"  - {err}")

            if not syntax_errors and not reference_errors and ui_components_defined:
                print("VERIFICATION PASSED: No syntax/reference errors and UIComponents is available.")
            else:
                print("VERIFICATION FAILED.")

        except Exception as e:
            print(f"Error checking buttons.html: {e}")

        browser.close()

if __name__ == "__main__":
    run()
