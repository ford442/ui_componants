from playwright.sync_api import sync_playwright

def verify_fixes():
    with sync_playwright() as p:
        # Launch with WebGPU enabled
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )
        context = browser.new_context()
        page = context.new_page()

        # --- 1. Verify Composite Blending (WGSL Error) ---
        print("Visiting Composite Blending page...")
        # Note: In Vite with src root, pages are usually at /pages/filename.html or just /filename.html depending on config
        # User memory says: "browser URLs must omit the src/ prefix (e.g., access http://localhost:5173/pages/file.html for src/pages/file.html)"

        try:
            page.goto("http://localhost:5173/pages/composite_blending.html")
            page.wait_for_timeout(2000) # Wait for shader compilation and render

            # Capture logs to check for errors
            logs = []
            page.on("console", lambda msg: logs.append(msg.text))

            page.screenshot(path="verification/composite_blending.png")
            print("Screenshot taken: verification/composite_blending.png")

            # Check for specific WGSL error in logs (it shouldn't be there now)
            error_found = any("error while parsing WGSL" in log for log in logs)
            if error_found:
                print("FAILURE: WGSL error still present in logs.")
                for log in logs:
                    if "error while parsing WGSL" in log:
                        print(log)
            else:
                print("SUCCESS: No WGSL parsing errors found.")

        except Exception as e:
            print(f"Error checking Composite Blending: {e}")

        # --- 2. Verify Switches (UIComponents Error) ---
        print("\nVisiting Switches page...")
        try:
            page.goto("http://localhost:5173/pages/switches.html")
            page.wait_for_timeout(2000) # Wait for modules to load and init

            # Capture logs
            logs_switches = []
            page.on("console", lambda msg: logs_switches.append(msg.text))

            page.screenshot(path="verification/switches.png")
            print("Screenshot taken: verification/switches.png")

            # Check for ReferenceError
            ref_error = any("UIComponents is not defined" in log for log in logs_switches)
            if ref_error:
                print("FAILURE: UIComponents ReferenceError found.")
            else:
                print("SUCCESS: UIComponents ReferenceError NOT found.")

            # Verify switches exist
            switches_count = page.locator(".toggle-switch").count()
            print(f"Found {switches_count} toggle switches.")
            if switches_count > 0:
                print("SUCCESS: Switches rendered.")
            else:
                print("FAILURE: No switches found.")

        except Exception as e:
            print(f"Error checking Switches: {e}")

        browser.close()

if __name__ == "__main__":
    verify_fixes()
