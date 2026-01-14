import os
from playwright.sync_api import sync_playwright

def verify_all_pages():
    pages_dir = "src/pages"
    files = [f for f in os.listdir(pages_dir) if f.endswith(".html")]

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--enable-unsafe-webgpu']
        )

        failed_pages = []

        for filename in files:
            page_url = f"http://localhost:5173/pages/{filename}"
            print(f"Verifying {filename}...")

            page = browser.new_page()
            errors = []
            page.on("console", lambda msg: print(f"[{filename}] Console: {msg.text}"))
            page.on("pageerror", lambda exc: errors.append(str(exc)))

            try:
                page.goto(page_url)
                page.wait_for_timeout(2000) # Wait for init

                if errors:
                    print(f"FAILED: {filename} has errors:")
                    for e in errors:
                        print(f"  - {e}")
                    failed_pages.append(filename)
                else:
                    print(f"PASSED: {filename}")

            except Exception as e:
                print(f"FAILED: {filename} exception: {e}")
                failed_pages.append(filename)
            finally:
                page.close()

        print("-" * 20)
        if failed_pages:
            print(f"Found {len(failed_pages)} failing pages:")
            for fp in failed_pages:
                print(f" - {fp}")
            exit(1)
        else:
            print("All pages passed.")

if __name__ == "__main__":
    verify_all_pages()
