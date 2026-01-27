import re
import subprocess

def check_js_in_html(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the module script
    match = re.search(r'<script type="module">([\s\S]*?)</script>', content)
    if not match:
        print("No module script found.")
        return

    js_content = match.group(1)

    # Write to a temp file
    with open('temp_check.js', 'w') as f:
        f.write(js_content)

    # Check syntax
    try:
        subprocess.check_output(['node', '--check', 'temp_check.js'], stderr=subprocess.STDOUT)
        print("Syntax OK")
    except subprocess.CalledProcessError as e:
        print("Syntax Error:")
        print(e.output.decode())

if __name__ == "__main__":
    check_js_in_html('src/pages/experiments.html')
