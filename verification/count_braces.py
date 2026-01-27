import re

with open('src/pages/experiments.html', 'r') as f:
    content = f.read()

match = re.search(r'<script type="module">([\s\S]*?)</script>', content)
if match:
    js = match.group(1)
    open_braces = js.count('{')
    close_braces = js.count('}')
    open_parens = js.count('(')
    close_parens = js.count(')')
    print(f"Braces: {open_braces} open, {close_braces} closed")
    print(f"Parens: {open_parens} open, {close_parens} closed")
else:
    print("No script found")
