#!/usr/bin/env python3
import os
import re
from pathlib import Path

def check_file_content(filepath, min_meaningful_lines=5):
    """Check if a file has meaningful content."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove comments and whitespace to check for real content
        lines = content.split('\n')
        meaningful_lines = 0
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines, comments
            if not stripped:
                continue
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                continue
            if stripped.startswith('#') and filepath.endswith('.py'):
                continue
            if stripped.startswith('<!--'):
                continue
            meaningful_lines += 1
        
        # Check for common placeholder patterns
        is_placeholder = any([
            'TODO' in content and meaningful_lines < 10,
            'FIXME' in content and meaningful_lines < 10,
            'placeholder' in content.lower() and meaningful_lines < 15,
            'coming soon' in content.lower(),
            'under construction' in content.lower(),
        ])
        
        file_size = os.path.getsize(filepath)
        
        return {
            'path': filepath,
            'size': file_size,
            'total_lines': len(lines),
            'meaningful_lines': meaningful_lines,
            'is_placeholder': is_placeholder,
            'is_empty': meaningful_lines == 0,
            'is_minimal': meaningful_lines < min_meaningful_lines
        }
    except Exception as e:
        return {
            'path': filepath,
            'error': str(e)
        }

def check_js_file(filepath):
    """Additional checks for JS files."""
    result = check_file_content(filepath)
    if 'error' in result:
        return result
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for basic JS patterns
        has_functions = 'function' in content or '=>' in content or 'class ' in content
        has_exports = 'export' in content or 'import' in content
        has_dom = 'document' in content or 'window' in content or 'canvas' in content.lower()
        has_logic = any(keyword in content for keyword in ['return', 'if', 'for', 'while', 'const', 'let', 'var'])
        
        result.update({
            'has_functions': has_functions,
            'has_exports': has_exports,
            'has_dom': has_dom,
            'has_logic': has_logic,
            'is_functional': has_functions and has_logic
        })
    except:
        pass
    
    return result

def check_html_file(filepath):
    """Additional checks for HTML files."""
    result = check_file_content(filepath)
    if 'error' in result:
        return result
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for basic HTML structure
        has_html_tag = '<html' in content.lower()
        has_body = '<body' in content.lower()
        has_content = len(re.findall(r'<[^>]+>[^<]+</[^>]+>', content)) > 2  # tags with content
        
        result.update({
            'has_html_tag': has_html_tag,
            'has_body': has_body,
            'has_content': has_content,
            'is_valid': has_html_tag and has_body
        })
    except:
        pass
    
    return result

def main():
    src_dir = Path('src')
    
    print("=" * 80)
    print("COMPONENT & EXPERIMENT AUDIT REPORT")
    print("=" * 80)
    
    # Check JS files
    print("\n📁 JAVASCRIPT FILES (src/js/)")
    print("-" * 80)
    js_files = sorted(src_dir.glob('js/*.js'))
    
    empty_js = []
    minimal_js = []
    placeholder_js = []
    
    for f in js_files:
        result = check_js_file(f)
        if 'error' in result:
            print(f"❌ ERROR: {f.name} - {result['error']}")
            continue
            
        if result['is_empty']:
            empty_js.append(result)
        elif result['is_placeholder']:
            placeholder_js.append(result)
        elif result['is_minimal']:
            minimal_js.append(result)
    
    if empty_js:
        print(f"\n🚫 EMPTY/BLANK JS FILES ({len(empty_js)}):")
        for r in empty_js:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes, {r['meaningful_lines']} meaningful lines)")
    
    if placeholder_js:
        print(f"\n⚠️  PLACEHOLDER JS FILES ({len(placeholder_js)}):")
        for r in placeholder_js:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes, {r['meaningful_lines']} meaningful lines)")
    
    if minimal_js:
        print(f"\n⚡ MINIMAL JS FILES (< 5 meaningful lines) ({len(minimal_js)}):")
        for r in minimal_js:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes, {r['meaningful_lines']} meaningful lines)")
    
    if not (empty_js or placeholder_js or minimal_js):
        print("✅ All JS files appear to have content")
    
    # Check HTML files
    print("\n" + "=" * 80)
    print("📁 HTML PAGES (src/pages/)")
    print("-" * 80)
    html_files = sorted(src_dir.glob('pages/*.html'))
    
    empty_html = []
    minimal_html = []
    invalid_html = []
    
    for f in html_files:
        result = check_html_file(f)
        if 'error' in result:
            print(f"❌ ERROR: {f.name} - {result['error']}")
            continue
            
        if result['is_empty']:
            empty_html.append(result)
        elif not result.get('is_valid', True):
            invalid_html.append(result)
        elif result['is_minimal']:
            minimal_html.append(result)
    
    if empty_html:
        print(f"\n🚫 EMPTY/BLANK HTML FILES ({len(empty_html)}):")
        for r in empty_html:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes)")
    
    if invalid_html:
        print(f"\n⚠️  INVALID HTML FILES (missing basic structure) ({len(invalid_html)}):")
        for r in invalid_html:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes)")
    
    if minimal_html:
        print(f"\n⚡ MINIMAL HTML FILES (< 5 meaningful lines) ({len(minimal_html)}):")
        for r in minimal_html:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes, {r['meaningful_lines']} meaningful lines)")
    
    if not (empty_html or invalid_html or minimal_html):
        print("✅ All HTML files appear to have content")
    
    # Check CSS files
    print("\n" + "=" * 80)
    print("📁 CSS FILES (src/css/)")
    print("-" * 80)
    css_files = sorted(src_dir.glob('css/*.css'))
    
    empty_css = []
    minimal_css = []
    
    for f in css_files:
        result = check_file_content(f)
        if 'error' in result:
            print(f"❌ ERROR: {f.name} - {result['error']}")
            continue
            
        if result['is_empty']:
            empty_css.append(result)
        elif result['is_minimal']:
            minimal_css.append(result)
    
    if empty_css:
        print(f"\n🚫 EMPTY/BLANK CSS FILES ({len(empty_css)}):")
        for r in empty_css:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes)")
    
    if minimal_css:
        print(f"\n⚡ MINIMAL CSS FILES (< 5 meaningful lines) ({len(minimal_css)}):")
        for r in minimal_css:
            print(f"   - {os.path.basename(r['path'])} ({r['size']} bytes, {r['meaningful_lines']} meaningful lines)")
    
    if not (empty_css or minimal_css):
        print("✅ All CSS files appear to have content")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_issues = (len(empty_js) + len(placeholder_js) + len(minimal_js) +
                   len(empty_html) + len(invalid_html) + len(minimal_html) +
                   len(empty_css) + len(minimal_css))
    
    print(f"Total JS files: {len(js_files)}")
    print(f"  - Empty: {len(empty_js)}")
    print(f"  - Placeholder: {len(placeholder_js)}")
    print(f"  - Minimal: {len(minimal_js)}")
    print(f"\nTotal HTML files: {len(html_files)}")
    print(f"  - Empty: {len(empty_html)}")
    print(f"  - Invalid: {len(invalid_html)}")
    print(f"  - Minimal: {len(minimal_html)}")
    print(f"\nTotal CSS files: {len(css_files)}")
    print(f"  - Empty: {len(empty_css)}")
    print(f"  - Minimal: {len(minimal_css)}")
    print(f"\n{'❌' if total_issues > 0 else '✅'} Total issues found: {total_issues}")

if __name__ == '__main__':
    main()
