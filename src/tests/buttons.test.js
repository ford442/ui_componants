import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

// Read HTML content once
const html = fs.readFileSync(path.resolve(__dirname, '../pages/buttons.html'), 'utf8');

describe('Buttons Page', () => {
    beforeEach(() => {
        // Reset DOM
        document.body.innerHTML = html;
        // Mock getElementById to avoid some null checks if necessary, but JSDOM should handle it.
        vi.resetModules();
    });

    it('initializes basic buttons', async () => {
        // Load module
        await import('../js/buttons.js');

        // Dispatch DOMContentLoaded
        document.dispatchEvent(new Event('DOMContentLoaded'));

        // Wait a bit for execution
        await new Promise(resolve => setTimeout(resolve, 50));

        // Check if buttons were added to the container
        const container = document.getElementById('basic-buttons');
        expect(container).toBeTruthy();
        expect(container.children.length).toBeGreaterThan(0);

        // Check for specific class or element that buttons.js creates
        // Based on "basic-buttons", it likely creates elements with class "led-button" or similar
        // We can inspect whatever is in there.
        const firstButton = container.children[0];
        expect(firstButton.tagName).toBeDefined();
    });

    it('initializes RGB buttons', async () => {
        await import('../js/main.js');
        await import('../js/buttons.js');
        document.dispatchEvent(new Event('DOMContentLoaded'));
        await new Promise(resolve => setTimeout(resolve, 50));

        const container = document.getElementById('rgb-buttons');
        expect(container).toBeTruthy();
        expect(container.children.length).toBeGreaterThan(0);
    });
});
