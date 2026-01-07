import { describe, it, expect, beforeEach, beforeAll, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

// Read HTML content once
const html = fs.readFileSync(path.resolve(__dirname, '../pages/buttons.html'), 'utf8');

describe('Buttons Page', () => {

  // --- ENVIRONMENT MOCKS (fix JSDOM / missing globals) ---
  beforeAll(() => {
    // 1. Mock the global UIComponents library
    global.UIComponents = {
      LEDButton: class {
        constructor(container, options) {
          const btn = document.createElement('button');
          btn.className = 'led-button-mock';
          btn.textContent = (options && options.label) || 'Mock Button';
          container.appendChild(btn);
        }
      },
      // --- ADDED THIS CLASS ---
      LayeredCanvas: class {
        constructor(container, options = {}) {
          this.container = container;
          this.width = options.width || 300;
          this.height = options.height || 150;
          this.layers = new Map();
          this.isAnimating = false;
        }

        addLayer(name, type = '2d', zIndex = 0) {
          const canvas = document.createElement('canvas');
          canvas.width = this.width;
          canvas.height = this.height;
          canvas.style.position = 'absolute';
          canvas.style.top = '0';
          canvas.style.left = '0';
          canvas.style.width = '100%';
          canvas.style.height = '100%';
          canvas.style.zIndex = zIndex;
          canvas.dataset.layer = name;
          this.container.appendChild(canvas);

          let context = null;
          if (type === '2d') {
            context = canvas.getContext('2d');
          } else {
            // For webgl/webgpu contexts in JSDOM, just leave context null
            context = null;
          }

          const layer = { canvas, context, type, zIndex, renderFn: null };
          this.layers.set(name, layer);
          return layer;
        }

        addSVGLayer(name, zIndex = 0) {
          const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          svg.setAttribute('width', '100%');
          svg.setAttribute('height', '100%');
          svg.style.position = 'absolute';
          svg.style.top = '0';
          svg.style.left = '0';
          svg.style.zIndex = zIndex;
          svg.style.pointerEvents = 'none';
          svg.dataset.layer = name;
          this.container.appendChild(svg);
          const layer = { element: svg, type: 'svg', zIndex };
          this.layers.set(name, layer);
          return layer;
        }

        getLayer(name) { return this.layers.get(name); }

        setRenderFunction(name, fn) {
          const layer = this.layers.get(name);
          if (layer) layer.renderFn = fn;
        }

        startAnimation() { this.isAnimating = true; }
        stopAnimation() { this.isAnimating = false; }

        resize(w, h) {
          this.width = w;
          this.height = h;
          this.layers.forEach(l => {
            if (l.canvas) {
              l.canvas.width = w;
              l.canvas.height = h;
            }
          });
        }
      }
    };

    // 2. Mock HTMLCanvasElement.prototype.getContext to fix "ctx.fillText is not a function"
    HTMLCanvasElement.prototype.getContext = vi.fn((contextId) => {
      if (contextId === '2d') {
        return {
          // Mock specific methods used by experiments
          fillText: vi.fn(),
          getImageData: vi.fn(() => ({
            data: new Uint8ClampedArray(100 * 100 * 4),
            width: 100,
            height: 100
          })),
          textBaseline: 'alphabetic',
          fillStyle: '#000000',
          fillRect: vi.fn(),
          clearRect: vi.fn(),
          beginPath: vi.fn(),
          arc: vi.fn(),
          fill: vi.fn(),
          measureText: vi.fn(() => ({ width: 10 })),

          // --- NEW MOCKS FOR TRANSFORMATION & STATE ---
          save: vi.fn(),
          restore: vi.fn(),
          translate: vi.fn(),
          rotate: vi.fn(),
          scale: vi.fn(),

          // Path drawing primitives
          moveTo: vi.fn(),
          lineTo: vi.fn(),
          stroke: vi.fn(),
          closePath: vi.fn(),
          lineWidth: 1,
          strokeStyle: '#000000',

          globalCompositeOperation: 'source-over',
          createLinearGradient: vi.fn(() => ({
            addColorStop: vi.fn()
          })),
          createRadialGradient: vi.fn(() => ({
            addColorStop: vi.fn()
          })),

          // --- PATH / STROKE METHODS ---
          moveTo: vi.fn(),
          lineTo: vi.fn(),
          stroke: vi.fn(),
          closePath: vi.fn(),
          setLineDash: vi.fn(),
          // strokeStyle and lineWidth are set by code directly
          // (we don't need to make them functions)
        };
      }
      return null;
    });

    // 3. Mock window.requestAnimationFrame to prevent infinite loops in tests
    global.requestAnimationFrame = vi.fn((callback) => setTimeout(callback, 0));
    global.cancelAnimationFrame = vi.fn((id) => clearTimeout(id));
  });
  // --- END MOCKS ---

    beforeEach(() => {
        // Reset DOM
        document.body.innerHTML = html;
        // Mock getElementById to avoid some null checks if necessary, but JSDOM should handle it.
        vi.resetModules();
    });

    it('initializes basic buttons', async () => {
        // Load modules (main must be loaded first because it defines UIComponents)
        await import('../js/main.js');
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
