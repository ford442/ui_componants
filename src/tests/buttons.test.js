import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

// Read HTML content once
const html = fs.readFileSync(path.resolve(__dirname, '../pages/buttons.html'), 'utf8');

describe('Buttons Page', () => {
    beforeEach(() => {
        // Reset DOM
        document.body.innerHTML = html;
        vi.resetModules();

        // Create a robust WebGL mock
        const createWebGLMock = () => ({
            getExtension: vi.fn(),
            createShader: vi.fn(),
            shaderSource: vi.fn(),
            compileShader: vi.fn(),
            getShaderParameter: vi.fn(() => true),
            getShaderInfoLog: vi.fn(() => ''),
            createProgram: vi.fn(),
            attachShader: vi.fn(),
            linkProgram: vi.fn(),
            getProgramParameter: vi.fn(() => true),
            getProgramInfoLog: vi.fn(() => ''),
            useProgram: vi.fn(),
            createBuffer: vi.fn(),
            bindBuffer: vi.fn(),
            bufferData: vi.fn(),
            enableVertexAttribArray: vi.fn(),
            vertexAttribPointer: vi.fn(),
            getAttribLocation: vi.fn(() => 0),
            getUniformLocation: vi.fn(() => 0),
            uniform1f: vi.fn(),
            uniform2f: vi.fn(),
            uniform3f: vi.fn(),
            uniform3fv: vi.fn(),
            clearColor: vi.fn(),
            clear: vi.fn(),
            viewport: vi.fn(),
            drawArrays: vi.fn(),
            drawElements: vi.fn(),
            enable: vi.fn(),
            blendFunc: vi.fn(),
            createVertexArray: vi.fn(),
            bindVertexArray: vi.fn(),
            // Constants
            VERTEX_SHADER: 35633,
            FRAGMENT_SHADER: 35632,
            COMPILE_STATUS: 35713,
            LINK_STATUS: 35714,
            ARRAY_BUFFER: 34962,
            element_ARRAY_BUFFER: 34963,
            STATIC_DRAW: 35044,
            FLOAT: 5126,
            COLOR_BUFFER_BIT: 16384,
            DEPTH_BUFFER_BIT: 256,
            TRIANGLES: 4,
            TRIANGLE_STRIP: 5,
            BLEND: 3042,
            SRC_ALPHA: 770,
            ONE_MINUS_SRC_ALPHA: 771,
            ONE: 1,
        });

        // Mock CanvasRenderingContext2D.fillText
        const getContextOriginal = HTMLCanvasElement.prototype.getContext;
        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (type, ...args) {
            // For WebGL, always return our mock in test env
            if (type === 'webgl' || type === 'webgl2') {
                return createWebGLMock();
            }

            const ctx = getContextOriginal.apply(this, [type, ...args]);
            if (type === '2d' && ctx) {
                if (!ctx.fillText) ctx.fillText = vi.fn();
                if (!ctx.measureText) ctx.measureText = vi.fn(() => ({ width: 0 }));
                if (!ctx.getImageData) ctx.getImageData = vi.fn(() => ({ data: new Uint8ClampedArray(4) }));
            }
            return ctx;
        });

        // Mock WebGL2Manager and WebGPUManager globals
        window.WebGL2Manager = class {
            constructor(canvas) {
                this.canvas = canvas;
                this.gl = canvas.getContext('webgl2'); // Uses test mock gl
            }
            createProgram(vs, fs) {
                // Minimal mock - return truthy program
                return {
                    useProgram: vi.fn(),
                    getUniformLocation: vi.fn(() => ({ uniform1f: vi.fn() })),
                };
            }
        };

        window.WebGPUManager = class {
            constructor(canvas) {
                this.canvas = canvas;
            }
            async init() {
                return true;
            }
        };

        // Mock UIComponents global
        vi.stubGlobal('UIComponents', {
            LEDButton: class {
                constructor(container, options) {
                    this.container = container;
                    this.element = document.createElement('div');
                    this.element.className = 'led-button-mock';
                    container.appendChild(this.element);
                    if (options.onToggle) this.onToggle = options.onToggle;
                }
                setOn(state) {
                    if (this.onToggle) this.onToggle(state);
                }
            },
            LayeredCanvas: class {
                constructor() { }
                addLayer() { return { context: createWebGLMock(), canvas: { width: 100, height: 100 } }; }
                addSVGLayer() { return { element: document.createElementNS('http://www.w3.org/2000/svg', 'svg') }; }
                setRenderFunction() { }
                startAnimation() { }
                getLayer() { }
            },
            WebGPUParticleSystem: class {
                constructor() { }
                init() { return Promise.resolve(true); }
                updateUniforms() { }
                render() { }
            },
            WebGPUVolumetricRenderer: class {
                constructor() { }
                init() { return Promise.resolve(true); }
                render() { }
            },
            ShaderUtils: {
                createProgram: vi.fn(() => ({})),
                vertexShader2D: 'void main() {}'
            }
        });

        // Mock navigator.gpu
        vi.stubGlobal('navigator', {
            ...navigator,
            gpu: {
                requestAdapter: vi.fn().mockResolvedValue({}),
                getPreferredCanvasFormat: vi.fn(() => 'bgra8unorm')
            }
        });

        // Mock WebGL2Manager and WebGPUManager globals
        window.WebGL2Manager = class {
            constructor(canvas) {
                this.canvas = canvas;
            }
        };

        window.WebGPUManager = class {
            constructor(canvas) {
                this.canvas = canvas;
            }
            async init() {
                return true;
            }
        };
    });

    it('initializes basic buttons', async () => {
        await import('../js/buttons.js');
        document.dispatchEvent(new Event('DOMContentLoaded'));
        await new Promise(resolve => setTimeout(resolve, 200));

        const container = document.getElementById('basic-buttons');
        expect(container).toBeTruthy();
        expect(container.children.length).toBeGreaterThan(0);

        const firstButton = container.querySelector('.led-button-mock');
        expect(firstButton).toBeTruthy();
    });

    it('initializes RGB buttons', async () => {
        // Reset module state if possible, or assume safe reuse
        vi.resetModules();
        await import('../js/buttons.js');

        document.dispatchEvent(new Event('DOMContentLoaded'));
        await new Promise(resolve => setTimeout(resolve, 200));

        const container = document.getElementById('rgb-buttons');
        expect(container).toBeTruthy();
        expect(container.children.length).toBeGreaterThan(0);
    });
});
