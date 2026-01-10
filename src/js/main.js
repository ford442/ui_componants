/**
 * Main JavaScript - UI Components Library
 * Provides shared utilities for canvas layer management and rendering contexts
 */

// Import global experiments (cursor trails, audio visualizer, holographic effects)
import './experiments-global.js';

// Feature detection for rendering contexts
const RenderingSupport = {
    webgl: false,
    webgl2: false,
    webgpu: false,

    async detect() {
        // Test WebGL
        const testCanvas = document.createElement('canvas');
        this.webgl = !!testCanvas.getContext('webgl');
        this.webgl2 = !!testCanvas.getContext('webgl2');

        // Test WebGPU
        if ('gpu' in navigator) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                this.webgpu = !!adapter;
            } catch (e) {
                this.webgpu = false;
            }
        }

        return this;
    },

    getStatus() {
        return {
            webgl: this.webgl,
            webgl2: this.webgl2,
            webgpu: this.webgpu
        };
    }
};

/**
 * LayeredCanvas - Manages multiple canvas layers with different rendering contexts
 */
class LayeredCanvas {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;
        this.layers = new Map();
        this.width = options.width || 300;
        this.height = options.height || 200;
        this.animationId = null;
        this.isAnimating = false;
    }

    /**
     * Add a new layer with specified rendering context
     * @param {string} name - Layer identifier
     * @param {string} type - 'webgl' | 'webgl2' | '2d' | 'webgpu'
     * @param {number} zIndex - Stack order
     */
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

        let context;
        let contextType = type;

        switch (type) {
            case 'webgl':
                context = canvas.getContext('webgl', {
                    alpha: true,
                    premultipliedAlpha: false
                });
                break;
            case 'webgl2':
                context = canvas.getContext('webgl2', {
                    alpha: true,
                    premultipliedAlpha: false
                });
                break;
            case 'webgpu':
                // WebGPU context setup is async, handled separately
                contextType = 'webgpu';
                context = null;
                break;
            default:
                context = canvas.getContext('2d');
                contextType = '2d';
        }

        this.container.appendChild(canvas);

        const layer = {
            canvas,
            context,
            type: contextType,
            zIndex,
            renderFn: null
        };

        this.layers.set(name, layer);
        return layer;
    }

    /**
     * Add an SVG layer
     */
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

        const layer = {
            element: svg,
            type: 'svg',
            zIndex
        };

        this.layers.set(name, layer);
        return layer;
    }

    /**
     * Get a layer by name
     */
    getLayer(name) {
        return this.layers.get(name);
    }

    /**
     * Set render function for a layer
     */
    setRenderFunction(name, fn) {
        const layer = this.layers.get(name);
        if (layer) {
            layer.renderFn = fn;
        }
    }

    /**
     * Start animation loop
     */
    startAnimation() {
        if (this.isAnimating) return;
        this.isAnimating = true;

        const animate = (timestamp) => {
            if (!this.isAnimating) return;

            this.layers.forEach((layer, name) => {
                if (layer.renderFn) {
                    layer.renderFn(layer, timestamp);
                }
            });

            this.animationId = requestAnimationFrame(animate);
        };

        this.animationId = requestAnimationFrame(animate);
    }

    /**
     * Stop animation loop
     */
    stopAnimation() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    /**
     * Resize all layers
     */
    resize(width, height) {
        this.width = width;
        this.height = height;

        this.layers.forEach((layer) => {
            if (layer.canvas) {
                layer.canvas.width = width;
                layer.canvas.height = height;
            }
        });
    }

    /**
     * Clear all layers
     */
    clearAll() {
        this.layers.forEach((layer) => {
            if (layer.type === '2d' && layer.context) {
                layer.context.clearRect(0, 0, this.width, this.height);
            } else if ((layer.type === 'webgl' || layer.type === 'webgl2') && layer.context) {
                const gl = layer.context;
                gl.clearColor(0, 0, 0, 0);
                gl.clear(gl.COLOR_BUFFER_BIT);
            }
        });
    }

    /**
     * Destroy and clean up
     */
    destroy() {
        this.stopAnimation();
        this.layers.forEach((layer) => {
            if (layer.canvas) {
                layer.canvas.remove();
            }
            if (layer.element) {
                layer.element.remove();
            }
        });
        this.layers.clear();
    }
}

/**
 * WebGL Shader Utilities
 */
const ShaderUtils = {
    /**
     * Create and compile a shader
     */
    createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    },

    /**
     * Create a shader program
     */
    createProgram(gl, vertexSource, fragmentSource) {
        const vertexShader = this.createShader(gl, gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

        if (!vertexShader || !fragmentShader) return null;

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    },

    // Common vertex shader for 2D effects
    vertexShader2D: `
        attribute vec4 a_position;
        attribute vec2 a_texCoord;
        varying vec2 v_texCoord;
        
        void main() {
            gl_Position = a_position;
            v_texCoord = a_texCoord;
        }
    `,

    // Glow effect fragment shader
    glowFragmentShader: `
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec3 u_color;
        uniform float u_intensity;
        
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            float glow = 1.0 - smoothstep(0.0, 0.5, dist);
            glow *= u_intensity;
            glow *= 0.8 + 0.2 * sin(u_time * 2.0);
            
            gl_FragColor = vec4(u_color * glow, glow);
        }
    `,

    // LED effect fragment shader
    ledFragmentShader: `
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec3 u_color;
        uniform float u_on;
        
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            // LED core
            float core = 1.0 - smoothstep(0.0, 0.15, dist);
            
            // Outer glow
            float glow = 1.0 - smoothstep(0.0, 0.4, dist);
            glow = pow(glow, 2.0);
            
            // Pulsing effect
            float pulse = 0.9 + 0.1 * sin(u_time * 3.0);
            
            float intensity = mix(0.1, 1.0, u_on);
            vec3 color = u_color * (core + glow * 0.5) * intensity * pulse;
            float alpha = (core + glow * 0.3) * intensity;
            
            gl_FragColor = vec4(color, alpha);
        }
    `
};

/**
 * LED Button Component
 */
class LEDButton {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        this.options = {
            width: options.width || 100,
            height: options.height || 60,
            color: options.color || [0, 1, 0.5], // RGB normalized
            glowColor: options.glowColor || options.color || [0, 1, 0.5],
            label: options.label || '',
            ...options
        };

        this.isOn = false;
        this.isPressed = false;
        this.canvas = null;
        this.gl = null;
        this.program = null;
        this.animationId = null;

        this.init();
    }

    init() {
        // Create wrapper
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'led-button';
        this.wrapper.style.width = this.options.width + 'px';
        this.wrapper.style.height = this.options.height + 'px';
        this.wrapper.style.position = 'relative';

        // Create canvas for WebGL glow effect
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.options.width * 2;
        this.canvas.height = this.options.height * 2;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.pointerEvents = 'none';

        // Create button element (CSS layer)
        this.button = document.createElement('button');
        this.button.className = 'led-button-element';
        this.button.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
            background: linear-gradient(145deg, #2a2a3a, #1a1a2a);
            border-radius: 10px;
            cursor: pointer;
            position: relative;
            z-index: 1;
            transition: all 0.1s ease;
            box-shadow: 
                inset 0 2px 4px rgba(255, 255, 255, 0.1),
                0 4px 8px rgba(0, 0, 0, 0.5);
        `;

        if (this.options.label) {
            this.button.textContent = this.options.label;
            this.button.style.color = '#888';
            this.button.style.fontSize = '0.8rem';
            this.button.style.fontWeight = 'bold';
        }

        this.wrapper.appendChild(this.canvas);
        this.wrapper.appendChild(this.button);
        this.container.appendChild(this.wrapper);

        // Initialize WebGL
        this.initWebGL();

        // Add event listeners
        this.addEventListeners();

        // Start animation
        this.animate();
    }

    initWebGL() {
        this.gl = this.canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
        if (!this.gl) return;

        const gl = this.gl;

        // Create program
        this.program = ShaderUtils.createProgram(
            gl,
            ShaderUtils.vertexShader2D,
            ShaderUtils.ledFragmentShader
        );

        if (!this.program) return;

        // Set up geometry
        const positions = new Float32Array([
            -1, -1, 0, 0,
            1, -1, 1, 0,
            -1, 1, 0, 1,
            1, 1, 1, 1,
        ]);

        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

        const positionLoc = gl.getAttribLocation(this.program, 'a_position');
        const texCoordLoc = gl.getAttribLocation(this.program, 'a_texCoord');

        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 16, 0);

        gl.enableVertexAttribArray(texCoordLoc);
        gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 16, 8);

        // Enable blending
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        // Get uniform locations
        this.uniforms = {
            time: gl.getUniformLocation(this.program, 'u_time'),
            resolution: gl.getUniformLocation(this.program, 'u_resolution'),
            color: gl.getUniformLocation(this.program, 'u_color'),
            on: gl.getUniformLocation(this.program, 'u_on')
        };
    }

    addEventListeners() {
        this.button.addEventListener('mousedown', () => {
            this.isPressed = true;
            this.button.style.transform = 'scale(0.98)';
            this.button.style.boxShadow = `
                inset 0 2px 4px rgba(255, 255, 255, 0.05),
                0 2px 4px rgba(0, 0, 0, 0.5)
            `;
        });

        this.button.addEventListener('mouseup', () => {
            this.isPressed = false;
            this.toggle();
            this.button.style.transform = 'scale(1)';
            this.button.style.boxShadow = `
                inset 0 2px 4px rgba(255, 255, 255, 0.1),
                0 4px 8px rgba(0, 0, 0, 0.5)
            `;
        });

        this.button.addEventListener('mouseleave', () => {
            if (this.isPressed) {
                this.isPressed = false;
                this.button.style.transform = 'scale(1)';
            }
        });
    }

    toggle() {
        this.isOn = !this.isOn;
        if (this.options.onToggle) {
            this.options.onToggle(this.isOn);
        }
    }

    setOn(value) {
        this.isOn = !!value;
    }

    animate() {
        const render = (timestamp) => {
            if (!this.gl || !this.program) {
                this.animationId = requestAnimationFrame(render);
                return;
            }

            const gl = this.gl;
            const time = timestamp * 0.001;

            gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(this.program);

            gl.uniform1f(this.uniforms.time, time);
            gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);
            gl.uniform3fv(this.uniforms.color, this.options.color);
            gl.uniform1f(this.uniforms.on, this.isOn ? 1.0 : 0.0);

            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            this.animationId = requestAnimationFrame(render);
        };

        this.animationId = requestAnimationFrame(render);
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.wrapper.remove();
    }
}

/**
 * Rotary Knob Component
 */
class RotaryKnob {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        this.options = {
            size: options.size || 80,
            min: options.min || 0,
            max: options.max || 100,
            value: options.value || 0,
            color: options.color || '#00ff88',
            label: options.label || '',
            ...options
        };

        this.value = this.options.value;
        this.rotation = this.valueToRotation(this.value);
        this.isDragging = false;
        this.startY = 0;
        this.startRotation = 0;

        this.init();
    }

    init() {
        // Create wrapper
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'knob-wrapper';
        this.wrapper.style.width = this.options.size + 'px';

        // Create label
        if (this.options.label) {
            this.labelEl = document.createElement('div');
            this.labelEl.className = 'knob-label';
            this.labelEl.textContent = this.options.label;
            this.wrapper.appendChild(this.labelEl);
        }

        // Create knob container
        this.knobContainer = document.createElement('div');
        this.knobContainer.style.cssText = `
            width: ${this.options.size}px;
            height: ${this.options.size}px;
            position: relative;
            cursor: grab;
        `;

        // Create SVG for knob visual
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', this.options.size);
        this.svg.setAttribute('height', this.options.size);
        this.svg.style.cssText = 'position: absolute; top: 0; left: 0;';

        const center = this.options.size / 2;
        const radius = this.options.size * 0.4;

        // Outer ring
        const outerRing = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        outerRing.setAttribute('cx', center);
        outerRing.setAttribute('cy', center);
        outerRing.setAttribute('r', radius + 5);
        outerRing.setAttribute('fill', 'none');
        outerRing.setAttribute('stroke', '#333');
        outerRing.setAttribute('stroke-width', '2');

        // Knob body
        const knobBody = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        knobBody.setAttribute('cx', center);
        knobBody.setAttribute('cy', center);
        knobBody.setAttribute('r', radius);
        knobBody.setAttribute('fill', 'url(#knobGradient)');

        // Gradient definition
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'radialGradient');
        gradient.setAttribute('id', 'knobGradient');
        gradient.innerHTML = `
            <stop offset="0%" stop-color="#4a4a5a"/>
            <stop offset="50%" stop-color="#3a3a4a"/>
            <stop offset="100%" stop-color="#2a2a3a"/>
        `;
        defs.appendChild(gradient);

        // Glow filter
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'glow');
        filter.innerHTML = `
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        `;
        defs.appendChild(filter);

        // Indicator line
        this.indicator = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        this.indicator.setAttribute('x1', center);
        this.indicator.setAttribute('y1', center - radius + 8);
        this.indicator.setAttribute('x2', center);
        this.indicator.setAttribute('y2', center - radius + 20);
        this.indicator.setAttribute('stroke', this.options.color);
        this.indicator.setAttribute('stroke-width', '3');
        this.indicator.setAttribute('stroke-linecap', 'round');
        this.indicator.setAttribute('filter', 'url(#glow)');

        // Indicator group for rotation
        this.indicatorGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.indicatorGroup.appendChild(this.indicator);
        this.indicatorGroup.style.transformOrigin = `${center}px ${center}px`;
        this.indicatorGroup.style.transform = `rotate(${this.rotation}deg)`;

        this.svg.appendChild(defs);
        this.svg.appendChild(outerRing);
        this.svg.appendChild(knobBody);
        this.svg.appendChild(this.indicatorGroup);

        this.knobContainer.appendChild(this.svg);

        // Create WebGL canvas for glow effect
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.options.size * 2;
        this.canvas.height = this.options.size * 2;
        this.canvas.style.cssText = `
            position: absolute;
            top: -25%;
            left: -25%;
            width: 150%;
            height: 150%;
            pointer-events: none;
            z-index: -1;
        `;
        this.knobContainer.appendChild(this.canvas);

        // Create value display
        this.valueEl = document.createElement('div');
        this.valueEl.className = 'knob-value';
        this.valueEl.textContent = this.value.toFixed(0);

        this.wrapper.appendChild(this.knobContainer);
        this.wrapper.appendChild(this.valueEl);
        this.container.appendChild(this.wrapper);

        // Initialize WebGL glow
        this.initGlow();

        // Add event listeners
        this.addEventListeners();
    }

    initGlow() {
        const gl = this.canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
        if (!gl) return;

        this.gl = gl;

        const fragmentShader = `
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec3 u_color;
            uniform float u_value;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(uv, center);
                
                float glow = 1.0 - smoothstep(0.2, 0.5, dist);
                glow = pow(glow, 3.0);
                glow *= u_value * 0.5;
                glow *= 0.9 + 0.1 * sin(u_time * 2.0);
                
                gl_FragColor = vec4(u_color * glow, glow * 0.5);
            }
        `;

        this.program = ShaderUtils.createProgram(gl, ShaderUtils.vertexShader2D, fragmentShader);
        if (!this.program) return;

        // Set up geometry
        const positions = new Float32Array([
            -1, -1, 0, 0,
            1, -1, 1, 0,
            -1, 1, 0, 1,
            1, 1, 1, 1,
        ]);

        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

        const positionLoc = gl.getAttribLocation(this.program, 'a_position');
        const texCoordLoc = gl.getAttribLocation(this.program, 'a_texCoord');

        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 16, 0);

        gl.enableVertexAttribArray(texCoordLoc);
        gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 16, 8);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        // Parse color
        const color = this.parseColor(this.options.color);

        this.glUniforms = {
            time: gl.getUniformLocation(this.program, 'u_time'),
            resolution: gl.getUniformLocation(this.program, 'u_resolution'),
            color: gl.getUniformLocation(this.program, 'u_color'),
            value: gl.getUniformLocation(this.program, 'u_value')
        };

        this.glColor = color;

        // Start animation
        this.animate();
    }

    parseColor(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        if (result) {
            return [
                parseInt(result[1], 16) / 255,
                parseInt(result[2], 16) / 255,
                parseInt(result[3], 16) / 255
            ];
        }
        return [0, 1, 0.5];
    }

    animate() {
        const render = (timestamp) => {
            if (!this.gl || !this.program) {
                this.animationId = requestAnimationFrame(render);
                return;
            }

            const gl = this.gl;
            const time = timestamp * 0.001;
            const normalizedValue = (this.value - this.options.min) / (this.options.max - this.options.min);

            gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(this.program);

            gl.uniform1f(this.glUniforms.time, time);
            gl.uniform2f(this.glUniforms.resolution, this.canvas.width, this.canvas.height);
            gl.uniform3fv(this.glUniforms.color, this.glColor);
            gl.uniform1f(this.glUniforms.value, normalizedValue);

            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            this.animationId = requestAnimationFrame(render);
        };

        this.animationId = requestAnimationFrame(render);
    }

    addEventListeners() {
        this.knobContainer.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.startY = e.clientY;
            this.startRotation = this.rotation;
            this.knobContainer.style.cursor = 'grabbing';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;

            const deltaY = this.startY - e.clientY;
            const newRotation = Math.max(-135, Math.min(135, this.startRotation + deltaY));
            this.rotation = newRotation;
            this.value = this.rotationToValue(newRotation);

            this.updateDisplay();
        });

        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.knobContainer.style.cursor = 'grab';
            }
        });

        // Touch support
        this.knobContainer.addEventListener('touchstart', (e) => {
            this.isDragging = true;
            this.startY = e.touches[0].clientY;
            this.startRotation = this.rotation;
            e.preventDefault();
        });

        document.addEventListener('touchmove', (e) => {
            if (!this.isDragging) return;

            const deltaY = this.startY - e.touches[0].clientY;
            const newRotation = Math.max(-135, Math.min(135, this.startRotation + deltaY));
            this.rotation = newRotation;
            this.value = this.rotationToValue(newRotation);

            this.updateDisplay();
        });

        document.addEventListener('touchend', () => {
            this.isDragging = false;
        });
    }

    valueToRotation(value) {
        const range = this.options.max - this.options.min;
        const normalized = (value - this.options.min) / range;
        return -135 + normalized * 270;
    }

    rotationToValue(rotation) {
        const normalized = (rotation + 135) / 270;
        const range = this.options.max - this.options.min;
        return this.options.min + normalized * range;
    }

    updateDisplay() {
        const center = this.options.size / 2;
        this.indicatorGroup.style.transform = `rotate(${this.rotation}deg)`;
        this.valueEl.textContent = this.value.toFixed(0);

        if (this.options.onChange) {
            this.options.onChange(this.value);
        }
    }

    setValue(value) {
        this.value = Math.max(this.options.min, Math.min(this.options.max, value));
        this.rotation = this.valueToRotation(this.value);
        this.updateDisplay();
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.wrapper.remove();
    }
}

/**
 * WebGPU Particle System
 * Manages GPU-accelerated particle simulations using compute shaders
 */
class WebGPUParticleSystem {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.options = {
            particleCount: options.particleCount || 10000,
            particleSize: options.particleSize || 2,
            color: options.color || [0, 1, 0.5, 1],
            physics: options.physics || 'attract', // 'attract', 'repel', 'orbit', 'fluid'
            ...options
        };

        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.computePipeline = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.initialized = false;
    }

    async init() {
        if (!navigator.gpu) {
            console.warn('WebGPU not supported');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.warn('WebGPU adapter not found');
                return false;
            }
            this.device = await adapter.requestDevice();

            this.context = this.canvas.getContext('webgpu');
            const format = navigator.gpu.getPreferredCanvasFormat();

            this.context.configure({
                device: this.device,
                format: format,
                alphaMode: 'premultiplied'
            });

            await this.createBuffers();
            await this.createPipelines(format);

            this.initialized = true;
            return true;
        } catch (e) {
            console.error('WebGPU initialization failed:', e);
            return false;
        }
    }

    async createBuffers() {
        // Create particle data
        const particleData = new Float32Array(this.options.particleCount * 8);

        for (let i = 0; i < this.options.particleCount; i++) {
            const offset = i * 8;
            // Position (x, y)
            particleData[offset + 0] = (Math.random() - 0.5) * 2;
            particleData[offset + 1] = (Math.random() - 0.5) * 2;
            // Velocity (vx, vy)
            particleData[offset + 2] = (Math.random() - 0.5) * 0.02;
            particleData[offset + 3] = (Math.random() - 0.5) * 0.02;
            // Color (r, g, b, a)
            particleData[offset + 4] = this.options.color[0];
            particleData[offset + 5] = this.options.color[1];
            particleData[offset + 6] = this.options.color[2];
            particleData[offset + 7] = this.options.color[3];
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
        this.particleBuffer.unmap();

        // Create uniform buffer
        this.uniformBuffer = this.device.createBuffer({
            size: 64, // 4x4 matrix + extra space
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    async createPipelines(format) {
        // Compute shader for particle physics
        const computeShader = `
            struct Particle {
                pos: vec2<f32>,
                vel: vec2<f32>,
                color: vec4<f32>
            }
            
            struct Uniforms {
                time: f32,
                deltaTime: f32,
                mouseX: f32,
                mouseY: f32,
                attractorStrength: f32,
                damping: f32
            }
            
            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }
                
                var particle = particles[index];
                
                // Apply attractor force
                let mousePos = vec2<f32>(uniforms.mouseX, uniforms.mouseY);
                let toMouse = mousePos - particle.pos;
                let dist = length(toMouse);
                
                if (dist > 0.01) {
                    let force = normalize(toMouse) * uniforms.attractorStrength / (dist * dist + 0.1);
                    particle.vel += force * uniforms.deltaTime;
                }
                
                // Apply damping
                particle.vel *= uniforms.damping;
                
                // Update position
                particle.pos += particle.vel * uniforms.deltaTime;
                
                // Boundary conditions (wrap around)
                if (particle.pos.x < -1.0) { particle.pos.x = 1.0; }
                if (particle.pos.x > 1.0) { particle.pos.x = -1.0; }
                if (particle.pos.y < -1.0) { particle.pos.y = 1.0; }
                if (particle.pos.y > 1.0) { particle.pos.y = -1.0; }
                
                particles[index] = particle;
            }
        `;

        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: computeShader }),
                entryPoint: 'main'
            }
        });

        // Render shader for drawing particles
        const renderShader = `
            struct Particle {
                pos: vec2<f32>,
                vel: vec2<f32>,
                color: vec4<f32>
            }
            
            @group(0) @binding(0) var<storage, read> particles: array<Particle>;
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>
            }
            
            @vertex
            fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                let particleIndex = vertexIndex / 6u;
                let particle = particles[particleIndex];
                
                let vertices = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(1.0, -1.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(-1.0, 1.0)
                );
                
                let localVertex = vertices[vertexIndex % 6u];
                let size = 0.005;
                let pos = particle.pos + localVertex * size;
                
                var output: VertexOutput;
                output.position = vec4<f32>(pos, 0.0, 1.0);
                output.color = particle.color;
                return output;
            }
            
            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
                return input.color;
            }
        `;

        this.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'vertexMain'
            },
            fragment: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'fragmentMain',
                targets: [{
                    format: format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        // Create bind groups
        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } }
            ]
        });
    }

    updateUniforms(time, deltaTime, mouseX = 0, mouseY = 0) {
        if (!this.initialized) return;

        const uniformData = new Float32Array([
            time,
            deltaTime,
            mouseX,
            mouseY,
            this.options.attractorStrength || 0.5,
            this.options.damping || 0.99,
            0, 0 // padding
        ]);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    render(time, deltaTime) {
        if (!this.initialized) return;

        this.updateUniforms(time, deltaTime);

        const commandEncoder = this.device.createCommandEncoder();

        // Compute pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.options.particleCount / 64));
        computePass.end();

        // Render pass
        const textureView = this.context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(this.options.particleCount * 6);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    destroy() {
        if (this.particleBuffer) this.particleBuffer.destroy();
        if (this.uniformBuffer) this.uniformBuffer.destroy();
    }
}

/**
 * WebGPU Compute Buffer Helper
 * Simplified buffer management for compute shader operations
 */
class WebGPUComputeBuffer {
    constructor(device, size, usage = GPUBufferUsage.STORAGE) {
        this.device = device;
        this.size = size;
        this.buffer = device.createBuffer({
            size: size,
            usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
    }

    write(data) {
        this.device.queue.writeBuffer(this.buffer, 0, data);
    }

    async read() {
        const readBuffer = this.device.createBuffer({
            size: this.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.buffer, 0, readBuffer, 0, this.size);
        this.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readBuffer.getMappedRange());
        const result = new Float32Array(data);
        readBuffer.unmap();
        readBuffer.destroy();

        return result;
    }

    destroy() {
        this.buffer.destroy();
    }
}

/**
 * WebGPU Volumetric Renderer
 * Raymarching and volumetric effects using compute shaders
 */
class WebGPUVolumetricRenderer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.options = {
            volumeSize: options.volumeSize || 128,
            raySteps: options.raySteps || 64,
            density: options.density || 0.5,
            ...options
        };

        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.initialized = false;
    }

    async init() {
        if (!navigator.gpu) {
            console.warn('WebGPU not supported');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.warn('WebGPU adapter not found');
                return false;
            }
            this.device = await adapter.requestDevice();

            this.context = this.canvas.getContext('webgpu');
            const format = navigator.gpu.getPreferredCanvasFormat();

            this.context.configure({
                device: this.device,
                format: format,
                alphaMode: 'premultiplied'
            });

            await this.createPipeline(format);

            this.initialized = true;
            return true;
        } catch (e) {
            console.error('WebGPU volumetric renderer initialization failed:', e);
            return false;
        }
    }

    async createPipeline(format) {
        const shaderCode = `
            struct Uniforms {
                time: f32,
                density: f32,
                raySteps: f32,
                _padding: f32
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>
            }
            
            @vertex
            fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                let pos = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(1.0, -1.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(-1.0, 1.0)
                );
                
                var output: VertexOutput;
                output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
                output.uv = pos[vertexIndex] * 0.5 + 0.5;
                return output;
            }
            
            fn noise3D(p: vec3<f32>) -> f32 {
                let i = floor(p);
                let f = fract(p);
                let u = f * f * (3.0 - 2.0 * f);
                
                return mix(
                    mix(
                        mix(hash(i + vec3<f32>(0.0, 0.0, 0.0)), 
                            hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
                        mix(hash(i + vec3<f32>(0.0, 1.0, 0.0)), 
                            hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
                    mix(
                        mix(hash(i + vec3<f32>(0.0, 0.0, 1.0)), 
                            hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
                        mix(hash(i + vec3<f32>(0.0, 1.0, 1.0)), 
                            hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y), u.z);
            }
            
            fn hash(p: vec3<f32>) -> f32 {
                let p3 = fract(p * 0.1031);
                let dot_val = dot(p3, vec3<f32>(p3.yz + 19.19, p3.x + 19.19));
                return fract((p3.x + p3.y) * dot_val);
            }
            
            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
                let rayOrigin = vec3<f32>(0.0, 0.0, -2.0);
                let rayDir = normalize(vec3<f32>(input.uv * 2.0 - 1.0, 1.0));
                
                var color = vec3<f32>(0.0);
                var alpha = 0.0;
                
                let steps = i32(uniforms.raySteps);
                let stepSize = 0.05;
                
                for (var i = 0; i < steps; i++) {
                    let t = f32(i) * stepSize;
                    let pos = rayOrigin + rayDir * t;
                    
                    let noiseValue = noise3D(pos * 2.0 + vec3<f32>(uniforms.time * 0.2));
                    let density = noiseValue * uniforms.density;
                    
                    if (density > 0.0) {
                        let col = vec3<f32>(
                            0.0,
                            0.5 + 0.5 * sin(uniforms.time + noiseValue * 3.14),
                            0.8 + 0.2 * cos(uniforms.time * 0.5)
                        );
                        
                        color += col * density * (1.0 - alpha);
                        alpha += density * 0.1 * (1.0 - alpha);
                        
                        if (alpha > 0.95) {
                            break;
                        }
                    }
                }
                
                return vec4<f32>(color, alpha);
            }
        `;

        this.uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: shaderCode }),
                entryPoint: 'vertexMain'
            },
            fragment: {
                module: this.device.createShaderModule({ code: shaderCode }),
                entryPoint: 'fragmentMain',
                targets: [{
                    format: format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } }
            ]
        });
    }

    render(time) {
        if (!this.initialized) return;

        const uniformData = new Float32Array([
            time,
            this.options.density,
            this.options.raySteps,
            0 // padding
        ]);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(6);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    destroy() {
        if (this.uniformBuffer) this.uniformBuffer.destroy();
    }
}

// Export for use in other modules
window.UIComponents = {
    RenderingSupport,
    LayeredCanvas,
    ShaderUtils,
    LEDButton,
    RotaryKnob,
    WebGPUParticleSystem,
    WebGPUComputeBuffer,
    WebGPUVolumetricRenderer
};

// Initialize rendering support detection
document.addEventListener('DOMContentLoaded', () => {
    RenderingSupport.detect().then(support => {
        console.log('Rendering Support:', support.getStatus());
    });
});
