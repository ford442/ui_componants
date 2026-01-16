/**
 * Neon City Experiment
 * Combines WebGL2 for procedural city rendering and WebGPU for compute-heavy effects.
 */

export class NeonCityExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.speed = 0.5;
        this.rainDensity = 0.7;
        this.mouse = { x: 0, y: 0 };
        this.colorMode = 0.0; // 0: Matrix, 1: RGB

        // WebGL2 State (City)
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.instanceCount = 2000;
        this.buildingDataArray = null; // CPU copy for raycasting
        this.buildingBuffer = null; // Buffer for building properties (pos, size)
        this.hoveredInstance = -1;
        this.pulseStartTime = -100.0; // Initialize far in past

        // WebGPU State (Rain)
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.rainPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.rainBuffer = null;
        this.uniformBuffer = null;
        this.numRainDrops = 10000;

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        // Setup container
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050510';

        // Controls
        const speedInput = document.getElementById('speed-control');
        if (speedInput) {
            speedInput.addEventListener('input', (e) => {
                this.speed = parseInt(e.target.value) / 100;
            });
        }
        const rainInput = document.getElementById('rain-control');
        if (rainInput) {
            rainInput.addEventListener('input', (e) => {
                this.rainDensity = parseInt(e.target.value) / 100;
                this.updateRainParams();
            });
        }
        const colorModeBtn = document.getElementById('color-mode-btn');
        if (colorModeBtn) {
            colorModeBtn.addEventListener('click', () => {
                this.colorMode = this.colorMode === 0.0 ? 1.0 : 0.0;
                colorModeBtn.textContent = this.colorMode === 0.0 ? 'Matrix' : 'Neon RGB';
                colorModeBtn.style.background = this.colorMode === 0.0 ? '#00f' : '#f0f';
            });
        }

        // Mouse Interaction
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            // Normalize to [-1, 1]
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            if (this.glCanvas) {
                this.checkIntersection();
            }
        });

        this.container.addEventListener('mouseleave', () => {
             // Reset or keep last position? Let's smoothly return to center logic if needed,
             // but for now keeping last pos is fine or resetting.
             // this.mouse.x = 0; this.mouse.y = 0;
        });

        this.container.addEventListener('click', () => {
            this.triggerPulse();
        });

        // 1. Init WebGL2 (City Layer)
        this.initWebGL2();

        // 2. Init WebGPU (Rain Layer)
        if (navigator.gpu) {
            try {
                await this.initWebGPU();
            } catch (e) {
                console.warn("NeonCity: WebGPU failed to init", e);
            }
        }

        // Initial resize to ensure canvases have correct size before rendering
        this.resize();

        this.isActive = true;
        this.animate();

        // Use ResizeObserver for responsive layout
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.container);
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }

        // Cleanup DOM
        if (this.glCanvas) this.glCanvas.remove();
        if (this.gpuCanvas) this.gpuCanvas.remove();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Cityscape)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Generate City Data (Instanced)
        // Position (x, z), Size (w, h, d), Color (r, g, b)
        // We pack this into attributes.
        // Let's keep it simple: Position X, Z, Scale Y (height), Random Seed
        const instanceData = new Float32Array(this.instanceCount * 4);
        this.buildingDataArray = instanceData; // Store reference for Raycasting
        const range = 200;

        for (let i = 0; i < this.instanceCount; i++) {
            const x = (Math.random() - 0.5) * range;
            const z = -Math.random() * range; // Extend into distance
            const h = Math.random() * 5.0 + 1.0; // Height
            const seed = Math.random();

            instanceData[i * 4 + 0] = x;
            instanceData[i * 4 + 1] = z;
            instanceData[i * 4 + 2] = h;
            instanceData[i * 4 + 3] = seed;
        }

        // Cube Vertices (Unit Cube)
        const vertices = new Float32Array([
            // Front face
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            // Back face
            -0.5, -0.5, -0.5,
            -0.5,  0.5, -0.5,
             0.5,  0.5, -0.5,
             0.5, -0.5, -0.5,
            // Top face
            -0.5,  0.5, -0.5,
            -0.5,  0.5,  0.5,
             0.5,  0.5,  0.5,
             0.5,  0.5, -0.5,
             // Bottom face
            -0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,
             0.5, -0.5,  0.5,
            -0.5, -0.5,  0.5,
            // Right face
             0.5, -0.5, -0.5,
             0.5,  0.5, -0.5,
             0.5,  0.5,  0.5,
             0.5, -0.5,  0.5,
             // Left face
            -0.5, -0.5, -0.5,
            -0.5, -0.5,  0.5,
            -0.5,  0.5,  0.5,
            -0.5,  0.5, -0.5,
        ]);

        const indices = new Uint16Array([
            0,  1,  2,      0,  2,  3,    // front
            4,  5,  6,      4,  6,  7,    // back
            8,  9,  10,     8,  10, 11,   // top
            12, 13, 14,     12, 14, 15,   // bottom
            16, 17, 18,     16, 18, 19,   // right
            20, 21, 22,     20, 22, 23    // left
        ]);

        // Create Program
        const vs = `#version 300 es
        layout(location=0) in vec3 a_position;
        layout(location=1) in vec4 a_instanceData; // x, z, height, seed

        uniform mat4 u_projection;
        uniform mat4 u_view;
        uniform float u_time;
        uniform float u_scrollSpeed;
        uniform float u_hoveredInstance;
        uniform float u_pulseTime;

        out vec3 v_color;
        out float v_dist;
        out vec3 v_worldPos;

        // Pseudo-random function
        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }

        void main() {
            vec3 pos = a_position;

            // Instance data
            float ix = a_instanceData.x;
            float iz = a_instanceData.y;
            float ih = a_instanceData.z;
            float iseed = a_instanceData.w;

            // Scroll Logic
            float offset = mod(u_time * u_scrollSpeed * 20.0, 200.0);
            float zPos = iz + offset;
            if (zPos > 10.0) {
                zPos -= 200.0;
            }

            // Scale building
            pos.y *= ih;
            pos.y += ih * 0.5; // Move base to y=0

            // Apply World Position
            vec3 worldPos = vec3(pos.x + ix, pos.y, pos.z + zPos);

            // Glitch Effect on Pulse
            if (u_pulseTime > 0.0) {
                float distFromCenter = length(worldPos.xz);
                float pulseRadius = u_pulseTime * 100.0;
                float pulseWidth = 20.0;
                float distDiff = abs(distFromCenter - pulseRadius);

                if (distDiff < pulseWidth) {
                    float glitchIntensity = smoothstep(pulseWidth, 0.0, distDiff);
                    // Jitter
                    float jitter = (random(vec2(worldPos.y, u_time)) - 0.5) * 2.0;
                    worldPos.x += jitter * glitchIntensity * 2.0;
                    // Vertical displacement
                    worldPos.y += sin(worldPos.x * 0.5 + u_time * 20.0) * glitchIntensity * 5.0;
                }
            }

            gl_Position = u_projection * u_view * vec4(worldPos, 1.0);

            // Color based on height and randomness
            float glow = 0.2 + 0.8 * iseed;
            vec3 buildingColor = mix(vec3(0.1, 0.0, 0.2), vec3(0.0, 0.8, 1.0), iseed);

            // Highlight Hovered
            if (abs(float(gl_InstanceID) - u_hoveredInstance) < 0.1) {
                buildingColor = vec3(1.0, 0.0, 0.8); // Neon Pink Highlight
                glow = 2.0;
            }

            // Cyber-Grid Pattern
            float grid = step(0.9, fract(worldPos.y * 0.5)) + step(0.9, fract(worldPos.x * 0.5));
            if (grid > 0.5) {
                buildingColor += vec3(0.1, 0.1, 0.2) * glow;
            }

            // Windows effect
            if (mod(worldPos.y * 2.0, 1.0) > 0.6 && mod(worldPos.x + worldPos.z, 2.0) > 1.2) {
                 buildingColor += vec3(0.8, 0.8, 0.5) * glow;
            }

            v_color = buildingColor;
            v_dist = length(worldPos.xz);
            v_worldPos = worldPos;
        }
        `;

        const fs = `#version 300 es
        precision highp float;

        in vec3 v_color;
        in float v_dist;
        in vec3 v_worldPos;

        uniform float u_pulseTime; // Time since pulse start

        out vec4 outColor;

        void main() {
            vec3 color = v_color;

            // Pulse Effect (Expanding Ring)
            float distFromCenter = length(v_worldPos.xz);
            float pulseRadius = u_pulseTime * 100.0; // Speed of pulse
            float pulseWidth = 15.0;

            float pulse = 0.0;
            if (u_pulseTime > 0.0) {
                float distDiff = abs(distFromCenter - pulseRadius);
                pulse = smoothstep(pulseWidth, 0.0, distDiff) * max(0.0, 1.0 - u_pulseTime * 0.2); // Fade out over distance
            }

            vec3 pulseColor = vec3(0.0, 1.0, 1.0); // Cyan Pulse
            color += pulseColor * pulse;

            // Fog
            float fogFactor = smoothstep(10.0, 120.0, v_dist); // Extended fog slightly
            vec3 fogColor = vec3(0.05, 0.05, 0.1);

            color = mix(color, fogColor, fogFactor);

            outColor = vec4(color, 1.0);
        }
        `;

        this.glProgram = this.createGLProgram(vs, fs);

        // VAO Setup
        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        // Geometry Buffer
        const geoBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, geoBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);

        // Instance Buffer
        const instBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, instBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, instanceData, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(1);
        this.gl.vertexAttribPointer(1, 4, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(1, 1); // Important: Per instance

        // Index Buffer
        const idxBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        this.gl.enable(this.gl.DEPTH_TEST);

        // Removed call to resizeGL() here to avoid setting undefined size
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fs));
            return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Data Rain)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2;
            pointer-events: none;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // Compute Shader: Update rain positions
        const computeCode = `
            struct Particle {
                pos: vec2f,
                speed: f32,
                len: f32,
            }
            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Uniforms {
                dt: f32,
                density: f32,
                mouseX: f32,
                mouseY: f32,
                wind: f32,
                colorMode: f32,
                pad1: f32,
                pad2: f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let i = id.x;
                if (i >= ${this.numRainDrops}) { return; }

                var p = particles[i];

                // Interaction: Repel from mouse
                let mousePos = vec2f(uniforms.mouseX, uniforms.mouseY);
                let dist = distance(p.pos, mousePos);
                if (dist < 0.3) {
                    let dir = normalize(p.pos - mousePos);
                    p.pos = p.pos + dir * (0.3 - dist) * 0.1;
                }

                // Wind Force
                p.pos.x = p.pos.x + uniforms.wind * uniforms.dt;

                // Fall down
                p.pos.y = p.pos.y - p.speed * uniforms.dt;

                // Reset if below screen
                if (p.pos.y < -1.2) {
                    p.pos.y = 1.2 + fract(p.speed * 123.45) * 0.5;
                    p.pos.x = (fract(p.pos.x * 67.89 + uniforms.dt) - 0.5) * 2.0;
                }
                // Wrap X (Wind effect)
                if (p.pos.x > 1.2) { p.pos.x -= 2.4; }
                if (p.pos.x < -1.2) { p.pos.x += 2.4; }

                particles[i] = p;
            }
        `;

        // Render Shader: Draw rain as lines
        const drawCode = `
            struct Particle {
                pos: vec2f,
                speed: f32,
                len: f32,
            }
            @group(0) @binding(0) var<storage, read> particles : array<Particle>;

            struct Uniforms {
                dt: f32,
                density: f32,
                mouseX: f32,
                mouseY: f32,
                wind: f32,
                colorMode: f32,
                pad1: f32,
                pad2: f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            struct VertexOutput {
                @builtin(position) pos : vec4f,
                @location(0) speed : f32,
                @location(1) colorMode : f32,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vIdx : u32, @builtin(instance_index) iIdx : u32) -> VertexOutput {
                let p = particles[iIdx];

                // Draw a vertical line (2 vertices)
                // Vertex 0: Top, Vertex 1: Bottom
                let yOffset = f32(vIdx) * p.len * 0.1;

                // Tilt based on wind
                let xOffset = yOffset * -uniforms.wind * 2.0;

                var out: VertexOutput;
                out.pos = vec4f(p.pos.x + xOffset, p.pos.y + yOffset, 0.0, 1.0);
                out.speed = p.speed;
                out.colorMode = uniforms.colorMode;
                return out;
            }

            @fragment
            fn fs_main(@location(0) speed : f32, @location(1) colorMode : f32) -> @location(0) vec4f {
                let alpha = clamp(speed * 0.5, 0.2, 0.8);

                var color = vec3f(0.0, 1.0, 0.5); // Default Matrix Green

                if (colorMode > 0.5) {
                    // RGB Mode (Cyberpunk)
                    // Color based on speed and random factor (simulated via speed)
                    if (speed > 1.2) {
                         color = vec3f(1.0, 0.0, 0.8); // Pink
                    } else if (speed > 0.8) {
                         color = vec3f(0.0, 0.8, 1.0); // Cyan
                    } else {
                         color = vec3f(0.5, 0.0, 1.0); // Purple
                    }
                }

                return vec4f(color, alpha);
            }
        `;

        // Init buffers
        const pSize = 4 * 4;
        const initData = new Float32Array(this.numRainDrops * 4);
        for(let i=0; i<this.numRainDrops; i++) {
            initData[i*4+0] = (Math.random() - 0.5) * 2.0; // x
            initData[i*4+1] = Math.random() * 2.0 - 1.0; // y
            initData[i*4+2] = 0.5 + Math.random(); // speed
            initData[i*4+3] = 0.5 + Math.random(); // len
        }

        this.rainBuffer = this.device.createBuffer({
            size: initData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.rainBuffer.getMappedRange()).set(initData);
        this.rainBuffer.unmap();

        // Uniforms: dt(4), density(4), mouseX(4), mouseY(4), wind(4), colorMode(4), pad1(4), pad2(4) -> 32 bytes
        this.uniformBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Layouts & Pipelines
        const computeBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // read-write
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        const renderBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.rainBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.rainBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // Compute Pipeline
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGLayout] }),
            compute: { module: this.device.createShaderModule({ code: computeCode }), entryPoint: 'main' }
        });

        // Render Pipeline
        this.rainPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBGLayout] }),
            vertex: {
                module: this.device.createShaderModule({ code: drawCode }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.device.createShaderModule({ code: drawCode }),
                entryPoint: 'fs_main',
                targets: [{ format, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                } }]
            },
            primitive: { topology: 'line-list' }
        });

        // Removed call to resizeGPU() here
    }

    updateRainParams() {
        // We could map this to uniform density if we implemented logic for it in shader
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth * dpr;
        const h = this.container.clientHeight * dpr;

        // Ensure non-zero dimensions
        if (w === 0 || h === 0) return;

        this.resizeGL(w, h);
        this.resizeGPU(w, h);
    }

    resizeGL(w, h) {
        if (!this.glCanvas) return;
        this.glCanvas.width = w;
        this.glCanvas.height = h;
        this.gl.viewport(0, 0, w, h);
    }

    resizeGPU(w, h) {
        if (!this.gpuCanvas) return;
        this.gpuCanvas.width = w;
        this.gpuCanvas.height = h;
    }

    triggerPulse() {
        const now = Date.now();
        this.pulseStartTime = (now - this.startTime) * 0.001;
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016; // Fixed timestep for simplicity

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Parallax Camera
            const camX = this.mouse.x * 5.0;
            const camY = 5.0 + this.mouse.y * 2.0;

            // Matrices
            const aspect = this.glCanvas.width / this.glCanvas.height;
            const projection = this.createPerspectiveMatrix(60, aspect, 0.1, 500.0);
            const view = this.createLookAtMatrix(
                [camX, camY, -20], // Eye (Parallax)
                [0, 0, 50],  // Target
                [0, 1, 0]    // Up
            );

            const projLoc = this.gl.getUniformLocation(this.glProgram, 'u_projection');
            const viewLoc = this.gl.getUniformLocation(this.glProgram, 'u_view');
            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const scrollLoc = this.gl.getUniformLocation(this.glProgram, 'u_scrollSpeed');
            const hoverLoc = this.gl.getUniformLocation(this.glProgram, 'u_hoveredInstance');
            const pulseLoc = this.gl.getUniformLocation(this.glProgram, 'u_pulseTime');

            this.gl.uniformMatrix4fv(projLoc, false, projection);
            this.gl.uniformMatrix4fv(viewLoc, false, view);
            this.gl.uniform1f(timeLoc, time);
            this.gl.uniform1f(scrollLoc, this.speed);
            this.gl.uniform1f(hoverLoc, this.hoveredInstance);

            const pulseElapsed = time - this.pulseStartTime;
            this.gl.uniform1f(pulseLoc, pulseElapsed);

            this.gl.clearColor(0.02, 0.02, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            // Draw 36 vertices (cube) * instanceCount
            this.gl.drawElementsInstanced(this.gl.TRIANGLES, 36, this.gl.UNSIGNED_SHORT, 0, this.instanceCount);
        }

        // 2. WebGPU Render
        if (this.device && this.rainPipeline) {
            // Calculate wind based on mouse X (center = 0)
            // mouse.x is [-1, 1]
            const wind = this.mouse.x * 0.5;

            // Update Uniforms
            const uData = new Float32Array([
                dt * (1.0 + this.speed * 5.0),
                this.rainDensity,
                this.mouse.x,
                this.mouse.y,
                wind,
                this.colorMode,
                0.0, // pad1
                0.0  // pad2
            ]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uData);

            const encoder = this.device.createCommandEncoder();

            // Compute
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numRainDrops / 64));
            cPass.end();

            // Render
            // Ensure width/height are valid (>0) to avoid validation errors
            if (this.gpuCanvas.width > 0 && this.gpuCanvas.height > 0) {
                const textureView = this.context.getCurrentTexture().createView();
                const rPass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: textureView,
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: 'load', // Load WebGL canvas content beneath? No, they are separate canvases.
                        storeOp: 'store'
                    }]
                });
                rPass.setPipeline(this.rainPipeline);
                rPass.setBindGroup(0, this.renderBindGroup);
                // Draw 2 vertices per instance * numRainDrops instances
                rPass.draw(2, this.numRainDrops);
                rPass.end();
            }

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    // Matrix Helpers
    createPerspectiveMatrix(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov * Math.PI / 360);
        const rangeInv = 1.0 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0
        ]);
    }

    createLookAtMatrix(eye, target, up) {
        let z = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
        const len = Math.sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
        if(len > 0) z = z.map(v => v / len);

        let x = [
            up[1]*z[2] - up[2]*z[1],
            up[2]*z[0] - up[0]*z[2],
            up[0]*z[1] - up[1]*z[0]
        ];
        const lenX = Math.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        if(lenX > 0) x = x.map(v => v / lenX);

        let y = [
            z[1]*x[2] - z[2]*x[1],
            z[2]*x[0] - z[0]*x[2],
            z[0]*x[1] - z[1]*x[0]
        ];
        // y is normalized if z and x are

        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]),
            -(y[0]*eye[0] + y[1]*eye[1] + y[2]*eye[2]),
            -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]),
            1
        ]);
    }

    checkIntersection() {
        if (!this.buildingDataArray) return;

        // Current Camera State
        const camX = this.mouse.x * 5.0;
        const camY = 5.0 + this.mouse.y * 2.0;
        const eye = [camX, camY, -20];
        const target = [0, 0, 50];
        const up = [0, 1, 0];

        // Recompute Camera Basis (Forward/Right/Up) manually
        let zAxis = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
        const lenZ = Math.sqrt(zAxis[0]*zAxis[0] + zAxis[1]*zAxis[1] + zAxis[2]*zAxis[2]);
        zAxis = zAxis.map(v => v / lenZ); // Forward (Camera looks down -Z relative to itself, but View matrix Z axis points back)

        let xAxis = [
            up[1]*zAxis[2] - up[2]*zAxis[1],
            up[2]*zAxis[0] - up[0]*zAxis[2],
            up[0]*zAxis[1] - up[1]*zAxis[0]
        ];
        const lenX = Math.sqrt(xAxis[0]*xAxis[0] + xAxis[1]*xAxis[1] + xAxis[2]*xAxis[2]);
        xAxis = xAxis.map(v => v / lenX);

        let yAxis = [
            zAxis[1]*xAxis[2] - zAxis[2]*xAxis[1],
            zAxis[2]*xAxis[0] - zAxis[0]*xAxis[2],
            zAxis[0]*xAxis[1] - zAxis[1]*xAxis[0]
        ];

        // Ray in World Space
        const fov = 60;
        const aspect = this.glCanvas.width / this.glCanvas.height;
        const tanFov = Math.tan(fov * Math.PI / 360);

        // Ray Direction
        // NDC (mouse.x, mouse.y) -> Camera Space -> World Space
        // Camera Space Dir = (mouse.x * aspect * tanFov, mouse.y * tanFov, -1)
        // World Space Dir = x * xAxis + y * yAxis - 1 * zAxis

        const dx = this.mouse.x * aspect * tanFov;
        const dy = this.mouse.y * tanFov;

        const rayDir = [
            xAxis[0] * dx + yAxis[0] * dy - zAxis[0],
            xAxis[1] * dx + yAxis[1] * dy - zAxis[1],
            xAxis[2] * dx + yAxis[2] * dy - zAxis[2]
        ];

        // Normalize rayDir is not strictly necessary for intersection test if we are consistent, but good practice
        // Actually, ray casting logic: Origin + t * Dir

        // Time logic for buildings
        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        let closestDist = Infinity;
        let hoveredId = -1;

        for (let i = 0; i < this.instanceCount; i++) {
            const ix = this.buildingDataArray[i * 4 + 0];
            const iz = this.buildingDataArray[i * 4 + 1];
            const ih = this.buildingDataArray[i * 4 + 2];

            // Recompute dynamic Z position
            // Must match shader logic exactly
            let offset = (time * this.speed * 20.0) % 200.0;
            let zPos = iz + offset;
            if (zPos > 10.0) {
                zPos -= 200.0;
            }

            // Building AABB
            // Center X: ix
            // Base Y: 0, Top Y: ih
            // Center Z: zPos
            // Width/Depth: 1.0 -> Radius 0.5

            const minX = ix - 0.5;
            const maxX = ix + 0.5;
            const minY = 0;
            const maxY = ih;
            const minZ = zPos - 0.5;
            const maxZ = zPos + 0.5;

            // Ray-AABB Intersection (Slab method)
            let tMin = -Infinity;
            let tMax = Infinity;

            const bounds = [[minX, maxX], [minY, maxY], [minZ, maxZ]];

            let hit = true;
            for (let axis = 0; axis < 3; axis++) {
                const origin = eye[axis];
                const dir = rayDir[axis];

                if (Math.abs(dir) < 1e-6) {
                    if (origin < bounds[axis][0] || origin > bounds[axis][1]) {
                        hit = false;
                        break;
                    }
                } else {
                    let t1 = (bounds[axis][0] - origin) / dir;
                    let t2 = (bounds[axis][1] - origin) / dir;
                    if (t1 > t2) [t1, t2] = [t2, t1];

                    tMin = Math.max(tMin, t1);
                    tMax = Math.min(tMax, t2);

                    if (tMin > tMax || tMax < 0) {
                        hit = false;
                        break;
                    }
                }
            }

            if (hit) {
                if (tMin < closestDist) {
                    closestDist = tMin;
                    hoveredId = i;
                }
            }
        }

        if (hoveredId !== this.hoveredInstance) {
            this.hoveredInstance = hoveredId;
            if (hoveredId !== -1) {
                console.log(`Hovered building: ${hoveredId}`);
            }
        }
    }
}

if (typeof window !== 'undefined') {
    window.NeonCityExperiment = NeonCityExperiment;
}
