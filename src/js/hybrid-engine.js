/**
 * Hybrid Rendering Engine
 * Demonstrates WebGL2 and WebGPU working in tandem.
 * - WebGL2: Renders a retro synthwave grid terrain.
 * - WebGPU: Renders a compute-driven particle swarm overlay.
 */

class HybridEngine {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 10000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        console.log("HybridEngine: Initializing...");

        // 1. Initialize WebGL2 Layer (Background)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("HybridEngine: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("HybridEngine: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("HybridEngine: WebGPU initialized successfully.");
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Retro Grid)
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
        if (!this.gl) {
            console.warn("HybridEngine: WebGL2 not supported.");
            return;
        }

        // Setup simple quad
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec2 a_position;

            uniform float u_time;

            out vec3 v_pos;

            void main() {
                // Plane deformation
                vec2 pos = a_position;

                // Scale up for grid density (8x8 units across the view)
                pos *= 8.0;

                float z = pos.y + 10.0; // Distance

                // Add wave movement
                float wave = sin(pos.x * 0.5 + u_time) * cos(pos.y * 0.5 + u_time) * 0.5;

                v_pos = vec3(pos.x, pos.y, wave);

                // Perspective projection
                gl_Position = vec4(a_position.x, a_position.y, 0.0, 1.0);

                // Better faux-3D projection (simplified)
                gl_Position = vec4(pos.x / z, (pos.y + wave * 0.5) / z * 2.0, 0.0, 1.0);
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Grid calculation using derivatives
                vec2 coord = v_pos.xy;

                // Thicker lines for better visibility
                // Use fwidth to keep line width constant in screen space
                vec2 deriv = fwidth(coord);
                vec2 grid = abs(fract(coord - 0.5) - 0.5) / deriv;
                float line = min(grid.x, grid.y);

                // Use smoothstep for softer, thicker lines
                float alpha = 1.0 - smoothstep(0.0, 1.5, line);

                // Dynamic Color
                vec3 color = vec3(0.5 + 0.5*sin(u_time), 0.8, 1.0 + 0.2*cos(u_time * 0.5));

                // Distance fog
                float dist = length(v_pos.xy);
                float fog = 1.0 - smoothstep(2.0, 15.0, dist);

                outColor = vec4(color, alpha * fog);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;

        this.resizeGL();
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('WebGL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('WebGL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Compute Particles)
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
            background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        let adapter;
        try {
            adapter = await navigator.gpu.requestAdapter();
        } catch (e) {
            console.warn("WebGPU Adapter request failed:", e);
            this.gpuCanvas.remove();
            return false;
        }

        if (!adapter) {
            this.gpuCanvas.remove();
            return false;
        }

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // COMPUTE SHADER
        const computeShaderCode = `
            struct Particle {
                pos : vec2f,
                vel : vec2f,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.numParticles}) {
                    return;
                }

                var p = particles[index];

                // Update position
                p.pos = p.pos + p.vel * params.dt;

                // Wrap around / Bounce
                if (p.pos.x < -1.5) { p.pos.x = 1.5; }
                if (p.pos.x > 1.5) { p.pos.x = -1.5; }
                if (p.pos.y < -1.0) { p.pos.y = 1.0; }
                if (p.pos.y > 1.0) { p.pos.y = -1.0; }

                // Swirl
                let dist = length(p.pos);
                let angle = atan2(p.pos.y, p.pos.x) + params.time * 0.5;

                // Modify velocity with noise-like swirl
                p.vel.x += sin(angle * 3.0) * 0.002;
                p.vel.y += cos(angle * 2.0) * 0.002;

                // Damping
                p.vel = p.vel * 0.99;

                particles[index] = p;
            }
        `;

        // RENDER SHADER
        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) particlePos : vec2f, @location(1) particleVel : vec2f) -> VertexOutput {
                var output : VertexOutput;
                output.position = vec4f(particlePos, 0.0, 1.0);

                let speed = length(particleVel) * 10.0;
                // Color based on speed (Cyan to White)
                output.color = vec4f(0.0, 1.0 - speed * 0.5, 1.0, 1.0);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialize Particles
        const particleUnitSize = 16; // 4 floats * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 4);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 4 + 0] = (Math.random() * 2 - 1); // x
            initialParticleData[i * 4 + 1] = (Math.random() * 2 - 1); // y
            initialParticleData[i * 4 + 2] = (Math.random() - 0.5) * 0.01; // vx
            initialParticleData[i * 4 + 3] = (Math.random() - 0.5) * 0.01; // vy
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 8, // 2 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        // Pipelines
        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{ format: presentationFormat }],
            },
            primitive: { topology: 'point-list' },
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        // Check if message already exists
        if (this.container.querySelector('.webgpu-error')) return;

        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(90deg, rgba(200, 50, 50, 0.8), rgba(100, 20, 20, 0.9));
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-family: monospace;
            z-index: 10;
            pointer-events: none;
            border: 1px solid rgba(255,100,100,0.5);
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available &mdash; Running Hybrid Mode (WebGL2 Only)";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        // Handle Device Pixel Ratio for crisp rendering
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        const displayWidth = Math.floor(width * dpr);
        const displayHeight = Math.floor(height * dpr);

        this.resizeGL(displayWidth, displayHeight);
        this.resizeGPU(displayWidth, displayHeight);
    }

    resizeGL(width, height) {
        if (!this.glCanvas) return;

        // Only change if needed to avoid flicker
        if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
            this.glCanvas.width = width;
            this.glCanvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    resizeGPU(width, height) {
        if (!this.gpuCanvas) return;

        if (this.gpuCanvas.width !== width || this.gpuCanvas.height !== height) {
            this.gpuCanvas.width = width;
            this.gpuCanvas.height = height;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            this.gl.uniform1f(timeLoc, time);

            this.gl.clearColor(0.05, 0.05, 0.1, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            // Update simulation params
            const params = new Float32Array([0.016, time]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render
            const textureView = this.context.getCurrentTexture().createView();
            const renderPassDescriptor = {
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            };

            const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);

        window.removeEventListener('resize', this.handleResize);

        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }

        if (this.device) {
            this.device.destroy();
        }

        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.HybridEngine = HybridEngine;
}
