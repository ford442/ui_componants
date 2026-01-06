/**
 * Cosmic String Instability
 * Demonstrates WebGL2 and WebGPU working in tandem.
 * - WebGL2: Renders a vibrating, glowing energy filament (the string) that distorts space.
 * - WebGPU: Simulates thousands of particles caught in the string's gravitational field.
 */

class CosmicStringExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // Interaction State
        this.mouse = { x: 0, y: 0 };
        this.isInteracting = false;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.numStringSegments = 200;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 30000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleTouchMove = this.onTouchMove.bind(this);
        this.handleTouchEnd = this.onTouchEnd.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        console.log("CosmicStringExperiment: Initializing...");

        // 1. Initialize WebGL2 Layer (Background String)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CosmicStringExperiment: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("CosmicStringExperiment: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("CosmicStringExperiment: WebGPU initialized successfully.");
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('touchmove', this.handleTouchMove, { passive: false });
        this.container.addEventListener('touchend', this.handleTouchEnd);
        this.container.addEventListener('mouseleave', this.handleTouchEnd);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = 1.0 - (e.clientY - rect.top) / rect.height; // WebGL Y is up

        // Map to [-1, 1]
        this.mouse.x = x * 2.0 - 1.0;
        this.mouse.y = y * 2.0 - 1.0;
        this.isInteracting = true;
    }

    onTouchMove(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = this.container.getBoundingClientRect();
        const x = (touch.clientX - rect.left) / rect.width;
        const y = 1.0 - (touch.clientY - rect.top) / rect.height;

        this.mouse.x = x * 2.0 - 1.0;
        this.mouse.y = y * 2.0 - 1.0;
        this.isInteracting = true;
    }

    onTouchEnd() {
        this.isInteracting = false;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Vibrating String)
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

        this.gl = this.glCanvas.getContext('webgl2', { alpha: false });
        if (!this.gl) {
            console.warn("CosmicStringExperiment: WebGL2 not supported.");
            return;
        }

        // Generate string geometry (vertical strip)
        const positions = [];
        const width = 0.05; // Base thickness
        for (let i = 0; i <= this.numStringSegments; i++) {
            const t = i / this.numStringSegments;
            const y = (t * 2.0) - 1.0; // -1 to 1

            // Left vertex
            positions.push(-width);
            positions.push(y);

            // Right vertex
            positions.push(width);
            positions.push(y);
        }

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec2 a_position;
            uniform float u_time;
            uniform float u_aspect;
            uniform vec2 u_mouse;
            uniform float u_interacting;

            out float v_intensity;
            out vec2 v_uv;

            void main() {
                vec2 pos = a_position;

                // Vertical coordinate (-1 to 1)
                float y = pos.y;

                // Vibration physics
                float freq1 = 5.0;
                float freq2 = 12.0;
                float freq3 = 25.0;

                // Standing wave pattern
                float displacement = sin(y * freq1 + u_time * 2.0) * 0.1
                                   + sin(y * freq2 - u_time * 5.0) * 0.05
                                   + sin(y * freq3 + u_time * 10.0) * 0.02;

                // Interaction: Pull towards mouse
                if (u_interacting > 0.5) {
                    float distY = abs(y - u_mouse.y);
                    float pull = exp(-distY * distY * 10.0) * 0.5; // Gaussian interaction window
                    displacement += (u_mouse.x - displacement) * pull * u_interacting;
                }

                // Modulate thickness based on energy flow
                float energy = sin(y * 10.0 - u_time * 8.0) * 0.5 + 0.5;
                float thickness = 0.02 + energy * 0.08;

                // Apply displacement to X
                float xOffset = displacement;

                // The vertex x is either -width or +width. We scale it by thickness.
                // We use sign(pos.x) to know which side we are on.
                float side = sign(pos.x);
                pos.x = xOffset + side * thickness;

                // Correct aspect ratio for width (make it look consistent on different screens)
                pos.x /= u_aspect;

                gl_Position = vec4(pos, 0.0, 1.0);

                v_intensity = energy;
                v_uv = vec2(side * 0.5 + 0.5, y * 0.5 + 0.5);
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in float v_intensity;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_mouse;
            uniform float u_interacting;

            out vec4 outColor;

            void main() {
                // Glow falloff from center of the string
                float dist = abs(v_uv.x - 0.5) * 2.0; // 0 at center, 1 at edges
                float glow = 1.0 - smoothstep(0.0, 1.0, dist);

                // Core color (hot white/blue)
                vec3 coreColor = vec3(0.8, 0.9, 1.0);

                // Outer glow (purple/magenta)
                vec3 glowColor = vec3(0.8, 0.2, 1.0);

                // Change color on interaction
                if (u_interacting > 0.5) {
                    glowColor = mix(glowColor, vec3(1.0, 0.4, 0.2), 0.5); // Orange-ish tint when touched
                }

                // Pulse intensity
                float pulse = 0.8 + 0.4 * sin(u_time * 10.0 + v_uv.y * 20.0);

                vec3 finalColor = mix(glowColor, coreColor, glow * glow);
                finalColor *= pulse * glow; // Fade out at edges

                // Add vertical streaks
                float streak = sin(v_uv.y * 100.0 + u_time * 20.0) * 0.1 + 0.9;
                finalColor *= streak;

                outColor = vec4(finalColor, 1.0);
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
    // WebGPU IMPLEMENTATION (Orbiting Particles)
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
                life : f32,
                dummy : f32, // Padding
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                isInteracting : f32,
                unused1 : f32,
                unused2 : f32,
                unused3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.numParticles}) {
                    return;
                }

                var p = particles[index];

                // Calculate string position (approximate match to WebGL)
                let y = p.pos.y;
                let t = params.time;
                var stringOffset = sin(y * 5.0 + t * 2.0) * 0.1
                                 + sin(y * 12.0 - t * 5.0) * 0.05;

                // Apply Mouse Interaction to String Position (approximate)
                if (params.isInteracting > 0.5) {
                     let distY = abs(y - params.mouseY);
                     let pull = exp(-distY * distY * 10.0) * 0.5;
                     stringOffset += (params.mouseX - stringOffset) * pull;
                }

                let dx = p.pos.x - stringOffset;
                let distSq = dx * dx + 0.001;
                let dist = sqrt(distSq);

                // Gravity towards string
                var force = -0.05 / distSq;
                if (force < -2.0) { force = -2.0; }

                let dirX = dx / dist;

                // Apply Gravity
                p.vel.x += dirX * force * params.dt;

                // Tangential force (Spiral)
                p.vel.y += sign(dx) * 2.0 * params.dt;

                // Mouse Repulsion/Attraction
                if (params.isInteracting > 0.5) {
                    let mDx = p.pos.x - params.mouseX;
                    let mDy = p.pos.y - params.mouseY;
                    let mDistSq = mDx*mDx + mDy*mDy + 0.001;

                    // Repel particles from mouse cursor
                    let repelForce = 0.5 / mDistSq;
                    if (repelForce > 5.0) { repelForce = 5.0; }

                    p.vel.x += (mDx / sqrt(mDistSq)) * repelForce * params.dt;
                    p.vel.y += (mDy / sqrt(mDistSq)) * repelForce * params.dt;
                }

                // Damping
                p.vel = p.vel * 0.98;

                // Update Pos
                p.pos = p.pos + p.vel * params.dt;

                // Boundaries - Respawn
                if (abs(p.pos.x) > 2.0 || abs(p.pos.y) > 1.2 || dist < 0.005) {
                    p.pos.x = (rand(vec2f(params.time, f32(index))) - 0.5) * 3.0;
                    p.pos.y = (rand(vec2f(f32(index), params.time)) - 0.5) * 2.0;
                    p.vel = vec2f(0.0, 0.0);
                }

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

                let speed = length(particleVel);
                // Color mapping
                let r = 0.2 + speed * 2.0;
                let g = 0.1 + speed * 0.5;
                let b = 1.0;

                output.color = vec4f(r, g, b, 1.0);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialize Particles
        // Struct: pos(2f), vel(2f), life(1f), dummy(1f) = 6 floats -> 24 bytes.
        // But stride must be aligned. Let's assume standard layout.
        const particleUnitSize = 24;
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 6);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 6 + 0] = (Math.random() * 4 - 2);
            initialParticleData[i * 6 + 1] = (Math.random() * 2 - 1);
            initialParticleData[i * 6 + 2] = 0;
            initialParticleData[i * 6 + 3] = 0;
            initialParticleData[i * 6 + 4] = Math.random();
            initialParticleData[i * 6 + 5] = 0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
        // Size needs to be 16-byte aligned.
        // 8 floats = 32 bytes.
        this.simParamBuffer = this.device.createBuffer({
            size: 32,
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
                        { shaderLocation: 0, offset: 0, format: 'float32x2' }, // pos
                        { shaderLocation: 1, offset: 8, format: 'float32x2' }, // vel
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
                    }
                }],
            },
            primitive: { topology: 'point-list' },
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;

        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(90deg, rgba(100, 50, 200, 0.8), rgba(50, 20, 100, 0.9));
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-family: monospace;
            z-index: 10;
            pointer-events: none;
            border: 1px solid rgba(150,100,255,0.5);
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available &mdash; Running String Simulation Only";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
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
            const aspectLoc = this.gl.getUniformLocation(this.glProgram, 'u_aspect');
            const mouseLoc = this.gl.getUniformLocation(this.glProgram, 'u_mouse');
            const interactingLoc = this.gl.getUniformLocation(this.glProgram, 'u_interacting');

            this.gl.uniform1f(timeLoc, time);
            this.gl.uniform1f(aspectLoc, this.glCanvas.width / this.glCanvas.height);
            this.gl.uniform2f(mouseLoc, this.mouse.x, this.mouse.y);
            this.gl.uniform1f(interactingLoc, this.isInteracting ? 1.0 : 0.0);

            this.gl.clearColor(0.02, 0.01, 0.05, 1.0); // Deep purple/black
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            // Enable additive blending for glow
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, (this.numStringSegments + 1) * 2);

            this.gl.disable(this.gl.BLEND);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            // Update simulation params
            // Struct SimParams: dt, time, mouseX, mouseY, isInteracting, unused1, unused2, unused3
            const params = new Float32Array([
                0.016,
                time,
                this.mouse.x,
                this.mouse.y,
                this.isInteracting ? 1.0 : 0.0,
                0.0,
                0.0,
                0.0
            ]);
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
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('touchmove', this.handleTouchMove);
        this.container.removeEventListener('touchend', this.handleTouchEnd);
        this.container.removeEventListener('mouseleave', this.handleTouchEnd);

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
    window.CosmicStringExperiment = CosmicStringExperiment;
}
