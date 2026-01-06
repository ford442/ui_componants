/**
 * Cherenkov Radiation Experiment
 * Hybrid WebGL2 (Volumetric Reactor Pool) + WebGPU (High Energy Particle Physics)
 */

class CherenkovRadiation {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };

        // WebGL2 State (Background Pool)
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;

        // WebGPU State (Particles)
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 100000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000510'; // Deep blue black

        console.log("CherenkovRadiation: Initializing...");

        // 1. Initialize WebGL2 Layer (Background Pool)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CherenkovRadiation: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("CherenkovRadiation: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("CherenkovRadiation: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Background Pool)
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
            console.warn("CherenkovRadiation: WebGL2 not supported.");
            return;
        }

        // Fullscreen Quad
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_mouse;
            uniform vec2 u_resolution;

            out vec4 outColor;

            // Simple noise
            float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }
            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));
                vec2 u = f * f * (3.0 - 2.0 * f);
                return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
            }

            void main() {
                vec2 uv = v_uv;
                if (u_resolution.y > 0.0) {
                    uv.x *= u_resolution.x / u_resolution.y;
                }

                // Deep reactor blue gradient
                vec3 col = vec3(0.0, 0.05, 0.15) * (1.0 - length(v_uv) * 0.5);

                // "Radiation" caustic glow
                float n = noise(uv * 4.0 + vec2(0.0, u_time * 0.2));
                float n2 = noise(uv * 8.0 - vec2(0.0, u_time * 0.5));

                float glow = (n + n2) * 0.5;

                // Mouse interaction glow
                float dist = length(uv - vec2(u_mouse.x * (u_resolution.x/u_resolution.y), u_mouse.y));
                float reaction = smoothstep(0.5, 0.0, dist);

                col += vec3(0.0, 0.2, 0.8) * glow * 0.2;
                col += vec3(0.4, 0.8, 1.0) * reaction * 0.4;

                // Scanlines
                col *= 0.9 + 0.1 * sin(v_uv.y * 200.0 + u_time * 10.0);

                outColor = vec4(col, 1.0);
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
    // WebGPU IMPLEMENTATION (Particle Physics)
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

        // WGSL Compute Shader
        const computeShaderCode = `
            struct Particle {
                pos : vec2f,
                vel : vec2f,
                life : f32, // 0-1
                pad : f32,  // Alignment
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                pad : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Pseudo-random
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Mouse interaction (Attractor)
                let mPos = vec2f(params.mouseX * params.aspect, params.mouseY);
                let pPosAspect = vec2f(p.pos.x * params.aspect, p.pos.y);

                let dir = mPos - pPosAspect;
                let dist = length(dir);
                let force = normalize(dir) / (dist * dist + 0.01) * 0.05;

                // Physics
                p.vel = p.vel + force * params.dt;

                // Constant "Radiation" jitter
                let jitter = vec2f(rand(p.pos + params.time) - 0.5, rand(p.pos - params.time) - 0.5) * 0.1;
                p.vel = p.vel + jitter * params.dt;

                // Drag
                p.vel = p.vel * 0.98;

                // Move
                p.pos = p.pos + p.vel * params.dt;

                // Respawn logic
                p.life = p.life - params.dt * 0.5;

                // Check bounds and life
                if (p.life <= 0.0 || abs(p.pos.x) > 1.2 || abs(p.pos.y) > 1.2) {
                    // Respawn near center or mouse
                    let angle = rand(vec2f(params.time, f32(index))) * 6.28;
                    let r = rand(vec2f(f32(index), params.time)) * 0.2;

                    // Bias spawn towards mouse if mouse is active (not 0,0 usually)
                    // Simplified: just spawn in a ring
                    p.pos = vec2f(cos(angle) * r, sin(angle) * r);

                    // Initial burst velocity
                    p.vel = vec2f(cos(angle), sin(angle)) * (0.5 + r);
                    p.life = 1.0;
                }

                particles[index] = p;
            }
        `;

        // WGSL Render Shader
        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f,
                @location(2) life : f32
            ) -> VertexOutput {
                var output : VertexOutput;
                output.position = vec4f(particlePos, 0.0, 1.0);

                let speed = length(particleVel);

                // Cherenkov Blue -> White
                // Low speed = dark blue
                // High speed = bright cyan/white

                let blue = vec3f(0.0, 0.1, 0.8);
                let cyan = vec3f(0.0, 0.8, 1.0);
                let white = vec3f(1.0, 1.0, 1.0);

                var c = mix(blue, cyan, smoothstep(0.0, 0.5, speed));
                c = mix(c, white, smoothstep(0.5, 1.2, speed));

                // Fade out by life
                let alpha = smoothstep(0.0, 0.2, life);

                output.color = vec4f(c, alpha);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                // Point shape
                return color;
            }
        `;

        // Buffer Setup
        const particleUnitSize = 24; // 6 floats (pos:2, vel:2, life:1, pad:1) * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 6);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 6 + 0] = (Math.random() * 2 - 1);
            initialParticleData[i * 6 + 1] = (Math.random() * 2 - 1);
            initialParticleData[i * 6 + 2] = 0;
            initialParticleData[i * 6 + 3] = 0;
            initialParticleData[i * 6 + 4] = Math.random(); // life
            initialParticleData[i * 6 + 5] = 0; // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 8 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

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
                        { shaderLocation: 2, offset: 16, format: 'float32' },  // life
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
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }],
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px;
            background: rgba(100, 20, 20, 0.9); color: white;
            padding: 8px 16px; border-radius: 4px; font-family: monospace;
            pointer-events: none;
        `;
        msg.innerHTML = "WebGPU Not Available (WebGL2 Only)";
        this.container.appendChild(msg);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width === 0 || height === 0) return;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

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

        // Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGPU
        if (this.device && this.renderPipeline && this.gpuCanvas.width > 0) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, time, this.mouse.x, this.mouse.y, aspect, 0, 0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            const textureView = this.context.getCurrentTexture().createView();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
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
        window.removeEventListener('mousemove', this.handleMouseMove);
        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

export { CherenkovRadiation };

if (typeof window !== 'undefined') {
    window.CherenkovRadiation = CherenkovRadiation;
}
