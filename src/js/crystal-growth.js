/**
 * Crystal Growth Experiment
 * Demonstrates Hybrid WebGL2 + WebGPU implementation.
 * - WebGL2: Renders a central "Seed Crystal" (Octahedron wireframe).
 * - WebGPU: Renders a "Growth Medium" particle swarm that accretes onto the crystal.
 */

class CrystalGrowthExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0, isPressed: false };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndexBuffer = null;
        this.glIndexCount = 0;

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
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050010'; // Deep purple/black void

        console.log("CrystalGrowthExperiment: Initializing...");

        // 1. Initialize WebGL2 Layer (Seed)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Swarm)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CrystalGrowthExperiment: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("CrystalGrowthExperiment: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("CrystalGrowthExperiment: WebGPU initialized successfully.");
        }

        this.isActive = true;

        // Event Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);

        this.resize();
        this.animate();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Seed Crystal)
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
            console.warn("CrystalGrowthExperiment: WebGL2 not supported.");
            return;
        }

        // Generate Octahedron Geometry
        // 6 vertices: (+-1, 0, 0), (0, +-1, 0), (0, 0, +-1)
        const positions = [
             1,  0,  0,
            -1,  0,  0,
             0,  1,  0,
             0, -1,  0,
             0,  0,  1,
             0,  0, -1
        ];

        // 8 faces, but we want wireframe lines.
        // 12 edges.
        const indices = [
            0, 2, 2, 1, 1, 3, 3, 0, // Equatorial square edges? No.
            // Upper pyramid
            0, 2, 2, 1, 1, 5, 5, 0, // Wait, let's just list edges
            // Top vertex (2) connects to 0, 1, 4, 5
            2, 0, 2, 4, 2, 1, 2, 5,
            // Bottom vertex (3) connects to 0, 1, 4, 5
            3, 0, 3, 4, 3, 1, 3, 5,
            // Middle ring: 0-4, 4-1, 1-5, 5-0
            0, 4, 4, 1, 1, 5, 5, 0
        ];

        this.glIndexCount = indices.length;

        // Setup VAO and Buffers
        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.glIndexBuffer = indexBuffer;
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform vec2 u_resolution;

            out vec3 v_pos;

            mat4 rotationX(float angle) {
                return mat4(1.0, 0.0, 0.0, 0.0,
                            0.0, cos(angle), -sin(angle), 0.0,
                            0.0, sin(angle), cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            mat4 rotationY(float angle) {
                return mat4(cos(angle), 0.0, sin(angle), 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            -sin(angle), 0.0, cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            mat4 rotationZ(float angle) {
                return mat4(cos(angle), -sin(angle), 0.0, 0.0,
                            sin(angle), cos(angle), 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            void main() {
                v_pos = a_position;

                // Slow rotation
                mat4 rot = rotationY(u_time * 0.2) * rotationZ(u_time * 0.1);
                vec4 pos = rot * vec4(a_position * 0.5, 1.0); // Scale down a bit

                // Perspective projection aspect ratio fix
                float aspect = u_resolution.x / u_resolution.y;
                pos.x /= aspect;

                gl_Position = pos;
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Crystal Blue/Purple
                vec3 color = vec3(0.5, 0.8, 1.0);

                // Pulse
                float pulse = 0.5 + 0.5 * sin(u_time * 3.0);
                color = mix(color, vec3(0.8, 0.2, 1.0), pulse * 0.5);

                outColor = vec4(color, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

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
    // WebGPU IMPLEMENTATION (Growth Particles)
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
                state : f32, // 0: Free, 1: Accreting
                dummy : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mousePos : vec2f,
                isPressed : f32,
                dummy : f32,
                dummy2 : f32,
                dummy3 : f32,
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

                let center = vec2f(0.0, 0.0);
                let diff = center - p.pos;
                let dist = length(diff);
                let dir = normalize(diff);

                // Attraction to center
                let attraction = dir * 0.3 * params.dt;
                p.vel += attraction;

                // Mouse Repulsion
                let mouseDiff = p.pos - params.mousePos;
                let mouseDist = length(mouseDiff);
                if (mouseDist < 0.4) {
                    let force = normalize(mouseDiff) * (0.4 - mouseDist) * 5.0;
                    if (params.isPressed > 0.5) {
                        force *= 3.0;
                    }
                    p.vel += force * params.dt;
                }

                // Damping
                p.vel *= 0.96;

                // Movement
                p.pos += p.vel * params.dt;

                // Crystal Boundary (Octahedron approximation or just sphere for simplicity)
                // Let's assume the crystal is roughly radius 0.25 (since vertex is at 0.5 scaled in WebGL)
                if (dist < 0.2) {
                    // "Freeze" or bounce heavily
                    p.vel *= -0.5;
                    p.pos = center - dir * 0.21; // Push out slightly
                    p.state = 1.0; // Crystalized
                } else {
                    p.state = max(0.0, p.state - params.dt); // Decay state
                }

                // Respawn if too far
                if (dist > 1.2) {
                    let r = rand(vec2f(params.time, f32(index)));
                    let theta = r * 6.28;
                    p.pos = vec2f(cos(theta), sin(theta)) * 1.1;
                    p.vel = vec2f(0.0, 0.0);
                    p.state = 0.0;
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
            fn vs_main(
                @builtin(vertex_index) vertexIndex : u32,
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f,
                @location(2) particleState : f32
            ) -> VertexOutput {
                var output : VertexOutput;

                output.position = vec4f(particlePos, 0.0, 1.0);

                let speed = length(particleVel);

                // Color:
                // Far: Dark Blue
                // Fast: Cyan
                // Crystalized (State > 0): White/Gold

                var c = mix(vec3f(0.0, 0.1, 0.3), vec3f(0.0, 0.8, 1.0), speed * 2.0);

                if (particleState > 0.5) {
                    c = mix(c, vec3f(1.0, 0.9, 0.5), particleState);
                }

                output.color = vec4f(c, 1.0);

                // Point size is implicit in point-list (1px usually), we rely on bloom/density or just large count.
                // Or we can use gl_PointSize equivalent if supported, but WebGPU usually requires manual quad expansion for size.
                // We'll stick to 1px points for the "swarm" look.

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleUnitSize = 32; // 8 floats * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            const theta = Math.random() * Math.PI * 2;
            const r = 0.5 + Math.random() * 0.5;
            initialParticleData[i * 8 + 0] = Math.cos(theta) * r;
            initialParticleData[i * 8 + 1] = Math.sin(theta) * r;
            initialParticleData[i * 8 + 2] = 0; // vel
            initialParticleData[i * 8 + 3] = 0;
            initialParticleData[i * 8 + 4] = 0; // state
            initialParticleData[i * 8 + 5] = 0;
            initialParticleData[i * 8 + 6] = 0;
            initialParticleData[i * 8 + 7] = 0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
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
                        { shaderLocation: 2, offset: 16, format: 'float32' },  // state
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
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
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
            background: linear-gradient(90deg, rgba(200, 50, 50, 0.8), rgba(100, 20, 20, 0.9));
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-family: monospace;
            z-index: 10;
            pointer-events: none;
            border: 1px solid rgba(255,100,100,0.5);
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available &mdash; Core Only Mode";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // EVENTS & LOOP
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

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    }

    onMouseDown(e) {
        this.mouse.isPressed = true;
    }

    onMouseUp(e) {
        this.mouse.isPressed = false;
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016;

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const resLoc = this.gl.getUniformLocation(this.glProgram, 'u_resolution');

            this.gl.uniform1f(timeLoc, time);
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.02, 0.0, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.glIndexBuffer);
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            const params = new Float32Array([
                dt,
                time,
                this.mouse.x,
                this.mouse.y,
                this.mouse.isPressed ? 1.0 : 0.0,
                0, 0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

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
        this.container.removeEventListener('mousedown', this.handleMouseDown);
        window.removeEventListener('mouseup', this.handleMouseUp);

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
    window.CrystalGrowthExperiment = CrystalGrowthExperiment;
}

export { CrystalGrowthExperiment };
