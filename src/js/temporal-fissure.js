/**
 * Temporal Fissure Experiment
 * Combines WebGL2 for a wireframe time-tunnel and WebGPU for chroniton particles.
 */

export class TemporalFissureExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.canvasSize = { width: 0, height: 0 };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.numTunnelSegments = 100;
        this.numTunnelSides = 32;
        this.tunnelIndexCount = 0;

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

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.mouse = { x: 0, y: 0 };

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050010';

        console.log("TemporalFissure: Initializing...");

        // 1. Initialize WebGL2 Layer (Background Tunnel)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("TemporalFissure: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("TemporalFissure: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("TemporalFissure: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Tunnel)
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
            console.warn("TemporalFissure: WebGL2 not supported.");
            return;
        }

        // Generate Tunnel Geometry
        const positions = [];
        const indices = [];

        for (let i = 0; i <= this.numTunnelSegments; i++) {
            const z = (i / this.numTunnelSegments) * 20.0 - 10.0; // Depth from -10 to 10
            for (let j = 0; j <= this.numTunnelSides; j++) {
                const theta = (j / this.numTunnelSides) * Math.PI * 2;
                const x = Math.cos(theta);
                const y = Math.sin(theta);
                positions.push(x, y, z);
            }
        }

        for (let i = 0; i < this.numTunnelSegments; i++) {
            for (let j = 0; j < this.numTunnelSides; j++) {
                const base = i * (this.numTunnelSides + 1) + j;
                const next = base + (this.numTunnelSides + 1);

                // Grid lines (Line strip simulation via triangle edges, but here we use LINES for wireframe look)
                indices.push(base, base + 1);
                indices.push(base, next);
            }
        }
        this.tunnelIndexCount = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;

            out float v_depth;

            void main() {
                vec3 pos = a_position;

                // Warp tunnel based on time and mouse
                float warp = sin(pos.z * 0.5 + u_time * 2.0) * 0.2;
                pos.x += warp + u_mouse.x * 0.5 * (pos.z + 10.0) / 20.0;
                pos.y += cos(pos.z * 0.5 + u_time * 1.5) * 0.2 + u_mouse.y * 0.5 * (pos.z + 10.0) / 20.0;

                // Infinite scroll effect
                pos.z = mod(pos.z + u_time * 5.0, 20.0) - 10.0;

                // Perspective projection
                float fov = 1.0;
                float aspect = u_resolution.x / u_resolution.y;
                float zToDist = 1.0 / (5.0 - pos.z); // Simple perspective

                gl_Position = vec4(pos.x * zToDist / aspect, pos.y * zToDist, pos.z, 1.0);
                if (pos.z > 4.0) gl_Position.w = 0.0; // Clip behind camera

                v_depth = pos.z;
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            uniform float u_time;
            out vec4 outColor;

            void main() {
                // Fade into distance
                float alpha = smoothstep(-10.0, 0.0, v_depth);

                // Pulsing color
                vec3 color = vec3(0.8, 0.2, 1.0); // Purple base
                color += vec3(0.2, 0.8, 1.0) * sin(u_time * 5.0 + v_depth) * 0.5;

                outColor = vec4(color, alpha * 0.5);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('TemporalFissure WebGL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('TemporalFissure WebGL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Chroniton Particles)
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

        const computeShaderCode = `
            struct Particle {
                pos : vec3f,
                vel : vec3f,
                life : f32,
                pad : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Random number generation
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                if (p.life <= 0.0) {
                    // Respawn at center (fissure source)
                    let r = rand(vec2f(params.time, f32(index))) * 0.5;
                    let theta = rand(vec2f(f32(index), params.time)) * 6.28;
                    p.pos = vec3f(cos(theta) * r, sin(theta) * r, -5.0); // Start deep in tunnel
                    p.vel = vec3f(cos(theta), sin(theta), 5.0 + rand(vec2f(p.pos.x, p.pos.y)) * 5.0); // Shoot forward
                    p.life = 1.0;
                } else {
                    // Update
                    p.pos = p.pos + p.vel * params.dt;

                    // Spiral effect
                    let spiral = vec3f(-p.pos.y, p.pos.x, 0.0) * 2.0;
                    p.vel = p.vel + spiral * params.dt;

                    // Mouse attraction/repulsion
                    let mousePos = vec3f(params.mouseX * 5.0, params.mouseY * 5.0, 0.0);
                    let diff = mousePos - p.pos;
                    let dist = length(diff);
                    if (dist < 3.0) {
                        p.vel = p.vel + normalize(diff) * 10.0 * params.dt;
                    }

                    p.life = p.life - params.dt * 0.5;
                }

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) life : f32,
            }

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            @vertex
            fn vs_main(
                @location(0) pos : vec3f,
                @location(1) vel : vec3f,
                @location(2) life : f32
            ) -> VertexOutput {
                var output : VertexOutput;

                // Perspective projection matching WebGL
                let zToDist = 1.0 / (5.0 - pos.z);

                output.position = vec4f(
                    pos.x * zToDist / params.aspect,
                    pos.y * zToDist,
                    0.0,
                    1.0
                );

                // Color based on life and velocity
                let speed = length(vel);
                let alpha = smoothstep(0.0, 0.2, life) * smoothstep(1.0, 0.8, life);
                output.color = vec4f(0.5, 0.8, 1.0, alpha); // Cyan/Blue
                output.life = life;

                // Size attenuation
                if (pos.z > 4.0 || pos.z < -10.0) {
                     output.position = vec4f(0.0, 0.0, 2.0, 1.0); // Clip
                }

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                if (color.a < 0.01) { discard; }
                return color;
            }
        `;

        const particleUnitSize = 32; // 8 floats * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 8 + 0] = 0; // x
            initialParticleData[i * 8 + 1] = 0; // y
            initialParticleData[i * 8 + 2] = -10; // z
            initialParticleData[i * 8 + 3] = 0; // vx
            initialParticleData[i * 8 + 4] = 0; // vy
            initialParticleData[i * 8 + 5] = 0; // vz
            initialParticleData[i * 8 + 6] = 0; // life
            initialParticleData[i * 8 + 7] = 0; // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 5 floats = 20 bytes -> padded to 32? (16 byte alignment)
                      // struct SimParams { dt, time, mouseX, mouseY } -> 16 bytes.
                      // Render shader has aspect -> +4 bytes = 20. Aligned to 32 works.
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        // Pipelines
        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }), // Need layout for uniform
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' }, // pos
                        { shaderLocation: 1, offset: 12, format: 'float32x3' }, // vel
                        { shaderLocation: 2, offset: 24, format: 'float32' },   // life
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{ format: presentationFormat, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                } }],
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
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(100, 20, 20, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: monospace;
            pointer-events: none;
        `;
        msg.innerHTML = "WebGPU Not Available (WebGL2 Only)";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

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
        if (width <= 0 || height <= 0) return;

        if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
            this.glCanvas.width = width;
            this.glCanvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    resizeGPU(width, height) {
        if (!this.gpuCanvas) return;
        if (width <= 0 || height <= 0) return;

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

            const resLoc = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            const mouseLoc = this.gl.getUniformLocation(this.glProgram, 'u_mouse');
            this.gl.uniform2f(mouseLoc, this.mouse.x, this.mouse.y);

            this.gl.clearColor(0.0, 0.0, 0.0, 0.0); // Transparent to let background show if any
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.tunnelIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0 && this.gpuCanvas.height > 0) {
            // Update simulation params
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, // dt
                time,  // time
                this.mouse.x, // mouseX
                this.mouse.y, // mouseY
                aspect, // aspect (matches struct in render shader? No, render shader has it separate? Wait)
                0, 0, 0 // padding
            ]);
            // struct SimParams in compute: dt, time, mouseX, mouseY (4 floats = 16 bytes)
            // struct SimParams in vertex: dt, time, mouseX, mouseY, aspect (5 floats -> 20 bytes -> 32 aligned)
            // I should use one struct definition or ensure alignment.
            // Let's check shaders again.
            // Compute: struct SimParams { dt, time, mouseX, mouseY } -> size 16.
            // Vertex: struct SimParams { dt, time, mouseX, mouseY, aspect } -> size 20 -> aligned 32.
            // The buffer size is 32.
            // I am writing 8 floats (32 bytes).
            // This covers both. 'aspect' is at float index 4.

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
            renderPass.setBindGroup(0, this.computeBindGroup); // Bind group 0 used in vertex
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
    window.TemporalFissureExperiment = TemporalFissureExperiment;
}
