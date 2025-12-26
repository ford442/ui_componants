/**
 * Quantum Data Stream
 * Demonstrates WebGL2 and WebGPU working in tandem.
 * - WebGL2: Renders a twisting tunnel structure.
 * - WebGPU: Renders a compute-driven data stream flowing through the tunnel.
 */

export class QuantumDataStream {
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
        this.glNumIndices = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 50000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        console.log("QuantumDataStream: Initializing...");

        // 1. Initialize WebGL2 Layer (Background Tunnel)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Particle Stream)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("QuantumDataStream: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("QuantumDataStream: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("QuantumDataStream: WebGPU initialized successfully.");
        }

        // Initialize size before starting loop
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Twisting Tunnel)
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
            console.warn("QuantumDataStream: WebGL2 not supported.");
            return;
        }

        // Ensure initial size matches container (or at least isn't 0x0 default)
        this.glCanvas.width = this.container.clientWidth;
        this.glCanvas.height = this.container.clientHeight;
        this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);

        // Create a cylinder/tube mesh
        const segments = 64;
        const rings = 64;
        const radius = 2.0;
        const length = 20.0;

        const positions = [];
        const indices = [];

        for (let r = 0; r <= rings; r++) {
            const z = (r / rings) * length - length / 2;
            for (let s = 0; s <= segments; s++) {
                const theta = (s / segments) * Math.PI * 2;
                const x = Math.cos(theta) * radius;
                const y = Math.sin(theta) * radius;
                positions.push(x, y, z);
            }
        }

        for (let r = 0; r < rings; r++) {
            for (let s = 0; s < segments; s++) {
                const a = r * (segments + 1) + s;
                const b = (r + 1) * (segments + 1) + s;
                const c = (r + 1) * (segments + 1) + (s + 1);
                const d = r * (segments + 1) + (s + 1);

                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }

        this.glNumIndices = indices.length;

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

            out vec3 v_pos;
            out float v_depth;

            void main() {
                vec3 pos = a_position;

                // Twist effect
                float angle = pos.z * 0.2 + u_time * 0.5;
                float c = cos(angle);
                float s = sin(angle);
                float x = pos.x * c - pos.y * s;
                float y = pos.x * s + pos.y * c;
                pos.x = x;
                pos.y = y;

                // Move camera through tunnel
                pos.z -= u_time * 5.0;
                pos.z = mod(pos.z + 10.0, 20.0) - 10.0; // Infinite loop

                v_pos = pos;
                v_depth = pos.z;

                // Perspective projection
                float fov = 1.0;
                float aspect = u_resolution.x / u_resolution.y;
                float zNear = 0.1;
                float zFar = 100.0;
                float f = 1.0 / tan(fov / 2.0);

                // Manual projection matrix for simplicity
                mat4 projection = mat4(
                    f / aspect, 0.0, 0.0, 0.0,
                    0.0, f, 0.0, 0.0,
                    0.0, 0.0, (zFar + zNear) / (zNear - zFar), -1.0,
                    0.0, 0.0, (2.0 * zFar * zNear) / (zNear - zFar), 0.0
                );

                gl_Position = projection * vec4(pos, 1.0);
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            in float v_depth;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Grid/Tech pattern
                float gridX = step(0.95, fract(v_pos.x * 2.0 + u_time));
                float gridY = step(0.95, fract(v_pos.y * 2.0));
                float gridZ = step(0.9, fract(v_pos.z * 0.5));

                vec3 color = vec3(0.0, 0.1, 0.2); // Base dark blue

                // Highlights
                color += vec3(0.0, 0.5, 1.0) * (gridX + gridY) * 0.5;
                color += vec3(0.5, 0.0, 1.0) * gridZ;

                // Distance fog
                float fog = smoothstep(10.0, 0.0, abs(v_depth));

                outColor = vec4(color * fog, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

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
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, size
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Simple random function
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

                // Update life
                p.pos.w -= params.dt * 0.5;

                // Reset if dead
                if (p.pos.w <= 0.0) {
                    let seed = vec2f(f32(index), params.time);
                    p.pos.x = (rand(seed) * 2.0 - 1.0) * 0.5;
                    p.pos.y = (rand(seed + 1.0) * 2.0 - 1.0) * 0.5;
                    p.pos.z = 10.0; // Start far away
                    p.pos.w = 1.0; // Life

                    p.vel.x = (rand(seed + 2.0) * 2.0 - 1.0) * 2.0;
                    p.vel.y = (rand(seed + 3.0) * 2.0 - 1.0) * 2.0;
                    p.vel.z = -20.0; // Move towards camera fast
                }

                // Spiral motion matching the tunnel
                let angle = p.pos.z * 0.2 + params.time * 0.5;
                let radius = 1.0 + sin(p.pos.z * 0.5) * 0.2;

                // Add velocity
                p.pos.z += p.vel.z * params.dt;

                // Keep particles inside the tunnel radius approx
                // We add some turbulence
                p.pos.x += sin(params.time * 2.0 + f32(index)) * 0.01;
                p.pos.y += cos(params.time * 2.0 + f32(index)) * 0.01;

                particles[index] = p;
            }
        `;

        // RENDER SHADER
        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) size : f32,
            }

            struct Uniforms {
               resolution: vec2f,
               time: f32,
            }
            @group(0) @binding(2) var<uniform> uniforms : Uniforms;


            @vertex
            fn vs_main(@location(0) particlePos : vec4f, @location(1) particleVel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                var pos = particlePos.xyz;

                // Apply same twist as WebGL2 to match visual
                let angle = pos.z * 0.2 + uniforms.time * 0.5;
                let c = cos(angle);
                let s = sin(angle);
                let x = pos.x * c - pos.y * s;
                let y = pos.x * s + pos.y * c;
                pos.x = x;
                pos.y = y;

                // Projection
                let fov = 1.0;
                let aspect = uniforms.resolution.x / uniforms.resolution.y;
                let f = 1.0 / tan(fov / 2.0);
                let zNear = 0.1;
                let zFar = 100.0;

                // Basic perspective
                let z = pos.z;

                // Fade out if behind camera or too far
                let alpha = smoothstep(0.0, 5.0, z + 5.0) * smoothstep(10.0, 5.0, z);

                output.position = vec4f(
                    pos.x * f / aspect,
                    pos.y * f,
                    (z * (zFar + zNear) / (zNear - zFar)) + (2.0 * zFar * zNear) / (zNear - zFar),
                    -z
                );

                // Point size attenuation
                // output.position.w is -z
                // We fake point size by passing it to fragment shader if we were using point sprites,
                // but WebGPU defaults to 1px points unless we use quads.
                // For simplicity in this point-list topology, we accept 1px or rely on specific extensions,
                // BUT standard WebGPU 'point-list' renders 1px squares.
                // To get larger points we usually need to expand geometry in vertex shader (billboards),
                // but let's stick to 1px 'data bits' for the 'stream' look, or use a trick.
                // Actually, let's just make them bright pixels.

                output.color = vec4f(0.5, 1.0, 1.0, alpha);

                // To support variable point size we would need 'triangle-list' and 4 vertices per particle.
                // For this experiment, a dense stream of pixels is acceptable and 'matrix-like'.

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialize Particles
        // Each particle: pos(4 floats), vel(4 floats) = 8 floats * 4 bytes = 32 bytes
        const particleUnitSize = 32;
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 8 + 0] = (Math.random() * 2 - 1) * 0.5; // x
            initialParticleData[i * 8 + 1] = (Math.random() * 2 - 1) * 0.5; // y
            initialParticleData[i * 8 + 2] = Math.random() * 20.0 - 10.0; // z
            initialParticleData[i * 8 + 3] = Math.random(); // life

            initialParticleData[i * 8 + 4] = 0; // vx
            initialParticleData[i * 8 + 5] = 0; // vy
            initialParticleData[i * 8 + 6] = -20.0; // vz
            initialParticleData[i * 8 + 7] = Math.random(); // size/extra
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer for Compute
        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 2 floats padded to 16 bytes alignment preference or just 8
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Uniform Buffer for Render
        this.renderUniformBuffer = this.device.createBuffer({
             size: 16, // vec2 + float + pad
             usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group for Compute
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

        // Bind Group for Render (Needs access to uniforms for projection)
        const renderBindGroupLayout = this.device.createBindGroupLayout({
             entries: [
                 { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
             ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 2, resource: { buffer: this.renderUniformBuffer } }
            ]
        });


        // Pipelines
        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });

        // We use additive blending for the particles, no depth test needed
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // vel
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one',
                            operation: 'add',
                        },
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
            this.gl.uniform1f(timeLoc, time);

            const resLoc = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.0, 0.0, 0.05, 1.0);
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glNumIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            // Update simulation params
            const params = new Float32Array([0.016, time, 0.0, 0.0]); // Padded
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            // Update render uniforms
            const renderUniforms = new Float32Array([this.gpuCanvas.width, this.gpuCanvas.height, time, 0.0]);
            this.device.queue.writeBuffer(this.renderUniformBuffer, 0, renderUniforms);

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
            renderPass.setBindGroup(0, this.renderBindGroup); // Bind uniforms
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
