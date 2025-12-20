/**
 * Cyber-Biology Interface Experiment
 * Demonstrates a hybrid rendering approach:
 * - WebGL2: Renders an organic, pulsating membrane surface using vertex displacement.
 * - WebGPU: Simulates "nanobots" or "data agents" swarming over the surface.
 */

class CyberBiologyExperiment {
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
        this.numIndices = 0;

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

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#0a0a0f';

        console.log("CyberBiology: Initializing...");

        // 1. Initialize WebGL2 Layer (Organic Surface)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Nanobots)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CyberBiology: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("CyberBiology: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("CyberBiology: WebGPU initialized successfully.");
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Organic Membrane)
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
            console.warn("CyberBiology: WebGL2 not supported.");
            return;
        }

        // Create a dense grid for the membrane
        const resolution = 100;
        const positions = [];
        const indices = [];

        for (let y = 0; y <= resolution; y++) {
            for (let x = 0; x <= resolution; x++) {
                const u = x / resolution;
                const v = y / resolution;
                // Center at 0,0, range -1 to 1
                positions.push(u * 2 - 1, v * 2 - 1);
            }
        }

        for (let y = 0; y < resolution; y++) {
            for (let x = 0; x < resolution; x++) {
                const i = y * (resolution + 1) + x;
                indices.push(i, i + 1, i + resolution + 1);
                indices.push(i + 1, i + resolution + 2, i + resolution + 1);
            }
        }

        this.numIndices = indices.length;

        // Buffers
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec2 a_position;
            uniform float u_time;
            out vec3 v_pos;
            out vec3 v_normal;

            // Simplex Noise (simplified)
            vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
            float snoise(vec2 v) {
                const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
                vec2 i  = floor(v + dot(v, C.yy) );
                vec2 x0 = v - i + dot(i, C.xx);
                vec2 i1;
                i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                vec4 x12 = x0.xyxy + C.xxzz;
                x12.xy -= i1;
                i = mod289(i);
                vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                    + i.x + vec3(0.0, i1.x, 1.0 ));
                vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
                m = m*m ;
                m = m*m ;
                vec3 x = 2.0 * fract(p * C.www) - 1.0;
                vec3 h = abs(x) - 0.5;
                vec3 ox = floor(x + 0.5);
                vec3 a0 = x - ox;
                m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
                vec3 g;
                g.x  = a0.x  * x0.x  + h.x  * x0.y;
                g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                return 130.0 * dot(m, g);
            }

            void main() {
                vec2 uv = a_position;

                // Organic movement
                float noise1 = snoise(uv * 2.0 + u_time * 0.2);
                float noise2 = snoise(uv * 4.0 - u_time * 0.5);

                float height = (noise1 * 0.6 + noise2 * 0.3);

                vec3 pos = vec3(uv.x, uv.y, height * 0.5);
                v_pos = pos;

                // Calc normal (approx)
                float d = 0.01;
                float h_x = (snoise((uv + vec2(d, 0.0)) * 2.0 + u_time * 0.2) * 0.6) +
                            (snoise((uv + vec2(d, 0.0)) * 4.0 - u_time * 0.5) * 0.3);
                float h_y = (snoise((uv + vec2(0.0, d)) * 2.0 + u_time * 0.2) * 0.6) +
                            (snoise((uv + vec2(0.0, d)) * 4.0 - u_time * 0.5) * 0.3);

                vec3 tangent = normalize(vec3(d, 0.0, (h_x - height) * 0.5));
                vec3 bitangent = normalize(vec3(0.0, d, (h_y - height) * 0.5));
                v_normal = normalize(cross(tangent, bitangent));

                // Perspective
                float z = 3.0 + pos.z; // Move camera back
                gl_Position = vec4(pos.x / z * 2.5, pos.y / z * 2.5, 0.0, 1.0);
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;
            in vec3 v_pos;
            in vec3 v_normal;
            uniform float u_time;
            out vec4 outColor;

            void main() {
                vec3 normal = normalize(v_normal);
                vec3 lightDir = normalize(vec3(sin(u_time), cos(u_time), 1.0));

                // Cell-shading style lighting
                float diff = max(dot(normal, lightDir), 0.0);

                vec3 baseColor = vec3(0.1, 0.8, 0.4); // Bio-green
                vec3 deepColor = vec3(0.0, 0.2, 0.1);

                // Rim light
                float rim = 1.0 - max(dot(normal, vec3(0.0, 0.0, 1.0)), 0.0);
                rim = pow(rim, 3.0);

                vec3 color = mix(deepColor, baseColor, diff * 0.8 + 0.2);
                color += vec3(0.5, 1.0, 0.8) * rim;

                // Grid overlay pattern
                vec2 grid = fract(v_pos.xy * 10.0);
                float lines = smoothstep(0.9, 0.95, grid.x) + smoothstep(0.9, 0.95, grid.y);
                color += vec3(0.2, 0.8, 0.5) * lines * 0.3;

                outColor = vec4(color, 1.0);
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
    // WebGPU IMPLEMENTATION (Nanobots)
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
                pad : f32,
                _pad2 : vec2f, // Pad to 32 bytes (8 floats) to match JS stride
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Simplified Simplex Noise for Compute Shader
            fn hash2(p: vec2f) -> vec2f {
                var p2 = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
                return -1.0 + 2.0 * fract(sin(p2) * 43758.5453123);
            }

            fn noise(p: vec2f) -> f32 {
                let K1 = 0.366025404; // (sqrt(3)-1)/2;
                let K2 = 0.211324865; // (3-sqrt(3))/6;

                let i = floor(p + (p.x + p.y) * K1);
                let a = p - i + (i.x + i.y) * K2;
                let o = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), a.x > a.y);
                let b = a - o + K2;
                let c = a - 1.0 + 2.0 * K2;

                let h = max(0.5 - vec3f(dot(a, a), dot(b, b), dot(c, c)), vec3f(0.0));
                let n = h * h * h * h * vec3f(dot(a, hash2(i + 0.0)), dot(b, hash2(i + o)), dot(c, hash2(i + 1.0)));

                return dot(n, vec3f(70.0));
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.numParticles}) {
                    return;
                }

                var p = particles[index];

                // Flow field based on noise
                let n = noise(p.pos * 3.0 + params.time * 0.2);
                let angle = n * 6.28;
                let flow = vec2f(cos(angle), sin(angle));

                // Attraction to center (0,0) loosely
                let toCenter = -p.pos * 0.5;

                // Update velocity
                p.vel = mix(p.vel, flow * 0.5 + toCenter, 0.05);
                p.pos += p.vel * params.dt;

                // Life cycle
                p.life -= params.dt * 0.5;
                if (p.life <= 0.0) {
                    p.life = 1.0;
                    p.pos = vec2f(
                        (fract(sin(params.time * 12.3 + f32(index)) * 43758.5453) - 0.5) * 2.0,
                        (fract(cos(params.time * 45.6 + f32(index)) * 31234.1234) - 0.5) * 2.0
                    );
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
                @location(1) life : f32,
            }

            struct SimParams {
                dt : f32,
                time : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            @vertex
            fn vs_main(
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f,
                @location(2) particleLife : f32
            ) -> VertexOutput {
                var output : VertexOutput;

                // Project 2D pos onto the "3D" surface matching WebGL2 logic approximately
                // WebGL logic:
                // float z = 3.0 + pos.z;
                // gl_Position = vec4(pos.x / z * 2.5, pos.y / z * 2.5, 0.0, 1.0);

                // Note: We don't have the exact height map here efficiently without sharing texture/buffer
                // So we approximate the depth for the particles to match the visual plane

                let z = 3.0; // Base depth
                output.position = vec4f(particlePos.x / z * 2.5, particlePos.y / z * 2.5, 0.0, 1.0);
                output.life = particleLife;

                // Color based on life
                let alpha = smoothstep(0.0, 0.2, particleLife) * smoothstep(1.0, 0.8, particleLife);
                output.color = vec4f(0.8, 1.0, 0.5, alpha); // Bright yellowish-green

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f, @location(1) life: f32) -> @location(0) vec4f {
                if (life <= 0.0) { discard; }
                return color;
            }
        `;

        // Initialize Particles
        // Structure: pos(2), vel(2), life(1), pad(1) = 6 floats * 4 bytes -> align to 8 floats for safety?
        // WGSL struct alignment rules: vec2 is 8 bytes.
        // pos (0), vel (8), life (16), pad (20) -> stride 24 bytes?
        // Wait, vec2<f32> has alignment 8.
        // offset 0: pos
        // offset 8: vel
        // offset 16: life
        // offset 20: pad
        // Total size 24.

        // Let's use 32 bytes stride (8 floats) just to be super safe and aligned to power of 2
        const particleUnitSize = 32;
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            const idx = i * 8;
            initialParticleData[idx + 0] = (Math.random() * 2 - 1); // x
            initialParticleData[idx + 1] = (Math.random() * 2 - 1); // y
            initialParticleData[idx + 2] = 0; // vx
            initialParticleData[idx + 3] = 0; // vy
            initialParticleData[idx + 4] = Math.random(); // life
            // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats (dt, time, pad, pad) to satisfy 16 byte alignment for uniform
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
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

        // Manually create pipeline layout to include the bind group for uniforms in vertex shader
        const renderPipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [computeBindGroupLayout]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: renderPipelineLayout,
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
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available &mdash; Running Hybrid Mode (WebGL2 Only)";
        this.container.appendChild(msg);
    }

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
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);

            this.gl.clearColor(0.02, 0.05, 0.03, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            // Enable depth testing for the surface
            this.gl.enable(this.gl.DEPTH_TEST);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            const params = new Float32Array([0.016, time, 0, 0]); // Padded to 16 bytes
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute Pass
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render Pass
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
            renderPass.setBindGroup(0, this.computeBindGroup); // Bind for uniforms in VS
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
    window.CyberBiologyExperiment = CyberBiologyExperiment;
}
