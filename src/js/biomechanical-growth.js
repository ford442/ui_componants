/**
 * Biomechanical Growth Experiment
 * Demonstrates WebGL2 and WebGPU working in tandem.
 * - WebGL2: Renders a pulsating biological membrane.
 * - WebGPU: Renders a compute-driven spore swarm overlay.
 */

class BiomechanicalGrowth {
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
        this.numParticles = options.numParticles || 10000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#1a0505'; // Dark fleshy background

        console.log("BiomechanicalGrowth: Initializing...");

        // 1. Initialize WebGL2 Layer (Background)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("BiomechanicalGrowth: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("BiomechanicalGrowth: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("BiomechanicalGrowth: WebGPU initialized successfully.");
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Biological Membrane)
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
            console.warn("BiomechanicalGrowth: WebGL2 not supported.");
            return;
        }

        // Create a dense grid for organic deformation
        const gridSize = 64;
        const positions = [];
        const indices = [];

        for (let y = 0; y <= gridSize; y++) {
            for (let x = 0; x <= gridSize; x++) {
                const u = x / gridSize;
                const v = y / gridSize;
                // Map to -1..1
                positions.push(u * 2 - 1, v * 2 - 1);
            }
        }

        for (let y = 0; y < gridSize; y++) {
            for (let x = 0; x < gridSize; x++) {
                const i = y * (gridSize + 1) + x;
                indices.push(i, i + 1, i + gridSize + 1);
                indices.push(i + 1, i + gridSize + 2, i + gridSize + 1);
            }
        }

        this.glIndexCount = indices.length;

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
            out float v_displacement;

            void main() {
                vec2 pos = a_position;

                // Organic movement
                float dist = length(pos);
                float angle = atan(pos.y, pos.x);

                // Pulsating effect
                float pulse = sin(u_time * 2.0 - dist * 4.0) * 0.1;

                // "Breathing" deformation
                float breathing = sin(u_time * 0.5) * 0.05;

                float z = pulse + breathing;

                // Add some twisting
                float twist = sin(angle * 3.0 + u_time) * 0.05 * dist;
                z += twist;

                v_pos = vec3(pos, z);
                v_displacement = z;

                gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            in float v_displacement;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Bio-texture generation
                vec2 coord = v_pos.xy * 4.0;

                float vein = sin(coord.x * 10.0 + sin(coord.y * 10.0 + u_time));
                vein += cos(coord.y * 8.0 + cos(coord.x * 8.0 + u_time));

                float tissue = smoothstep(-0.5, 0.8, vein);

                // Colors: Dark red background, bright red/pink veins
                vec3 baseColor = vec3(0.2, 0.05, 0.05);
                vec3 veinColor = vec3(0.8, 0.2, 0.3);
                vec3 highlightColor = vec3(1.0, 0.4, 0.5);

                vec3 color = mix(baseColor, veinColor, tissue);

                // Add pulse highlight
                float pulseHighlight = smoothstep(0.05, 0.1, v_displacement);
                color = mix(color, highlightColor, pulseHighlight * 0.5);

                // Vignette
                float dist = length(v_pos.xy);
                color *= 1.0 - smoothstep(0.5, 1.5, dist);

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

        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

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
    // WebGPU IMPLEMENTATION (Spore Particles)
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
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Simple pseudo-random function
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

                // Spore movement logic
                // Particles drift and are affected by a flow field based on the "veins"

                let angle = params.time * 0.1;
                let flow = vec2f(
                    sin(p.pos.y * 5.0 + params.time),
                    cos(p.pos.x * 5.0 + params.time)
                );

                // Add some chaotic movement
                p.vel = p.vel * 0.95 + flow * 0.001;

                // Repel from center slightly
                let dist = length(p.pos);
                if (dist < 0.2) {
                    p.vel += normalize(p.pos) * 0.01;
                }

                p.pos = p.pos + p.vel * params.dt;
                p.life -= params.dt * 0.2;

                // Respawn
                if (p.life <= 0.0 || abs(p.pos.x) > 1.2 || abs(p.pos.y) > 1.2) {
                    let r = rand(vec2f(params.time, f32(index)));
                    let theta = r * 6.28;
                    let d = 0.1 + rand(vec2f(r, params.time)) * 0.2;
                    p.pos = vec2f(cos(theta) * d, sin(theta) * d);
                    p.vel = vec2f(cos(theta), sin(theta)) * 0.1;
                    p.life = 1.0 + rand(vec2f(f32(index), params.time));
                }

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

            @vertex
            fn vs_main(
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f,
                @location(2) life : f32
            ) -> VertexOutput {
                var output : VertexOutput;
                output.position = vec4f(particlePos, 0.0, 1.0);

                // Color based on life (Green/Yellow to faded)
                let alpha = smoothstep(0.0, 0.2, life);
                output.color = vec4f(0.8, 1.0, 0.2, alpha);
                output.size = 2.0 + life * 2.0;

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialize Particles
        // Struct is pos(2f), vel(2f), life(1f), dummy(1f) = 6 floats -> aligned to 8 floats for 16-byte alignment?
        // vec2f is 8 bytes. f32 is 4 bytes.
        // pos: 0, vel: 8, life: 16. Next struct starts at 20? No, stride must be multiple of 16 usually for array elements if they contain vec3/4?
        // Actually for storage buffers standard layout rules apply.
        // Let's use 4 floats per particle to be safe if life fits in z/w of vel?
        // Or explicitly pad.
        // Struct size: 2*4 + 2*4 + 4 + 4(padding) = 24 bytes? No.
        // Let's just use 4 floats: pos.x, pos.y, vel.x, vel.y and put life elsewhere or pack it?
        // Let's stick to the struct defined in shader: pos(vec2), vel(vec2), life(f32), dummy(f32).
        // Size = 8 + 8 + 4 + 4 = 24 bytes. But WGSL arrays of structs have stride requirements.
        // Usually power of 2 is safest. 32 bytes (8 floats).

        const particleUnitSize = 32; // 8 floats * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            // Pos
            initialParticleData[i * 8 + 0] = (Math.random() * 2 - 1);
            initialParticleData[i * 8 + 1] = (Math.random() * 2 - 1);
            // Vel
            initialParticleData[i * 8 + 2] = (Math.random() - 0.5) * 0.1;
            initialParticleData[i * 8 + 3] = (Math.random() - 0.5) * 0.1;
            // Life
            initialParticleData[i * 8 + 4] = Math.random();
            // Padding
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
            size: 16, // 2 floats + padding to 16 bytes min uniform size
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

            // Clear with dark red/black
            this.gl.clearColor(0.1, 0.02, 0.02, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            const params = new Float32Array([0.016, time, 0, 0]); // Padded to 16 bytes
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
    window.BiomechanicalGrowth = BiomechanicalGrowth;
}
