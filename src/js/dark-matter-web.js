/**
 * Dark Matter Web Experiment
 * Demonstrates Hybrid WebGL2 + WebGPU implementation.
 * - WebGL2: Renders a "Cosmic Web" of interconnected nodes (GL_LINES).
 * - WebGPU: Simulates particles (Dark Energy) flowing along the filaments of the web.
 */

class DarkMatterWebExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // Graph Data
        this.nodeCount = 40;
        this.connectionDistance = 0.6;
        this.nodes = null;
        this.lines = null; // Float32Array [x1, y1, z1, x2, y2, z2, ...] for WebGL
        this.gpuLines = null; // Float32Array [x1, y1, z1, 0, x2, y2, z2, 0, ...] for WebGPU alignment

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glVertexCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.lineBuffer = null;
        this.numParticles = options.numParticles || 30000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        // Generate Graph Data
        this.initGraph();

        console.log("DarkMatterWeb: Initializing...");

        // 1. Initialize WebGL2 Layer (The Web)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (The Flow)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("DarkMatterWeb: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("DarkMatterWeb: WebGPU initialized successfully.");
        }

        this.isActive = true;
        window.addEventListener('resize', this.handleResize);
        this.resize();
        this.animate();
    }

    initGraph() {
        const nodes = [];
        // Generate random nodes in a sphere
        for (let i = 0; i < this.nodeCount; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);
            const r = Math.pow(Math.random(), 0.5) * 1.5; // Bias towards center slightly

            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);
            nodes.push({ x, y, z });
        }

        const lines = [];
        const gpuLines = []; // Padding for vec4 alignment

        for (let i = 0; i < this.nodeCount; i++) {
            for (let j = i + 1; j < this.nodeCount; j++) {
                const n1 = nodes[i];
                const n2 = nodes[j];
                const dx = n1.x - n2.x;
                const dy = n1.y - n2.y;
                const dz = n1.z - n2.z;
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

                if (dist < this.connectionDistance) {
                    // WebGL: Packed vec3
                    lines.push(n1.x, n1.y, n1.z);
                    lines.push(n2.x, n2.y, n2.z);

                    // WebGPU: vec4 (xyz + pad)
                    gpuLines.push(n1.x, n1.y, n1.z, 0);
                    gpuLines.push(n2.x, n2.y, n2.z, 0);
                }
            }
        }

        this.lines = new Float32Array(lines);
        this.gpuLines = new Float32Array(gpuLines);
        this.glVertexCount = lines.length / 3;
        this.lineCount = gpuLines.length / 8; // Each line is 2 points * 4 floats
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (The Web)
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
            console.warn("DarkMatterWeb: WebGL2 not supported.");
            return;
        }

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.lines, this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec3 a_position;
            uniform float u_time;
            uniform float u_aspect;

            out float v_depth;

            mat4 rotationY(float angle) {
                return mat4(cos(angle), 0.0, sin(angle), 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            -sin(angle), 0.0, cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            mat4 rotationX(float angle) {
                return mat4(1.0, 0.0, 0.0, 0.0,
                            0.0, cos(angle), -sin(angle), 0.0,
                            0.0, sin(angle), cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            void main() {
                vec3 pos = a_position;

                // Slow rotation
                mat4 rot = rotationY(u_time * 0.1) * rotationX(u_time * 0.05);
                vec4 p = rot * vec4(pos, 1.0);

                // Perspective
                float scale = 1.0 / (p.z + 4.0);
                p.x *= scale / u_aspect;
                p.y *= scale;

                gl_Position = vec4(p.x, p.y, 0.0, 1.0);
                v_depth = p.z;
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            out vec4 outColor;

            void main() {
                // Distance fade
                float alpha = 1.0 - smoothstep(-1.0, 1.0, v_depth);
                outColor = vec4(0.3, 0.4, 0.6, alpha * 0.3); // Dim blue-ish web
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;

        // Enable blending
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE); // Additive
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
    // WebGPU IMPLEMENTATION (The Flow)
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

        let adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // 1. Storage Buffer: Lines
        // Each line is 2 points: vec4 start, vec4 end. Total 32 bytes per line.
        const lineBufferSize = this.gpuLines.byteLength;
        this.lineBuffer = this.device.createBuffer({
            size: lineBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.lineBuffer, 0, this.gpuLines);

        // 2. Storage Buffer: Particles
        // struct Particle { pos: vec4f, vel: vec4f }
        // We use 'vel' to store: x=lineIndex, y=progress(0..1), z=speed, w=unused
        const particleUnitSize = 32; // 8 floats
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialData = new Float32Array(this.numParticles * 8);

        for(let i=0; i<this.numParticles; i++) {
            // Pick random line
            const lineIdx = Math.floor(Math.random() * this.lineCount);
            const progress = Math.random();
            const speed = 0.5 + Math.random() * 1.5;

            // Calc init pos
            const startIdx = lineIdx * 8; // 8 floats per line (2 vec4s)
            const endIdx = startIdx + 4;

            // Linear interp
            const t = progress;
            const x = this.gpuLines[startIdx] * (1-t) + this.gpuLines[endIdx] * t;
            const y = this.gpuLines[startIdx+1] * (1-t) + this.gpuLines[endIdx+1] * t;
            const z = this.gpuLines[startIdx+2] * (1-t) + this.gpuLines[endIdx+2] * t;

            initialData[i*8+0] = x;
            initialData[i*8+1] = y;
            initialData[i*8+2] = z;
            initialData[i*8+3] = 1.0; // alpha/life

            initialData[i*8+4] = lineIdx;
            initialData[i*8+5] = progress;
            initialData[i*8+6] = speed;
            initialData[i*8+7] = 0; // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        // 3. Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 16, // dt, time, aspect, count
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // SHADERS
        const computeShaderCode = `
            struct Line {
                start: vec4f,
                end: vec4f,
            }

            struct Particle {
                pos: vec4f, // xyz, life
                data: vec4f, // lineIndex, progress, speed, pad
            }

            struct SimParams {
                dt: f32,
                time: f32,
                aspect: f32,
                lineCount: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<storage, read> lines: array<Line>;
            @group(0) @binding(2) var<uniform> params: SimParams;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                var lineIdx = u32(p.data.x);
                var progress = p.data.y;
                var speed = p.data.z;

                // Move
                progress += speed * params.dt;

                // Check end
                if (progress >= 1.0) {
                    // Pick new random line
                    let r = rand(vec2f(params.time, f32(index)));
                    lineIdx = u32(r * params.lineCount);
                    // Ensure valid
                    if (lineIdx >= u32(params.lineCount)) { lineIdx = 0u; }

                    progress = 0.0;
                    p.data.x = f32(lineIdx);
                }

                p.data.y = progress;

                // Update Pos
                let l = lines[lineIdx];
                let pos = mix(l.start.xyz, l.end.xyz, progress);
                p.pos = vec4f(pos, 1.0);

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct SimParams {
                dt: f32,
                time: f32,
                aspect: f32,
                lineCount: f32,
            }

            @group(0) @binding(2) var<uniform> params: SimParams;

            // Matrix functions for rotation (Must match WebGL)
            fn rotationY(angle: f32) -> mat4x4f {
                let c = cos(angle);
                let s = sin(angle);
                return mat4x4f(
                    c, 0.0, -s, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    s, 0.0, c, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            }

            fn rotationX(angle: f32) -> mat4x4f {
                let c = cos(angle);
                let s = sin(angle);
                return mat4x4f(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, c, -s, 0.0,
                    0.0, s, c, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            }

            @vertex
            fn vs_main(@location(0) particlePos : vec4f, @location(1) particleData : vec4f) -> VertexOutput {
                var output : VertexOutput;

                var pos = vec4f(particlePos.xyz, 1.0);

                // Apply same rotation as WebGL
                let rot = rotationY(params.time * 0.1) * rotationX(params.time * 0.05);
                pos = rot * pos;

                // Perspective
                let scale = 1.0 / (pos.z + 4.0);
                pos.x = pos.x * scale / params.aspect;
                pos.y = pos.y * scale;
                pos.z = 0.0;
                pos.w = 1.0;

                output.position = pos;

                // Color based on speed
                let speed = particleData.z;
                let t = (speed - 0.5) / 1.5; // norm
                output.color = mix(vec4f(0.0, 0.5, 1.0, 0.8), vec4f(1.0, 0.2, 0.8, 1.0), t);

                // Point size trick for WebGPU?
                // WebGPU 'point-list' renders 1px points usually.
                // For this demo, small points are fine.

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Create Pipelines
        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        const drawModule = this.device.createShaderModule({ code: drawShaderCode });

        // Bind Group Layout
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // data
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

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.lineBuffer } },
                { binding: 2, resource: { buffer: this.simParamBuffer } },
            ],
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px;
            color: #fff; background: rgba(100,0,0,0.8); padding: 10px;
            border-radius: 5px; font-family: sans-serif; pointer-events: none;
        `;
        msg.innerHTML = "WebGPU Not Available - Web Mode Only";
        this.container.appendChild(msg);
    }

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        if (this.glCanvas) {
            this.glCanvas.width = width * dpr;
            this.glCanvas.height = height * dpr;
            this.gl.viewport(0, 0, width * dpr, height * dpr);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = width * dpr;
            this.gpuCanvas.height = height * dpr;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const aspect = width / height;

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const aspectLoc = this.gl.getUniformLocation(this.glProgram, 'u_aspect');

            this.gl.uniform1f(timeLoc, time);
            this.gl.uniform1f(aspectLoc, aspect);

            this.gl.clearColor(0.0, 0.0, 0.02, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.LINES, 0, this.glVertexCount);
        }

        // 2. WebGPU Render
        if (this.device && this.context && this.renderPipeline) {
            const params = new Float32Array([0.016, time, aspect, this.lineCount]);
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
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });

            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup); // Re-use same bindgroup as it has uniforms too?
            // Wait, my renderPipeline layout matches computeBindGroupLayout?
            // Yes: `layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] })`
            // And vertex shader needs Uniforms at binding 2.

            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

if (typeof window !== 'undefined') {
    window.DarkMatterWebExperiment = DarkMatterWebExperiment;
}

export { DarkMatterWebExperiment };
