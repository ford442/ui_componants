/**
 * Spectral Loom Experiment
 * Combines WebGL2 for vertical "Warp" threads and WebGPU for horizontal "Weft" particles.
 */

export class SpectralLoomExperiment {
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
        this.numThreads = 150;
        this.threadSegments = 64;
        this.indexCount = 0;

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

        // Interaction
        this.mouse = { x: 0, y: 0 };
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020205';

        // 1. Initialize WebGL2 Layer (Warp Threads)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Weft Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("SpectralLoom: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

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
    // WebGL2 IMPLEMENTATION (Warp Threads)
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

        // Generate Geometry: Vertical lines
        const positions = [];
        const indices = [];

        for (let i = 0; i < this.numThreads; i++) {
            // Normalized X from -1 to 1
            const x = (i / (this.numThreads - 1)) * 2.0 - 1.0;

            for (let j = 0; j <= this.threadSegments; j++) {
                const y = (j / this.threadSegments) * 2.0 - 1.0;
                // We'll modify Z and X slightly in shader
                positions.push(x, y, 0.0);
            }
        }

        for (let i = 0; i < this.numThreads; i++) {
            const offset = i * (this.threadSegments + 1);
            for (let j = 0; j < this.threadSegments; j++) {
                indices.push(offset + j, offset + j + 1);
            }
        }
        this.indexCount = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // VS: Deform lines into sine waves
        const vsSource = `#version 300 es
            in vec3 a_position;
            uniform float u_time;
            uniform vec2 u_mouse;

            void main() {
                vec3 pos = a_position;

                // Standing wave pattern
                float wave = sin(pos.y * 5.0 + u_time * 2.0) * 0.05;
                float wave2 = sin(pos.y * 10.0 - u_time * 3.0) * 0.02;

                // Mouse influence
                float d = distance(pos.xy, u_mouse);
                float mouseForce = smoothstep(0.5, 0.0, d) * 0.2;

                pos.x += wave + wave2 + (pos.x - u_mouse.x) * mouseForce;

                // 3D wiggle
                pos.z += sin(pos.y * 8.0 + u_time) * 0.1;

                gl_Position = vec4(pos, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            uniform float u_time;
            out vec4 outColor;

            void main() {
                // Vertical gradient
                float alpha = 0.3 + 0.2 * sin(u_time * 3.0);
                outColor = vec4(0.4, 0.8, 1.0, alpha); // Cyan warp
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
    }

    createGLProgram(vs, fs) {
        const vShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vShader, vs);
        this.gl.compileShader(vShader);
        if (!this.gl.getShaderParameter(vShader, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vShader));
            return null;
        }

        const fShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fShader, fs);
        this.gl.compileShader(fShader);
        if (!this.gl.getShaderParameter(fShader, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fShader));
            return null;
        }

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vShader);
        this.gl.attachShader(prog, fShader);
        this.gl.linkProgram(prog);
        return prog;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Weft Particles)
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

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied',
        });

        // Compute Shader: Move particles horizontally
        const computeCode = `
            struct Particle {
                pos: vec4f,
                vel: vec4f,
            }

            struct Params {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: Params;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Move horizontally
                p.pos.x = p.pos.x + p.vel.x * params.dt;

                // Weave pattern (Sine wave on Z)
                let weave = sin(p.pos.x * 10.0 + params.time) * 0.1;
                p.pos.z = weave;

                // Mouse influence (Repulsion)
                let d = distance(vec2f(p.pos.x, p.pos.y), vec2f(params.mouseX, params.mouseY));
                if (d < 0.3) {
                     let push = normalize(vec2f(p.pos.x, p.pos.y) - vec2f(params.mouseX, params.mouseY));
                     p.pos.y = p.pos.y + push.y * params.dt * 2.0;
                }

                // Wrap around X
                if (p.pos.x > 1.0) {
                    p.pos.x = -1.0;
                    p.pos.y = fract(sin(params.time * 0.1 + f32(i)) * 43758.5453) * 2.0 - 1.0;
                }
                if (p.pos.x < -1.0) {
                     p.pos.x = 1.0;
                }

                // Constrain Y
                if (p.pos.y > 1.0) { p.pos.y = -1.0; }
                if (p.pos.y < -1.0) { p.pos.y = 1.0; }

                particles[i] = p;
            }
        `;

        // Render Shader
        const drawCode = `
            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs_main(@location(0) pos: vec4f, @location(1) vel: vec4f) -> VertexOutput {
                var out: VertexOutput;
                // Simple orthographic projection
                out.position = vec4f(pos.x, pos.y, pos.z, 1.0);

                // Color based on velocity/direction
                let speed = abs(vel.x);
                out.color = vec4f(1.0, 0.2 + speed, 0.5, 0.8);
                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const particleSize = 32; // 2 * vec4f (16 bytes each) = 32 bytes
        const totalSize = this.numParticles * particleSize;
        const pData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            pData[i*8+0] = Math.random() * 2 - 1; // x
            pData[i*8+1] = Math.random() * 2 - 1; // y
            pData[i*8+2] = 0; // z
            pData[i*8+3] = 1; // w

            pData[i*8+4] = 0.2 + Math.random() * 0.3; // vx (speed)
            pData[i*8+5] = 0; // vy
            pData[i*8+6] = 0; // vz
            pData[i*8+7] = 0; // vw
        }

        this.particleBuffer = this.device.createBuffer({
            size: totalSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, pData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        const drawModule = this.device.createShaderModule({ code: drawCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                        { shaderLocation: 1, offset: 16, format: 'float32x4' } // vel
                    ]
                }]
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "WebGPU Not Supported";
        msg.style.cssText = "position: absolute; top: 10px; right: 10px; color: red; font-family: sans-serif;";
        this.container.appendChild(msg);
    }

    resize() {
        if (this.glCanvas) {
            this.glCanvas.width = this.container.clientWidth;
            this.glCanvas.height = this.container.clientHeight;
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = this.container.clientWidth;
            this.gpuCanvas.height = this.container.clientHeight;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) / 1000;

        // WebGL2 Render
        if (this.gl) {
            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.useProgram(this.glProgram);

            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.indexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // WebGPU Render
        if (this.device) {
            const params = new Float32Array([0.016, time, this.mouse.x, this.mouse.y]);
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
                    storeOp: 'store'
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(this.animate.bind(this));
    }

    destroy() {
        this.isActive = false;
        cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
    }
}
