/**
 * Hyper-Dimensional Cube (Tesseract) Experiment
 * Combines WebGL2 for the wireframe 4D structure and WebGPU for inter-dimensional particles.
 */

class HyperCubeExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };

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
        this.numParticles = options.numParticles || 50000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        console.log("HyperCubeExperiment: Initializing...");

        // 1. Initialize WebGL2 Layer (Background Structure)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("HyperCubeExperiment: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("HyperCubeExperiment: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("HyperCubeExperiment: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        // Normalize mouse coordinates to [-1, 1] (UV space centered)
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Wireframe Tesseract)
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
            console.warn("HyperCubeExperiment: WebGL2 not supported.");
            return;
        }

        // Tesseract Geometry Generation
        // 16 vertices, 4D coords
        const vertices = [];
        for (let i = 0; i < 16; i++) {
            const x = (i & 1) ? 1 : -1;
            const y = (i & 2) ? 1 : -1;
            const z = (i & 4) ? 1 : -1;
            const w = (i & 8) ? 1 : -1;
            vertices.push(x, y, z, w);
        }

        // Edges: Connect vertices that differ by 1 bit
        const indices = [];
        for (let i = 0; i < 16; i++) {
            for (let bit = 0; bit < 4; bit++) {
                const neighbor = i ^ (1 << bit);
                if (i < neighbor) { // Avoid duplicates
                    indices.push(i, neighbor);
                }
            }
        }
        this.numIndices = indices.length;

        // Buffers
        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Shader Source
        const vsSource = `#version 300 es
            in vec4 a_position;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;

            out float v_depth;

            // 4D Rotations
            void rotateZW(inout vec4 p, float t) {
                float c = cos(t), s = sin(t);
                float z = p.z, w = p.w;
                p.z = z*c - w*s;
                p.w = z*s + w*c;
            }
            void rotateXW(inout vec4 p, float t) {
                float c = cos(t), s = sin(t);
                float x = p.x, w = p.w;
                p.x = x*c - w*s;
                p.w = x*s + w*c;
            }
            void rotateYW(inout vec4 p, float t) {
                float c = cos(t), s = sin(t);
                float y = p.y, w = p.w;
                p.y = y*c - w*s;
                p.w = y*s + w*c;
            }
            // 3D Rotations
            void rotateXY(inout vec4 p, float t) {
                float c = cos(t), s = sin(t);
                float x = p.x, y = p.y;
                p.x = x*c - y*s;
                p.y = x*s + y*c;
            }
            void rotateXZ(inout vec4 p, float t) {
                float c = cos(t), s = sin(t);
                float x = p.x, z = p.z;
                p.x = x*c - z*s;
                p.z = x*s + z*c;
            }

            void main() {
                vec4 p = a_position;

                // Animate rotations
                float t = u_time * 0.5;
                rotateZW(p, t * 0.7);
                rotateXW(p, t * 0.5);
                rotateYW(p, t * 0.3);

                // Mouse influence
                rotateXY(p, u_mouse.x * 2.0);
                rotateXZ(p, u_mouse.y * 2.0);

                // 4D to 3D Projection (Stereographic-ish)
                float wDistance = 2.5;
                float wScale = 1.0 / (wDistance - p.w);
                vec3 p3 = p.xyz * wScale;

                // 3D to 2D Projection (Perspective)
                float aspect = u_resolution.x / u_resolution.y;
                vec3 camPos = vec3(0.0, 0.0, -3.0);
                vec3 viewPos = p3 - camPos; // Camera is at -3 looking at 0

                // Perspective
                float fov = 1.2;
                float scale = 1.0 / tan(fov * 0.5);

                float x = viewPos.x * scale / aspect;
                float y = viewPos.y * scale;
                float z = viewPos.z; // Depth

                // Basic perspective divide (assuming viewPos.z is positive distance from camera)
                // In this setup, p3 is near 0. camPos is -3. viewPos.z is ~3.

                gl_Position = vec4(x, y, 0.0, z); // z is w-component for divide

                v_depth = p.w; // Use 4th dim depth for color
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            out vec4 outColor;

            void main() {
                // Color based on W-depth
                vec3 col1 = vec3(0.0, 1.0, 0.8); // Cyan
                vec3 col2 = vec3(1.0, 0.0, 0.5); // Magenta

                float t = smoothstep(-1.0, 1.0, v_depth);
                vec3 col = mix(col1, col2, t);

                // Fade out distant edges
                float alpha = 0.6 + 0.4 * t;

                outColor = vec4(col, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 4, this.gl.FLOAT, false, 0, 0);

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
    // WebGPU IMPLEMENTATION (Inter-dimensional Particles)
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
                pos : vec4f, // x, y, z, w
                vel : vec4f, // vx, vy, vz, vw
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

            // Rotation helpers
            fn rotateZW(p: vec4f, t: f32) -> vec4f {
                let c = cos(t); let s = sin(t);
                return vec4f(p.x, p.y, p.z*c - p.w*s, p.z*s + p.w*c);
            }
            fn rotateXW(p: vec4f, t: f32) -> vec4f {
                let c = cos(t); let s = sin(t);
                return vec4f(p.x*c - p.w*s, p.y, p.z, p.x*s + p.w*c);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // 4D Attraction Point (The Origin for now, or drifting)
                // Particles drift in 4D space

                // Rotational field based on position
                let dist = length(p.pos);

                // Add velocity towards origin if too far
                if (dist > 2.0) {
                    p.vel = p.vel - p.pos * 0.01;
                }

                // Swirling in XW and YZ planes
                let sw = 0.5;
                p.vel = p.vel + vec4f(
                    -p.pos.w * sw, // Rotate X-W
                    -p.pos.z * sw, // Rotate Y-Z
                     p.pos.y * sw,
                     p.pos.x * sw
                ) * 0.01;

                // Mouse interaction (Repel from mouse ray in 3D projection)
                // This is rough approximation
                let mx = params.mouseX;
                let my = params.mouseY;
                let dx = p.pos.x - mx;
                let dy = p.pos.y - my;
                let d2 = dx*dx + dy*dy;
                if (d2 < 0.2) {
                    p.vel.x = p.vel.x + dx * 0.1;
                    p.vel.y = p.vel.y + dy * 0.1;
                }

                // Damping
                p.vel = p.vel * 0.96;
                p.pos = p.pos + p.vel * params.dt;

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
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

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            // 4D Rotations match WebGL
            fn rotateZW(p_in: vec4f, t: f32) -> vec4f {
                var p = p_in;
                let c = cos(t); let s = sin(t);
                let z = p.z; let w = p.w;
                p.z = z*c - w*s;
                p.w = z*s + w*c;
                return p;
            }
            fn rotateXW(p_in: vec4f, t: f32) -> vec4f {
                var p = p_in;
                let c = cos(t); let s = sin(t);
                let x = p.x; let w = p.w;
                p.x = x*c - w*s;
                p.w = x*s + w*c;
                return p;
            }
            fn rotateYW(p_in: vec4f, t: f32) -> vec4f {
                var p = p_in;
                let c = cos(t); let s = sin(t);
                let y = p.y; let w = p.w;
                p.y = y*c - w*s;
                p.w = y*s + w*c;
                return p;
            }
            fn rotateXY(p_in: vec4f, t: f32) -> vec4f {
                var p = p_in;
                let c = cos(t); let s = sin(t);
                let x = p.x; let y = p.y;
                p.x = x*c - y*s;
                p.y = x*s + y*c;
                return p;
            }
            fn rotateXZ(p_in: vec4f, t: f32) -> vec4f {
                var p = p_in;
                let c = cos(t); let s = sin(t);
                let x = p.x; let z = p.z;
                p.x = x*c - z*s;
                p.z = x*s + z*c;
                return p;
            }

            @vertex
            fn vs_main(@location(0) particlePos : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Apply the same 4D Rotation as WebGL structure
                let t = params.time * 0.5;
                var p = particlePos;

                p = rotateZW(p, t * 0.7);
                p = rotateXW(p, t * 0.5);
                p = rotateYW(p, t * 0.3);

                // Mouse Rotation
                p = rotateXY(p, params.mouseX * 2.0);
                p = rotateXZ(p, params.mouseY * 2.0);

                // 4D Projection
                let wDistance = 2.5;
                let wScale = 1.0 / (wDistance - p.w);
                let p3 = p.xyz * wScale;

                // 3D Projection
                let camPos = vec3f(0.0, 0.0, -3.0);
                let viewPos = p3 - camPos;

                let fov = 1.2;
                let scale = 1.0 / tan(fov * 0.5);

                let x = viewPos.x * scale / params.aspect;
                let y = viewPos.y * scale;
                let z = viewPos.z;

                output.position = vec4f(x, y, p.w * 0.1, z); // z is w for perspective

                // Color
                let alpha = smoothstep(2.0, 0.0, length(viewPos));
                output.color = vec4f(0.5, 0.2, 1.0, 0.6); // Purple particles

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleUnitSize = 32; // vec4 pos + vec4 vel
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            // Random 4D points
            initialParticleData[i*8+0] = (Math.random() - 0.5) * 2.0;
            initialParticleData[i*8+1] = (Math.random() - 0.5) * 2.0;
            initialParticleData[i*8+2] = (Math.random() - 0.5) * 2.0;
            initialParticleData[i*8+3] = (Math.random() - 0.5) * 2.0;
            // Velocity
            initialParticleData[i*8+4] = 0;
            initialParticleData[i*8+5] = 0;
            initialParticleData[i*8+6] = 0;
            initialParticleData[i*8+7] = 0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: initialParticleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

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

        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
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
        msg.innerText = "WebGPU Not Available";
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

        if (this.glCanvas) {
            this.glCanvas.width = displayWidth;
            this.glCanvas.height = displayHeight;
            this.gl.viewport(0, 0, displayWidth, displayHeight);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = displayWidth;
            this.gpuCanvas.height = displayHeight;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // WebGL Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);

            this.gl.clearColor(0.0, 0.0, 0.0, 0.0); // Transparent
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // WebGPU Render
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0) {
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
                }],
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup);
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
    window.HyperCubeExperiment = HyperCubeExperiment;
}
