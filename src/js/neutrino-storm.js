/**
 * Neutrino Storm Experiment
 * Hybrid WebGL2 (Detector Tank) + WebGPU (Cherenkov Radiation Simulation)
 */

export class NeutrinoStormExperiment {
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

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020205';

        console.log("NeutrinoStorm: Initializing...");

        // 1. Initialize WebGL2 Layer (Detector Tank Structure)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Cherenkov Radiation)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("NeutrinoStorm: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("NeutrinoStorm: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        }

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
    // WebGL2 IMPLEMENTATION (Detector Tank Wireframe)
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
            console.warn("NeutrinoStorm: WebGL2 not supported.");
            return;
        }

        // Generate Cylinder Wireframe
        const segments = 32;
        const rings = 10;
        const vertices = [];
        const indices = [];

        // Vertices
        for (let r = 0; r <= rings; r++) {
            const y = (r / rings) * 2 - 1; // -1 to 1
            for (let s = 0; s < segments; s++) {
                const angle = (s / segments) * Math.PI * 2;
                const x = Math.cos(angle);
                const z = Math.sin(angle);
                vertices.push(x, y, z);
            }
        }

        // Indices (Lines)
        for (let r = 0; r <= rings; r++) {
            for (let s = 0; s < segments; s++) {
                const current = r * segments + s;
                const next = r * segments + (s + 1) % segments;

                // Horizontal ring lines
                indices.push(current, next);

                // Vertical lines
                if (r < rings) {
                    const below = (r + 1) * segments + s;
                    indices.push(current, below);
                }
            }
        }
        this.numIndices = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec3 a_position;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;

            out float v_y;

            mat4 perspective(float fov, float aspect, float near, float far) {
                float f = 1.0 / tan(fov / 2.0);
                return mat4(
                    f / aspect, 0.0, 0.0, 0.0,
                    0.0, f, 0.0, 0.0,
                    0.0, 0.0, (far + near) / (near - far), -1.0,
                    0.0, 0.0, (2.0 * far * near) / (near - far), 0.0
                );
            }

            void main() {
                v_y = a_position.y;

                // Rotate cylinder
                float t = u_time * 0.2;
                float c = cos(t);
                float s = sin(t);

                // Mouse Interaction (Tilt)
                float mx = u_mouse.x * 0.5;
                float my = u_mouse.y * 0.5;

                vec3 pos = a_position;

                // Y-Axis Rotation (Auto)
                float x = pos.x * c - pos.z * s;
                float z = pos.x * s + pos.z * c;
                pos.x = x;
                pos.z = z;

                // X-Axis Rotation (Mouse Tilt)
                float cx = cos(my);
                float sx = sin(my);
                float y = pos.y * cx - pos.z * sx;
                z = pos.y * sx + pos.z * cx;
                pos.y = y;
                pos.z = z;

                // Camera transform
                pos.z -= 4.0; // Move back

                float aspect = u_resolution.x / u_resolution.y;
                mat4 proj = perspective(1.0, aspect, 0.1, 100.0);

                gl_Position = proj * vec4(pos, 1.0);
                gl_PointSize = 4.0;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in float v_y;
            out vec4 outColor;

            void main() {
                // Gold/Copper color for PMTs
                vec3 col = vec3(0.8, 0.6, 0.2);
                float alpha = 0.3 + 0.2 * sin(v_y * 10.0);
                outColor = vec4(col, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) return null;

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) return null;

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);
        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Cherenkov Particles)
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
        if (!adapter) {
            this.gpuCanvas.remove();
            return false;
        }
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied',
        });

        const computeShaderCode = `
            struct Particle {
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, type
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

            // Random helpers
            fn hash(p: u32) -> f32 {
                var p_ = p;
                p_ = (p_ << 13u) ^ p_;
                p_ = p_ * (p_ * p_ * 15731u + 789221u) + 1376312589u;
                return f32(p_ & 0x7fffffffu) / f32(0x7fffffffu);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Update life
                p.pos.w = p.pos.w - params.dt * 2.0;

                // Respawn if dead
                if (p.pos.w <= 0.0) {
                    // Random respawn
                    // We want cone bursts.
                    // Let's use the index to assign it to a "group"
                    // Group ID based on time to simulate bursts

                    let seed = hash(index + u32(params.time * 1000.0));

                    // Only spawn if probability met (controls density)
                    if (seed > 0.98) {
                         // Random position inside cylinder
                        let theta = hash(index * 2u) * 6.28;
                        let r = sqrt(hash(index * 3u));
                        let h = hash(index * 4u) * 2.0 - 1.0;

                        p.pos.x = r * cos(theta);
                        p.pos.z = r * sin(theta);
                        p.pos.y = h;

                        // Direction: Cherenkov Cone
                        // Assume neutrino travels down (Y axis) or mouse direction
                        let nuDir = normalize(vec3f(params.mouseX, -1.0, params.mouseY));

                        // Emit at angle ~42 degrees (Cherenkov angle)
                        // Create a random vector, project to plane perp to nuDir, then add nuDir component

                        let randDir = normalize(vec3f(hash(index*5u)-0.5, hash(index*6u)-0.5, hash(index*7u)-0.5));
                        // Force away from nuDir

                        // Simplified: Just explosion with bias
                        p.vel.x = (hash(index*8u) - 0.5) + nuDir.x;
                        p.vel.y = (hash(index*9u) - 0.5) + nuDir.y;
                        p.vel.z = (hash(index*10u) - 0.5) + nuDir.z;

                        p.vel = normalize(p.vel) * 2.0; // Speed

                        p.pos.w = 1.0; // Reset Life
                    }
                } else {
                    // Move
                    p.pos.x = p.pos.x + p.vel.x * params.dt;
                    p.pos.y = p.pos.y + p.vel.y * params.dt;
                    p.pos.z = p.pos.z + p.vel.z * params.dt;
                }

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

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

            @vertex
            fn vs_main(@location(0) particlePos : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Reuse WebGL cylinder rotation logic for alignment
                let t = params.time * 0.2;
                let c = cos(t); let s = sin(t);
                let mx = params.mouseX * 0.5; let my = params.mouseY * 0.5;

                var pos = particlePos.xyz;

                // Y-Rotate
                let x = pos.x * c - pos.z * s;
                let z = pos.x * s + pos.z * c;
                pos.x = x; pos.z = z;

                // X-Rotate (Tilt)
                let cx = cos(my); let sx = sin(my);
                let y = pos.y * cx - pos.z * sx;
                pos.y = y;
                pos.z = pos.y * sx + pos.z * cx;

                pos.z = pos.z - 4.0; // Camera offset

                // Project
                let fov = 1.0;
                let f = 1.0 / tan(fov / 2.0);
                let px = (pos.x * f) / params.aspect;
                let py = pos.y * f;
                let pz = pos.z;

                // Perspective divide handled by w
                output.position = vec4f(px, py, pz * 0.01, -pz);

                // Color (Cherenkov Blue)
                let life = particlePos.w;
                let alpha = smoothstep(0.0, 0.2, life) * smoothstep(1.0, 0.8, life);

                output.color = vec4f(0.0, 0.5, 1.0, alpha);

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleUnitSize = 32; // vec4 pos (w=life) + vec4 vel
        const initialData = new Float32Array(this.numParticles * 8);

        // Init to dead
        for (let i = 0; i < this.numParticles; i++) {
            initialData[i*8 + 3] = -1.0; // Life < 0
        }

        this.particleBuffer = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

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
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos + life
                    ],
                }],
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

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width === 0 || height === 0) return;

        this.canvasSize.width = width;
        this.canvasSize.height = height;
        const dw = Math.floor(width * dpr);
        const dh = Math.floor(height * dpr);

        if (this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            this.gl.viewport(0, 0, dw, dh);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = dw;
            this.gpuCanvas.height = dh;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);

            this.gl.clearColor(0, 0, 0, 0); // Transparent
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);

            // Draw lines
            this.gl.drawElements(this.gl.LINES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);

            // Draw points
            this.gl.drawElements(this.gl.POINTS, this.numIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // WebGPU
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
                }]
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
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.NeutrinoStormExperiment = NeutrinoStormExperiment;
}
