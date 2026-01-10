/**
 * Acoustic Levitation Experiment
 * Combines WebGL2 for emitter plates/grid and WebGPU for particle levitation physics.
 */

class AcousticLevitation {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };
        this.frequency = 4.0; // Wave frequency
        this.phase = 0.0;     // Phase shift

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glGridVao = null;

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
        this.handleMouseDown = this.onMouseDown.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#0a0a0f';

        console.log("AcousticLevitation: Initializing...");

        // 1. Initialize WebGL2 Layer (Background Emitters)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("AcousticLevitation: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("AcousticLevitation: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("AcousticLevitation: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;

        // Modulate frequency based on X
        this.frequency = 4.0 + (x + 1.0) * 4.0;
    }

    onMouseDown(e) {
        // Shift phase on click
        this.phase += Math.PI / 2;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Emitters)
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
            console.warn("AcousticLevitation: WebGL2 not supported.");
            return;
        }

        // --- Emitter Plates Shader ---
        const vsSource = `#version 300 es
            in vec3 a_position;
            in vec2 a_uv;

            uniform mat4 u_projection;
            uniform mat4 u_view;
            uniform mat4 u_model;

            out vec2 v_uv;
            out vec3 v_pos;

            void main() {
                v_uv = a_uv;
                v_pos = a_position;
                gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec2 v_uv;
            in vec3 v_pos;
            uniform float u_time;
            uniform float u_frequency;

            out vec4 outColor;

            float hexDist(vec2 p) {
                p = abs(p);
                return max(p.x, dot(p, normalize(vec2(1.0, 1.73))));
            }

            void main() {
                // Hexagon pattern
                vec2 st = v_uv * 10.0;
                vec2 r = vec2(1.0, 1.73);
                vec2 h = r * 0.5;
                vec2 a = mod(st, r) - h;
                vec2 b = mod(st - h, r) - h;
                vec2 gv = dot(a, a) < dot(b, b) ? a : b;
                float x = atan(gv.x, gv.y);
                float y = 0.5 - hexDist(gv);
                float hexVal = smoothstep(0.0, 0.1, 0.5 - length(gv));

                // Pulse emission
                float pulse = sin(u_time * 5.0 - length(v_uv - 0.5) * 10.0) * 0.5 + 0.5;

                vec3 color = vec3(0.1, 0.2, 0.4);
                color += vec3(0.5, 0.7, 1.0) * hexVal * pulse;

                // Edge glow
                float edge = smoothstep(0.4, 0.5, abs(v_uv.x - 0.5)) + smoothstep(0.4, 0.5, abs(v_uv.y - 0.5));
                color += vec3(0.2, 0.5, 1.0) * edge;

                outColor = vec4(color, 0.8);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        // Create Plane Geometry (for emitters)
        const positions = new Float32Array([
            -1, 0, -1,  1, 0, -1,  -1, 0, 1,  1, 0, 1
        ]);
        const uvs = new Float32Array([
            0, 0,  1, 0,  0, 1,  1, 1
        ]);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        const uvBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, uvBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, uvs, this.gl.STATIC_DRAW);
        const uvLoc = this.gl.getAttribLocation(this.glProgram, 'a_uv');
        this.gl.enableVertexAttribArray(uvLoc);
        this.gl.vertexAttribPointer(uvLoc, 2, this.gl.FLOAT, false, 0, 0);
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
    // WebGPU IMPLEMENTATION (Levitating Particles)
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
                pos : vec4f, // x, y, z, w (padding/life)
                vel : vec4f, // vx, vy, vz, vw
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                frequency : f32,
                phase : f32,
                mouseX : f32,
                mouseY : f32,
                pad1 : f32,
                pad2 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Standing wave pressure field
                // Force directs particles towards nodes where sin(ky + phi) is minimal/zero pressure
                // Pressure P(y) ~ cos(k*y)
                // Force F = -grad(P^2) ~ sin(2*k*y)

                let k = params.frequency;
                let y = p.pos.y;
                let forceY = -sin(2.0 * k * y + params.phase) * 5.0; // Vertical Acoustic Force

                // Radial containment (weak)
                let r = length(p.pos.xz);
                let forceRadial = -p.pos.xz * 1.0;

                // Mouse interaction (repel)
                let mousePos = vec2f(params.mouseX * 2.0, params.mouseY * 2.0); // Rough mapping
                // For simplicity in this 3D view, let's map mouse X to rotation or some disturbance

                // Apply forces
                p.vel.y += forceY * params.dt;
                p.vel.x += forceRadial.x * params.dt;
                p.vel.z += forceRadial.y * params.dt;

                // Drag/Damping
                p.vel *= 0.95;

                // Update position
                p.pos += p.vel * params.dt;

                // Boundaries
                if (p.pos.y > 1.2) { p.pos.y = -1.2; }
                if (p.pos.y < -1.2) { p.pos.y = 1.2; }
                if (length(p.pos.xz) > 1.5) {
                    p.pos.x = 0.0;
                    p.pos.z = 0.0;
                }

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct Uniforms {
                viewProjection : mat4x4f,
            }
            @group(0) @binding(0) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @location(0) particlePos : vec4f,
                @location(1) particleVel : vec4f
            ) -> VertexOutput {
                var output : VertexOutput;
                output.position = uniforms.viewProjection * vec4f(particlePos.xyz, 1.0);

                // Color based on velocity/pressure
                let speed = length(particleVel.xyz);
                let val = abs(particlePos.y); // visualize nodes vs antinodes

                let c1 = vec3f(0.2, 0.5, 1.0);
                let c2 = vec3f(1.0, 1.0, 1.0);
                let col = mix(c1, c2, speed * 2.0);

                output.color = vec4f(col, 1.0);

                // Point size attenuation approximation
                // WebGPU points are fixed size usually unless using quads, but let's rely on primitive point size if available or just fixed
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
            initialParticleData[i * 8 + 0] = (Math.random() * 2 - 1) * 0.5; // x
            initialParticleData[i * 8 + 1] = (Math.random() * 2 - 1);       // y
            initialParticleData[i * 8 + 2] = (Math.random() * 2 - 1) * 0.5; // z
            initialParticleData[i * 8 + 3] = 1.0; // w
            initialParticleData[i * 8 + 4] = 0; // vx
            initialParticleData[i * 8 + 5] = 0; // vy
            initialParticleData[i * 8 + 6] = 0; // vz
            initialParticleData[i * 8 + 7] = 0; // vw
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffers
        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 8 floats aligned
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.viewProjBuffer = this.device.createBuffer({
            size: 64, // mat4x4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Groups
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

        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.viewProjBuffer } },
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
            padding: 8px 16px; border-radius: 4px; font-family: monospace; pointer-events: none;
        `;
        msg.innerHTML = "WebGPU Not Available";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON & RENDER LOOP
    // ========================================================================

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width === 0 || height === 0) return;

        const dpr = window.devicePixelRatio || 1;
        this.canvasSize = { width, height };

        if (this.glCanvas) {
            this.glCanvas.width = width * dpr;
            this.glCanvas.height = height * dpr;
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = width * dpr;
            this.gpuCanvas.height = height * dpr;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // Camera setup (Orbital)
        const aspect = this.canvasSize.width / this.canvasSize.height;
        const camRadius = 3.5;
        const camX = Math.sin(time * 0.2) * camRadius;
        const camZ = Math.cos(time * 0.2) * camRadius;
        const camY = 0.5 + this.mouse.y;

        const viewMatrix = this.lookAt([camX, camY, camZ], [0, 0, 0], [0, 1, 0]);
        const projectionMatrix = this.perspective(45 * Math.PI / 180, aspect, 0.1, 100.0);
        const viewProjectionMatrix = this.multiplyMatrices(projectionMatrix, viewMatrix);

        // 1. WebGL2 Render (Emitters)
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_frequency'), this.frequency);

            // Matrix uniforms
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_projection'), false, projectionMatrix);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_view'), false, viewMatrix);

            this.gl.clearColor(0.0, 0.0, 0.0, 0.0); // Transparent background for WebGL (container has color)
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

            // Top Emitter
            let modelTop = this.translation(0, 1.2, 0);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_model'), false, modelTop);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

            // Bottom Emitter
            let modelBot = this.translation(0, -1.2, 0);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_model'), false, modelBot);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. WebGPU Render (Particles)
        if (this.device && this.context && this.renderPipeline) {
            // Update Sim Params
            const params = new Float32Array([
                0.016, // dt
                time,
                this.frequency,
                this.phase,
                this.mouse.x,
                this.mouse.y,
                0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);
            this.device.queue.writeBuffer(this.viewProjBuffer, 0, viewProjectionMatrix);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute Pass
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render Pass
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
            renderPass.setBindGroup(0, this.renderBindGroup);
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
        this.container.removeEventListener('mousedown', this.handleMouseDown);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }

    // Matrix Math Helpers
    multiplyMatrices(a, b) {
        const out = new Float32Array(16);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) sum += a[i * 4 + k] * b[k * 4 + j];
                out[i * 4 + j] = sum;
            }
        }
        return out;
    }
    // Column-major construction for GLSL
    translation(tx, ty, tz) {
        return new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            tx, ty, tz, 1
        ]);
    }
    perspective(fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ]);
    }
    lookAt(eye, center, up) {
        let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
        let len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        z0 *= len; z1 *= len; z2 *= len;
        let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
        len = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
        if (!len) { x0 = 0; x1 = 0; x2 = 0; } else { len = 1 / len; x0 *= len; x1 *= len; x2 *= len; }
        let y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
        len = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
        if (!len) { y0 = 0; y1 = 0; y2 = 0; } else { len = 1 / len; y0 *= len; y1 *= len; y2 *= len; }
        return new Float32Array([
            x0, y0, z0, 0,
            x1, y1, z1, 0,
            x2, y2, z2, 0,
            -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]), -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]), -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]), 1
        ]);
    }
}

if (typeof window !== 'undefined') {
    window.AcousticLevitation = AcousticLevitation;
}
