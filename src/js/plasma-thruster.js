/**
 * Plasma Thruster Experiment
 * Hybrid WebGL2 + WebGPU visualization.
 * - WebGL2: Renders a "Magnetic Nozzle" (Ring Assembly).
 * - WebGPU: Renders a high-velocity "Ion Stream" particle simulation.
 */

export class PlasmaThrusterExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0, thrust: 0.5 }; // x,y for gimbal, thrust for power

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
        this.renderUniformBuffer = null;
        this.renderBindGroup = null;
        this.numParticles = options.numParticles || 40000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000'; // Deep space black

        console.log("PlasmaThruster: Initializing...");

        // 1. Initialize WebGL2 Layer (Nozzle)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Ion Stream)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("PlasmaThruster: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("PlasmaThruster: WebGPU initialized successfully.");
        }

        this.isActive = true;

        // Event Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);

        // Initial resize
        this.resize();
        this.animate();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Magnetic Nozzle)
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
            console.warn("PlasmaThruster: WebGL2 not supported.");
            return;
        }

        // Generate Nozzle Geometry (Stack of Rings)
        // We'll approximate a ring as a thin cylinder segment or torus-like structure
        // Actually, let's just make a few cylinders of varying radii
        const positions = [];
        const indices = [];
        let indexOffset = 0;

        // Function to create a ring
        const addRing = (radius, zPos, thickness, height) => {
            const segments = 48;
            for (let i = 0; i <= segments; i++) {
                const theta = (i / segments) * Math.PI * 2;
                const c = Math.cos(theta);
                const s = Math.sin(theta);

                // Inner top
                positions.push(c * radius, s * radius, zPos + height/2);
                // Outer top
                positions.push(c * (radius + thickness), s * (radius + thickness), zPos + height/2);
                // Inner bottom
                positions.push(c * radius, s * radius, zPos - height/2);
                // Outer bottom
                positions.push(c * (radius + thickness), s * (radius + thickness), zPos - height/2);
            }

            for (let i = 0; i < segments; i++) {
                const base = indexOffset + i * 4;
                // Top face
                indices.push(base, base + 1, base + 5);
                indices.push(base, base + 5, base + 4);
                // Bottom face
                indices.push(base + 2, base + 6, base + 7);
                indices.push(base + 2, base + 7, base + 3);
                // Inner face
                indices.push(base, base + 4, base + 6);
                indices.push(base, base + 6, base + 2);
                // Outer face
                indices.push(base + 1, base + 3, base + 7);
                indices.push(base + 1, base + 7, base + 5);
            }
            indexOffset += (segments + 1) * 4;
        };

        // Create 3 rings for the nozzle
        addRing(1.0, 0.0, 0.2, 0.3);
        addRing(0.8, 0.6, 0.2, 0.3);
        addRing(0.6, 1.2, 0.2, 0.3);

        this.glNumIndices = indices.length;

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
            uniform vec2 u_mouse; // x: gimbal, y: thrust

            out vec3 v_pos;
            out float v_glow;

            // Rotation matrix
            mat4 rotateX(float angle) {
                float c = cos(angle);
                float s = sin(angle);
                return mat4(
                    1, 0, 0, 0,
                    0, c, -s, 0,
                    0, s, c, 0,
                    0, 0, 0, 1
                );
            }

            mat4 rotateY(float angle) {
                float c = cos(angle);
                float s = sin(angle);
                return mat4(
                    c, 0, s, 0,
                    0, 1, 0, 0,
                    -s, 0, c, 0,
                    0, 0, 0, 1
                );
            }

            void main() {
                vec3 pos = a_position;

                // Gimbal rotation based on mouse X
                float gimbalAngle = u_mouse.x * 0.5; // -0.5 to 0.5 rad
                pos = (rotateY(gimbalAngle) * vec4(pos, 1.0)).xyz;

                // Base rotation to view it nicely
                pos = (rotateX(0.5) * vec4(pos, 1.0)).xyz;

                // Move back
                pos.z -= 4.0;

                v_pos = pos;

                // Glow pulsates with thrust (mouse Y)
                v_glow = 0.5 + 0.5 * sin(u_time * 5.0) * u_mouse.y;

                // Perspective
                float aspect = u_resolution.x / u_resolution.y;
                float fov = 1.0;
                float f = 1.0 / tan(fov/2.0);
                float zNear = 0.1;
                float zFar = 100.0;

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
            in float v_glow;

            out vec4 outColor;

            void main() {
                // Metallic / Energy look
                vec3 baseColor = vec3(0.1, 0.1, 0.15);

                // Rim lighting
                vec3 normal = normalize(cross(dFdx(v_pos), dFdy(v_pos)));
                vec3 viewDir = normalize(-v_pos);
                float rim = 1.0 - max(dot(normal, viewDir), 0.0);
                rim = pow(rim, 3.0);

                vec3 glowColor = vec3(0.2, 0.6, 1.0);
                if (v_glow > 0.8) glowColor = vec3(0.5, 0.8, 1.0);

                outColor = vec4(baseColor + glowColor * rim * (1.0 + v_glow), 1.0);
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
    // WebGPU IMPLEMENTATION (Ion Stream)
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
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied',
        });

        // COMPUTE SHADER
        const computeCode = `
            struct Particle {
                pos: vec4f, // x, y, z, life
                vel: vec4f, // vx, vy, vz, initialSpeed
            }

            struct SimParams {
                dt: f32,
                time: f32,
                thrust: f32, // Controlled by mouse Y
                gimbal: f32, // Controlled by mouse X
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: SimParams;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            mat2 rotate(angle: f32) {
                let c = cos(angle);
                let s = sin(angle);
                return mat2x2(c, -s, s, c);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.numParticles}) { return; }

                var p = particles[index];

                // Update Life
                // Higher thrust = faster life decay (faster particles)
                p.pos.w -= params.dt * (0.5 + params.thrust);

                // Reset
                if (p.pos.w <= 0.0) {
                    let seed = vec2f(f32(index), params.time);

                    // Spawn at nozzle origin (approx)
                    // The nozzle rings are at z ~ 0 to 1.2
                    // We spawn at 1.2 (exit)
                    let r = sqrt(rand(seed)) * 0.5;
                    let theta = rand(seed + 1.0) * 6.28;

                    p.pos.x = cos(theta) * r;
                    p.pos.y = sin(theta) * r;
                    p.pos.z = 1.2; // Exit of nozzle
                    p.pos.w = 1.0; // Life

                    // Initial Velocity
                    let speed = 5.0 + rand(seed + 2.0) * 10.0 * params.thrust;
                    p.vel.w = speed;
                    p.vel.x = (rand(seed + 3.0) - 0.5) * 0.5; // Slight spread
                    p.vel.y = (rand(seed + 4.0) - 0.5) * 0.5;
                    p.vel.z = speed; // Shoot forward (+Z in local nozzle space)
                }

                // Apply Gimbal Rotation to Velocity vector
                // We just rotate the "ideal" forward vector
                // But since particles persist, we should probably rotate their position/vel frame
                // OR just rotate the emission and let them fly straight?
                // Let's rotate emission velocity.
                // Wait, if I rotate nozzle, the stream should follow.
                // Simple approach: Apply rotation to velocity every frame? No, that curves them.
                // Apply rotation to position in render shader? Yes, that matches the rigid body rotation of nozzle.
                // But the stream should lag behind? That's complex.
                // Let's just rotate the simulation frame in the vertex shader to match the nozzle.
                // So here we simulate in "Nozzle Space" (particles move +Z).

                // Physics in Nozzle Space
                p.pos.x += p.vel.x * params.dt;
                p.pos.y += p.vel.y * params.dt;
                p.pos.z += p.vel.z * params.dt;

                // Turbulence / Noise
                p.pos.x += sin(p.pos.z * 2.0 + params.time * 5.0) * 0.01;

                particles[index] = p;
            }
        `;

        // RENDER SHADER
        const drawCode = `
            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec4f,
                @location(1) life: f32,
            }

            struct Uniforms {
                resolution: vec2f,
                time: f32,
                gimbal: f32, // To rotate the stream with nozzle
            }
            @group(0) @binding(2) var<uniform> uniforms: Uniforms;

            // Same rotation functions as Compute/WebGL
            mat4 rotateX(angle: f32) {
                let c = cos(angle);
                let s = sin(angle);
                return mat4x4(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, c, -s, 0.0,
                    0.0, s, c, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            }

            mat4 rotateY(angle: f32) {
                let c = cos(angle);
                let s = sin(angle);
                return mat4x4(
                    c, 0.0, s, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    -s, 0.0, c, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            }

            @vertex
            fn vs_main(@location(0) pPos: vec4f, @location(1) pVel: vec4f) -> VertexOutput {
                var output: VertexOutput;
                var pos = pPos.xyz;

                // Rotate stream to match nozzle gimbal
                // Gimbal is Y-axis rotation
                let gimbalAngle = uniforms.gimbal * 0.5;
                let rotY = rotateY(gimbalAngle);

                // View rotation (fixed X tilt)
                let rotX = rotateX(0.5);

                var p4 = vec4f(pos, 1.0);
                p4 = rotY * p4;
                p4 = rotX * p4;

                pos = p4.xyz;
                pos.z -= 4.0; // Same translation as WebGL

                // Perspective
                let aspect = uniforms.resolution.x / uniforms.resolution.y;
                let fov = 1.0;
                let f = 1.0 / tan(fov/2.0);
                let zNear = 0.1;
                let zFar = 100.0;
                let z = pos.z;

                output.position = vec4f(
                    pos.x * f / aspect,
                    pos.y * f,
                    (z * (zFar + zNear) / (zNear - zFar)) + (2.0 * zFar * zNear) / (zNear - zFar),
                    -z
                );

                // Color based on heat/velocity
                let speed = pVel.w; // stored in w
                let life = pPos.w;

                // Blue core -> Purple -> Red tail
                let heat = smoothstep(0.0, 1.0, life);
                let col1 = vec3f(1.0, 0.2, 0.1); // Red/Orange
                let col2 = vec3f(0.1, 0.5, 1.0); // Blue
                let c = mix(col1, col2, heat);

                output.color = vec4f(c, life); // Fade out with life
                output.life = life;

                return output;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f, @location(1) life: f32) -> @location(0) vec4f {
                if (life <= 0.0) { discard; }
                return color;
            }
        `;

        // Initialize Buffers
        const particleUnitSize = 32;
        const totalSize = this.numParticles * particleUnitSize;
        this.particleBuffer = this.device.createBuffer({
            size: totalSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        // Compute Params Uniform (16 bytes)
        this.simParamBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Render Uniforms (16 bytes: res vec2, time f32, gimbal f32)
        this.renderUniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Groups
        const computeBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        const renderBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBGLayout,
            entries: [
                { binding: 2, resource: { buffer: this.renderUniformBuffer } },
            ],
        });

        // Pipelines
        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBGLayout] }),
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
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    },
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
            background: rgba(200,50,50,0.8);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            pointer-events: none;
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // LOGIC
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dw = Math.floor(w * dpr);
        const dh = Math.floor(h * dpr);

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

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;

        // Map x to -1..1 for gimbal
        this.mouse.x = x * 2 - 1;
        // Map y to 0..1 for thrust (inverted, bottom is low thrust?)
        // Let's say top is high thrust (1.0), bottom is low (0.0)
        this.mouse.thrust = 1.0 - y;
        this.mouse.y = this.mouse.thrust; // for WebGL uniform mapping
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016;

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.thrust);

            this.gl.clearColor(0, 0, 0, 0); // Transparent so we can see CSS BG if needed
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glNumIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. WebGPU Render
        if (this.device && this.context && this.renderPipeline) {
            // Write Sim Params (dt, time, thrust, gimbal)
            // Layout: f32, f32, f32, f32 (16 bytes aligned)
            const params = new Float32Array([dt, time, this.mouse.thrust, this.mouse.x]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            // Write Render Uniforms (res.x, res.y, time, gimbal)
            const rUniforms = new Float32Array([this.gpuCanvas.width, this.gpuCanvas.height, time, this.mouse.x]);
            this.device.queue.writeBuffer(this.renderUniformBuffer, 0, rUniforms);

            const encoder = this.device.createCommandEncoder();

            // Compute Pass
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            // Render Pass
            const texView = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: texView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBindGroup);
            rPass.setVertexBuffer(0, this.particleBuffer);
            rPass.draw(this.numParticles);
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}
