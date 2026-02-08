/**
 * Quantum Tunneling Experiment
 * Combines WebGL2 for the Potential Barrier visualization and WebGPU for particle probability simulation.
 */

class QuantumTunneling {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0.5, y: 0.5 }; // Normalized 0-1
        this.canvasSize = { width: 0, height: 0 };

        // Simulation Parameters
        this.barrierWidth = 0.2;
        this.barrierPotential = 0.8; // Normalized Energy Height
        this.particleEnergy = 0.6;   // Constant particle energy

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;

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

        // UI
        this.infoPanel = null;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#0a0a0f';

        console.log("QuantumTunneling: Initializing...");

        // UI Layer
        this.infoPanel = document.createElement('div');
        this.infoPanel.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            color: rgba(100, 200, 255, 0.9);
            font-family: monospace;
            font-size: 11px;
            pointer-events: none;
            z-index: 10;
            background: rgba(0, 0, 0, 0.6);
            padding: 8px;
            border: 1px solid rgba(100, 200, 255, 0.3);
            border-radius: 4px;
            line-height: 1.5;
        `;
        this.container.appendChild(this.infoPanel);

        // 1. Initialize WebGL2 Layer (Barrier)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("QuantumTunneling: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("QuantumTunneling: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("QuantumTunneling: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        this.mouse.y = Math.max(0, Math.min(1, 1.0 - ((e.clientY - rect.top) / rect.height))); // Invert Y so up is higher potential

        // Map mouse to params
        this.barrierWidth = 0.05 + this.mouse.x * 0.4;
        this.barrierPotential = this.mouse.y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Potential Barrier)
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
            console.warn("QuantumTunneling: WebGL2 not supported.");
            return;
        }

        // --- Barrier Shader ---
        const vsSource = `#version 300 es
            in vec3 a_position;
            in vec3 a_normal;

            uniform mat4 u_projection;
            uniform mat4 u_view;
            uniform mat4 u_model;

            out vec3 v_normal;
            out vec3 v_worldPos;

            void main() {
                v_normal = mat3(u_model) * a_normal;
                vec4 worldPos = u_model * vec4(a_position, 1.0);
                v_worldPos = worldPos.xyz;
                gl_Position = u_projection * u_view * worldPos;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_normal;
            in vec3 v_worldPos;
            uniform float u_potential; // 0-1 mapped to color intensity
            uniform float u_time;

            out vec4 outColor;

            void main() {
                vec3 normal = normalize(v_normal);

                // 1. 3D Grid Pattern
                float gridSize = 4.0;
                vec3 pos = v_worldPos * gridSize;
                vec3 f = fract(pos);
                // Simple grid lines using step
                float lineThickness = 0.05;
                vec3 gridVec = step(1.0 - lineThickness, f);
                float grid = max(max(gridVec.x, gridVec.y), gridVec.z);

                // 2. Scanline Effect (Vertical)
                float scanSpeed = 2.0;
                float scan = smoothstep(0.0, 0.1, abs(sin(v_worldPos.y * 3.0 - u_time * scanSpeed)));
                float scanLine = pow(scan, 8.0);

                // 3. Fresnel Effect
                float fresnel = pow(1.0 - abs(dot(normal, vec3(0.0, 0.0, 1.0))), 2.5);

                // Colors
                vec3 colorBase = vec3(0.0, 0.05, 0.15); // Deep blue
                vec3 colorGrid = vec3(0.0, 0.6, 0.8);   // Cyan grid
                vec3 colorHigh = vec3(1.0, 0.2, 0.5);   // Magenta/Red for high potential

                // Mix grid color based on potential
                vec3 activeGridColor = mix(colorGrid, colorHigh, u_potential);

                vec3 finalColor = colorBase;
                finalColor += activeGridColor * grid * 0.6;
                finalColor += activeGridColor * scanLine * 0.5;
                finalColor += activeGridColor * fresnel * 0.8;

                // Pulse based on time
                float pulse = sin(u_time * 4.0) * 0.5 + 0.5;
                finalColor += activeGridColor * pulse * 0.1 * u_potential;

                float alpha = 0.1 + grid * 0.2 + scanLine * 0.3 + fresnel * 0.5;
                alpha = clamp(alpha, 0.0, 0.85);

                outColor = vec4(finalColor, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        // Create Cube Geometry
        const positions = new Float32Array([
            // Front face
            -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
            // Back face
            -0.5, -0.5, -0.5,  -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,   0.5, -0.5, -0.5,
            // Top face
            -0.5,  0.5, -0.5,  -0.5,  0.5,  0.5,   0.5,  0.5,  0.5,   0.5,  0.5, -0.5,
            // Bottom face
            -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, -0.5,  0.5,  -0.5, -0.5,  0.5,
            // Right face
             0.5, -0.5, -0.5,   0.5,  0.5, -0.5,   0.5,  0.5,  0.5,   0.5, -0.5,  0.5,
            // Left face
            -0.5, -0.5, -0.5,  -0.5, -0.5,  0.5,  -0.5,  0.5,  0.5,  -0.5,  0.5, -0.5,
        ]);

        const indices = new Uint16Array([
            0,  1,  2,      0,  2,  3,    // front
            4,  5,  6,      4,  6,  7,    // back
            8,  9,  10,     8,  10, 11,   // top
            12, 13, 14,     12, 14, 15,   // bottom
            16, 17, 18,     16, 18, 19,   // right
            20, 21, 22,     20, 22, 23    // left
        ]);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        const normBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, normBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW); // Reusing positions as mock normals
        const normLoc = this.gl.getAttribLocation(this.glProgram, 'a_normal');
        this.gl.enableVertexAttribArray(normLoc);
        this.gl.vertexAttribPointer(normLoc, 3, this.gl.FLOAT, false, 0, 0);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);
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
    // WebGPU IMPLEMENTATION (Particles)
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
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, state (0=incident, 1=reflected, 2=tunneled)
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                barrierWidth : f32,
                barrierPotential : f32,
                seed : f32,
                energy : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Simple random hash
            fn hash(p: u32) -> f32 {
                var p1 = p;
                p1 = (p1 << 13u) ^ p1;
                return (1.0 - f32((p1 * (p1 * p1 * 15731u + 789221u) + 1376312589u) & 0x7fffffffu) / 1073741824.0);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Movement
                p.pos.x += p.vel.x * params.dt;
                p.pos.y += p.vel.y * params.dt;
                p.pos.z += p.vel.z * params.dt;

                let barrierXMin = -params.barrierWidth * 0.5;
                let barrierXMax = params.barrierWidth * 0.5;

                // Check Barrier Interaction
                // Only if state is 0 (Incident) and we just entered the barrier region
                if (p.vel.w < 0.5 && p.pos.x > barrierXMin && p.pos.x < barrierXMax) {

                    let energy = params.energy;
                    let potential = params.barrierPotential + 0.1;

                    var prob = 1.0;

                    if (energy < potential) {
                        // Tunneling regime
                        let k = 10.0; // Coupling constant
                        let w = params.barrierWidth;
                        let diff = potential - energy;
                        prob = exp(-k * w * sqrt(diff));
                    } else {
                        prob = 0.98;
                    }

                    // Random check
                    let rnd = hash(index + u32(params.time * 1000.0));

                    if (rnd < prob) {
                        // Tunnel!
                        p.vel.w = 2.0; // State: Tunneled
                        p.vel.x *= 0.8;
                    } else {
                        // Reflect!
                        p.vel.w = 1.0; // State: Reflected
                        p.vel.x = -abs(p.vel.x);
                        p.pos.x = barrierXMin - 0.01;
                    }
                }

                // Reset logic
                if (p.pos.x > 3.0 || p.pos.x < -3.0 || abs(p.pos.y) > 2.0) {
                    let rnd1 = hash(index * 3u + u32(params.time));
                    let rnd2 = hash(index * 7u + u32(params.time));

                    p.pos.x = -2.5 - rnd1 * 0.5;
                    p.pos.y = (rnd2 - 0.5) * 2.0;
                    p.pos.z = (hash(index) - 0.5) * 2.0;

                    p.vel.x = 2.0 + rnd1 * 0.5;
                    p.vel.y = 0.0;
                    p.vel.z = 0.0;
                    p.vel.w = 0.0; // Reset to incident
                }

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) uv : vec2f,
            }

            struct Uniforms {
                viewProjection : mat4x4f,
                cameraRight : vec4f,
                cameraUp : vec4f,
            }
            @group(0) @binding(0) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @builtin(vertex_index) vertexIndex : u32,
                @location(0) particlePos : vec4f,
                @location(1) particleVel : vec4f
            ) -> VertexOutput {
                var output : VertexOutput;

                // Quad
                var pos = array<vec2f, 6>(
                    vec2f(-1.0, -1.0),
                    vec2f( 1.0, -1.0),
                    vec2f(-1.0,  1.0),
                    vec2f(-1.0,  1.0),
                    vec2f( 1.0, -1.0),
                    vec2f( 1.0,  1.0)
                );

                let uv = pos[vertexIndex];
                let size = 0.03;

                let worldPos = particlePos.xyz +
                               uniforms.cameraRight.xyz * uv.x * size +
                               uniforms.cameraUp.xyz * uv.y * size;

                output.position = uniforms.viewProjection * vec4f(worldPos, 1.0);
                output.uv = uv;

                // Color based on state (vel.w)
                let state = particleVel.w;
                var col = vec3f(0.0, 0.6, 1.0); // Incident (Blue)

                if (state > 0.5 && state < 1.5) {
                    col = vec3f(1.0, 0.5, 0.0); // Reflected (Orange)
                } else if (state > 1.5) {
                    col = vec3f(1.0, 0.0, 0.5); // Tunneled (Magenta)
                }

                output.color = vec4f(col, 1.0);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f, @location(1) uv : vec2f) -> @location(0) vec4f {
                let r = length(uv);
                if (r > 1.0) { discard; }

                // Soft glowing particle
                // Gaussian falloff
                let glow = exp(-r * r * 4.0);
                let core = smoothstep(0.4, 0.0, r);

                let intensity = glow * 0.7 + core * 0.8;

                return vec4f(color.rgb, intensity * color.a);
            }
        `;

        const particleUnitSize = 32; // 8 floats
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 8 + 0] = (Math.random() - 0.5) * 4.0; // x
            initialParticleData[i * 8 + 1] = (Math.random() - 0.5) * 2.0; // y
            initialParticleData[i * 8 + 2] = (Math.random() - 0.5) * 2.0; // z
            initialParticleData[i * 8 + 3] = 1.0; // life
            initialParticleData[i * 8 + 4] = 2.0 + Math.random(); // vx
            initialParticleData[i * 8 + 5] = 0.0; // vy
            initialParticleData[i * 8 + 6] = 0.0; // vz
            initialParticleData[i * 8 + 7] = 0.0; // state
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniforms
        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 8 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.viewProjBuffer = this.device.createBuffer({
            size: 64 + 16 + 16,
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
                    stepMode: 'instance',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
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
            primitive: { topology: 'triangle-list' },
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

        // Update Info Panel
        if (this.infoPanel) {
            let energy = this.particleEnergy;
            let potential = this.barrierPotential + 0.1;
            let prob = 0;
            let region = "Incident";

            if (energy < potential) {
                region = "Tunneling";
                let k = 10.0;
                let w = this.barrierWidth;
                let diff = potential - energy;
                prob = Math.exp(-k * w * Math.sqrt(diff));
            } else {
                region = "Passing";
                prob = 0.98;
            }

            this.infoPanel.innerHTML = `
                <div style="margin-bottom:4px; font-weight:bold; color:white">QUANTUM DATA</div>
                Barrier Width: ${this.barrierWidth.toFixed(2)}<br>
                Potential V:   ${this.barrierPotential.toFixed(2)}<br>
                Regime:        ${region}<br>
                Transmission:  ${(prob * 100).toFixed(4)}%
            `;
        }

        // Camera setup
        const aspect = this.canvasSize.width / this.canvasSize.height;
        const camZ = 5.0;
        const viewMatrix = this.lookAt([0, 0, camZ], [0, 0, 0], [0, 1, 0]);
        const projectionMatrix = this.perspective(45 * Math.PI / 180, aspect, 0.1, 100.0);

        // 1. WebGL2 Render (Barrier)
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_potential'), this.barrierPotential);

            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_projection'), false, projectionMatrix);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_view'), false, viewMatrix);

            let h = 2.0;
            let w = this.barrierWidth;
            let model = new Float32Array([
                w, 0, 0, 0,
                0, h, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            ]);

            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_model'), false, model);

            this.gl.clearColor(0.0, 0.0, 0.0, 0.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

            // Draw cube (36 vertices)
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, 36, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. WebGPU Render (Particles)
        if (this.device && this.context && this.renderPipeline) {
            // Update Sim Params
            const params = new Float32Array([
                0.016, // dt
                time,
                this.barrierWidth,
                this.barrierPotential,
                Math.random(), // seed
                this.particleEnergy, // energy
                0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            // Update ViewProj
            const viewProjectionMatrix = this.multiplyMatrices(projectionMatrix, viewMatrix);
            const right = [viewMatrix[0], viewMatrix[4], viewMatrix[8], 0];
            const up = [viewMatrix[1], viewMatrix[5], viewMatrix[9], 0];

            const uniformData = new Float32Array(16 + 4 + 4);
            uniformData.set(viewProjectionMatrix, 0);
            uniformData.set(right, 16);
            uniformData.set(up, 20);

            this.device.queue.writeBuffer(this.viewProjBuffer, 0, uniformData);

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
            renderPass.draw(6, this.numParticles, 0, 0);
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

        if (this.infoPanel) {
            this.infoPanel.remove();
            this.infoPanel = null;
        }

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

export { QuantumTunneling };
