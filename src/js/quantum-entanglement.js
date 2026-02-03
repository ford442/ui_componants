/**
 * Quantum Entanglement Experiment
 * Demonstrates Hybrid WebGL2 + WebGPU implementation.
 * - WebGL2: Renders two rotating Torus "Entangled Rings".
 * - WebGPU: Renders a particle swarm transferring between them.
 */

class QuantumEntanglementExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0, isPressed: false };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndexBuffer = null;
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
        this.numParticles = options.numParticles || 30000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020105'; // Deep dark background

        console.log("QuantumEntanglement: Initializing...");

        // 1. Initialize WebGL2 Layer (Rings)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("QuantumEntanglement: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("QuantumEntanglement: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("QuantumEntanglement: WebGPU initialized successfully.");
        }

        this.isActive = true;

        // Event Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);

        this.resize();
        this.animate();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Torus Rings)
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
            console.warn("QuantumEntanglement: WebGL2 not supported.");
            return;
        }

        // Generate Torus Geometry
        const { positions, indices } = this.createTorusGeometry(0.3, 0.05, 32, 16);
        this.glIndexCount = indices.length;

        // Setup VAO and Buffers
        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.glIndexBuffer = indexBuffer;
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform vec2 u_resolution;
            uniform float u_offsetX; // -0.5 or 0.5

            out vec3 v_pos;
            out vec3 v_normal;

            mat4 rotationX(float angle) {
                return mat4(1.0, 0.0, 0.0, 0.0,
                            0.0, cos(angle), -sin(angle), 0.0,
                            0.0, sin(angle), cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            mat4 rotationY(float angle) {
                return mat4(cos(angle), 0.0, sin(angle), 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            -sin(angle), 0.0, cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

             mat4 rotationZ(float angle) {
                return mat4(cos(angle), -sin(angle), 0.0, 0.0,
                            sin(angle), cos(angle), 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            void main() {
                v_pos = a_position;

                // Rotate the torus
                // Different rotation direction based on offset for visual interest
                float dir = sign(u_offsetX);
                mat4 rot = rotationY(u_time * 0.5 * dir) * rotationX(u_time * 0.3);

                vec4 pos = rot * vec4(a_position, 1.0);

                // Shift position
                pos.x += u_offsetX;

                // Perspective projection aspect ratio fix
                float aspect = u_resolution.x / u_resolution.y;
                pos.x /= aspect;

                gl_Position = pos;
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            uniform float u_offsetX;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Color based on offset
                vec3 color;
                if (u_offsetX < 0.0) {
                    // Left: Cyan
                    color = vec3(0.0, 0.8, 1.0);
                } else {
                    // Right: Magenta
                    color = vec3(1.0, 0.0, 0.8);
                }

                // Simple wireframe-ish glow
                // We don't have normals passed perfectly for wireframe, but let's just make it glowy
                float pulse = 0.5 + 0.5 * sin(u_time * 2.0);

                // Add some variation
                color += pulse * 0.2;

                outColor = vec4(color, 0.6); // Semi-transparent
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;

        // Enable blending for transparency
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        this.resizeGL();
    }

    createTorusGeometry(radius, tube, radialSegments, tubularSegments) {
        const positions = [];
        const indices = [];

        for (let j = 0; j <= radialSegments; j++) {
            for (let i = 0; i <= tubularSegments; i++) {
                const u = i / tubularSegments * Math.PI * 2;
                const v = j / radialSegments * Math.PI * 2;

                const centerX = radius * Math.cos(u);
                const centerY = radius * Math.sin(u);

                const x = (radius + tube * Math.cos(v)) * Math.cos(u);
                const y = (radius + tube * Math.cos(v)) * Math.sin(u);
                const z = tube * Math.sin(v);

                positions.push(x, y, z);
            }
        }

        for (let j = 1; j <= radialSegments; j++) {
            for (let i = 1; i <= tubularSegments; i++) {
                const a = (tubularSegments + 1) * j + i - 1;
                const b = (tubularSegments + 1) * (j - 1) + i - 1;
                const c = (tubularSegments + 1) * (j - 1) + i;
                const d = (tubularSegments + 1) * j + i;

                // Wireframe lines: a-b, b-c?
                // Just triangles for now, but draw with LINE_STRIP or LINES
                // Let's use LINES topology in draw call for wireframe look
                indices.push(a, b);
                indices.push(b, c);
                indices.push(c, d);
                indices.push(d, a);
            }
        }

        return { positions, indices };
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
    // WebGPU IMPLEMENTATION (Entangled Particles)
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
                state : f32, // 0: Left, 1: Right, 2: Transit L->R, 3: Transit R->L
                target : f32, // Progress of transit (0..1) or random seed
                dummy1: f32,
                dummy2: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mousePos : vec2f,
                isPressed : f32,
                aspect : f32,
                dummy : f32, // Padding
                dummy2 : f32, // Padding
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

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

                // Centers adjusted for aspect ratio in vertex shader, but here we work in simulation space
                // Aspect ratio is applied at rendering time or simulation?
                // Let's keep simulation space -1..1 (x) and apply aspect ratio during render.
                // However, our WebGL rings are at +/- 0.5 * aspect corrected?
                // The WebGL shader does: pos.x += offset; pos.x /= aspect;
                // So the visual center in NDC is offset/aspect.
                // If aspect is 16/9 (~1.77), then 0.5 becomes 0.28.
                // To match, we should define simulation centers as +/- 0.5 * params.aspect.
                // Wait, if I want them to align, I should match the logic.

                let leftCenter = vec2f(-0.5 * params.aspect, 0.0);
                let rightCenter = vec2f(0.5 * params.aspect, 0.0);

                // States
                // 0: Orbit Left
                // 1: Orbit Right
                // 2: Transit to Right
                // 3: Transit to Left

                if (p.state < 0.5) { // State 0 (Left)
                    let diff = p.pos - leftCenter;
                    let dist = length(diff);
                    let dir = normalize(diff);

                    // Orbit
                    let tangent = vec2f(-dir.y, dir.x);
                    p.vel = mix(p.vel, tangent * 0.5, 0.1);
                    p.vel += -dir * 0.5 * params.dt; // Centripetal

                    p.pos += p.vel * params.dt;

                    // Interaction: Mouse click on left side
                    if (params.isPressed > 0.5 && params.mousePos.x < 0.0) {
                         let mouseDist = distance(p.pos, params.mousePos);
                         if (mouseDist < 0.5) {
                             p.state = 2.0; // Trigger transit
                             p.target = 0.0; // Reset progress
                         }
                    }

                } else if (p.state < 1.5) { // State 1 (Right)
                    let diff = p.pos - rightCenter;
                    let dist = length(diff);
                    let dir = normalize(diff);

                    // Orbit
                    let tangent = vec2f(-dir.y, dir.x);
                    p.vel = mix(p.vel, tangent * 0.5, 0.1);
                    p.vel += -dir * 0.5 * params.dt; // Centripetal

                    p.pos += p.vel * params.dt;

                    // Interaction: Mouse click on right side
                    if (params.isPressed > 0.5 && params.mousePos.x > 0.0) {
                         let mouseDist = distance(p.pos, params.mousePos);
                         if (mouseDist < 0.5) {
                             p.state = 3.0; // Trigger transit back
                             p.target = 0.0;
                         }
                    }

                } else if (p.state < 2.5) { // State 2 (Transit L->R)
                    p.target += params.dt * 2.0; // Speed of transfer
                    if (p.target >= 1.0) {
                        p.state = 1.0; // Arrived
                        p.pos = rightCenter + (p.pos - leftCenter); // Teleport relatively? Or just snap
                    } else {
                         // Lerp with some noise
                         let t = p.target;
                         let basePos = mix(leftCenter, rightCenter, t);
                         let noise = vec2f(rand(vec2f(t, f32(index))), rand(vec2f(t+1.0, f32(index)))) - 0.5;
                         p.pos = basePos + noise * 0.2 * sin(t * 3.14);
                    }
                } else { // State 3 (Transit R->L)
                    p.target += params.dt * 2.0;
                    if (p.target >= 1.0) {
                        p.state = 0.0; // Arrived
                        p.pos = leftCenter + (p.pos - rightCenter);
                    } else {
                         let t = p.target;
                         let basePos = mix(rightCenter, leftCenter, t);
                         let noise = vec2f(rand(vec2f(t, f32(index))), rand(vec2f(t+1.0, f32(index)))) - 0.5;
                         p.pos = basePos + noise * 0.2 * sin(t * 3.14);
                    }
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

            struct SimParams {
                dt : f32,
                time : f32,
                mousePos : vec2f,
                isPressed : f32,
                aspect : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            @vertex
            fn vs_main(
                @builtin(vertex_index) vertexIndex : u32,
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f,
                @location(2) particleState : f32,
            ) -> VertexOutput {
                var output : VertexOutput;

                // Convert simulation space to NDC
                // Sim space x is already scaled by aspect logic in compute?
                // Wait, in compute I used aspect to position centers.
                // So particlePos is in "World Space" where X is wide.
                // To get to NDC, we divide X by aspect.

                var pos = particlePos;
                pos.x /= params.aspect;

                output.position = vec4f(pos, 0.0, 1.0);

                // Color based on state
                if (particleState < 0.5) {
                    // Left: Cyan
                    output.color = vec4f(0.0, 1.0, 1.0, 1.0);
                } else if (particleState < 1.5) {
                    // Right: Magenta
                    output.color = vec4f(1.0, 0.0, 1.0, 1.0);
                } else {
                    // Transit: White/Energy
                    output.color = vec4f(1.0, 1.0, 1.0, 1.0);
                }

                output.size = 2.0;
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
            // Randomly assign to Left (0) or Right (1)
            const state = Math.random() > 0.5 ? 1.0 : 0.0;
            const centerX = (state === 0.0 ? -0.5 : 0.5); // * aspect happens in loop, we just need initial near center
            // Approx aspect for init
            const aspect = this.container.clientWidth / this.container.clientHeight;

            const theta = Math.random() * Math.PI * 2;
            const r = 0.2 + Math.random() * 0.1;

            initialParticleData[i * 8 + 0] = (centerX * aspect) + Math.cos(theta) * r; // Pos X
            initialParticleData[i * 8 + 1] = Math.sin(theta) * r; // Pos Y

            initialParticleData[i * 8 + 2] = 0; // Vel X
            initialParticleData[i * 8 + 3] = 0; // Vel Y

            initialParticleData[i * 8 + 4] = state; // State
            initialParticleData[i * 8 + 5] = 0; // Target
            initialParticleData[i * 8 + 6] = 0; // Pad
            initialParticleData[i * 8 + 7] = 0; // Pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
        // dt(4) + time(4) + mousePos(8) + isPressed(4) + aspect(4) + pad(8) -> 32 bytes
        this.simParamBuffer = this.device.createBuffer({
            size: 32,
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
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }), // Need params in VS
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' }, // pos
                        { shaderLocation: 1, offset: 8, format: 'float32x2' }, // vel
                        { shaderLocation: 2, offset: 16, format: 'float32' },  // state
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
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available &mdash; Core Only Mode";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // EVENTS & LOOP
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

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        // Normalize mouse to -1..1
        // Need to match simulation space which scales X by aspect
        const aspect = rect.width / rect.height;

        const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);

        this.mouse.x = ndcX * aspect; // Sim space X
        this.mouse.y = ndcY; // Sim space Y
    }

    onMouseDown(e) {
        this.mouse.isPressed = true;
    }

    onMouseUp(e) {
        this.mouse.isPressed = false;
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016; // Fixed step approximation

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const resLoc = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
            const offsetLoc = this.gl.getUniformLocation(this.glProgram, 'u_offsetX');

            this.gl.uniform1f(timeLoc, time);
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.02, 0.01, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.glIndexBuffer);

            // Draw Left Ring
            this.gl.uniform1f(offsetLoc, -0.5);
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);

            // Draw Right Ring
            this.gl.uniform1f(offsetLoc, 0.5);
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            // Update Uniforms
            const aspect = this.gpuCanvas.width / this.gpuCanvas.height;
            // dt(4), time(4), mouseX(4), mouseY(4), isPressed(4), aspect(4), pad(4), pad(4)
            const params = new Float32Array([
                dt,
                time,
                this.mouse.x,
                this.mouse.y,
                this.mouse.isPressed ? 1.0 : 0.0,
                aspect,
                0, 0 // Padding
            ]);
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
            renderPass.setBindGroup(0, this.computeBindGroup); // Need for VS
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
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('mousedown', this.handleMouseDown);
        window.removeEventListener('mouseup', this.handleMouseUp);

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
    window.QuantumEntanglementExperiment = QuantumEntanglementExperiment;
}

export { QuantumEntanglementExperiment };
