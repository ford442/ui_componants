/**
 * Quantum Stabilizer Experiment
 * Demonstrates Hybrid WebGL2 + WebGPU implementation.
 * - WebGL2: Renders a rotating "Quantum Core" (Icosahedron wireframe).
 * - WebGPU: Renders a "Containment Field" particle swarm.
 */

class QuantumStabilizer {
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
        this.numParticles = options.numParticles || 20000;

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
        this.container.style.background = '#000005'; // Deep void black

        console.log("QuantumStabilizer: Initializing...");

        // 1. Initialize WebGL2 Layer (Core)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Field)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("QuantumStabilizer: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("QuantumStabilizer: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("QuantumStabilizer: WebGPU initialized successfully.");
        }

        this.isActive = true;

        // Event Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp); // Window to catch release outside

        this.resize();
        this.animate();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Quantum Core)
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
            console.warn("QuantumStabilizer: WebGL2 not supported.");
            return;
        }

        // Generate Icosahedron Geometry
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;
        const positions = [
            -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0,
            0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t,
            t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1
        ];

        // Normalize positions to sphere
        for(let i=0; i<positions.length; i+=3) {
            const length = Math.sqrt(positions[i]**2 + positions[i+1]**2 + positions[i+2]**2);
            positions[i] /= length;
            positions[i+1] /= length;
            positions[i+2] /= length;
        }

        const indices = [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
        ];

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

            out vec3 v_pos;

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

            void main() {
                v_pos = a_position;

                // Rotate
                mat4 rot = rotationY(u_time * 0.5) * rotationX(u_time * 0.3);
                vec4 pos = rot * vec4(a_position, 1.0);

                // Scale pulse
                float pulse = 0.5 + 0.1 * sin(u_time * 2.0);
                pos.xyz *= pulse;

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
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Neon core look
                vec3 color = vec3(0.0, 1.0, 0.8); // Cyan

                // Add some variation based on position
                float glow = sin(v_pos.x * 10.0 + u_time * 5.0) * 0.5 + 0.5;
                color += vec3(0.2, 0.0, 0.5) * glow; // Purple accents

                outColor = vec4(color, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

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
    // WebGPU IMPLEMENTATION (Containment Field)
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
                dummy : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mousePos : vec2f,
                isPressed : f32,
                dummy : f32, // Padding
                dummy2 : f32, // Padding
                dummy3 : f32, // Padding
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

                // Orbit logic
                let center = vec2f(0.0, 0.0);
                let diff = center - p.pos;
                let dist = length(diff);
                let dir = normalize(diff);

                // Tangential force (orbit)
                let tangent = vec2f(-dir.y, dir.x);

                // Attraction force
                let attractionStrength = 0.5;
                let attraction = dir * attractionStrength * params.dt;

                // Orbit velocity target
                let orbitSpeed = 0.8;
                let targetVel = tangent * orbitSpeed;

                // Mix current velocity towards target orbit velocity
                p.vel = mix(p.vel, targetVel, 0.05);
                p.vel += attraction;

                // Mouse interaction (Repel / Destabilize)
                let mouseDiff = p.pos - params.mousePos;
                let mouseDist = length(mouseDiff);
                if (mouseDist < 0.4) {
                    let force = normalize(mouseDiff) * (0.4 - mouseDist) * 10.0;
                    if (params.isPressed > 0.5) {
                         force *= 5.0; // Stronger blast on click
                    }
                    p.vel += force * params.dt;
                }

                // Drag
                p.vel *= 0.98;

                p.pos += p.vel * params.dt;

                // Reset if lost
                if (dist > 1.5 || dist < 0.01) {
                    let r = rand(vec2f(params.time, f32(index)));
                    let theta = r * 6.28;
                    let d = 0.5 + rand(vec2f(r, params.time)) * 0.2;
                    p.pos = vec2f(cos(theta) * d, sin(theta) * d);
                    // Initial tangent velocity
                    p.vel = vec2f(-sin(theta), cos(theta)) * 0.5;
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
                @builtin(vertex_index) vertexIndex : u32,
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f
            ) -> VertexOutput {
                var output : VertexOutput;

                // Billboard quad generation from vertex_index (0..5)
                // We'll just draw points for simplicity using GL_POINTS equivalent topology?
                // Wait, WebGPU needs explicit quad expansion or point-list topology.
                // Using point-list is easiest if supported, but typically triangle-list is better for sizing.
                // Let's stick to point-list as in previous successful experiments, defaulting size in FS or just single pixel?
                // Actually, renderPipeline topology: 'point-list' works but size is fixed to 1px in standard WebGPU (unlike OpenGL).
                // To get variable size points, we need to generate quads.
                // But for "Containment Field" sparks, single pixels or small quads might be fine.
                // Let's try 'point-list' first. If it's too small, I'll switch to quads.
                // Actually, let's just use point-list.

                output.position = vec4f(particlePos, 0.0, 1.0);

                // Color based on velocity
                let speed = length(particleVel);
                let energy = smoothstep(0.0, 1.0, speed);

                output.color = mix(vec4f(0.0, 0.5, 1.0, 0.8), vec4f(1.0, 0.2, 0.8, 1.0), energy);
                output.size = 2.0; // Not used in point-list really without extension

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
            // Pos
            const theta = Math.random() * Math.PI * 2;
            const r = 0.5 + Math.random() * 0.2;
            initialParticleData[i * 8 + 0] = Math.cos(theta) * r;
            initialParticleData[i * 8 + 1] = Math.sin(theta) * r;
            // Vel (Tangential)
            initialParticleData[i * 8 + 2] = -Math.sin(theta) * 0.5;
            initialParticleData[i * 8 + 3] = Math.cos(theta) * 0.5;
            // Life/Padding
            initialParticleData[i * 8 + 4] = 1.0;
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
        // Size: dt(4) + time(4) + mousePos(8) + isPressed(4) + pad(12) -> 32 bytes aligned
        this.simParamBuffer = this.device.createBuffer({
            size: 32,
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
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
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

            this.gl.uniform1f(timeLoc, time);
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.0, 0.0, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            // Ensure index buffer is bound (safeguard)
            this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.glIndexBuffer);
            // Draw lines for wireframe
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            // Update Uniforms
            // dt(4), time(4), mouseX(4), mouseY(4), isPressed(4), pad(4), pad(4), pad(4)
            const params = new Float32Array([
                dt,
                time,
                this.mouse.x,
                this.mouse.y,
                this.mouse.isPressed ? 1.0 : 0.0,
                0, 0, 0 // Padding
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
    window.QuantumStabilizer = QuantumStabilizer;
}

export { QuantumStabilizer };
