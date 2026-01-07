/**
 * Hybrid Cyber Glow Experiment
 * Combines WebGL2 for a raymarched neon tunnel and WebGPU for a massive particle swarm.
 */

class HybridCyberGlow {
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

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 100000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        console.log("HybridCyberGlow: Initializing...");

        // 1. Initialize WebGL2 Layer (Background)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("HybridCyberGlow: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("HybridCyberGlow: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("HybridCyberGlow: WebGPU initialized successfully.");
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
    // WebGL2 IMPLEMENTATION (Raymarched Tunnel)
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
            console.warn("HybridCyberGlow: WebGL2 not supported.");
            return;
        }

        // Fullscreen quad
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_mouse;
            uniform vec2 u_resolution;

            out vec4 outColor;

            mat2 rot(float a) {
                float s = sin(a);
                float c = cos(a);
                return mat2(c, -s, s, c);
            }

            float map(vec3 p) {
                // Twist
                p.xy *= rot(p.z * 0.1 + u_time * 0.2);

                // Box tunnel
                vec3 q = abs(p) - vec3(2.0, 2.0, 1000.0); // Infinite z
                // Repetition
                vec3 p2 = mod(p, 4.0) - 2.0;

                float box = length(max(abs(p2) - vec3(0.5, 0.5, 0.5), 0.0));

                // Ground plane variation
                float ground = p.y + 2.0 + sin(p.x * 2.0 + p.z * 0.5) * 0.2;

                return min(length(p.xy) - 1.5, box); // Simple tunnel
            }

            void main() {
                vec2 uv = v_uv;
                if (u_resolution.y > 0.0) {
                    uv.x *= u_resolution.x / u_resolution.y;
                }

                // Camera
                vec3 ro = vec3(0.0, 0.0, u_time * 5.0);
                vec3 rd = normalize(vec3(uv + u_mouse * 0.5, 1.0));

                // Raymarching
                float t = 0.0;
                float d = 0.0;
                int steps = 0;
                for(int i = 0; i < 64; i++) {
                    vec3 p = ro + rd * t;
                    d = 2.0 - length(p.xy); // Cylindrical tunnel SDF approx
                    // Add grid structure
                    float grid = length(max(abs(mod(p, 2.0) - 1.0) - 0.1, 0.0));
                    d = min(d, grid);

                    if (d < 0.001 || t > 50.0) break;
                    t += d * 0.5;
                    steps = i;
                }

                vec3 col = vec3(0.0);

                if (t < 50.0) {
                    float glow = 1.0 - float(steps) / 64.0;
                    vec3 p = ro + rd * t;

                    // Grid color
                    vec3 gridColor = vec3(0.0, 1.0, 0.8) * (sin(p.z * 0.5 + u_time) * 0.5 + 0.5);
                    col = gridColor * glow * 2.0;

                    // Fog
                    col = mix(col, vec3(0.05, 0.0, 0.1), smoothstep(0.0, 40.0, t));
                }

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 2, this.gl.FLOAT, false, 0, 0);

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
    // WebGPU IMPLEMENTATION (Particle Swarm)
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

        // WGSL
        const computeShaderCode = `
            struct Particle {
                pos : vec3f,
                vel : vec3f,
                life : f32,
                pad : f32, // align to 32 bytes (3+3+1+1)*4 = 32
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                pad1 : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Simple hash for randomization
            fn hash(p: f32) -> f32 {
                var p2 = fract(p * .1031);
                p2 = p2 * (p2 + 33.33);
                p2 = p2 * (p2 + p2);
                return fract(p2);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Mouse interaction (projected into pseudo-3D space)
                let mousePos = vec3f(params.mouseX * 5.0, params.mouseY * 5.0, 0.0);

                // Flow towards camera
                p.vel.z = p.vel.z - 2.0 * params.dt; // Move towards negative Z (screen)

                // Spiral motion
                let angle = atan2(p.pos.y, p.pos.x);
                let radius = length(p.pos.xy);
                let spiralForce = vec3f(-sin(angle), cos(angle), 0.0) * (2.0 / (radius + 0.1));

                // Mouse attraction
                let toMouse = mousePos - p.pos;
                let distMouse = length(toMouse);
                let mouseForce = normalize(toMouse) * (1.0 / (distMouse + 0.1)) * 5.0;

                p.vel = p.vel + (spiralForce + mouseForce) * params.dt;

                // Damping
                p.vel = p.vel * 0.98;

                p.pos = p.pos + p.vel * params.dt;

                // Respawn if behind camera or too far
                if (p.pos.z < -2.0 || length(p.pos.xy) > 10.0) {
                    // Respawn far away
                    let t = params.time + f32(index) * 0.0001;
                    p.pos = vec3f(
                        (hash(t) - 0.5) * 10.0,
                        (hash(t + 13.0) - 0.5) * 10.0,
                        20.0 + hash(t * 2.0) * 10.0 // Far Z
                    );
                    p.vel = vec3f(0.0, 0.0, -5.0); // Fast forward
                }

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(
                @location(0) particlePos : vec3f,
                @location(1) particleVel : vec3f
            ) -> VertexOutput {
                var output : VertexOutput;

                // Simple perspective projection
                let fov = 1.0;
                let projection = mat4x4f(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 1.0,
                    0.0, 0.0, -1.0, 0.0
                );

                // Scale point size based on Z
                let zDist = max(0.1, particlePos.z + 5.0); // Offset for camera

                // Manual projection for point sprite
                // We'll just render points for now
                let pos = vec4f(particlePos.x, particlePos.y, particlePos.z - 5.0, 1.0);

                // Basic perspective divide manually for control
                let x = pos.x / -pos.z;
                let y = pos.y / -pos.z;

                // Keep Z for depth test if needed, but we are doing additive blending

                output.position = vec4f(x, y, 0.0, 1.0);

                // Color based on velocity/depth
                let speed = length(particleVel);
                let depthFade = smoothstep(20.0, 0.0, particlePos.z);

                output.color = vec4f(1.0, 0.2, 0.8, 1.0) * depthFade;
                output.color.g = min(1.0, speed * 0.2);

                // Point size trick in WebGPU requires specific primitive topology or quad expasion
                // For simplicity in this 'point-list' mode, we just draw 1px points.
                // To do better, we'd need triangle strip expansion.
                // But let's stick to points to see if it works first.

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffer setup
        const particleUnitSize = 32; // 8 floats * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 8 + 0] = (Math.random() * 10 - 5); // x
            initialParticleData[i * 8 + 1] = (Math.random() * 10 - 5); // y
            initialParticleData[i * 8 + 2] = Math.random() * 20 + 5;   // z
            initialParticleData[i * 8 + 3] = 0; // vx
            initialParticleData[i * 8 + 4] = 0; // vy
            initialParticleData[i * 8 + 5] = -2; // vz
            initialParticleData[i * 8 + 6] = 1.0; // life
            initialParticleData[i * 8 + 7] = 0.0; // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Groups & Pipelines
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
                        { shaderLocation: 0, offset: 0, format: 'float32x3' }, // pos
                        { shaderLocation: 1, offset: 12, format: 'float32x3' }, // vel
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
        msg.innerHTML = "WebGPU Not Available (WebGL2 Only)";
        this.container.appendChild(msg);
    }

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

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
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
        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.HybridCyberGlow = HybridCyberGlow;
}
