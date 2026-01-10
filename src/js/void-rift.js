/**
 * Void Rift Experiment
 * Hybrid WebGL2 (Gravitational Lensing) + WebGPU (Accretion Disk Simulation)
 */

class VoidRift {
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
        this.container.style.background = '#000000';

        console.log("VoidRift: Initializing...");

        // 1. Initialize WebGL2 Layer (Background + Lensing)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Accretion Disk)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("VoidRift: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("VoidRift: WebGPU not enabled/supported. Running in WebGL2-only mode.");
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
    // WebGL2 IMPLEMENTATION (Black Hole Lensing)
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
            console.warn("VoidRift: WebGL2 not supported.");
            return;
        }

        // Full screen quad
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
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;

            out vec4 outColor;

            // Simple noise for background stars/nebula
            float random(vec2 st) {
                return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
            }

            float noise(in vec2 st) {
                vec2 i = floor(st);
                vec2 f = fract(st);
                float a = random(i);
                float b = random(i + vec2(1.0, 0.0));
                float c = random(i + vec2(0.0, 1.0));
                float d = random(i + vec2(1.0, 1.0));
                vec2 u = f * f * (3.0 - 2.0 * f);
                return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
            }

            float fbm(in vec2 st) {
                float value = 0.0;
                float amplitude = .5;
                float frequency = 0.;
                for (int i = 0; i < 5; i++) {
                    value += amplitude * noise(st);
                    st *= 2.;
                    amplitude *= .5;
                }
                return value;
            }

            void main() {
                vec2 uv = v_uv;
                if (u_resolution.y > 0.0) {
                    uv.x *= u_resolution.x / u_resolution.y;
                }

                // Camera shift by mouse
                vec2 center = u_mouse * 0.1;
                vec2 p = uv - center;
                float r = length(p);
                float radius = 0.35; // Schwarzschild radius (visual)

                // Lensing
                // Light bends around the mass.
                // Simple approximation: distort UVs based on inverse distance
                float distortion = 0.05 / (r - radius + 0.05);
                distortion = clamp(distortion, 0.0, 2.0);

                vec2 distortedUV = uv - center - normalize(p) * distortion * 0.3;

                // Background (Stars/Nebula)
                vec3 bg = vec3(0.0);

                // Stars
                float stars = pow(random(distortedUV * 50.0), 20.0);
                bg += vec3(stars);

                // Nebula
                float n = fbm(distortedUV * 3.0 + u_time * 0.05);
                vec3 nebColor = mix(vec3(0.05, 0.0, 0.1), vec3(0.2, 0.0, 0.2), n);
                bg += nebColor;

                // Event Horizon
                // Black hole mask
                float mask = smoothstep(radius, radius - 0.01, r);

                // Accretion Disk Glow (WebGL side - subtle underlay)
                float disk = 1.0 / (abs(r - radius * 1.5) * 10.0);
                vec3 diskColor = vec3(0.8, 0.3, 0.1) * disk * 0.2;

                vec3 finalColor = bg + diskColor;

                // Blackout center
                finalColor = mix(finalColor, vec3(0.0), mask);

                // Photon Ring (bright rim)
                float ring = smoothstep(0.02, 0.0, abs(r - radius));
                finalColor += vec3(1.0, 0.8, 0.6) * ring;

                outColor = vec4(finalColor, 1.0);
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
            console.error('VoidRift GL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('VoidRift GL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);
        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Particle Simulation)
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

        // WGSL: Compute Shader
        const computeShaderCode = `
            struct Particle {
                pos : vec2f,
                vel : vec2f,
                life : f32,
                pad : f32,
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

            fn random(st: vec2f) -> f32 {
                return fract(sin(dot(st, vec2f(12.9898, 78.233))) * 43758.5453123);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Adjust positions for aspect ratio logic
                let center = vec2f(params.mouseX * params.aspect * 0.1, params.mouseY * 0.1);

                // Gravity
                let p_real = vec2f(p.pos.x * params.aspect, p.pos.y);
                let toCenter = center - p_real;
                let distSq = dot(toCenter, toCenter);
                let dist = sqrt(distSq);
                let dir = normalize(toCenter);

                let force = dir * (0.01 / (distSq + 0.01));

                // Tangential velocity (Orbit)
                let tangent = vec2f(-dir.y, dir.x);
                let orbitSpeed = 2.0 / (dist + 0.1);

                // Combine forces: Gravity pulls in, but maintain orbit velocity
                // We update velocity to point somewhat towards center but mostly tangent
                // Actually, let's just use F = ma
                p.vel = p.vel + force * params.dt;

                // Drag / Chaos
                p.vel = p.vel * 0.995;

                // Add some noise/turbulence
                let seed = p.pos * params.time;
                p.vel = p.vel + vec2f(random(seed) - 0.5, random(seed + 1.0) - 0.5) * 0.005;

                // Update Pos
                p.pos = p.pos + p.vel * params.dt;

                // Event Horizon Reset
                // If too close, reset to outer rim
                let currentPosReal = vec2f(p.pos.x * params.aspect, p.pos.y);
                let d = distance(currentPosReal, center);

                if (d < 0.35 || d > 2.5) {
                    // Reset
                    let angle = random(vec2f(params.time, f32(index))) * 6.28;
                    let r = 1.5 + random(vec2f(f32(index), params.time)) * 0.5;

                    let rx = cos(angle) * r;
                    let ry = sin(angle) * r;

                    // Correct for aspect in spawning?
                    // Store normalized pos.
                    p.pos = vec2f(rx / params.aspect, ry) + center;

                    // Initial velocity: Tangent
                    let spawnDir = normalize(vec2f(rx, ry));
                    let spawnTangent = vec2f(-spawnDir.y, spawnDir.x);
                    p.vel = spawnTangent * (0.5 + random(vec2f(f32(index), 1.0)) * 0.5);
                }

                particles[index] = p;
            }
        `;

        // WGSL: Render Shader
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
            fn vs_main(
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f
            ) -> VertexOutput {
                var output : VertexOutput;
                output.position = vec4f(particlePos, 0.0, 1.0);

                let speed = length(particleVel);

                // Color mapping:
                // Low speed (outer) = Red/Dark
                // High speed (inner) = White/Blue/Bright

                // Distance from center roughly correlates with speed inverse
                // Let's use speed.

                let cOuter = vec3f(0.8, 0.2, 0.1);
                let cInner = vec3f(0.9, 0.9, 1.0);

                let t = smoothstep(0.0, 1.5, speed);
                let col = mix(cOuter, cInner, t);

                // Alpha based on speed too?
                let alpha = smoothstep(0.1, 1.0, speed);

                output.color = vec4f(col, alpha);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleUnitSize = 24; // 6 floats (pos:2, vel:2, life:1, pad:1) * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialData = new Float32Array(this.numParticles * 6);

        for (let i = 0; i < this.numParticles; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = 1.0 + Math.random() * 1.0;
            initialData[i * 6 + 0] = Math.cos(angle) * r; // x
            initialData[i * 6 + 1] = Math.sin(angle) * r; // y

            // Tangent velocity
            const speed = 0.5;
            initialData[i * 6 + 2] = -Math.sin(angle) * speed; // vx
            initialData[i * 6 + 3] = Math.cos(angle) * speed; // vy
            initialData[i * 6 + 4] = 1.0; // life
            initialData[i * 6 + 5] = 0.0; // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

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

        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
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
        msg.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(100, 20, 20, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: monospace;
            pointer-events: none;
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

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);

            this.gl.clearColor(0, 0, 0, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. WebGPU Render
        if (this.device && this.context && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, time, this.mouse.x, this.mouse.y, aspect, 0, 0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

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
            renderPass.setBindGroup(0, this.computeBindGroup); // Need uniform params in vertex shader
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

export { VoidRift };

if (typeof window !== 'undefined') {
    window.VoidRift = VoidRift;
}
