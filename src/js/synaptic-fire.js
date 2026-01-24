/**
 * Synaptic Fire Experiment
 * Hybrid visualization of a biological synapse.
 * WebGL2: Raymarched organic blobs (Axon/Dendrite).
 * WebGPU: Particle system for neurotransmitters.
 */

export class SynapticFireExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0, isDown: false };
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
        this.numParticles = options.numParticles || 50000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);
        this.animate = this.animate.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#00050a';

        // 1. Initialize WebGL2 Layer (Organic Background)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("SynapticFire: WebGPU initialization error:", e);
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
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);

        // Touch support
        this.container.addEventListener('touchmove', (e) => {
            if (e.touches && e.touches[0]) {
                e.preventDefault();
                const touch = e.touches[0];
                this.updateMousePos(touch.clientX, touch.clientY);
            }
        }, { passive: false });

        this.container.addEventListener('touchstart', (e) => {
             this.mouse.isDown = true;
             if (e.touches && e.touches[0]) {
                 const touch = e.touches[0];
                 this.updateMousePos(touch.clientX, touch.clientY);
             }
        }, { passive: false });

        this.container.addEventListener('touchend', () => {
             this.mouse.isDown = false;
        });
    }

    updateMousePos(clientX, clientY) {
        const rect = this.container.getBoundingClientRect();
        const x = (clientX - rect.left) / rect.width * 2 - 1;
        const y = -((clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    onMouseMove(e) {
        this.updateMousePos(e.clientX, e.clientY);
    }

    onMouseDown() {
        this.mouse.isDown = true;
    }

    onMouseUp() {
        this.mouse.isDown = false;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (SDF Raymarching)
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

        // Fullscreen quad
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 2, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;

        const vsSource = `#version 300 es
            layout(location = 0) in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            out vec4 outColor;

            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;
            uniform float u_pulse; // For action potential flash

            // SDF Primitives
            float sdSphere(vec3 p, float s) {
                return length(p) - s;
            }

            // Smooth union
            float smin(float a, float b, float k) {
                float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
                return mix(b, a, h) - k * h * (1.0 - h);
            }

            float map(vec3 p) {
                // Top Blob (Pre-synaptic Axon)
                vec3 p1 = p - vec3(0.0, 2.8, 0.0);
                // Distort
                p1.x += sin(p1.y * 5.0 + u_time) * 0.1;
                p1.z += cos(p1.y * 4.0 + u_time * 1.3) * 0.1;
                float d1 = sdSphere(p1, 2.0);

                // Bottom Blob (Post-synaptic Dendrite)
                vec3 p2 = p - vec3(0.0, -2.8, 0.0);
                p2.x += cos(p2.y * 4.5 - u_time * 0.8) * 0.1;
                float d2 = sdSphere(p2, 1.8);

                return smin(d1, d2, 0.8);
            }

            vec3 calcNormal(vec3 p) {
                const float eps = 0.001;
                const vec2 h = vec2(eps, 0);
                return normalize(vec3(map(p+h.xyy) - map(p-h.xyy),
                                      map(p+h.yxy) - map(p-h.yxy),
                                      map(p+h.yyx) - map(p-h.yyx)));
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;

                // Camera
                vec3 ro = vec3(0.0, 0.0, 5.0);
                vec3 rd = normalize(vec3(uv, -1.0));

                // Raymarch
                float t = 0.0;
                float dist = 0.0;
                for(int i = 0; i < 64; i++) {
                    vec3 p = ro + rd * t;
                    dist = map(p);
                    if(dist < 0.001 || t > 10.0) break;
                    t += dist;
                }

                vec3 col = vec3(0.0, 0.05, 0.1); // Deep background

                if(t < 10.0) {
                    vec3 p = ro + rd * t;
                    vec3 n = calcNormal(p);
                    vec3 lightPos = vec3(2.0, 2.0, 5.0);
                    vec3 lightDir = normalize(lightPos - p);

                    float diff = max(dot(n, lightDir), 0.0);
                    float rim = 1.0 - max(dot(n, -rd), 0.0);
                    rim = pow(rim, 3.0);

                    // Organic colors
                    vec3 membraneColor = vec3(0.1, 0.6, 0.4);
                    vec3 flashColor = vec3(0.4, 0.9, 1.0);

                    vec3 finalColor = mix(membraneColor, flashColor, u_pulse * 0.5);

                    col = finalColor * (diff * 0.8 + 0.2) + rim * flashColor;
                }

                // Vignette
                float vig = 1.0 - length(uv) * 0.5;
                col *= vig;

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
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
    // WebGPU IMPLEMENTATION (Neurotransmitters)
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
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        const computeShaderCode = `
            struct Particle {
                pos : vec4f,
                vel : vec4f,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                isDown : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Neurotransmitter behavior:
                // Start near top (Axon). Drift down to bottom (Dendrite).
                // If clicked, "burst" = high velocity, more chaotic.

                // Life cycle check (using w component of pos as life/state?)
                // Actually let's just recycle them when they go off screen.

                // Forces
                var force = vec3f(0.0);

                // 1. Gravity/Flow (Downwards)
                force.y -= 0.5;

                // 2. Electric Field (Mouse influence)
                let mousePos = vec3f(params.mouseX * 3.0 * params.aspect, params.mouseY * 3.0, 0.0);
                let dist = distance(p.pos.xyz, mousePos);

                // Repel from mouse (Electric repulsion)
                if (dist < 1.5) {
                    let pushDir = normalize(p.pos.xyz - mousePos);
                    force += pushDir * (1.0 - dist / 1.5) * 10.0;
                }

                // 3. Burst (Action Potential)
                if (params.isDown > 0.5) {
                    // Explode outwards from top center if they are near the top
                    if (p.pos.y > 0.5) {
                        force += normalize(p.pos.xyz - vec3f(0.0, 1.0, 0.0)) * 20.0;
                    }
                }

                // Brownian Motion (Noise)
                let noise = vec3f(
                    rand(vec2f(p.pos.y, params.time)) - 0.5,
                    rand(vec2f(p.pos.z, params.time * 1.1)) - 0.5,
                    rand(vec2f(p.pos.x, params.time * 0.9)) - 0.5
                );
                force += noise * 5.0;

                // Integration
                p.vel += vec4f(force * params.dt, 0.0);
                p.vel *= 0.95; // Drag

                p.pos += p.vel * params.dt;

                // Recycling
                // If below -3.0 (absorbed) or too far out
                if (p.pos.y < -3.5 || abs(p.pos.x) > 4.0 * params.aspect || abs(p.pos.z) > 2.0) {
                    // Respawn at top (Axon terminal)
                    // Random position in a sphere near (0, 1.5, 0)
                    let r = pow(rand(vec2f(params.time, f32(index))), 0.33) * 1.0;
                    let theta = rand(vec2f(f32(index), params.time)) * 6.28;
                    let phi = acos(2.0 * rand(vec2f(params.time, f32(index) + 1.0)) - 1.0);

                    let offset = vec3f(
                        r * sin(phi) * cos(theta),
                        r * sin(phi) * sin(theta), // y offset relative to center
                        r * cos(phi)
                    );

                    // Flatten sphere to disk/blob
                    offset.y *= 0.3;

                    p.pos = vec4f(offset + vec3f(0.0, 1.5, 0.0), 1.0);
                    p.vel = vec4f(0.0, -1.0, 0.0, 0.0); // Initial downward velocity
                }

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
                isDown : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(
                @builtin(vertex_index) vertexIndex : u32,
                @location(0) particlePos : vec4f,
                @location(1) particleVel : vec4f
            ) -> VertexOutput {
                var output : VertexOutput;

                // Instanced Billboarding (Quad)
                // 0--1
                // | /|
                // |/ |
                // 2--3
                let cornerIndex = vertexIndex % 6u;
                var offset = vec2f(0.0, 0.0);
                if (cornerIndex == 0u || cornerIndex == 4u) { offset = vec2f(-1.0, -1.0); }
                else if (cornerIndex == 1u) { offset = vec2f(1.0, -1.0); }
                else if (cornerIndex == 2u || cornerIndex == 3u) { offset = vec2f(-1.0, 1.0); }
                else if (cornerIndex == 5u) { offset = vec2f(1.0, 1.0); }

                let pos = particlePos.xyz;
                let camPos = vec3f(0.0, 0.0, 5.0);
                let viewPos = pos - camPos;

                // Billboard size
                let size = 0.03;

                // Simple perspective projection
                let fov = 1.0;
                let f = 1.0 / tan(fov / 2.0);

                // Apply billboard offset in view space (before projection) or screen space?
                // For simplicity, let's do screen space offset after projection approximation
                // But correct way for billboard is to add offset to position orthogonal to view vector.

                // Let's cheat and add to x/y in view space assuming camera looks down -Z
                let finalPos = vec3f(pos.x + offset.x * size, pos.y + offset.y * size, pos.z);
                let finalViewPos = finalPos - camPos;

                let x = finalViewPos.x * f / params.aspect;
                let y = finalViewPos.y * f;
                let z = finalViewPos.z;
                let w = -z;

                output.position = vec4f(x, y, z * 0.1, w);

                // Color based on activity
                // "Resting" = Blue/Cyan
                // "Active" (Fast) = White/Yellow
                let speed = length(particleVel.xyz);
                let t = smoothstep(0.0, 5.0, speed);
                let col = mix(vec3f(0.2, 0.6, 1.0), vec3f(1.0, 0.9, 0.5), t);

                // Alpha fade near edges/depth
                let alpha = 0.8;

                output.color = vec4f(col, alpha);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                // Circular particle
                // Since we don't pass UVs easily in this hacky billboard, let's just output square or use gl_PointCoord equivalent if we were using points.
                // But we are using quads. Wait, I didn't pass UVs.
                // It's fine, small squares look like pixels/dust.
                return color;
            }
        `;

        const particleUnitSize = 32;
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        // Initialize particles
        for (let i = 0; i < this.numParticles; i++) {
             // Start random
             initialParticleData[i*8+0] = (Math.random() - 0.5) * 4.0;
             initialParticleData[i*8+1] = (Math.random() - 0.5) * 4.0;
             initialParticleData[i*8+2] = (Math.random() - 0.5) * 2.0;
             initialParticleData[i*8+3] = 1.0;
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
                    stepMode: 'instance', // Instanced drawing!
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{ format: presentationFormat, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                } }],
            },
            primitive: { topology: 'triangle-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.innerHTML = "WebGPU Not Available (WebGL2 Only)";
        msg.style.cssText = "position: absolute; bottom: 10px; right: 10px; color: red;";
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
            if (this.gl) this.gl.viewport(0, 0, displayWidth, displayHeight);
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

        // ----------------- WebGL2 Render -----------------
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_pulse'), this.mouse.isDown ? 1.0 : 0.0);

            this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // ----------------- WebGPU Render -----------------
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, time, this.mouse.x, this.mouse.y, aspect, this.mouse.isDown ? 1.0 : 0.0, 0, 0
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
            // Draw 6 vertices (1 quad) per instance (particle)
            renderPass.draw(6, this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(this.animate);
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mouseup', this.handleMouseUp);

        if (this.container) {
            this.container.removeEventListener('mousemove', this.handleMouseMove);
            this.container.removeEventListener('mousedown', this.handleMouseDown);
            this.container.innerHTML = '';
        }

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
    }
}

if (typeof window !== 'undefined') {
    window.SynapticFireExperiment = SynapticFireExperiment;
}
