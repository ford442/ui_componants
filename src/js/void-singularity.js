/**
 * Void Singularity Experiment
 * Demonstrates Hybrid WebGL2 + WebGPU rendering.
 * - WebGL2: Renders a gravitational lensing effect (black hole distortion) using a fragment shader on a full-screen quad.
 * - WebGPU: Simulates a massive accretion disk particle swarm using Compute Shaders.
 */

export class VoidSingularityExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

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
        this.numParticles = options.numParticles || 60000;

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        console.log("VoidSingularity: Initializing...");

        // 1. Initialize WebGL2 (Background & Lensing)
        this.initWebGL2();

        // 2. Initialize WebGPU (Particle Swarm)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("VoidSingularity: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("VoidSingularity: WebGPU not available. Running WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();
        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Lensing Shader)
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

        // Full screen quad
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_resolution;
            out vec4 outColor;

            // Simple noise function
            float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
            float noise(vec2 p) {
                vec2 i = floor(p); vec2 f = fract(p);
                vec2 u = f*f*(3.0-2.0*f);
                return mix(mix(hash(i + vec2(0.0,0.0)), hash(i + vec2(1.0,0.0)), u.x),
                           mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), u.x), u.y);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                // Black Hole Center
                float dist = length(uv);
                float radius = 0.2;
                float accretion = 0.35;

                // Event Horizon (Black circle)
                float hole = smoothstep(radius, radius - 0.01, dist);

                // Gravitational Lensing Distortion
                // Bend light near the hole
                float bend = 0.05 / (dist - radius * 0.5);
                vec2 distortedUV = uv * (1.0 - bend);

                // Background Stars (Distorted)
                float starNoise = noise(distortedUV * 50.0 + u_time * 0.05);
                float stars = step(0.98, starNoise) * (starNoise - 0.98) * 50.0;

                // Accretion Disk Glow (WebGL fallback visual)
                float disk = smoothstep(radius, accretion, dist) * smoothstep(accretion + 0.2, accretion, dist);
                vec3 diskColor = vec3(1.0, 0.6, 0.2) * disk * 2.0;

                // Combine
                vec3 color = vec3(stars);
                color += diskColor;
                color = mix(color, vec3(0.0), hole); // Black hole consumes all

                outColor = vec4(color, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
    }

    createGLProgram(vs, fs) {
        const createShader = (type, source) => {
            const s = this.gl.createShader(type);
            this.gl.shaderSource(s, source);
            this.gl.compileShader(s);
            if (!this.gl.getShaderParameter(s, this.gl.COMPILE_STATUS)) {
                console.error('WebGL Shader Error:', this.gl.getShaderInfoLog(s));
                return null;
            }
            return s;
        };
        const p = this.gl.createProgram();
        const v = createShader(this.gl.VERTEX_SHADER, vs);
        const f = createShader(this.gl.FRAGMENT_SHADER, fs);
        if(!v || !f) return null;

        this.gl.attachShader(p, v);
        this.gl.attachShader(p, f);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Accretion Disk Particles)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2; pointer-events: none; background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) { this.gpuCanvas.remove(); return false; }

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied',
        });

        // Compute Shader: Newtonian Gravity + Spiral
        const computeShader = `
            struct Particle {
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, mass
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Params {
                dt : f32,
                time : f32,
            }
            @group(0) @binding(1) var<uniform> params : Params;

            // Random
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.numParticles}) { return; }

                var p = particles[index];
                let dt = params.dt;

                // Physics: Gravity well at (0,0,0)
                let dist = length(p.pos.xyz);
                let dir = normalize(-p.pos.xyz);
                let force = 5.0 / (dist * dist + 0.1); // Inverse square law

                // Tangential force to maintain orbit
                let tangent = normalize(vec3f(-dir.y, dir.x, 0.0)); // Simple 2D orbit plane approx

                // Apply forces
                p.vel.x += (dir.x * force + tangent.x * 2.0) * dt;
                p.vel.y += (dir.y * force + tangent.y * 2.0) * dt;
                p.vel.z += (dir.z * force) * dt;

                // Drag / Friction
                p.vel.x *= 0.99;
                p.vel.y *= 0.99;
                p.vel.z *= 0.99;

                // Update Pos
                p.pos.x += p.vel.x * dt;
                p.pos.y += p.vel.y * dt;
                p.pos.z += p.vel.z * dt;

                // Event Horizon Check (Reset if sucked in)
                if (dist < 0.5 || dist > 25.0) {
                    let seed = vec2f(f32(index), params.time);
                    let angle = rand(seed) * 6.28;
                    let r = 8.0 + rand(seed + 1.0) * 5.0;

                    p.pos.x = cos(angle) * r;
                    p.pos.y = sin(angle) * r;
                    p.pos.z = (rand(seed + 2.0) - 0.5) * 0.5; // Thin disk
                    p.pos.w = 1.0; // Life

                    // Initial orbital velocity
                    let speed = 4.0;
                    p.vel.x = -sin(angle) * speed;
                    p.vel.y = cos(angle) * speed;
                    p.vel.z = 0.0;
                }

                particles[index] = p;
            }
        `;

        // Render Shader
        const drawShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct Uniforms {
                resolution : vec2f,
                time : f32,
            }
            @group(0) @binding(2) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Simple perspective projection
                let aspect = uniforms.resolution.x / uniforms.resolution.y;
                let z = pos.z - 15.0; // Camera z offset

                output.position = vec4f(
                    pos.x / abs(z) / aspect,
                    pos.y / abs(z),
                    0.0,
                    1.0
                );

                // Color based on velocity (heat)
                let speed = length(vel.xyz);
                let heat = smoothstep(0.0, 8.0, speed);
                output.color = mix(
                    vec4f(0.8, 0.4, 0.1, 1.0), // Cool orange
                    vec4f(0.4, 0.8, 1.0, 1.0), // Hot blue
                    heat
                );

                // Point size hack for point-list (usually always 1px in WebGPU w/o extension)
                output.position.w = 1.0;

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const particleSize = 32; // 8 floats
        const initialData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            // Initial distribution
            const angle = Math.random() * Math.PI * 2;
            const r = 5.0 + Math.random() * 5.0;
            initialData[i*8+0] = Math.cos(angle) * r;
            initialData[i*8+1] = Math.sin(angle) * r;
            initialData[i*8+2] = (Math.random() - 0.5) * 0.5;
            initialData[i*8+3] = 1.0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: this.numParticles * particleSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.renderUniformBuffer = this.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Bind Groups
        const computeLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });
        this.computeBindGroup = this.device.createBindGroup({
            layout: computeLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ]
        });

        const renderLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });
        this.renderBindGroup = this.device.createBindGroup({
            layout: renderLayout,
            entries: [
                { binding: 2, resource: { buffer: this.renderUniformBuffer } }
            ]
        });

        // Pipelines
        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: drawShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "⚠️ WebGPU Not Available (Running Hybrid Mode)";
        msg.style.cssText = "position:absolute; bottom:20px; right:20px; color:white; background:rgba(200,50,50,0.8); padding:10px; border-radius:8px;";
        this.container.appendChild(msg);
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        if (this.glCanvas) {
            this.glCanvas.width = Math.floor(w * dpr);
            this.glCanvas.height = Math.floor(h * dpr);
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = Math.floor(w * dpr);
            this.gpuCanvas.height = Math.floor(h * dpr);
        }
    }

    animate() {
        if (!this.isActive) return;
        const time = (Date.now() - this.startTime) * 0.001;

        // Render WebGL2
        if (this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGPU
        if (this.device && this.renderPipeline) {
            this.device.queue.writeBuffer(this.simParamBuffer, 0, new Float32Array([0.016, time, 0, 0]));
            this.device.queue.writeBuffer(this.renderUniformBuffer, 0, new Float32Array([this.gpuCanvas.width, this.gpuCanvas.height, time, 0]));

            const encoder = this.device.createCommandEncoder();

            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            const textureView = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
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

    destroy() {
        this.isActive = false;
        cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

// Global exposure for dashboard
if (typeof window !== 'undefined') {
    window.VoidSingularityExperiment = VoidSingularityExperiment;
}
