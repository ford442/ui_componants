/**
 * Gravitational Nebula Experiment
 * Combines WebGL2 (Raymarched Black Hole) and WebGPU (N-Body Particle System).
 */

class GravitationalNebula {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // WebGL2 State (Background / Black Hole)
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;

        // WebGPU State (Particle Swarm)
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.uniformBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 100000;

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        // 1. Initialize WebGL2 (The Black Hole)
        this.initWebGL2();

        // 2. Initialize WebGPU (The Accretion Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("GravitationalNebula: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;

        window.addEventListener('resize', this.handleResize);
        this.animate();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Raymarched Black Hole)
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

        // Fullscreen Quad
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

            // Simple Raymarch / SDF
            float sdSphere(vec3 p, float s) {
                return length(p) - s;
            }

            void main() {
                vec2 p = (v_uv * 2.0 - 1.0) * (u_resolution / u_resolution.y);

                vec3 ro = vec3(0.0, 2.0, -4.0); // Ray origin
                vec3 rd = normalize(vec3(p, 1.5)); // Ray direction

                // Tilt the camera view slightly down
                float angle = 0.3;
                float s = sin(angle);
                float c = cos(angle);
                mat3 rotX = mat3(
                    1.0, 0.0, 0.0,
                    0.0, c, -s,
                    0.0, s, c
                );
                ro = rotX * ro;
                rd = rotX * rd;

                vec3 col = vec3(0.0);
                float t = 0.0;
                float minDist = 100.0;

                // Raymarching Loop
                for(int i=0; i<64; i++) {
                    vec3 pos = ro + rd * t;

                    // Black hole core (sphere at 0,0,0)
                    float d = length(pos) - 0.5;

                    // Gravitational Lensing approx (bend ray towards center)
                    // Real lensing is hard, we cheat by accumulating 'glow' based on distance to center
                    float glowDist = length(pos);
                    col += vec3(0.05, 0.02, 0.1) * (0.02 / (abs(d) + 0.01));

                    // Accretion Disk (flat ring)
                    float diskH = abs(pos.y);
                    float diskR = length(pos.xz);
                    if (diskR > 0.6 && diskR < 2.5) {
                        float density = exp(-diskH * 10.0) * exp(-(diskR - 1.0)*2.0);
                        col += vec3(1.0, 0.6, 0.2) * density * 0.05;
                    }

                    t += max(d * 0.5, 0.02); // Small steps for volumetric feel
                    if(d < 0.01) {
                        col = vec3(0.0); // Hit Event Horizon
                        break;
                    }
                    if(t > 10.0) break;
                }

                // Starfield background
                if (length(col) < 0.1) {
                    float stars = fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453);
                    if (stars > 0.995) col += vec3(1.0);
                }

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.resizeGL();
    }

    createGLProgram(vs, fs) {
        const vShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vShader, vs);
        this.gl.compileShader(vShader);
        if (!this.gl.getShaderParameter(vShader, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vShader));
            return null;
        }

        const fShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fShader, fs);
        this.gl.compileShader(fShader);
        if (!this.gl.getShaderParameter(fShader, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fShader));
            return null;
        }

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vShader);
        this.gl.attachShader(prog, fShader);
        this.gl.linkProgram(prog);
        return prog;
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
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        const computeShader = `
            struct Particle {
                pos : vec4f,
                vel : vec4f,
            }

            struct Uniforms {
                dt : f32,
                time : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                let dist = length(p.pos.xyz);

                // Gravity: F = G*M / r^2
                // We simplify. Pull towards center.
                let dir = normalize(-p.pos.xyz);
                let force = 2.0 / (dist * dist + 0.1);

                p.vel.x += dir.x * force * uniforms.dt;
                p.vel.y += dir.y * force * uniforms.dt;
                p.vel.z += dir.z * force * uniforms.dt;

                // Update Pos
                p.pos.x += p.vel.x * uniforms.dt;
                p.pos.y += p.vel.y * uniforms.dt;
                p.pos.z += p.vel.z * uniforms.dt;

                // Event Horizon Reset
                if (dist < 0.5 || dist > 10.0) {
                    // Reset to outer rim
                    let angle = uniforms.time + f32(index);
                    let r = 3.0 + (fract(sin(f32(index))*43758.54) * 2.0);
                    p.pos.x = cos(angle) * r;
                    p.pos.z = sin(angle) * r;
                    p.pos.y = (fract(cos(f32(index))*23421.2) - 0.5) * 0.2; // Thin disk

                    // Tangential Velocity for orbit
                    let v = 1.5;
                    p.vel.x = -sin(angle) * v;
                    p.vel.z = cos(angle) * v;
                    p.vel.y = 0.0;
                }

                particles[index] = p;
            }
        `;

        const renderShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Camera Transform (Match WebGL2)
                // ro = (0, 2, -4) looking at (0,0,0)
                // Tilt angle 0.3 rads

                let angle = 0.3;
                let c = cos(angle);
                let s = sin(angle);

                // Apply Camera Rotation (inverse of ray rot)
                // World -> Camera
                let rx = pos.x;
                let ry = pos.y * c - pos.z * s;
                let rz = pos.y * s + pos.z * c;

                // Translate
                let tx = rx;
                let ty = ry - 2.0; // Move camera up = move world down
                let tz = rz + 4.0; // Move camera back = move world forward

                // Project
                // Simple perspective: x' = x/z
                let z = tz;
                output.position = vec4f(tx / z * 2.5, ty / z * 2.5, 0.0, 1.0);

                // Color based on speed
                let speed = length(vel.xyz);
                let hot = vec3f(1.0, 0.8, 0.5);
                let cold = vec3f(0.5, 0.1, 0.2);
                let col = mix(cold, hot, clamp(speed - 1.0, 0.0, 1.0));

                output.color = vec4f(col, 1.0);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Init Particles
        const data = new Float32Array(this.numParticles * 8); // pos(4), vel(4)
        for(let i=0; i<this.numParticles; i++) {
            const idx = i*8;
            const angle = Math.random() * Math.PI * 2;
            const r = 2.0 + Math.random() * 2.0;

            // Pos
            data[idx] = Math.cos(angle) * r;
            data[idx+1] = (Math.random() - 0.5) * 0.2; // Y spread
            data[idx+2] = Math.sin(angle) * r;
            data[idx+3] = 1.0;

            // Vel (Tangent)
            const v = Math.sqrt(2.0 / r); // Orbital velocity approx
            data[idx+4] = -Math.sin(angle) * v;
            data[idx+5] = 0.0;
            data[idx+6] = Math.cos(angle) * v;
            data[idx+7] = 0.0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, data);

        this.uniformBuffer = this.device.createBuffer({
            size: 16, // 2 floats * 4 bytes + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ],
        });

        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                        { shaderLocation: 1, offset: 16, format: 'float32x4' } // vel
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{ format: format, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
                } }]
            },
            primitive: { topology: 'point-list' },
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(200, 50, 50, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-family: monospace;
            z-index: 10;
            pointer-events: none;
        `;
        msg.textContent = "WebGPU Not Supported - Showing Background Only";
        this.container.appendChild(msg);
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.resizeGL(w * dpr, h * dpr);
        this.resizeGPU(w * dpr, h * dpr);
    }

    resizeGL(w, h) {
        if(!this.glCanvas) return;
        this.glCanvas.width = w;
        this.glCanvas.height = h;
        this.gl.viewport(0, 0, w, h);
    }

    resizeGPU(w, h) {
        if(!this.gpuCanvas) return;
        this.gpuCanvas.width = w;
        this.gpuCanvas.height = h;
    }

    animate() {
        if(!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;

        // WebGL2 Render
        if(this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // WebGPU Render
        if(this.device && this.renderPipeline) {
            const uniforms = new Float32Array([0.016, time, 0, 0]); // dt, time, padding
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

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
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

if(typeof window !== 'undefined') {
    window.GravitationalNebula = GravitationalNebula;
}
