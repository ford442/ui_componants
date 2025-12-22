/**
 * Temporal Data Core
 * Demonstrates a hybrid WebGL2 Raymarched artifact with WebGPU Particle Vortex.
 * - WebGL2: Renders a raymarched, twisting torus artifact representing the "Core".
 * - WebGPU: Renders a temporal distortion field (particles) reacting to the core.
 */

class TemporalCore {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // Parameters
        this.params = {
            speed: 1.0,
            distortion: 0.5,
            color: [0.2, 0.8, 1.0]
        };

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
        this.numParticles = options.numParticles || 20000;

        // Bind resize handler
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050510';

        // 1. Initialize WebGL2 Layer (The Core)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (The Field)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("TemporalCore: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Raymarched Twisted Core)
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
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_resolution;
            out vec4 outColor;

            // Rotation matrix
            mat2 rot(float a) {
                float s = sin(a);
                float c = cos(a);
                return mat2(c, -s, s, c);
            }

            // SDF Primitives
            float sdTorus(vec3 p, vec2 t) {
                vec2 q = vec2(length(p.xz) - t.x, p.y);
                return length(q) - t.y;
            }

            // Scene Mapping
            float map(vec3 p) {
                // Twist
                float k = 2.0 * sin(u_time * 0.5); // Twist amount
                float c = cos(k * p.y);
                float s = sin(k * p.y);
                mat2 m = mat2(c, -s, s, c);
                vec3 q = vec3(m * p.xz, p.y);

                // Rotate entire object
                q.xz *= rot(u_time * 0.3);
                q.xy *= rot(u_time * 0.2);

                return sdTorus(q, vec2(1.0, 0.3));
            }

            // Normals
            vec3 calcNormal(vec3 p) {
                float d = 0.001;
                return normalize(vec3(
                    map(p + vec3(d, 0, 0)) - map(p - vec3(d, 0, 0)),
                    map(p + vec3(0, d, 0)) - map(p - vec3(0, d, 0)),
                    map(p + vec3(0, 0, d)) - map(p - vec3(0, 0, d))
                ));
            }

            void main() {
                vec2 uv = v_uv;
                uv.x *= u_resolution.x / u_resolution.y;

                vec3 ro = vec3(0.0, 0.0, -4.0); // Ray origin
                vec3 rd = normalize(vec3(uv, 1.0)); // Ray direction

                float t = 0.0;
                float d = 0.0;

                // Raymarching loop
                int i;
                for(i = 0; i < 64; i++) {
                    vec3 p = ro + rd * t;
                    d = map(p);
                    t += d;
                    if(d < 0.001 || t > 20.0) break;
                }

                vec3 col = vec3(0.0);

                if(d < 0.001) {
                    vec3 p = ro + rd * t;
                    vec3 n = calcNormal(p);

                    // Simple lighting
                    vec3 lightPos = vec3(2.0, 2.0, -3.0);
                    vec3 l = normalize(lightPos - p);
                    float diff = max(dot(n, l), 0.0);
                    float amb = 0.1;

                    // Iridescence based on normal
                    vec3 irid = 0.5 + 0.5 * cos(u_time + p.xyx + vec3(0, 2, 4));

                    col = vec3(diff + amb) * irid;

                    // Rim light
                    float rim = 1.0 - max(dot(n, -rd), 0.0);
                    rim = pow(rim, 3.0);
                    col += vec3(0.5, 0.8, 1.0) * rim;
                }

                // Distance fog
                col = mix(col, vec3(0.05, 0.05, 0.1), 1.0 - exp(-0.1 * t * t));

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
        this.resizeGL();
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('TemporalCore WebGL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('TemporalCore WebGL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Vortex Particles)
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

        if (!navigator.gpu) return false;

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

        // Compute Shader
        const computeShaderCode = `
            struct Particle {
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, unused
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                speed : f32,
                _pad : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            // Pseudo-random function
            fn hash(u: u32) -> f32 {
                var x = u;
                x = (x ^ 61u) ^ (x >> 16u);
                x = x * 9u;
                x = x ^ (x >> 4u);
                x = x * 668265261u;
                x = x ^ (x >> 15u);
                return f32(x) / 4294967296.0;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Update Life
                p.pos.w -= params.dt * 0.2; // Decay

                // Respawn if dead
                if (p.pos.w <= 0.0) {
                    p.pos.w = 1.0;
                    // Respawn in a shell around the core
                    let theta = hash(index + u32(params.time * 1000.0)) * 6.28;
                    let phi = hash(index + 1u + u32(params.time * 1000.0)) * 3.14;
                    let r = 2.5 + hash(index + 2u) * 0.5;

                    p.pos.x = r * sin(phi) * cos(theta);
                    p.pos.y = r * sin(phi) * sin(theta);
                    p.pos.z = r * cos(phi);

                    p.vel = vec4f(0.0);
                }

                // Vortex Force field
                // Particles are attracted to the center but spiral around Y axis
                let center = vec3f(0.0);
                let diff = center - p.pos.xyz;
                let dist = length(diff);
                let dir = normalize(diff);

                // Tangent force (Spiral)
                let tangent = normalize(cross(vec3f(0.0, 1.0, 0.0), p.pos.xyz));

                let speed = params.speed;

                // Forces
                let attraction = dir * (1.0 / (dist + 0.1)) * 2.0;
                let swirl = tangent * 4.0;

                // Apply forces
                p.vel.x += (attraction.x + swirl.x) * params.dt * speed;
                p.vel.y += (attraction.y + swirl.y) * params.dt * speed;
                p.vel.z += (attraction.z + swirl.z) * params.dt * speed;

                // Damping
                p.vel.x *= 0.98;
                p.vel.y *= 0.98;
                p.vel.z *= 0.98;

                // Update Pos
                p.pos.x += p.vel.x * params.dt;
                p.pos.y += p.vel.y * params.dt;
                p.pos.z += p.vel.z * params.dt;

                particles[index] = p;
            }
        `;

        // Render Shader
        const renderShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) particlePos : vec4f, @location(1) particleVel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Simple perspective projection (simulating camera at 0,0,-4)
                let pos = particlePos.xyz;
                let viewZ = pos.z - 4.0;

                // Projection parameters
                let fov = 1.5;
                let aspect = 1.0; // Normalized aspect in this space
                let scale = 1.0 / -viewZ;

                output.position = vec4f(
                    pos.x * scale * aspect,
                    pos.y * scale,
                    0.0,
                    1.0
                );

                // Color based on velocity and life
                let speed = length(particleVel.xyz);
                let life = particlePos.w;

                let energy = speed * 2.0;
                output.color = vec4f(
                    0.2 + energy,   // Red
                    0.5 + energy * 0.5, // Green
                    1.0,            // Blue
                    life * smoothstep(2.5, 3.5, abs(viewZ)) // Fade if too close/far
                );

                // Point size
                output.position.w = 1.0;

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialize Particles
        const particleUnitSize = 32; // 8 floats * 4 bytes = 32 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        // Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group Layout
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

        const renderModule = this.device.createShaderModule({ code: renderShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos + life
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // vel
                    ],
                }],
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one' }, // Additive blending
                        alpha: { srcFactor: 'zero', dstFactor: 'one' },
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
            position: absolute; bottom: 20px; right: 20px; color: #f55;
            border: 1px solid #f55; padding: 5px 10px; border-radius: 4px;
            font-size: 12px; pointer-events: none;
        `;
        msg.innerText = "WebGPU not supported - Showing WebGL2 Core only";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
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

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time * this.params.speed);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.02, 0.02, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. WebGPU Render
        if (this.device && this.context && this.renderPipeline) {
            // Update uniforms
            const params = new Float32Array([0.016, time, this.params.speed, 0]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render
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

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();

        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.TemporalCore = TemporalCore;
}
