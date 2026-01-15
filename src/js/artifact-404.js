/**
 * Artifact 404 Experiment
 * Hybrid WebGL2 (Glitching Geometry) + WebGPU (Chaos Particle Swarm)
 */

export class Artifact404 {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0.0, y: 0.0 };
        this.glitchIntensity = 0.0;
        this.canvasSize = { width: 0, height: 0 };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.numIndices = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.uniformBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.particleCount || 40000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        console.log("Artifact404: Initializing...");

        // 1. Initialize WebGL2 Layer (The Artifact)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (The Swarm)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("Artifact404: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Normalize mouse to [-1, 1]
        this.mouse.x = (e.clientX / width) * 2 - 1;
        this.mouse.y = -(e.clientY / height) * 2 + 1;

        // Glitch intensity increases as mouse moves right
        // Map mouse X from [-1, 1] to [0, 1] roughly
        this.glitchIntensity = Math.max(0, this.mouse.x * 0.5 + 0.5);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Glitchy Icosahedron)
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

        // Generate Icosahedron Geometry
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;
        const vertices = [
            -1,  t,  0,    1,  t,  0,   -1, -t,  0,    1, -t,  0,
             0, -1,  t,    0,  1,  t,    0, -1, -t,    0,  1, -t,
             t,  0, -1,    t,  0,  1,   -t,  0, -1,   -t,  0,  1
        ];

        // Normalize vertices to project onto sphere
        for(let i=0; i<vertices.length; i+=3) {
            const length = Math.sqrt(vertices[i]**2 + vertices[i+1]**2 + vertices[i+2]**2);
            vertices[i] /= length;
            vertices[i+1] /= length;
            vertices[i+2] /= length;
        }

        const indices = [
            0, 11, 5,    0, 5, 1,    0, 1, 7,    0, 7, 10,   0, 10, 11,
            1, 5, 9,     5, 11, 4,   11, 10, 2,  10, 7, 6,   7, 1, 8,
            3, 9, 4,     3, 4, 2,    3, 2, 6,    3, 6, 8,    3, 8, 9,
            4, 9, 5,     2, 4, 11,   6, 2, 10,   8, 6, 7,    9, 8, 1
        ];

        this.numIndices = indices.length;

        // Shaders
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_projection;
            uniform float u_time;
            uniform float u_glitch;

            out vec3 v_normal;
            out vec3 v_pos;
            out float v_noise;

            // Simple pseudo-random
            float random(vec2 st) {
                return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
            }

            void main() {
                vec3 pos = a_position;

                // Glitch effect: Displace vertices randomly based on time and glitch intensity
                float noise = random(vec2(u_time * 5.0, float(gl_VertexID)));

                if (noise < u_glitch * 0.5) {
                    pos += normalize(pos) * (noise * 0.5); // Spike out
                }

                // Jitter position
                if (noise > 0.95 - u_glitch * 0.2) {
                     pos.x += (random(vec2(u_time, 0.0)) - 0.5) * u_glitch;
                }

                v_normal = pos;
                v_pos = pos;
                v_noise = noise;

                gl_Position = u_projection * u_view * u_model * vec4(pos, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_normal;
            in vec3 v_pos;
            in float v_noise;

            uniform float u_time;
            uniform float u_glitch;

            out vec4 outColor;

            void main() {
                // Holographic Rim Lighting
                vec3 viewDir = vec3(0.0, 0.0, 1.0); // Simplified view dir
                vec3 normal = normalize(v_normal);

                float rim = 1.0 - max(dot(viewDir, normal), 0.0);
                rim = pow(rim, 3.0);

                vec3 baseColor = vec3(0.1, 0.0, 0.2); // Dark purple
                vec3 glowColor = vec3(1.0, 0.0, 0.5); // Neon pink/red

                // Glitch bands
                float band = sin(v_pos.y * 20.0 + u_time * 10.0);
                if (band > 0.9 && u_glitch > 0.2) {
                    glowColor = vec3(0.0, 1.0, 1.0); // Cyan glitch flash
                }

                vec3 finalColor = mix(baseColor, glowColor, rim);

                // Wireframe-ish feel via alpha
                float alpha = rim * 0.8 + 0.1;

                // Intense core if glitching
                if (u_glitch > 0.5 && v_noise > 0.8) {
                    finalColor += vec3(1.0);
                    alpha = 1.0;
                }

                outColor = vec4(finalColor, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);

        // Buffers
        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const posBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const loc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 3, this.gl.FLOAT, false, 0, 0);

        const idxBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if(!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error("VS Error:", this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if(!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error("FS Error:", this.gl.getShaderInfoLog(fs));
            return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Chaos Swarm)
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
            alphaMode: 'premultiplied',
        });

        // Compute Shader: Chaos Attractor
        const computeCode = `
            struct Particle {
                pos: vec4f, // x, y, z, w (unused)
                vel: vec4f, // vx, vy, vz, life
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Uniforms {
                time: f32,
                dt: f32,
                glitch: f32,
                mouseX: f32,
            }
            @group(0) @binding(1) var<uniform> u : Uniforms;

            // Lorenz-like attractor or similar chaotic system
            fn attractor(p: vec3f) -> vec3f {
                let sigma = 10.0;
                let rho = 28.0;
                let beta = 8.0 / 3.0;

                // Modified chaos based on glitch
                let r = rho * (1.0 + u.glitch * 5.0); // Wildly vary rho

                let dx = sigma * (p.y - p.x);
                let dy = p.x * (r - p.z) - p.y;
                let dz = p.x * p.y - beta * p.z;

                return vec3f(dx, dy, dz);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let i = GlobalInvocationID.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];
                var pos = p.pos.xyz;
                var vel = p.vel.xyz;

                // Orbit logic
                // Center is (0,0,0)

                // Apply a simple curl noise or orbital velocity
                let dist = length(pos);
                let dir = normalize(pos);

                // Tangent force
                let tangent = cross(dir, vec3f(0.0, 1.0, 0.0));

                // Attraction to center (the artifact)
                let attraction = -dir * (1.0 / (dist + 0.1)) * 5.0;

                // Mouse repulsion/attraction
                // Not implementing full 3D mouse picking, just abstract "Agitation"

                var force = attraction + tangent * 2.0;

                // Add Chaos
                if (u.glitch > 0.1) {
                    // Random jumps
                    if (fract(sin(u.time * 100.0 + f32(i)) * 43758.5453) < 0.01 * u.glitch) {
                         pos = pos * 1.5; // Explode outward
                    }

                    // Turbulent noise force
                    force += vec3f(
                        sin(pos.y * 5.0 + u.time),
                        sin(pos.z * 5.0 + u.time),
                        sin(pos.x * 5.0 + u.time)
                    ) * 10.0 * u.glitch;
                }

                vel = vel + force * u.dt;
                vel = vel * 0.95; // Damping

                pos = pos + vel * u.dt;

                // Reset if lost
                if (length(pos) > 20.0) {
                    pos = normalize(pos) * 5.0;
                    vel = vec3f(0.0);
                }

                particles[i].pos = vec4f(pos, 1.0);
                particles[i].vel = vec4f(vel, 1.0);
            }
        `;

        const renderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct Uniforms {
                time: f32,
                dt: f32,
                glitch: f32,
                mouseX: f32,
            }
            @group(0) @binding(1) var<uniform> u : Uniforms;

            // Standard camera uniforms
            struct Camera {
                viewProj: mat4x4f,
            }
            @group(0) @binding(2) var<uniform> cam : Camera;

            @vertex
            fn vs_main(@location(0) pos : vec4f) -> VertexOutput {
                var output : VertexOutput;

                output.position = cam.viewProj * vec4f(pos.xyz, 1.0);

                // Point size emulation not available in WGSL directly like gl_PointSize in all implementations,
                // but we render as points topology.

                // Color based on velocity/chaos
                var c = vec3f(0.0, 1.0, 0.5); // Cyan
                if (u.glitch > 0.3) {
                    c = vec3f(1.0, 0.0, 0.5); // Red/Pink
                }

                // Fade distant points
                let dist = output.position.z;
                let alpha = 1.0 / (dist * 0.1 + 1.0);

                output.color = vec4f(c, alpha);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Create Particles
        const pData = new Float32Array(this.numParticles * 8); // 4 pos, 4 vel
        for(let i=0; i<this.numParticles; i++) {
            // Sphere distribution
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 3.0 + Math.random() * 2.0;

            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);

            pData[i*8+0] = x;
            pData[i*8+1] = y;
            pData[i*8+2] = z;
            pData[i*8+3] = 1.0;

            pData[i*8+4] = 0;
            pData[i*8+5] = 0;
            pData[i*8+6] = 0;
            pData[i*8+7] = 1;
        }

        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(pData);
        this.particleBuffer.unmap();

        // Uniforms
        this.uniformBuffer = this.device.createBuffer({
            size: 32, // 4 floats -> 16 bytes, padded to 32? No, 16 is fine, but lets align 16
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.cameraBuffer = this.device.createBuffer({
            size: 64, // mat4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Layouts
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                // Binding 2 not needed in compute
            ]
        });

        // Wait, I need binding 2 in Vertex for Camera
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                // Binding 0 is Vertex Buffer (implicit in pipeline usually, or explicit if storage)
                // Actually, render pipeline doesn't use storage buffer binding for position if using vertex attributes.
                // But let's check my shader. VS uses @location(0) pos, so it comes from vertex buffer.
                // But it also uses Uniform u (binding 1) and Cam (binding 2).

                // However, bind group layouts must match.
                // Let's separate them or just include unused ones?

                // Re-design:
                // Group 0: Storage(0), Uniforms(1), Camera(2)
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // Not used in VS?
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout, // Reuse same layout for simplicity if compatible
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
                { binding: 2, resource: { buffer: this.cameraBuffer } }, // Bind it even if compute doesn't use it, to match layout
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        const renderModule = this.device.createShaderModule({ code: renderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32, // 8 floats * 4 bytes
                    stepMode: 'vertex',
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] // We only need pos
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        // ... (Similar to other classes)
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.canvasSize.width = w;
        this.canvasSize.height = h;

        if (this.glCanvas) {
            this.glCanvas.width = w * dpr;
            this.glCanvas.height = h * dpr;
            this.gl.viewport(0, 0, w * dpr, h * dpr);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = w * dpr;
            this.gpuCanvas.height = h * dpr;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016;

        // Model Rotation
        const rotationSpeed = 0.5 + this.glitchIntensity * 2.0;
        const rotY = time * rotationSpeed;
        const rotX = Math.sin(time * 0.5) * 0.2 + this.glitchIntensity;

        // Matrices
        const aspect = this.canvasSize.width / this.canvasSize.height;
        const projection = this.createPerspectiveMatrix(60, aspect, 0.1, 100.0);

        // Jitter camera on glitch
        let camZ = 8.0;
        if (this.glitchIntensity > 0.5 && Math.random() < 0.1) {
            camZ += (Math.random() - 0.5) * 0.5;
        }

        const view = this.createLookAtMatrix([0, 0, camZ], [0, 0, 0], [0, 1, 0]);

        // Model Matrix (Rotate Icosahedron)
        const model = new Float32Array([
            Math.cos(rotY), 0, -Math.sin(rotY), 0,
            0, 1, 0, 0,
            Math.sin(rotY), 0, Math.cos(rotY), 0,
            0, 0, 0, 1
        ]);
        // Simple rotation matrix application is incomplete here, but for now just Y rot is fine.
        // Actually lets do a proper rotation helper or just inline simple one.

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_projection'), false, projection);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_view'), false, view);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_model'), false, model);

            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_glitch'), this.glitchIntensity);

            this.gl.clearColor(0, 0, 0, 0); // Transparent!
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.computePipeline) {
            // Update Uniforms
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([
                time, dt, this.glitchIntensity, this.mouse.x
            ]));

            // Update Camera Buffer (View * Proj)
            const vp = this.multiplyMatrices(projection, view);
            this.device.queue.writeBuffer(this.cameraBuffer, 0, vp);

            const encoder = this.device.createCommandEncoder();

            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            const texView = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: texView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setVertexBuffer(0, this.particleBuffer);
            rPass.setBindGroup(0, this.computeBindGroup);
            rPass.draw(this.numParticles);
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);
        // cleanup context
    }

    // Math Helpers
    createPerspectiveMatrix(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov * Math.PI / 360);
        const rangeInv = 1.0 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0
        ]);
    }

    createLookAtMatrix(eye, target, up) {
         let z = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
        const len = Math.sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
        if(len > 0) z = z.map(v => v / len);

        let x = [
            up[1]*z[2] - up[2]*z[1],
            up[2]*z[0] - up[0]*z[2],
            up[0]*z[1] - up[1]*z[0]
        ];
        const lenX = Math.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        if(lenX > 0) x = x.map(v => v / lenX);

        let y = [
            z[1]*x[2] - z[2]*x[1],
            z[2]*x[0] - z[0]*x[2],
            z[0]*x[1] - z[1]*x[0]
        ];

        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]),
            -(y[0]*eye[0] + y[1]*eye[1] + y[2]*eye[2]),
            -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]),
            1
        ]);
    }

    multiplyMatrices(a, b) {
        const out = new Float32Array(16);
        for (let r = 0; r < 4; ++r) {
            for (let c = 0; c < 4; ++c) {
                let sum = 0;
                for (let k = 0; k < 4; ++k) {
                    sum += b[r * 4 + k] * a[k * 4 + c];
                }
                out[r * 4 + c] = sum;
            }
        }
        return out;
    }
}
