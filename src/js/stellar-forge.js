/**
 * Stellar Forge Experiment
 * Combines WebGL2 (Star Sphere) and WebGPU (Particle Accretion Disk).
 */

export class StellarForge {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };
        this.rotation = { x: 0, y: 0 };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
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
        this.numParticles = options.numParticles || 100000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000005';

        // 1. Initialize WebGL2 (Background Star)
        this.initWebGL2();

        // 2. Initialize WebGPU (Foreground Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("StellarForge: WebGPU initialization error:", e);
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
        const rect = this.container.getBoundingClientRect();
        // Normalize -1 to 1
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);

        // Update rotation target based on mouse
        this.rotation.x = this.mouse.y * 0.5;
        this.rotation.y = this.mouse.x * 0.5;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (The Star)
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

        // Generate Sphere Mesh
        const { vertices, indices } = this.createSphere(1.0, 32, 32);
        this.glIndexCount = indices.length;

        // Buffers
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_projection;
            uniform float u_time;

            out vec3 v_normal;
            out vec3 v_pos;
            out float v_displacement;

            // Simple noise
            float hash(float n) { return fract(sin(n) * 753.5453123); }
            float noise(vec3 x) {
                vec3 p = floor(x);
                vec3 f = fract(x);
                f = f * f * (3.0 - 2.0 * f);
                float n = p.x + p.y * 157.0 + 113.0 * p.z;
                return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                               mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
                           mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                               mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
            }

            void main() {
                v_normal = normalize(a_position);

                // Pulsating displacement
                float n = noise(a_position * 2.0 + u_time * 0.5);
                v_displacement = n;

                vec3 pos = a_position + v_normal * (n * 0.1);
                v_pos = pos;

                gl_Position = u_projection * u_view * u_model * vec4(pos, 1.0);
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_normal;
            in vec3 v_pos;
            in float v_displacement;

            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Star Surface Color
                vec3 coreColor = vec3(1.0, 0.4, 0.1); // Orange/Red
                vec3 hotColor = vec3(1.0, 0.9, 0.5);  // Yellow/White

                float activity = v_displacement; // 0 to 1 approx
                vec3 color = mix(coreColor, hotColor, activity);

                // Fresnel Glow
                vec3 viewDir = normalize(vec3(0.0, 0.0, 5.0) - v_pos); // Approximate view
                float fresnel = pow(1.0 - max(dot(v_normal, viewDir), 0.0), 3.0);

                color += vec3(1.0, 0.6, 0.2) * fresnel * 2.0;

                outColor = vec4(color, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        // VAO Setup
        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;

        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.CULL_FACE);
    }

    createSphere(radius, widthSegments, heightSegments) {
        const vertices = [];
        const indices = [];

        for (let y = 0; y <= heightSegments; y++) {
            const v = y / heightSegments;
            const theta = v * Math.PI;

            for (let x = 0; x <= widthSegments; x++) {
                const u = x / widthSegments;
                const phi = u * 2 * Math.PI;

                const px = radius * Math.sin(theta) * Math.cos(phi);
                const py = radius * Math.cos(theta);
                const pz = radius * Math.sin(theta) * Math.sin(phi);

                vertices.push(px, py, pz);
            }
        }

        for (let y = 0; y < heightSegments; y++) {
            for (let x = 0; x < widthSegments; x++) {
                const first = (y * (widthSegments + 1)) + x;
                const second = first + widthSegments + 1;

                indices.push(first, second, first + 1);
                indices.push(second, second + 1, first + 1);
            }
        }
        return { vertices: new Float32Array(vertices), indices: new Uint16Array(indices) };
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) return null;

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) return null;

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
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2; pointer-events: none; background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        const wgslCommon = `
            struct Particle {
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, mass
            }
            struct Params {
                modelViewProj : mat4x4f,
                time : f32,
                dt : f32,
                radius : f32, // Star radius
                pad : f32,
            }
        `;

        const computeShader = `
            ${wgslCommon}
            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : Params;

            // Random number generator
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];

                // Gravity Center
                let center = vec3f(0.0, 0.0, 0.0);
                let distVec = center - p.pos.xyz;
                let dist = length(distVec);

                // Accretion Disk Physics
                // Tangential velocity for orbit + inward pull
                let dir = normalize(distVec);

                // Gravity pull
                let gravity = dir * (10.0 / (dist * dist + 0.1));

                // Tangential push (Rotation around Y axis)
                // Cross product of UP (0,1,0) and Dir gives tangent
                let tangent = normalize(cross(vec3f(0.0, 1.0, 0.0), dir));

                // Combined Force: Gravity + Tangential
                var force = gravity * 0.5 + tangent * 0.8;

                // Jets: If near center and high vertical position, shoot up/down
                if (dist < params.radius * 1.5 && abs(p.pos.y) > 0.5) {
                    let signY = sign(p.pos.y);
                    force = vec3f(0.0, signY * 15.0, 0.0);
                }

                p.vel = vec4f(p.vel.xyz + force.xyz * params.dt, p.vel.w);

                // Drag / Damping
                p.vel = vec4f(p.vel.xyz * 0.98, p.vel.w);

                // Update Pos
                p.pos = vec4f(p.pos.xyz + p.vel.xyz * params.dt, p.pos.w);

                // Respawn if too close or too far
                if (dist < params.radius * 0.8 || dist > 20.0) {
                    let angle = rand(vec2f(p.pos.x, params.time)) * 6.28;
                    let r = 5.0 + rand(vec2f(p.pos.z, params.time)) * 5.0;
                    p.pos = vec4f(
                        cos(angle) * r,
                        (rand(vec2f(p.pos.y, params.time)) - 0.5) * 0.5, // Thin disk
                        sin(angle) * r,
                        1.0
                    );

                    // Initial Orbital Velocity
                    let initialVelDir = normalize(cross(vec3f(0.0, 1.0, 0.0), normalize(vec3f(0.0,0.0,0.0) - p.pos.xyz)));
                    p.vel = vec4f(initialVelDir * 3.0, 1.0);
                }

                particles[idx] = p;
            }
        `;

        const renderShader = `
            ${wgslCommon}
            @group(0) @binding(1) var<uniform> params : Params;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                output.position = params.modelViewProj * vec4f(pos.xyz, 1.0);

                // Point size based on distance?
                // WebGPU points are always 1px unless using textured quads expansion.
                // For this demo, 1px points are fine for high density.

                let speed = length(vel.xyz);

                // Color ramp: Orange -> Blue (Hotter)
                let cCold = vec3f(1.0, 0.3, 0.0);
                let cHot = vec3f(0.2, 0.8, 1.0);

                let t = smoothstep(0.0, 8.0, speed);
                output.color = vec4f(mix(cCold, cHot, t), 1.0); // Additive blending handles alpha

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const pSize = 32; // 8 floats * 4 bytes
        this.particleBuffer = this.device.createBuffer({
            size: this.numParticles * pSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        // Init Particles
        const initData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = 5.0 + Math.random() * 5.0;
            const x = Math.cos(angle) * r;
            const z = Math.sin(angle) * r;
            const y = (Math.random() - 0.5) * 0.5; // Disk height

            initData[i*8+0] = x;
            initData[i*8+1] = y;
            initData[i*8+2] = z;
            initData[i*8+3] = 1.0; // Life

            // Initial velocity tangent
            const dir = Math.atan2(z, x);
            const vSpeed = 2.0;
            initData[i*8+4] = Math.sin(dir) * vSpeed; // vx
            initData[i*8+5] = 0.0;
            initData[i*8+6] = -Math.cos(dir) * vSpeed; // vz
            initData[i*8+7] = 1.0; // Mass
        }
        this.device.queue.writeBuffer(this.particleBuffer, 0, initData);

        this.simParamBuffer = this.device.createBuffer({
            size: 80, // mat4 (64) + time, dt, radius, pad (16)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Layouts
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        const renderModule = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: pSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' },
            depthStencil: { // Enable depth testing so particles can be behind the star (if star was in GPU, but it's separate layer. Just depth sort against self?)
                // Actually, for additive particles, depth write is usually off, depth test is on.
                depthWriteEnabled: false,
                depthCompare: 'less',
                format: 'depth24plus',
            }
        });

        // Depth texture for particles
        this.depthTexture = this.device.createTexture({
            size: [this.gpuCanvas.width, this.gpuCanvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `position: absolute; bottom: 20px; right: 20px; color: red; font-family: monospace;`;
        msg.textContent = "WebGPU Not Available - Particles Disabled";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        if (w === 0 || h === 0) return;

        this.canvasSize.width = w;
        this.canvasSize.height = h;

        const dw = Math.floor(w * dpr);
        const dh = Math.floor(h * dpr);

        if (this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            this.gl.viewport(0, 0, dw, dh);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = dw;
            this.gpuCanvas.height = dh;
            // Recreate depth texture on resize
            if (this.device) {
                if (this.depthTexture) this.depthTexture.destroy();
                this.depthTexture = this.device.createTexture({
                    size: [dw, dh],
                    format: 'depth24plus',
                    usage: GPUTextureUsage.RENDER_ATTACHMENT,
                });
            }
        }
    }

    // Simple Matrix Helper (Column Major)
    perspective(fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        return [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ];
    }

    // LookAt
    lookAt(eye, center, up) {
        const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
        const len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        const zx = z0 * len, zy = z1 * len, zz = z2 * len;

        const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
        const lenX = 1 / Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
        const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;

        const y0 = zy * xz - zz * xy, y1 = zz * xx - zx * xz, y2 = zx * xy - zy * xx;
        const lenY = 1 / Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
        const yx = y0 * lenY, yy = y1 * lenY, yz = y2 * lenY;

        return [
            xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
            -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
            -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
            1
        ];
    }

    multiply(a, b) {
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

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const aspect = this.canvasSize.width / this.canvasSize.height;

        // Camera Logic
        const camX = Math.sin(time * 0.1) * 15.0;
        const camZ = Math.cos(time * 0.1) * 15.0;
        const view = this.lookAt([camX, 8.0 + this.rotation.x * 5, camZ], [0, 0, 0], [0, 1, 0]);
        const proj = this.perspective(Math.PI / 4, aspect, 0.1, 100.0);

        // Model Matrix for Star (Rotate it)
        const model = [
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 // Identity for now
        ];

        // 1. Render WebGL2 (Star)
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Upload Uniforms
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_model'), false, new Float32Array(model));
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_view'), false, new Float32Array(view));
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_projection'), false, new Float32Array(proj));
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);

            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
            this.gl.clearColor(0, 0, 0, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU (Particles)
        if (this.device && this.renderPipeline) {
            // Combine ViewProj for GPU
            const viewProj = this.multiply(view, proj); // Note: Manual matrix multiply is col-major sensitive
            // Correct multiplication: Proj * View. My multiply func is (a * b), so if I pass (view, proj), it calculates View * Proj (row major logic) or...
            // Standard GL: P * V * M * v.
            // My multiply function:
            // out[row i, col j] = sum(a[row i, k] * b[row k, col j])
            // This is standard matrix mult.
            // If arrays are column major: A * B means B is applied first then A.
            // So we want Proj * View.
            // But let's check my multiply implementation again.
            // a[i*4 + k] implies `i` is column if column-major? No, usually `idx = col * 4 + row`.
            // If input is flat array [0..15], `a[i*4 + k]` treats `i` as major index.
            // If I want Proj * View, I should check how I structured it.
            // Let's just trust standard implementation:

            // Re-implement multiply for Column-Major arrays (standard WebGL/WebGPU)
            const multiplyCM = (a, b) => {
                const out = new Float32Array(16);
                for (let col = 0; col < 4; col++) {
                   for (let row = 0; row < 4; row++) {
                       let sum = 0;
                       for (let k = 0; k < 4; k++) {
                           // A[row, k] * B[k, col]
                           // A index: k*4 + row
                           // B index: col*4 + k
                           sum += a[k * 4 + row] * b[col * 4 + k];
                       }
                       out[col * 4 + row] = sum;
                   }
                }
                return out;
            };

            const pv = multiplyCM(proj, view); // Proj * View

            // Update Uniforms
            const params = new Float32Array(20); // 80 bytes
            params.set(pv, 0); // 0-15: Matrix
            params[16] = time;
            params[17] = 0.016; // dt
            params[18] = 1.0; // Star Radius

            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute Pass
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render Pass
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: this.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                }
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup); // Re-use bind group for uniforms
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
    window.StellarForge = StellarForge;
}
