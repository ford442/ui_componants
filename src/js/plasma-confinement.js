/**
 * Plasma Confinement Field Experiment
 * Demonstrates Hybrid WebGL2 + WebGPU rendering.
 * - WebGL2: Raymarched glass containment vessel with refraction.
 * - WebGPU: High-energy plasma particles simulated via Compute Shaders.
 */

export class PlasmaConfinement {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouseX = 0;
        this.mouseY = 0;
        // NEW: Confinement Strength
        this.confinementStrength = 1.0;

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
        this.numParticles = options.numParticles || 30000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        // 1. Initialize WebGL2 Layer (Background Container)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground Plasma)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("PlasmaConfinement: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        this.container.addEventListener('mouseup', this.handleMouseUp);
        // Also support touch
        this.container.addEventListener('touchstart', this.handleMouseDown);
        this.container.addEventListener('touchend', this.handleMouseUp);


        // Initial resize
        this.resize();
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        // Normalize mouse to -1 to 1
        this.mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    }

    onMouseDown(e) {
        // Invert strength to repel/explode
        this.confinementStrength = -2.0;
    }

    onMouseUp(e) {
        // Restore confinement
        this.confinementStrength = 1.0;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Raymarched Container)
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

        this.gl = this.glCanvas.getContext('webgl2', { alpha: false });
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
                float s = sin(a), c = cos(a);
                return mat2(c, -s, s, c);
            }

            // SDF for a hollow sphere/capsule
            float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
                vec3 pa = p - a, ba = b - a;
                float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                return length(pa - ba * h) - r;
            }

            float map(vec3 p) {
                // Rotate the scene
                p.xz *= rot(u_time * 0.2);
                p.yz *= rot(0.1);

                // Main confinement chamber (Sphere)
                float d = length(p) - 1.2;

                // Cut out inner to make it a shell (hollow)
                // We are rendering the shell, so distance is abs
                // But simple raymarch of just the surface:
                return abs(d) - 0.02; // Thickness 0.02
            }

            // Normal calculation
            vec3 calcNormal(vec3 p) {
                const float h = 0.001;
                const vec2 k = vec2(1, -1);
                return normalize(k.xyy * map(p + k.xyy * h) +
                                 k.yyx * map(p + k.yyx * h) +
                                 k.yxy * map(p + k.yxy * h) +
                                 k.xxx * map(p + k.xxx * h));
            }

            void main() {
                vec2 uv = v_uv;
                // Correct aspect ratio
                uv.x *= u_resolution.x / u_resolution.y;

                // Camera setup
                vec3 ro = vec3(0.0, 0.0, 3.5);
                vec3 rd = normalize(vec3(uv, -1.0));

                float t = 0.0;
                float tmax = 20.0;
                float d = 0.0;

                vec3 col = vec3(0.0);
                float glow = 0.0;

                // Raymarch
                for(int i = 0; i < 64; i++) {
                    vec3 p = ro + rd * t;
                    d = map(p);
                    if (d < 0.001 || t > tmax) break;
                    t += d;
                    glow += 0.01 / (0.01 + abs(d)); // Accumulate glow near surface
                }

                if (t < tmax) {
                    vec3 p = ro + rd * t;
                    vec3 n = calcNormal(p);
                    vec3 r = reflect(rd, n);

                    // Glassy appearance
                    float fresnel = pow(1.0 + dot(rd, n), 3.0);

                    // Base color
                    vec3 baseColor = vec3(0.1, 0.3, 0.5);

                    // Reflection (fake env)
                    vec3 refCol = vec3(0.8) * pow(max(dot(r, vec3(0,1,0)), 0.0), 10.0);

                    col = baseColor * 0.2 + refCol * fresnel + vec3(0.0, 0.5, 1.0) * glow * 0.05;

                    // Transparency hint
                    col += vec3(0.05, 0.1, 0.15);
                } else {
                    // Background glow
                    col = vec3(0.0, 0.02, 0.05) * glow * 0.5;
                }

                // Add center glow
                float centerGlow = 1.0 / (length(uv) + 0.1);
                col += vec3(0.0, 0.2, 0.4) * centerGlow * 0.05;

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
            console.error('Plasma VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('Plasma FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Plasma Particles)
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

        // Compute Shader: Particle Physics
        const computeShader = `
            struct Particle {
                pos : vec4f, // xyz, w=life/energy
                vel : vec4f, // xyz, w=padding
            }

            struct Params {
                dt : f32,
                time : f32,
                mouseX: f32,
                mouseY: f32,
                confinementStrength: f32,
                pad1: f32,
                pad2: f32,
                pad3: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : Params;

            // Simple noise function
            fn hash(p: u32) -> f32 {
                var p2 = p;
                p2 = (p2 << 13u) ^ p2;
                return (1.0 - f32((p2 * (p2 * p2 * 15731u + 789221u) + 1376312589u) & 0x7fffffffu) / 1073741824.0);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let i = GlobalInvocationID.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Update Position
                p.pos.x += p.vel.x * params.dt;
                p.pos.y += p.vel.y * params.dt;
                p.pos.z += p.vel.z * params.dt;

                // Confinement Force (Sphere Radius ~ 1.2)
                let radius = 1.1;
                let dist = length(p.pos.xyz);

                // Rotation force (swirl) around Y axis
                let angle = atan2(p.pos.z, p.pos.x);
                let swirlForce = 0.5;
                p.vel.x += -sin(angle) * swirlForce * params.dt;
                p.vel.z += cos(angle) * swirlForce * params.dt;

                // Attraction/Repulsion center
                let centerDir = normalize(-p.pos.xyz);
                // params.confinementStrength is 1.0 (attract) or -2.0 (repel)
                p.vel.x += centerDir.x * 0.8 * params.confinementStrength * params.dt;
                p.vel.y += centerDir.y * 0.8 * params.confinementStrength * params.dt;
                p.vel.z += centerDir.z * 0.8 * params.confinementStrength * params.dt;

                // Mouse Repulsion (Local Interaction)
                // Project mouse to approximate world coords
                let mouseWorld = vec3f(params.mouseX * 2.0, params.mouseY * 2.0, 0.0);
                let mouseDist = distance(p.pos.xyz, mouseWorld);
                if (mouseDist < 0.5) {
                    let repelDir = normalize(p.pos.xyz - mouseWorld);
                    let force = (1.0 - mouseDist/0.5) * 5.0;
                    p.vel.x += repelDir.x * force * params.dt;
                    p.vel.y += repelDir.y * force * params.dt;
                    p.vel.z += repelDir.z * force * params.dt;
                }

                // Turbulence
                p.vel.x += (hash(i + u32(params.time * 100.0)) - 0.5) * 0.1;
                p.vel.y += (hash(i + 1u + u32(params.time * 100.0)) - 0.5) * 0.1;
                p.vel.z += (hash(i + 2u + u32(params.time * 100.0)) - 0.5) * 0.1;

                // Boundary condition (Soft bounce if outside radius)
                if (dist > radius) {
                    // Only bounce if we are trying to confine
                    if (params.confinementStrength > 0.0) {
                        let n = normalize(-p.pos.xyz);
                        p.vel.x += n.x * 2.0 * params.dt;
                        p.vel.y += n.y * 2.0 * params.dt;
                        p.vel.z += n.z * 2.0 * params.dt;
                        // Dampen
                        p.vel.x *= 0.9;
                        p.vel.y *= 0.9;
                        p.vel.z *= 0.9;
                    }
                }

                // Global Damping
                p.vel.x *= 0.99;
                p.vel.y *= 0.99;
                p.vel.z *= 0.99;

                // Update energy/life for color pulsing
                p.pos.w = sin(params.time * 2.0 + f32(i) * 0.01) * 0.5 + 0.5;

                particles[i] = p;
            }
        `;

        // Render Shader
        const drawShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) pos : vec3f,
            }

            struct Uniforms {
                modelViewProjectionMatrix : mat4x4f,
            }
            @group(0) @binding(0) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @location(0) particlePos : vec4f,
                @location(1) particleVel : vec4f
            ) -> VertexOutput {
                var output : VertexOutput;
                output.position = uniforms.modelViewProjectionMatrix * vec4f(particlePos.xyz, 1.0);

                // Size attenuation based on depth could go here but using point-list for now
                // Actually WebGPU doesn't do point size in vertex shader by default like gl_PointSize
                // We'll just render single pixels or use a geometry shader approach if needed
                // But point-list renders 1px points.

                // Color based on velocity/energy
                let energy = length(particleVel.xyz);
                let life = particlePos.w;

                // Cyan/Electric Blue core, purple edge
                let coreColor = vec3f(0.5, 0.9, 1.0);
                let edgeColor = vec3f(0.8, 0.2, 1.0);

                output.color = vec4f(mix(edgeColor, coreColor, energy * 2.0), 0.8 * life);
                output.pos = particlePos.xyz;
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const pSize = 32; // 8 floats * 4 bytes
        const pBufferData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            // Random position inside sphere
            let u = Math.random();
            let v = Math.random();
            let theta = 2 * Math.PI * u;
            let phi = Math.acos(2 * v - 1);
            let r = Math.cbrt(Math.random()) * 1.0;

            pBufferData[i*8+0] = r * Math.sin(phi) * Math.cos(theta);
            pBufferData[i*8+1] = r * Math.sin(phi) * Math.sin(theta);
            pBufferData[i*8+2] = r * Math.cos(phi);
            pBufferData[i*8+3] = Math.random(); // life

            pBufferData[i*8+4] = 0; // vx
            pBufferData[i*8+5] = 0; // vy
            pBufferData[i*8+6] = 0; // vz
            pBufferData[i*8+7] = 0; // padding
        }

        this.particleBuffer = this.device.createBuffer({
            size: pBufferData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, pBufferData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // increased size to accommodate new params (8 floats = 32 bytes)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Compute Pipeline
        const computeModule = this.device.createShaderModule({ code: computeShader });
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });
        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ]
        });

        // Render Pipeline
        const drawModule = this.device.createShaderModule({ code: drawShader });
        this.uniformBuffer = this.device.createBuffer({
            size: 64, // 4x4 matrix
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }]
        });
        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: pSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    ]
                }]
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
                }]
            },
            primitive: { topology: 'point-list' },
            depthStencil: undefined // No depth test for additive blending particles
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `
            position: absolute; bottom: 10px; right: 10px;
            color: #ff5555; background: rgba(0,0,0,0.8);
            padding: 5px 10px; border-radius: 4px; font-size: 12px; pointer-events: none;
        `;
        msg.textContent = "WebGPU Not Supported";
        this.container.appendChild(msg);
    }

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        if (this.glCanvas) {
            this.glCanvas.width = width * dpr;
            this.glCanvas.height = height * dpr;
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = width * dpr;
            this.gpuCanvas.height = height * dpr;
        }
    }

    getProjectionMatrix(aspect) {
        // Simple perspective matrix
        const fov = 60 * Math.PI / 180;
        const near = 0.1;
        const far = 100.0;
        const f = 1.0 / Math.tan(fov / 2);

        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0
        ]);
    }

    getViewMatrix(time) {
        // Rotate camera around
        const radius = 3.5;
        const camX = Math.sin(time * 0.2) * radius;
        const camZ = Math.cos(time * 0.2) * radius;

        // LookAt (0,0,0) from (camX, 0, camZ)
        // Simply: Translation * Rotation
        // But manual matrix construction is safer

        // Forward = normalize(target - eye) = normalize(-eye)
        const eye = [camX, 0, camZ];
        const target = [0, 0, 0];
        const up = [0, 1, 0];

        const zAxis = [eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]; // forward (actually backwards)
        let len = Math.hypot(zAxis[0], zAxis[1], zAxis[2]);
        zAxis[0]/=len; zAxis[1]/=len; zAxis[2]/=len;

        const xAxis = [
            up[1]*zAxis[2] - up[2]*zAxis[1],
            up[2]*zAxis[0] - up[0]*zAxis[2],
            up[0]*zAxis[1] - up[1]*zAxis[0]
        ];
        len = Math.hypot(xAxis[0], xAxis[1], xAxis[2]);
        xAxis[0]/=len; xAxis[1]/=len; xAxis[2]/=len;

        const yAxis = [
            zAxis[1]*xAxis[2] - zAxis[2]*xAxis[1],
            zAxis[2]*xAxis[0] - zAxis[0]*xAxis[2],
            zAxis[0]*xAxis[1] - zAxis[1]*xAxis[0]
        ];

        return [
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -(xAxis[0]*eye[0] + xAxis[1]*eye[1] + xAxis[2]*eye[2]),
            -(yAxis[0]*eye[0] + yAxis[1]*eye[1] + yAxis[2]*eye[2]),
            -(zAxis[0]*eye[0] + zAxis[1]*eye[1] + zAxis[2]*eye[2]),
            1
        ];
    }

    multiplyMatrices(a, b) {
        const out = new Float32Array(16);
        for(let r=0; r<4; r++) {
            for(let c=0; c<4; c++) {
                let sum = 0;
                for(let i=0; i<4; i++) {
                    sum += a[i*4+r] * b[c*4+i];
                }
                out[c*4+r] = sum;
            }
        }
        return out;
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.viewport(0, 0, width * dpr, height * dpr);
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), width * dpr, height * dpr);

            // Clear but keep background color
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. Render WebGPU
        if (this.device && this.renderPipeline) {
            // Update simulation params: dt, time, mouseX, mouseY, confinementStrength, pad...
            // Size is 32 bytes (8 floats)
            const params = new Float32Array([
                0.016, time, this.mouseX, this.mouseY,
                this.confinementStrength, 0, 0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            // Update Camera Uniforms
            const aspect = width / height;
            const proj = this.getProjectionMatrix(aspect);
            const view = this.getViewMatrix(time);
            const mvp = this.multiplyMatrices(view, proj);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, mvp);

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
            renderPass.setBindGroup(0, this.renderBindGroup);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('mousedown', this.handleMouseDown);
        this.container.removeEventListener('mouseup', this.handleMouseUp);
        this.container.removeEventListener('touchstart', this.handleMouseDown);
        this.container.removeEventListener('touchend', this.handleMouseUp);

        // Clean up GL
        if(this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if(ext) ext.loseContext();
        }
        // Clean up WebGPU
        if(this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.PlasmaConfinement = PlasmaConfinement;
}
