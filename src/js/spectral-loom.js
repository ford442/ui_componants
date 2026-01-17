
/**
 * Spectral Loom Experiment
 * Hybrid WebGL2 (Warp Threads) + WebGPU (Weft Particles)
 */

export class SpectralLoomExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.threadCount = options.threadCount || 200;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.vertexCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.particlePipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.renderBindGroup = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.numParticles = 40000;

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        // Setup container
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050510'; // Dark loom background

        // Mouse Interaction
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        });

        // 1. Init WebGL2
        this.initWebGL2();

        // 2. Init WebGPU
        if (navigator.gpu) {
            try {
                await this.initWebGPU();
            } catch (e) {
                console.warn("SpectralLoom: WebGPU failed to init", e);
            }
        }

        // Initial resize
        this.resize();

        this.isActive = true;
        this.animate = this.animate.bind(this);
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);

        // Cleanup WebGL
        if (this.gl) {
            this.gl.deleteProgram(this.glProgram);
            // ... buffer cleanup
        }

        // Cleanup WebGPU
        if (this.device) {
            this.device.destroy();
        }

        this.container.innerHTML = '';
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Warp Threads)
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

        // Generate Threads (Lines)
        // A set of vertical lines
        const vertices = [];
        const count = this.threadCount;
        const width = 40.0;
        const height = 30.0;
        const segments = 50; // Points per line for flexibility

        for (let i = 0; i < count; i++) {
            const x = (i / (count - 1)) * width - (width / 2);
            for (let j = 0; j < segments; j++) {
                const y = (j / (segments - 1)) * height - (height / 2);
                // x, y, z, threadIndex
                vertices.push(x, y, 0, i);

                // If not last segment, push next point to create line segment
                if (j < segments - 1) {
                    const nextY = ((j + 1) / (segments - 1)) * height - (height / 2);
                    vertices.push(x, nextY, 0, i);
                }
            }
        }

        // We are drawing GL_LINES, so we need 2 points per segment.
        // The loop above pushes start point, then if not last, pushes next point.
        // Wait, for GL_LINES we need pairs.
        // 0-1, 1-2, 2-3...
        // My loop:
        // j=0: push P0. if j<last, push P1. -> P0, P1
        // j=1: push P1. if j<last, push P2. -> P1, P2
        // Correct.

        this.vertexCount = vertices.length / 4; // 4 components per vertex

        // Shaders
        const vs = `#version 300 es
        layout(location=0) in vec4 a_position; // x,y,z, threadIndex

        uniform mat4 u_projection;
        uniform mat4 u_view;
        uniform float u_time;
        uniform vec2 u_mouse;

        out float v_intensity;

        void main() {
            vec3 pos = a_position.xyz;
            float threadIdx = a_position.w;

            // Wave effect
            float wave1 = sin(pos.y * 0.5 + u_time + threadIdx * 0.1) * 0.5;
            float wave2 = cos(pos.y * 1.5 - u_time * 0.5 + threadIdx * 0.05) * 0.2;

            // Mouse interaction
            float dMouse = distance(pos.xy, u_mouse * vec2(20.0, 15.0)); // Approximate world scale
            float push = exp(-dMouse * 0.5) * 2.0;

            // Apply deformation to Z and X
            pos.z += wave1 + wave2;
            pos.x += push * (pos.x - u_mouse.x * 20.0); // Push away from mouse

            gl_Position = u_projection * u_view * vec4(pos, 1.0);

            // Brightness based on movement
            v_intensity = 0.5 + abs(wave1 + wave2);
            v_intensity += push;
        }
        `;

        const fs = `#version 300 es
        precision highp float;

        in float v_intensity;
        out vec4 outColor;

        void main() {
            vec3 color = vec3(0.2, 0.5, 1.0); // Cyan/Blue base
            color *= v_intensity;

            // Add some "gold" highlights for the spectral feel
            color += vec3(1.0, 0.8, 0.2) * pow(v_intensity, 4.0) * 0.2;

            outColor = vec4(color, 0.4); // Transparent
        }
        `;

        this.glProgram = this.createGLProgram(vs, fs);

        // Buffers
        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 4, this.gl.FLOAT, false, 0, 0);

        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('GL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('GL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Weft Particles)
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
        if (!adapter) return;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // Compute Shader
        const computeCode = `
            struct Particle {
                pos: vec4f, // x, y, z, life
                vel: vec4f, // vx, vy, vz, size
            }
            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Uniforms {
                time: f32,
                dt: f32,
                mouseX: f32,
                mouseY: f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            fn hash(seed: u32) -> f32 {
                var x = seed;
                x = ((x >> 16u) ^ x) * 0.45f;
                x = ((x >> 16u) ^ x) * 0.45f;
                x = (x >> 16u) ^ x;
                return f32(x) / 4294967295.0;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let i = id.x;
                if (i >= ${this.numParticles}) { return; }

                var p = particles[i];

                // Update Life
                p.pos.w = p.pos.w - uniforms.dt * 0.3; // Decay

                // Respawn
                if (p.pos.w <= 0.0) {
                    let seed = i + u32(uniforms.time * 1000.0);
                    // Spawn at left or right edge randomly
                    let side = select(-1.0, 1.0, hash(seed) > 0.5);

                    p.pos.x = side * 30.0;
                    p.pos.y = (hash(seed + 1u) - 0.5) * 40.0;
                    p.pos.z = (hash(seed + 2u) - 0.5) * 10.0;
                    p.pos.w = 1.0; // Reset life

                    // Velocity: Horizontal towards center
                    p.vel.x = -side * (5.0 + hash(seed + 3u) * 5.0);
                    p.vel.y = (hash(seed + 4u) - 0.5) * 2.0;
                    p.vel.z = (hash(seed + 5u) - 0.5) * 2.0;
                    p.vel.w = 0.5 + hash(seed + 6u) * 1.5; // Size
                } else {
                    // Physics
                    p.pos.x = p.pos.x + p.vel.x * uniforms.dt;
                    p.pos.y = p.pos.y + p.vel.y * uniforms.dt;
                    p.pos.z = p.pos.z + p.vel.z * uniforms.dt;

                    // Weave motion (Sine wave on Z)
                    p.pos.z = p.pos.z + sin(p.pos.x * 0.5 + uniforms.time * 5.0) * 0.1;

                    // Mouse Attraction
                    let mx = uniforms.mouseX * 25.0;
                    let my = -uniforms.mouseY * 20.0;
                    let mouseTarget = vec3f(mx, my, 0.0);

                    let dir = mouseTarget - p.pos.xyz;
                    let dist = length(dir);
                    if (dist < 15.0) {
                        let force = normalize(dir) * (15.0 - dist) * 0.5;
                        p.vel.x = p.vel.x + force.x * uniforms.dt;
                        p.vel.y = p.vel.y + force.y * uniforms.dt;
                        p.vel.z = p.vel.z + force.z * uniforms.dt;
                    }
                }

                particles[i] = p;
            }
        `;

        // Render Shader
        const drawCode = `
            struct Particle {
                pos: vec4f,
                vel: vec4f,
            }
            @group(0) @binding(0) var<storage, read> particles : array<Particle>;

            struct VertexOutput {
                @builtin(position) pos : vec4f,
                @location(0) life : f32,
                @location(1) size : f32,
                @location(2) colorVar : f32,
            }

            @group(0) @binding(1) var<uniform> uniforms : Uniforms;
            struct Uniforms {
                time: f32,
                dt: f32,
                mouseX: f32,
                mouseY: f32,
                viewProj: mat4x4f,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vIdx : u32) -> VertexOutput {
                let p = particles[vIdx];

                var out: VertexOutput;
                out.pos = uniforms.viewProj * vec4f(p.pos.xyz, 1.0);
                out.life = p.pos.w;
                out.size = p.vel.w;
                out.colorVar = abs(p.vel.x); // Color based on speed

                return out;
            }

            @fragment
            fn fs_main(@location(0) life : f32, @location(1) size : f32, @location(2) speed : f32) -> @location(0) vec4f {
                let alpha = life;

                // Color palette: Gold to Purple based on speed
                var color = mix(vec3f(1.0, 0.8, 0.2), vec3f(0.8, 0.2, 1.0), smoothstep(5.0, 15.0, speed));

                // Add center glow
                // Since we are rendering points, we can't easily do UV based glow without gl_PointCoord which is not in WGSL yet (or requires configuration).
                // We'll just output flat color with alpha.

                return vec4f(color, alpha);
            }
        `;

        // Initialize Buffers
        const initData = new Float32Array(this.numParticles * 8);
        this.particleBuffer = this.device.createBuffer({
            size: initData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 80, // Same alignment as before
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Layouts & Bind Groups
        const computeBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        const renderBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGLayout] }),
            compute: { module: this.device.createShaderModule({ code: computeCode }), entryPoint: 'main' }
        });

        this.particlePipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBGLayout] }),
            vertex: {
                module: this.device.createShaderModule({ code: drawCode }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.device.createShaderModule({ code: drawCode }),
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth * dpr;
        const h = this.container.clientHeight * dpr;

        if (w === 0 || h === 0) return;

        if (this.glCanvas) {
            this.glCanvas.width = w;
            this.glCanvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = w;
            this.gpuCanvas.height = h;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016;

        // Camera Math
        const camX = this.mouse.x * 5.0;
        const camY = this.mouse.y * 5.0;
        const aspect = (this.glCanvas?.width || 1) / (this.glCanvas?.height || 1);

        const proj = this.createPerspectiveMatrix(60, aspect, 0.1, 500.0);
        const view = this.createLookAtMatrix([camX, camY, 40], [0, 0, 0], [0, 1, 0]);
        const viewProj = this.multiplyMatrices(proj, view);

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const uProj = this.gl.getUniformLocation(this.glProgram, 'u_projection');
            const uView = this.gl.getUniformLocation(this.glProgram, 'u_view');
            const uTime = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const uMouse = this.gl.getUniformLocation(this.glProgram, 'u_mouse');

            this.gl.uniformMatrix4fv(uProj, false, proj);
            this.gl.uniformMatrix4fv(uView, false, view);
            this.gl.uniform1f(uTime, time);
            this.gl.uniform2f(uMouse, this.mouse.x, this.mouse.y);

            this.gl.clearColor(0.02, 0.01, 0.05, 0.0); // Transparent clear
            this.gl.clear(this.gl.COLOR_BUFFER_BIT); // No Depth clear needed if 2D overlay, but lines are 3D
            // Wait, lines are 3D. We should clear depth too?
            // Actually, if we want transparency, we should be careful with depth.
            // But let's just clear both for now.
            // Note: Canvas is on top of bg color but below WebGPU canvas.

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.LINES, 0, this.vertexCount);
        }

        // 2. WebGPU Render
        if (this.device && this.particlePipeline) {
            const uniforms = new Float32Array(20);
            uniforms[0] = time;
            uniforms[1] = dt;
            uniforms[2] = this.mouse.x;
            uniforms[3] = this.mouse.y;
            uniforms.set(viewProj, 4);

            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

            const encoder = this.device.createCommandEncoder();

            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            if (this.gpuCanvas.width > 0 && this.gpuCanvas.height > 0) {
                const textureView = this.context.getCurrentTexture().createView();
                const rPass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: textureView,
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: 'load',
                        storeOp: 'store'
                    }]
                });
                rPass.setPipeline(this.particlePipeline);
                rPass.setBindGroup(0, this.renderBindGroup);
                rPass.draw(this.numParticles);
                rPass.end();
            }

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(this.animate);
    }

    // Matrix Utils (Copied for simplicity)
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
        const len = Math.hypot(...z);
        if(len > 0) z = z.map(v => v / len);
        else z = [0, 0, 1];

        let x = [
            up[1]*z[2] - up[2]*z[1],
            up[2]*z[0] - up[0]*z[2],
            up[0]*z[1] - up[1]*z[0]
        ];
        const lenX = Math.hypot(...x);
        if(lenX > 0) x = x.map(v => v / lenX);
        else x = [1, 0, 0];

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
        let out = new Float32Array(16);
        const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

        const b00 = b[0], b01 = b[1], b02 = b[2], b03 = b[3];
        const b10 = b[4], b11 = b[5], b12 = b[6], b13 = b[7];
        const b20 = b[8], b21 = b[9], b22 = b[10], b23 = b[11];
        const b30 = b[12], b31 = b[13], b32 = b[14], b33 = b[15];

        out[0] = b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30;
        out[1] = b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31;
        out[2] = b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32;
        out[3] = b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33;

        out[4] = b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30;
        out[5] = b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31;
        out[6] = b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32;
        out[7] = b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33;

        out[8] = b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30;
        out[9] = b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31;
        out[10] = b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32;
        out[11] = b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33;

        out[12] = b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30;
        out[13] = b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31;
        out[14] = b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32;
        out[15] = b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33;

        return out;
    }
}

if (typeof window !== 'undefined') {
    window.SpectralLoomExperiment = SpectralLoomExperiment;
}
