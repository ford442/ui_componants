/**
 * Quantum Tensor Experiment
 *
 * Concept: A hybrid experiment visualizing a "Quantum Field" around a central "Tensor Core".
 * - WebGL2: Renders the static/stable geometry (Core, Grid).
 * - WebGPU: Simulates the chaotic quantum fluctuations (Particle Swarm) via Compute Shaders.
 */

export class QuantumTensorExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            particleCount: options.particleCount || 100000,
            ...options
        };

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.params = {
            chaos: 0.5,     // Controls particle noise
            attraction: 0.5 // Controls pull towards center
        };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.indexCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.computePipeline = null;
        this.renderPipeline = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.computeBindGroup = null;
        this.renderBindGroup = null;

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020205';

        // Event Listeners
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        });

        const chaosInput = document.getElementById('chaos-control');
        if (chaosInput) {
            chaosInput.addEventListener('input', (e) => this.params.chaos = e.target.value / 100);
        }

        const attractionInput = document.getElementById('attraction-control');
        if (attractionInput) {
            attractionInput.addEventListener('input', (e) => this.params.attraction = e.target.value / 100);
        }

        // 1. Initialize WebGL2 (Core Structure)
        this.initWebGL2();

        // 2. Initialize WebGPU (Particle Field)
        if (navigator.gpu) {
            try {
                await this.initWebGPU();
            } catch (e) {
                console.warn("QuantumTensor: WebGPU init failed:", e);
                this.addWebGPUNotSupportedMessage();
            }
        } else {
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();
        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 Implementation (The Core)
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

        // --- Icosahedron Geometry ---
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;
        const vertices = new Float32Array([
            -1, t, 0,  1, t, 0, -1, -t, 0,  1, -t, 0,
            0, -1, t,  0, 1, t,  0, -1, -t,  0, 1, -t,
            t, 0, -1,  t, 0, 1, -t, 0, -1, -t, 0, 1
        ]);

        // Define triangles for Icosahedron
        const indices = new Uint16Array([
            0, 11, 5,   0, 5, 1,   0, 1, 7,   0, 7, 10,  0, 10, 11,
            1, 5, 9,    5, 11, 4,  11, 10, 2, 10, 7, 6,  7, 1, 8,
            3, 9, 4,    3, 4, 2,   3, 2, 6,   3, 6, 8,   3, 8, 9,
            4, 9, 5,    2, 4, 11,  6, 2, 10,  8, 6, 7,   9, 8, 1
        ]);
        this.indexCount = indices.length;

        // Vertex Shader
        const vs = `#version 300 es
        in vec3 a_position;

        uniform mat4 u_projection;
        uniform mat4 u_view;
        uniform mat4 u_model;
        uniform float u_time;

        out vec3 v_pos;
        out vec3 v_normal;

        void main() {
            // Add some "breathing" vertex displacement
            vec3 pos = a_position;
            float pulse = sin(u_time * 2.0 + pos.y * 2.0) * 0.1;
            pos += normalize(pos) * pulse;

            v_pos = pos;
            v_normal = normalize(pos);
            gl_Position = u_projection * u_view * u_model * vec4(pos, 1.0);
        }
        `;

        // Fragment Shader
        const fs = `#version 300 es
        precision highp float;

        in vec3 v_pos;
        in vec3 v_normal;
        uniform float u_time;

        out vec4 outColor;

        void main() {
            // Holographic aesthetic
            vec3 viewDir = normalize(vec3(0.0, 0.0, 5.0) - v_pos); // Approx view pos
            float rim = 1.0 - max(dot(v_normal, viewDir), 0.0);
            rim = pow(rim, 3.0);

            vec3 coreColor = vec3(0.1, 0.2, 0.4);
            vec3 glowColor = vec3(0.0, 1.0, 0.8);

            // Scanline effect
            float scan = sin(v_pos.y * 20.0 - u_time * 5.0) * 0.5 + 0.5;

            vec3 finalColor = mix(coreColor, glowColor, rim * scan);

            outColor = vec4(finalColor, 0.6 * rim + 0.2);
        }
        `;

        this.glProgram = this.createGLProgram(vs, fs);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if(!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vs)); return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if(!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fs)); return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU Implementation (The Field)
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
        if (!adapter) throw new Error("No Adapter");
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // --- Compute Shader ---
        const computeCode = `
            struct Particle {
                pos: vec4f, // xyz, w=life
                vel: vec4f, // xyz, w=unused
            }

            struct SimParams {
                dt: f32,
                time: f32,
                chaos: f32,
                attraction: f32,
                mouseX: f32,
                mouseY: f32,
                pad1: f32,
                pad2: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: SimParams;

            fn hash(p: vec3f) -> vec3f {
                var p3 = fract(p * vec3f(.1031, .1030, .0973));
                p3 += dot(p3, p3.yzx + 33.33);
                return fract((p3.xxy + p3.yzz) * p3.zyx) * 2.0 - 1.0;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];
                let center = vec3f(0.0, 0.0, 0.0);

                // 1. Attraction to Center
                let dirToCenter = center - p.pos.xyz;
                let dist = length(dirToCenter);
                let force = normalize(dirToCenter) * params.attraction * (1.0 / (dist + 0.1));

                // 2. Mouse Influence (Repulsion)
                // Project mouse 2D to rough 3D cylinder
                let mousePos = vec3f(params.mouseX * 5.0, params.mouseY * 5.0, p.pos.z);
                let dirToMouse = p.pos.xyz - mousePos;
                let distMouse = length(dirToMouse);
                if (distMouse < 2.0) {
                    force += normalize(dirToMouse) * 2.0 * (2.0 - distMouse);
                }

                // 3. Chaos / Curl Noise approx
                let noise = hash(p.pos.xyz * 0.5 + params.time * 0.1) * params.chaos;

                p.vel = p.vel.xyz * 0.96 + force * params.dt + noise * params.dt;
                p.pos = vec4f(p.pos.xyz + p.vel.xyz * params.dt, p.pos.w);

                // 4. Reset if lost
                if (dist > 20.0) {
                    p.pos = vec4f((hash(vec3f(f32(i), params.time, 0.0)) * 5.0), 1.0);
                    p.vel = vec4f(0.0);
                }

                particles[i] = p;
            }
        `;

        // --- Render Shader ---
        const renderCode = `
            struct Particle {
                pos: vec4f,
                vel: vec4f,
            }
            @group(0) @binding(0) var<storage, read> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> viewProj: mat4x4f;

            struct VertexOutput {
                @builtin(position) Position: vec4f,
                @location(0) Color: vec4f,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) iIdx: u32) -> VertexOutput {
                var output: VertexOutput;
                let p = particles[iIdx];

                // Simple Point Rendering (Expanded to Quad in future? For now just GL_POINT equivalent logic via point-list topology)
                // Note: WebGPU 'point-list' renders 1 pixel points. For larger particles we need billboards.
                // But let's stick to point-list for performance/simplicity first.

                output.Position = viewProj * vec4f(p.pos.xyz, 1.0);

                let speed = length(p.vel.xyz);
                let c1 = vec3f(0.0, 0.5, 1.0); // Blue
                let c2 = vec3f(1.0, 0.2, 0.8); // Pink
                let col = mix(c1, c2, clamp(speed * 0.5, 0.0, 1.0));

                output.Color = vec4f(col, 0.8);
                return output;
            }

            @fragment
            fn fs_main(@location(0) Color: vec4f) -> @location(0) vec4f {
                return Color;
            }
        `;

        // Buffers
        const numParticles = this.options.particleCount;
        const particleSize = 32; // 2x vec4f
        const initData = new Float32Array(numParticles * 8);
        for(let i=0; i<numParticles; i++) {
            initData[i*8+0] = (Math.random() - 0.5) * 10.0;
            initData[i*8+1] = (Math.random() - 0.5) * 10.0;
            initData[i*8+2] = (Math.random() - 0.5) * 10.0;
            initData[i*8+3] = 1.0; // Life
        }

        this.particleBuffer = this.device.createBuffer({
            size: initData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(initData);
        this.particleBuffer.unmap();

        this.uniformBuffer = this.device.createBuffer({
            size: 32, // Struct SimParams aligned
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.viewProjBuffer = this.device.createBuffer({
            size: 64, // mat4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Pipeline
        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'main' }
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // Render Pipeline
        const renderModule = this.device.createShaderModule({ code: renderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.viewProjBuffer } }
            ]
        });
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px;
            background: rgba(100,0,0,0.8); color: white; padding: 10px;
            border: 1px solid red; border-radius: 4px; pointer-events: none;
        `;
        msg.innerHTML = "WebGPU Not Supported - Running Reduced Mode";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Logic Loop
    // ========================================================================

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

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

        // Camera Math (Orbit)
        // Camera rotates around center
        const radius = 10.0;
        const camX = Math.sin(time * 0.2) * radius + this.mouse.x * 2.0;
        const camY = Math.sin(time * 0.1) * 2.0 + this.mouse.y * 2.0;
        const camZ = Math.cos(time * 0.2) * radius;

        const fov = 60 * Math.PI / 180;
        const aspect = this.container.clientWidth / this.container.clientHeight;
        const near = 0.1;
        const far = 100.0;

        // Perspective Matrix
        const f = 1.0 / Math.tan(fov / 2);
        const rangeInv = 1.0 / (near - far);
        const proj = new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0
        ]);

        // View Matrix (LookAt)
        // Eye: [camX, camY, camZ], Target: [0,0,0], Up: [0,1,0]
        const eye = [camX, camY, camZ];
        const target = [0, 0, 0];
        const up = [0, 1, 0];

        const zAxis = normalize(sub(eye, target));
        const xAxis = normalize(cross(up, zAxis));
        const yAxis = cross(zAxis, xAxis);

        const view = new Float32Array([
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1
        ]);

        // WebGPU ViewProj (View * Proj) - Column Major
        const viewProj = multiplyMatrices(proj, view);


        // --- Render WebGL2 ---
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Model Matrix (Rotate the Core)
            const model = new Float32Array([
                Math.cos(time), 0, Math.sin(time), 0,
                0, 1, 0, 0,
                -Math.sin(time), 0, Math.cos(time), 0,
                0, 0, 0, 1
            ]);

            const locs = {
                proj: this.gl.getUniformLocation(this.glProgram, 'u_projection'),
                view: this.gl.getUniformLocation(this.glProgram, 'u_view'),
                model: this.gl.getUniformLocation(this.glProgram, 'u_model'),
                time: this.gl.getUniformLocation(this.glProgram, 'u_time'),
            };

            this.gl.uniformMatrix4fv(locs.proj, false, proj);
            this.gl.uniformMatrix4fv(locs.view, false, view);
            this.gl.uniformMatrix4fv(locs.model, false, model);
            this.gl.uniform1f(locs.time, time);

            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

            this.gl.clearColor(0.02, 0.02, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.indexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // --- Render WebGPU ---
        if (this.device && this.renderPipeline) {
            // Write Uniforms
            const params = new Float32Array([dt, time, this.params.chaos, this.params.attraction, this.mouse.x, this.mouse.y, 0, 0]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, params);
            this.device.queue.writeBuffer(this.viewProjBuffer, 0, viewProj);

            const encoder = this.device.createCommandEncoder();

            // Compute Pass
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.options.particleCount / 64));
            cPass.end();

            // Render Pass
            const textureView = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: {r:0,g:0,b:0,a:0},
                    loadOp: 'load', // Load WebGL content? No, separate canvas overlay.
                    storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBindGroup);
            rPass.draw(this.options.particleCount); // Points
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        if(this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if(this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

// Math Helpers
function normalize(v) {
    const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    return len > 0 ? [v[0]/len, v[1]/len, v[2]/len] : [0,0,0];
}
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross(a, b) {
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ];
}
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function multiplyMatrices(a, b) {
    // a * b (Column Major)
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                sum += b[i * 4 + k] * a[k * 4 + j];
            }
            out[i * 4 + j] = sum;
        }
    }
    return out;
}

if (typeof window !== 'undefined') {
    window.QuantumTensorExperiment = QuantumTensorExperiment;
}
