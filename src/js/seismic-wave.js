/**
 * Seismic Wave Experiment
 * Hybrid Rendering:
 * - WebGL2: Renders static "Sensor Stations" (red pillars) on the terrain.
 * - WebGPU: Simulates wave propagation on a grid using Compute Shaders (Wave Equation).
 */

export class SeismicWaveExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            gridSize: options.gridSize || 100,
            ...options
        };

        this.isActive = false;
        this.animationId = null;
        this.time = 0;

        // WebGL2
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.uMatrixLoc = null;

        // WebGPU
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.computePipeline = null;
        this.renderPipeline = null;
        this.stateBufferA = null; // Ping-pong buffers if needed, or state
        this.stateBufferB = null;
        this.uniformBuffer = null;
        this.computeBindGroupA = null;
        this.computeBindGroupB = null;
        this.frame = 0;

        // Interaction
        this.mouse = { x: 0, y: 0, active: false };

        // Bind methods
        this.resize = this.resize.bind(this);
        this.animate = this.animate.bind(this);
        this.onMouseMove = this.onMouseMove.bind(this);
        this.onMouseDown = this.onMouseDown.bind(this);
        this.onMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        // 1. WebGL2 (Static Elements)
        this.initWebGL2();

        // 2. WebGPU (Dynamic Waves)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("SeismicWave: WebGPU error", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        // Interaction
        this.container.addEventListener('mousemove', this.onMouseMove);
        this.container.addEventListener('mousedown', this.onMouseDown);
        this.container.addEventListener('mouseup', this.onMouseUp);
        window.addEventListener('resize', this.resize);

        this.isActive = true;
        this.animate();

        // Initial resize to set sizes
        this.resize();
    }

    // ========================================================================
    // Interaction
    // ========================================================================

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    onMouseDown(e) {
        this.mouse.active = true;
        this.onMouseMove(e);
    }

    onMouseUp(e) {
        this.mouse.active = false;
    }

    // ========================================================================
    // WebGL2 (Static Sensors)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position:absolute; top:0; left:0; width:100%; height:100%; z-index:1; pointer-events:none;';
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        // Simple Pillar Shader
        const vs = `#version 300 es
            in vec3 a_pos;
            in vec3 a_offset;
            uniform mat4 u_matrix;
            void main() {
                vec3 pos = a_pos * vec3(0.05, 0.5, 0.05) + a_offset;
                gl_Position = u_matrix * vec4(pos, 1.0);
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            out vec4 outColor;
            void main() {
                outColor = vec4(1.0, 0.2, 0.2, 0.8);
            }
        `;

        this.glProgram = this.createProgram(vs, fs);
        this.uMatrixLoc = this.gl.getUniformLocation(this.glProgram, 'u_matrix');
        const offsetLoc = this.gl.getAttribLocation(this.glProgram, 'a_offset');
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_pos');

        // Cube Geometry
        const cubeVerts = new Float32Array([
            // Front face
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            // Back face
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,
        ]);

        // Indices for lines
        const cubeIndices = new Uint16Array([
            0, 1, 1, 2, 2, 3, 3, 0, // Front
            4, 5, 5, 6, 6, 7, 7, 4, // Back
            0, 4, 1, 7, 2, 6, 3, 5  // Connectors
        ]);

        // Instance offsets (Sensor locations)
        const offsets = new Float32Array([
            -0.5, 0, -0.5,
             0.5, 0, -0.5,
            -0.5, 0,  0.5,
             0.5, 0,  0.5,
             0.0, 0,  0.0
        ]);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        // Vertices
        const vBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, cubeVerts, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        // Indices
        const iBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuf);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, cubeIndices, this.gl.STATIC_DRAW);

        // Instanced Offsets
        const oBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, oBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, offsets, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(offsetLoc);
        this.gl.vertexAttribPointer(offsetLoc, 3, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(offsetLoc, 1);
    }

    createProgram(vsSrc, fsSrc) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSrc);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vs));
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSrc);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fs));
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Wave Simulation)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position:absolute; top:0; left:0; width:100%; height:100%; z-index:0;';
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: format, alphaMode: 'premultiplied' });

        const N = this.options.gridSize;
        const totalPoints = N * N;

        // --- Data ---
        // State: [height, velocity, padding, padding] per point
        const initialData = new Float32Array(totalPoints * 4);

        this.stateBufferA = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.stateBufferA, 0, initialData);

        this.stateBufferB = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.stateBufferB, 0, initialData);

        // Uniforms: MVP (64) + Params (4: dt, damping, mouseX, mouseY) + GridSize (4: N, time, active, pad)
        // Total aligned 4 floats.
        // Struct:
        // mvp: mat4x4
        // params: vec4 (dt, damping, mouseX, mouseY)
        // grid: vec4 (N, time, clickActive, padding)
        const uniformSize = 64 + 16 + 16;
        this.uniformBuffer = this.device.createBuffer({
            size: uniformSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // --- Compute Shader (Wave Equation) ---
        const computeShader = `
            struct State {
                height : f32,
                vel : f32,
                pad1 : f32,
                pad2 : f32,
            }

            struct Uniforms {
                mvp : mat4x4f,
                params : vec4f, // dt, damping, mouseX, mouseY
                grid : vec4f,   // N, time, clickActive, pad
            }

            @group(0) @binding(0) var<storage, read> inputState : array<State>;
            @group(0) @binding(1) var<storage, read_write> outputState : array<State>;
            @group(0) @binding(2) var<uniform> uniforms : Uniforms;

            fn getIdx(x: i32, y: i32, N: i32) -> i32 {
                let xx = clamp(x, 0, N - 1);
                let yy = clamp(y, 0, N - 1);
                return yy * N + xx;
            }

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let N = i32(uniforms.grid.x);
                let x = i32(id.x);
                let y = i32(id.y);

                if (x >= N || y >= N) { return; }

                let idx = y * N + x;
                var s = inputState[idx];

                // Wave equation: acceleration = Laplacian * c^2
                // Laplacian using finite difference
                let h = s.height;
                let h_l = inputState[getIdx(x - 1, y, N)].height;
                let h_r = inputState[getIdx(x + 1, y, N)].height;
                let h_u = inputState[getIdx(x, y - 1, N)].height;
                let h_d = inputState[getIdx(x, y + 1, N)].height;

                let laplacian = h_l + h_r + h_u + h_d - 4.0 * h;

                // Update velocity
                let c2 = 0.05; // speed squared
                let damping = uniforms.params.y;
                let dt = uniforms.params.x;

                s.vel += (c2 * laplacian - damping * s.vel) * dt;

                // Add interaction force
                // Map mouse (-1 to 1) to grid coords
                // Mouse is in normalized device coords (x: -1..1, y: -1..1)
                // Grid is 0..N-1 mapped to -1..1 in render
                // But we need to check distance in world space or grid space.
                // Let's assume grid covers -1 to 1.
                let gx = (f32(x) / f32(N)) * 2.0 - 1.0;
                let gy = (f32(y) / f32(N)) * 2.0 - 1.0;

                let mx = uniforms.params.z;
                let my = uniforms.params.w;
                let clickActive = uniforms.grid.z;

                let dist = distance(vec2f(gx, gy), vec2f(mx, my));
                if (clickActive > 0.5 && dist < 0.2) {
                    s.vel += 10.0 * dt * (1.0 - dist / 0.2);
                }

                // Random Drops
                if (fract(sin(uniforms.grid.y * 12.9898 + f32(idx)) * 43758.5453) > 0.999) {
                     s.vel += 2.0;
                }

                // Update height
                s.height += s.vel * dt;

                // Damping for stability
                s.height *= 0.999;

                outputState[idx] = s;
            }
        `;

        const computeModule = this.device.createShaderModule({ code: computeShader });
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        // Group A: Read A, Write B
        this.computeBindGroupA = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.stateBufferA } },
                { binding: 1, resource: { buffer: this.stateBufferB } },
                { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // Group B: Read B, Write A
        this.computeBindGroupB = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.stateBufferB } },
                { binding: 1, resource: { buffer: this.stateBufferA } },
                { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // --- Render Shader ---
        const drawShader = `
            struct Uniforms {
                mvp : mat4x4f,
                params : vec4f,
                grid : vec4f,
            }

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) height : f32,
            }

            @group(0) @binding(0) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @builtin(vertex_index) vIdx : u32,
                @location(0) height : f32,
                @location(1) vel : f32
            ) -> VertexOutput {
                var output : VertexOutput;
                let N = i32(uniforms.grid.x);

                // Reconstruct grid position from index
                let x = f32(i32(vIdx) % N);
                let y = f32(i32(vIdx) / N);

                // Map to -1..1
                let u = x / f32(N - 1) * 2.0 - 1.0;
                let v = y / f32(N - 1) * 2.0 - 1.0;

                let pos = vec3f(u, height * 0.3, v);
                output.position = uniforms.mvp * vec4f(pos, 1.0);
                output.height = height;
                return output;
            }

            @fragment
            fn fs_main(@location(0) height : f32) -> @location(0) vec4f {
                // Color based on height
                let c = (height + 0.5);
                return vec4f(0.2 + c*0.8, 0.4 + c*0.2, 1.0 - c*0.5, 1.0);
            }
        `;

        const renderModule = this.device.createShaderModule({ code: drawShader });
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 16, // 4 * 4 bytes
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32' }, // height
                        { shaderLocation: 1, offset: 4, format: 'float32' }  // vel
                        // padding at 8, 12 ignored
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "⚠️ WebGPU Not Available (Waves require WebGPU)";
        msg.style.cssText = "position:absolute; top:20px; left:50%; transform:translateX(-50%); color:#ffaa00; background:rgba(0,0,0,0.8); padding:10px 20px; border-radius:8px; pointer-events:none;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Loop
    // ========================================================================

    resize() {
        const w = this.container.clientWidth || window.innerWidth;
        const h = this.container.clientHeight || window.innerHeight;
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

    resizeGPU() {
        // Handled in resize
    }

    animate() {
        if (!this.isActive) return;
        this.time += 0.016;

        // Matrix Setup
        const aspect = this.container.clientWidth / this.container.clientHeight;
        const fov = 60 * Math.PI / 180;
        const f = 1.0 / Math.tan(fov / 2);
        const zNear = 0.1;
        const zFar = 100.0;
        const proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (zFar + zNear) / (zNear - zFar), -1,
            0, 0, (2 * zFar * zNear) / (zNear - zFar), 0
        ];

        // Orbit camera
        const r = 2.5;
        const cx = Math.sin(this.time * 0.2) * r;
        const cz = Math.cos(this.time * 0.2) * r;
        const cy = 1.8;

        const zAxis = this.normalize([cx, cy, cz]);
        const xAxis = this.normalize(this.cross([0,1,0], zAxis));
        const yAxis = this.cross(zAxis, xAxis);

        const view = [
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -this.dot(xAxis, [cx,cy,cz]), -this.dot(yAxis, [cx,cy,cz]), -this.dot(zAxis, [cx,cy,cz]), 1
        ];

        const mvp = this.multiplyMatrices(proj, view);

        // --- Render WebGL2 (Sensors) ---
        if (this.gl) {
            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(this.uMatrixLoc, false, mvp);

            this.gl.bindVertexArray(this.glVao);
            // Draw 5 instances of the cube
            this.gl.drawElementsInstanced(this.gl.LINES, 24, this.gl.UNSIGNED_SHORT, 0, 5);
        }

        // --- Render WebGPU (Waves) ---
        if (this.device && this.context) {
            // Update Uniforms
            const uniformData = new Float32Array(24); // Size 96 bytes needed?
            // Layout:
            // mvp (0-63)
            // params (64-79): dt, damping, mx, my
            // grid (80-95): N, time, active, pad

            uniformData.set(mvp, 0);
            // We need to write to the buffer with correct offset
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

            const paramsData = new Float32Array([
                0.016, // dt
                0.02,  // damping
                this.mouse.x,
                this.mouse.y
            ]);
            this.device.queue.writeBuffer(this.uniformBuffer, 64, paramsData);

            const gridData = new Float32Array([
                this.options.gridSize,
                this.time,
                this.mouse.active ? 1.0 : 0.0,
                0.0
            ]);
            this.device.queue.writeBuffer(this.uniformBuffer, 80, gridData);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute Step
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            // Toggle buffers
            const bindGroup = (this.frame % 2 === 0) ? this.computeBindGroupA : this.computeBindGroupB;
            computePass.setBindGroup(0, bindGroup);
            const workGroupSize = 8;
            const groups = Math.ceil(this.options.gridSize / workGroupSize);
            computePass.dispatchWorkgroups(groups, groups);
            computePass.end();

            // Render Step
            const textureView = this.context.getCurrentTexture().createView();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'load', // Draw ON TOP of WebGL (which is under it)
                    storeOp: 'store'
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.renderBindGroup);

            // Read from the buffer we just wrote to (output of compute)
            // If frame 0: input A, output B. We render B.
            const renderBuffer = (this.frame % 2 === 0) ? this.stateBufferB : this.stateBufferA;
            renderPass.setVertexBuffer(0, renderBuffer);

            const totalPoints = this.options.gridSize * this.options.gridSize;
            renderPass.draw(totalPoints);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);

            // Swap for next frame
            this.frame++;
        }

        this.animationId = requestAnimationFrame(this.animate);
    }

    // Math Helpers
    normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if(len === 0) return [0,0,0];
        return [v[0]/len, v[1]/len, v[2]/len];
    }

    cross(a, b) {
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ];
    }

    dot(a, b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    multiplyMatrices(a, b) {
        const out = [];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += b[i * 4 + k] * a[k * 4 + j];
                }
                out.push(sum);
            }
        }
        return out;
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.resize);
        this.container.removeEventListener('mousemove', this.onMouseMove);
        this.container.removeEventListener('mousedown', this.onMouseDown);
        this.container.removeEventListener('mouseup', this.onMouseUp);
        // clean up GPU/GL resources if needed
    }
}

// Global Export
if (typeof window !== 'undefined') {
    window.SeismicWaveExperiment = SeismicWaveExperiment;
}
