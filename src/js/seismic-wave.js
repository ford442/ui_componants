/**
 * Seismic Wave Terrain Experiment
 * Hybrid Rendering:
 * - WebGL2: Renders static "Sensor Stations" (geometric markers).
 * - WebGPU: Simulates elastic wave propagation on a wireframe grid using a compute shader.
 */

export class SeismicWaveExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            gridSize: options.gridSize || 128,
            gridSpacing: options.gridSpacing || 0.1,
            ...options
        };

        this.isActive = false;
        this.animationId = null;

        // Interaction
        this.mousePos = { x: -1000, y: -1000 };
        this.isMouseDown = false;
        this.lastClickTime = 0;

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
        this.heightBuffer1 = null; // Current height state
        this.heightBuffer2 = null; // Previous height state (for wave eq)
        this.uniformBuffer = null;
        this.computeBindGroup1 = null; // Read 1, Write 2
        this.computeBindGroup2 = null; // Read 2, Write 1
        this.frameParity = 0; // Toggle between 0 and 1

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020202';

        // 1. WebGL2 (Static Objects)
        this.initWebGL2();

        // 2. WebGPU (Dynamic Terrain)
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

        // Interaction Listeners
        this.container.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.container.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.container.addEventListener('mouseup', () => this.onMouseUp());
        this.container.addEventListener('mouseleave', () => this.onMouseUp());

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mousePos = { x, y };
    }

    onMouseDown(e) {
        this.isMouseDown = true;
        this.lastClickTime = performance.now();
    }

    onMouseUp() {
        this.isMouseDown = false;
    }

    // ========================================================================
    // WebGL2 (Static Sensor Stations)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position:absolute; top:0; left:0; width:100%; height:100%; z-index:1;';
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        // Simple Pyramid Shader
        const vs = `#version 300 es
            in vec3 a_pos;
            in vec3 a_offset;
            in vec3 a_color;
            uniform mat4 u_matrix;
            out vec3 v_color;
            void main() {
                vec3 pos = a_pos * 0.5 + a_offset; // Scale and position
                gl_Position = u_matrix * vec4(pos, 1.0);
                v_color = a_color;
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            in vec3 v_color;
            out vec4 outColor;
            void main() {
                outColor = vec4(v_color, 0.8);
            }
        `;

        this.glProgram = this.createProgram(vs, fs);
        this.uMatrixLoc = this.gl.getUniformLocation(this.glProgram, 'u_matrix');

        // Pyramid Geometry
        // Base (-1,-1) to (1,1), Tip at (0,1,0)
        const verts = new Float32Array([
            // Base
            -1, 0, -1,  1, 0, -1,  1, 0, 1,
            -1, 0, -1,  1, 0, 1, -1, 0, 1,
            // Sides
            -1, 0, -1,  1, 0, -1,  0, 2, 0,
             1, 0, -1,  1, 0,  1,  0, 2, 0,
             1, 0,  1, -1, 0,  1,  0, 2, 0,
            -1, 0,  1, -1, 0, -1,  0, 2, 0,
        ]);

        // Instanced Data: 4 towers at corners
        const range = 4.0;
        const offsets = new Float32Array([
            -range, -2, -range,
             range, -2, -range,
            -range, -2,  range,
             range, -2,  range
        ]);

        const colors = new Float32Array([
            1.0, 0.2, 0.2,
            0.2, 1.0, 0.2,
            0.2, 0.2, 1.0,
            1.0, 1.0, 0.0
        ]);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        // Geometry Buffer
        const buf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, verts, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_pos');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        // Offset Buffer (Instanced)
        const offsetBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, offsetBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, offsets, this.gl.STATIC_DRAW);
        const offLoc = this.gl.getAttribLocation(this.glProgram, 'a_offset');
        this.gl.enableVertexAttribArray(offLoc);
        this.gl.vertexAttribPointer(offLoc, 3, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(offLoc, 1);

        // Color Buffer (Instanced)
        const colBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, colBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, colors, this.gl.STATIC_DRAW);
        const colLoc = this.gl.getAttribLocation(this.glProgram, 'a_color');
        this.gl.enableVertexAttribArray(colLoc);
        this.gl.vertexAttribPointer(colLoc, 3, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(colLoc, 1);

        this.resizeGL();
    }

    createProgram(vsSrc, fsSrc) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSrc);
        this.gl.compileShader(vs);
        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSrc);
        this.gl.compileShader(fs);
        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Deformable Grid)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position:absolute; top:0; left:0; width:100%; height:100%; z-index:2; pointer-events:none;';
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: format, alphaMode: 'premultiplied' });

        const N = this.options.gridSize;
        const count = N * N;

        // Buffers for Height Field (Ping-Pong)
        // Just storing float height
        const initialData = new Float32Array(count); // Zeros
        this.heightBuffer1 = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.heightBuffer1, 0, initialData);

        this.heightBuffer2 = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.heightBuffer2, 0, initialData);

        // Uniform Buffer
        this.uniformBuffer = this.device.createBuffer({
            size: 144, // Mat4(64) + Params(16) + Mouse(16) + Padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // --- Compute Shader (Wave Equation) ---
        // h_new = 2*h - h_old + c^2 * dt^2 * laplacian(h) - damping * (h - h_old)
        // Actually simpler Verlet integration for waves:
        // acceleration = laplacian(h) * speed - velocity * damping
        // For simplicity on GPU without explicit velocity buffer (using 2 time steps):
        // h_new = 2*h_cur - h_prev + (d^2/dt^2) * laplacian
        // Let's use 3 buffers? Or just Height/Velocity.
        // Let's use Height and Velocity in one struct to be robust.

        // Revised Buffer: Struct { h: f32, v: f32 }
        const simData = new Float32Array(count * 2);
        this.simBuffer = this.device.createBuffer({
            size: simData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX // Vertex to read height
        });
        // We actually only need one buffer if we update in place carefully, but race conditions.
        // Let's stick to standard Compute approach: Read Buffer -> Write Buffer.
        // Wait, if I use `var<storage, read_write>` I can do in-place if careful, but for wave prop I need neighbors.
        // So I need ping-pong.

        // Let's do Ping-Pong with just `h` and `v` stored together.
        // Ping-Pong Buffer A and B.
        this.simBufferA = this.device.createBuffer({
            size: simData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX
        });
        this.simBufferB = this.device.createBuffer({
            size: simData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX
        });

        const computeShader = `
            struct Node {
                h : f32,
                v : f32,
            }

            struct Uniforms {
                mvp : mat4x4f,
                params : vec4f, // x=gridSize, y=dt, z=damping, w=speed
                mouse : vec4f, // x,y world pos, z=click strength, w=time
            }

            @group(0) @binding(0) var<storage, read> input : array<Node>;
            @group(0) @binding(1) var<storage, read_write> output : array<Node>;
            @group(0) @binding(2) var<uniform> uniforms : Uniforms;

            fn getIdx(x: i32, y: i32) -> i32 {
                let N = i32(uniforms.params.x);
                let cx = clamp(x, 0, N-1);
                let cy = clamp(y, 0, N-1);
                return cy * N + cx;
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let N = i32(uniforms.params.x);
                let x = i32(id.x);
                let y = i32(id.y);

                if (x >= N || y >= N) { return; }

                let idx = y * N + x;
                var node = input[idx];

                // Laplacian (Neighbor average - center)
                let h_left = input[getIdx(x-1, y)].h;
                let h_right = input[getIdx(x+1, y)].h;
                let h_up = input[getIdx(x, y-1)].h;
                let h_down = input[getIdx(x, y+1)].h;

                let laplacian = (h_left + h_right + h_up + h_down - 4.0 * node.h);

                // Wave Physics
                let dt = uniforms.params.y;
                let damping = uniforms.params.z;
                let speed = uniforms.params.w;

                let accel = laplacian * speed - node.v * damping;
                node.v += accel * dt;
                node.h += node.v * dt;

                // Mouse Interaction (Gaussian impulse)
                let gridX = (f32(x) / f32(N) - 0.5) * 10.0; // Map 0..N to -5..5 world space
                let gridY = (f32(y) / f32(N) - 0.5) * 10.0;

                let dist = distance(vec2f(gridX, gridY), uniforms.mouse.xy);
                if (uniforms.mouse.z > 0.0 && dist < 1.0) {
                    node.v += uniforms.mouse.z * (1.0 - dist) * 10.0 * dt;
                }

                output[idx] = node;
            }
        `;

        const computeModule = this.device.createShaderModule({ code: computeShader });
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        // Group 1: Read A, Write B
        this.computeBindGroup1 = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.simBufferA } },
                { binding: 1, resource: { buffer: this.simBufferB } },
                { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // Group 2: Read B, Write A
        this.computeBindGroup2 = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.simBufferB } },
                { binding: 1, resource: { buffer: this.simBufferA } },
                { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // --- Render Shader ---
        // We need an Index Buffer to draw the grid lines
        // For N*N nodes, we have horizontal and vertical lines.
        // N*(N-1) horizontal segments, N*(N-1) vertical segments.
        const indices = [];
        for (let y = 0; y < N; y++) {
            for (let x = 0; x < N; x++) {
                const i = y * N + x;
                if (x < N - 1) { indices.push(i, i + 1); } // Horizontal
                if (y < N - 1) { indices.push(i, i + N); } // Vertical
            }
        }
        this.numIndices = indices.length;
        this.indexBuffer = this.device.createBuffer({
            size: new Uint32Array(indices).byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, new Uint32Array(indices));


        const drawShader = `
            struct Uniforms {
                mvp : mat4x4f,
                params : vec4f,
            }
            @group(0) @binding(0) var<uniform> uniforms : Uniforms;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) height : f32,
            }

            @vertex
            fn vs_main(
                @builtin(vertex_index) vIdx : u32,
                @location(0) h : f32,
                @location(1) v : f32
            ) -> VertexOutput {
                var output : VertexOutput;

                let N = u32(uniforms.params.x);
                let x_idx = vIdx % N;
                let y_idx = vIdx / N;

                // Map grid index to world space (-5 to 5)
                let x = (f32(x_idx) / f32(N) - 0.5) * 10.0;
                let z = (f32(y_idx) / f32(N) - 0.5) * 10.0;

                output.position = uniforms.mvp * vec4f(x, h * 2.0, z, 1.0);
                output.height = h;

                return output;
            }

            @fragment
            fn fs_main(@location(0) height : f32) -> @location(0) vec4f {
                // Color based on height (heat map)
                let val = height * 2.0; // scale up
                let r = max(0.0, val);
                let b = max(0.0, -val);
                let g = 0.5 - abs(val);
                return vec4f(0.2 + r, 0.4 + g, 0.6 + b, 1.0);
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
                    arrayStride: 8, // h(f32) + v(f32) = 8 bytes
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32' }, // h
                        { shaderLocation: 1, offset: 4, format: 'float32' }  // v
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
                        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'line-list' }
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "⚠️ WebGPU Not Available (Seismic Wave Terrain runs in reduced mode)";
        msg.style.cssText = "position:absolute; bottom:10px; right:10px; color:white; background:rgba(255,0,0,0.5); padding:5px; border-radius:4px;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Loop
    // ========================================================================

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        this.resizeGL(w*dpr, h*dpr);
        this.resizeGPU(w*dpr, h*dpr);
    }

    resizeGL(w, h) {
        if(this.glCanvas) {
            this.glCanvas.width = w;
            this.glCanvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
    }

    resizeGPU(w, h) {
        if(this.gpuCanvas) {
            this.gpuCanvas.width = w;
            this.gpuCanvas.height = h;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = performance.now() * 0.001;

        // Camera
        const camX = Math.sin(time * 0.1) * 8.0;
        const camZ = Math.cos(time * 0.1) * 8.0;
        const camY = 6.0;

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

        const zAxis = this.normalize([camX, camY, camZ]);
        const xAxis = this.normalize(this.cross([0,1,0], zAxis));
        const yAxis = this.cross(zAxis, xAxis);

        const view = [
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -this.dot(xAxis, [camX,camY,camZ]), -this.dot(yAxis, [camX,camY,camZ]), -this.dot(zAxis, [camX,camY,camZ]), 1
        ];

        const mvp = this.multiplyMatrices(proj, view);

        // Interaction Check
        // Project mouse ray to plane y=0 roughly (simplified)
        // Actually, we just pass the raw mouse screen coords or normalized device coords?
        // The shader logic `distance(vec2f(gridX, gridY), uniforms.mouse.xy)` implies world space.
        // We need to unproject mouse NDC to world space.
        // For prototype, let's just map NDC directly to World X/Z loosely since camera rotates.
        // Actually, let's pass a "click" signal that just randomly excites a spot if I can't do full raycasting here easily.
        // Let's do a simple "Impact" at center if clicked, or just random drops.

        let clickStrength = 0.0;
        let targetX = 0;
        let targetZ = 0;

        if (this.isMouseDown) {
            clickStrength = 5.0;
            // Hacky raycast: Assume looking at 0,0,0 somewhat.
            // Let's just create waves from the center for now to ensure it works,
            // or use a pre-calculated wandering point.
            targetX = Math.sin(time * 3) * 3;
            targetZ = Math.cos(time * 2) * 3;
        }

        // --- Render WebGL2 ---
        if(this.gl) {
            this.gl.clearColor(0.01, 0.01, 0.02, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(this.uMatrixLoc, false, mvp);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArraysInstanced(this.gl.TRIANGLES, 0, 18, 4);
        }

        // --- Render WebGPU ---
        if(this.device && this.context) {
            // Update Uniforms
            // struct Uniforms { mvp, params(grid, dt, damp, speed), mouse(x,y,str,time) }
            // Size: 64 + 16 + 16 = 96 bytes. Aligned to 16 bytes.
            const uniformData = new Float32Array(36); // allocate plenty
            uniformData.set(mvp, 0); // 0-15
            uniformData[16] = this.options.gridSize;
            uniformData[17] = 0.016; // dt
            uniformData[18] = 0.98; // damping
            uniformData[19] = 150.0; // speed

            uniformData[20] = targetX;
            uniformData[21] = targetZ;
            uniformData[22] = clickStrength;
            uniformData[23] = time;

            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute Pass
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            // Ping Pong: If parity 0, Read A Write B.
            const bindGroup = (this.frameParity === 0) ? this.computeBindGroup1 : this.computeBindGroup2;
            computePass.setBindGroup(0, bindGroup);
            const wg = Math.ceil(this.options.gridSize / 16);
            computePass.dispatchWorkgroups(wg, wg);
            computePass.end();

            // Render Pass
            const textureView = this.context.getCurrentTexture().createView();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.renderBindGroup);
            // Draw using the buffer we just WROTE to (B if parity 0)
            const vertexBuffer = (this.frameParity === 0) ? this.simBufferB : this.simBufferA;
            renderPass.setVertexBuffer(0, vertexBuffer);
            renderPass.setIndexBuffer(this.indexBuffer, 'uint32');
            renderPass.drawIndexed(this.numIndices);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);

            // Swap
            this.frameParity = 1 - this.frameParity;
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        return [v[0]/len, v[1]/len, v[2]/len];
    }
    cross(a, b) {
        return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]];
    }
    dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    multiplyMatrices(a, b) {
        const out = [];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) { sum += b[i * 4 + k] * a[k * 4 + j]; }
                out.push(sum);
            }
        }
        return out;
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
    }
}

if (typeof window !== 'undefined') {
    window.SeismicWaveExperiment = SeismicWaveExperiment;
}
