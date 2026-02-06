/**
 * Silicon Life Experiment
 * Hybrid Voxel Automata (3D Game of Life)
 *
 * - WebGL2: Renders the boundary wireframe container.
 * - WebGPU: Computes the cellular automata state and renders instanced voxel cubes.
 */

export class SiliconLifeExperiment {
    constructor(container) {
        this.container = container;

        this.isActive = false;
        this.animationId = null;
        this.startTime = Date.now();
        this.frame = 0;

        // Simulation Config
        this.gridSize = 32;
        this.totalCells = this.gridSize * this.gridSize * this.gridSize;
        this.updateInterval = 5; // Update sim every N frames to slow it down

        // Interaction
        this.mouse = { x: 0, y: 0, isPressed: false };
        this.rotation = { x: 0, y: 0 };
        this.targetRotation = { x: 0, y: 0 };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.computePipeline = null;
        this.renderPipeline = null;
        this.stateBufferA = null;
        this.stateBufferB = null;
        this.uniformBuffer = null;
        this.bindGroupA = null; // Read A, Write B
        this.bindGroupB = null; // Read B, Write A
        this.useBufferA = true; // Current state is in A

        // Handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.background = '#020202';
        this.container.style.overflow = 'hidden';

        // 1. WebGL2 Init
        this.initWebGL2();

        // 2. WebGPU Init
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("SiliconLife: WebGPU error", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.resize();
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);
    }

    // ========================================================================
    // WebGL2 Implementation (Container Frame)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none;`;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Create Cube Wireframe (1x1x1 centered)
        // 8 corners
        const corners = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ];

        // Lines
        const indices = [
            0,1, 1,2, 2,3, 3,0, // Back face
            4,5, 5,6, 6,7, 7,4, // Front face
            0,4, 1,5, 2,6, 3,7  // Connecting edges
        ];

        const vertices = [];
        indices.forEach(i => vertices.push(...corners[i]));

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
        this.glVertexCount = vertices.length / 3;

        const vs = `#version 300 es
            in vec3 a_pos;
            uniform mat4 u_mvp;
            void main() {
                gl_Position = u_mvp * vec4(a_pos * 2.0, 1.0); // Scale up to size 4 (+-2) to fit 32 grid (mapped to -1..1 * scale)
            }
        `;

        const fs = `#version 300 es
            precision mediump float;
            out vec4 color;
            void main() {
                color = vec4(0.0, 1.0, 0.5, 0.3);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);
    }

    createGLProgram(vsSrc, fsSrc) {
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
    // WebGPU Implementation (Voxel Automata)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2;`;
        this.container.appendChild(this.gpuCanvas); // Put GPU on top to capture events effectively if needed, but events are on container

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // --- SHADERS ---

        const commonWGSL = `
            struct Uniforms {
                mvp : mat4x4f,
                time : f32,
                gridSize : f32,
                mousePack : vec4f, // x,y, isPressed, padding
            }
        `;

        const computeWGSL = `
            ${commonWGSL}
            @group(0) @binding(0) var<uniform> uniforms : Uniforms;
            @group(0) @binding(1) var<storage, read> stateIn : array<u32>;
            @group(0) @binding(2) var<storage, read_write> stateOut : array<u32>;

            fn getIdx(x: i32, y: i32, z: i32) -> u32 {
                let s = i32(uniforms.gridSize);
                // Wrap around
                let wx = (x + s) % s;
                let wy = (y + s) % s;
                let wz = (z + s) % s;
                return u32(wx + wy * s + wz * s * s);
            }

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let idx = id.x;
                let s = i32(uniforms.gridSize);
                let total = u32(s * s * s);
                if (idx >= total) { return; }

                // Decode position
                let x = i32(idx) % s;
                let y = (i32(idx) / s) % s;
                let z = i32(idx) / (s * s);

                // Mouse Injection
                if (uniforms.mousePack.z > 0.5) {
                    // Random noise injection at click
                    if (rand(vec2f(f32(idx)*0.01, uniforms.time)) > 0.95) {
                         stateOut[idx] = 1u;
                         return;
                    }
                }

                // Count neighbors (Moore neighborhood 3x3x3 - center)
                var neighbors = 0u;
                for (var dz = -1; dz <= 1; dz++) {
                    for (var dy = -1; dy <= 1; dy++) {
                        for (var dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0 && dz == 0) { continue; }
                            let nIdx = getIdx(x + dx, y + dy, z + dz);
                            if (stateIn[nIdx] > 0u) {
                                neighbors++;
                            }
                        }
                    }
                }

                let current = stateIn[idx];
                var next = 0u;

                // Rules: 4555 (Survive 4, Born 5, Survive 5, Born 5) -> "Amoeba" / "Builder" variants
                // Standard Life 3D varies. Let's try 5..7 survive, 6 born?
                // Rule 445: Survive 4, Born 4..5 ?
                // Let's try: Survive 4, Born 5. (Classic 4555)

                // If alive
                if (current > 0u) {
                    if (neighbors >= 4u && neighbors <= 5u) {
                        next = 1u;
                    }
                } else {
                    // If dead
                    if (neighbors == 5u) { // Born
                        next = 1u;
                    }
                }

                stateOut[idx] = next;
            }
        `;

        const renderWGSL = `
            ${commonWGSL}
            @group(0) @binding(0) var<uniform> uniforms : Uniforms;
            @group(0) @binding(1) var<storage, read> state : array<u32>;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) uv : vec2f,
            }

            @vertex
            fn vs_main(
                @builtin(vertex_index) vIdx : u32,
                @builtin(instance_index) iIdx : u32
            ) -> VertexOutput {
                var output : VertexOutput;

                let active = state[iIdx];
                if (active == 0u) {
                    output.position = vec4f(0.0);
                    return output;
                }

                let s = i32(uniforms.gridSize);
                let x = i32(iIdx) % s;
                let y = (i32(iIdx) / s) % s;
                let z = i32(iIdx) / (s * s);

                // Cube Vertices
                // 0: -1,-1, 1
                // ... (Triangle Strip or Index based?)
                // Let's generate cube corners on fly (36 verts for triangles)
                // Simplified: Just 8 corners indexed?
                // Let's use a hardcoded array of 36 vertices for a cube (triangles)

                // Cube -0.5 to 0.5
                var pos = vec3f(0.0);
                // Triangles... too long to hardcode in WGSL cleanly without array.
                // Let's use logic.

                // Box Geometry indices for 36 vertices
                // Front, Back, Top, Bottom, Right, Left
                // Just use a simple billboard? No, needs to be voxel.
                // Okay, let's assume we pass cube geometry vertices in vertex buffer?
                // For simplicity here, let's construct vertices from vIdx (0..35)

                // Trick: Just render points as expanded quads? No, needs 3D.
                // Let's hack a cube.
                let cubeVerts = array<vec3f, 36>(
                    vec3f(-0.4,-0.4, 0.4), vec3f( 0.4,-0.4, 0.4), vec3f( 0.4, 0.4, 0.4), // Front 1
                    vec3f(-0.4,-0.4, 0.4), vec3f( 0.4, 0.4, 0.4), vec3f(-0.4, 0.4, 0.4), // Front 2
                    vec3f( 0.4,-0.4,-0.4), vec3f(-0.4,-0.4,-0.4), vec3f(-0.4, 0.4,-0.4), // Back 1
                    vec3f( 0.4,-0.4,-0.4), vec3f(-0.4, 0.4,-0.4), vec3f( 0.4, 0.4,-0.4), // Back 2
                    vec3f(-0.4, 0.4,-0.4), vec3f(-0.4, 0.4, 0.4), vec3f( 0.4, 0.4, 0.4), // Top 1
                    vec3f(-0.4, 0.4,-0.4), vec3f( 0.4, 0.4, 0.4), vec3f( 0.4, 0.4,-0.4), // Top 2
                    vec3f(-0.4,-0.4, 0.4), vec3f(-0.4,-0.4,-0.4), vec3f( 0.4,-0.4,-0.4), // Bottom 1
                    vec3f(-0.4,-0.4, 0.4), vec3f( 0.4,-0.4,-0.4), vec3f( 0.4,-0.4, 0.4), // Bottom 2
                    vec3f( 0.4,-0.4, 0.4), vec3f( 0.4,-0.4,-0.4), vec3f( 0.4, 0.4,-0.4), // Right 1
                    vec3f( 0.4,-0.4, 0.4), vec3f( 0.4, 0.4,-0.4), vec3f( 0.4, 0.4, 0.4), // Right 2
                    vec3f(-0.4,-0.4,-0.4), vec3f(-0.4,-0.4, 0.4), vec3f(-0.4, 0.4, 0.4), // Left 1
                    vec3f(-0.4,-0.4,-0.4), vec3f(-0.4, 0.4, 0.4), vec3f(-0.4, 0.4,-0.4)  // Left 2
                );

                let vPos = cubeVerts[vIdx];

                // Map grid 0..31 to -2..2
                let gx = (f32(x) / uniforms.gridSize) * 4.0 - 2.0;
                let gy = (f32(y) / uniforms.gridSize) * 4.0 - 2.0;
                let gz = (f32(z) / uniforms.gridSize) * 4.0 - 2.0;

                // Scale voxel
                let scale = 4.0 / uniforms.gridSize;

                let worldPos = vec3f(gx, gy, gz) + vPos * scale;

                output.position = uniforms.mvp * vec4f(worldPos, 1.0);

                // Color based on position
                let r = f32(x) / uniforms.gridSize;
                let g = f32(y) / uniforms.gridSize;
                let b = f32(z) / uniforms.gridSize;

                // Add shading based on normal approx (vPos)
                let light = normalize(vec3f(1.0, 2.0, 3.0));
                let normal = normalize(vPos); // Cube center is 0,0,0 local
                let diff = max(dot(normal, light), 0.2);

                output.color = vec4f(vec3f(r, g, b) * diff + 0.1, 1.0);

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // --- BUFFERS ---

        // State Buffers
        const stateSize = this.totalCells * 4; // u32
        this.stateBufferA = this.device.createBuffer({ size: stateSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.stateBufferB = this.device.createBuffer({ size: stateSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

        // Init random state
        const initialData = new Uint32Array(this.totalCells);
        for(let i=0; i<this.totalCells; i++) {
            initialData[i] = Math.random() > 0.8 ? 1 : 0;
        }
        this.device.queue.writeBuffer(this.stateBufferA, 0, initialData);

        // Uniform Buffer
        this.uniformBuffer = this.device.createBuffer({
            size: 96, // mat4 (64) + floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // --- PIPELINES ---

        // Compute
        const computeModule = this.device.createShaderModule({ code: computeWGSL });
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        this.bindGroupA = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.stateBufferA } }, // Read A
                { binding: 2, resource: { buffer: this.stateBufferB } }  // Write B
            ]
        });

        this.bindGroupB = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.stateBufferB } }, // Read B
                { binding: 2, resource: { buffer: this.stateBufferA } }  // Write A
            ]
        });

        // Render
        const renderModule = this.device.createShaderModule({ code: renderWGSL });
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: { module: renderModule, entryPoint: 'vs_main' },
            fragment: {
                module: renderModule, entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: { // Additive for glowy look
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list' },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            }
        });

        this.depthTexture = this.device.createTexture({
            size: [this.gpuCanvas.width, this.gpuCanvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Render Bind Groups (One for A, One for B)
        this.renderBindGroupA = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.stateBufferA } } // Render from A
            ]
        });

        this.renderBindGroupB = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.stateBufferB } } // Render from B
            ]
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "WebGPU Not Supported - Simulation Disabled";
        msg.style.cssText = `position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: red; font-family: monospace; background: rgba(0,0,0,0.8); padding: 20px; border: 1px solid red;`;
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Logic
    // ========================================================================

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

            if (this.device) {
                if (this.depthTexture) this.depthTexture.destroy();
                this.depthTexture = this.device.createTexture({
                    size: [this.gpuCanvas.width, this.gpuCanvas.height],
                    format: 'depth24plus',
                    usage: GPUTextureUsage.RENDER_ATTACHMENT,
                });
            }
        }
    }

    onMouseMove(e) {
        if (e.buttons === 1) { // Rotate
            this.targetRotation.y += e.movementX * 0.01;
            this.targetRotation.x += e.movementY * 0.01;
        }

        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    onMouseDown() { this.mouse.isPressed = true; }
    onMouseUp() { this.mouse.isPressed = false; }

    getMVP(aspect) {
        const fov = 60 * Math.PI / 180;
        const f = 1.0 / Math.tan(fov / 2);
        const near = 0.1;
        const far = 100.0;
        const proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0
        ];

        // Smooth rotation
        this.rotation.x += (this.targetRotation.x - this.rotation.x) * 0.1;
        this.rotation.y += (this.targetRotation.y - this.rotation.y) * 0.1;

        const camDist = 6.0;
        const cx = Math.sin(this.rotation.y) * Math.cos(this.rotation.x) * camDist;
        const cy = Math.sin(this.rotation.x) * camDist;
        const cz = Math.cos(this.rotation.y) * Math.cos(this.rotation.x) * camDist;

        const eye = [cx, cy, cz];
        const center = [0, 0, 0];
        const up = [0, 1, 0];

        // LookAt
        const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
        const len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        const zx = z0 * len, zy = z1 * len, zz = z2 * len;
        const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
        const lenX = 1 / Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
        const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;
        const y0 = zy * xz - zz * xy, y1 = zz * xx - zx * xz, y2 = zx * xy - zy * xx;
        const lenY = 1 / Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
        const yx = y0 * lenY, yy = y1 * lenY, yz = y2 * lenY;

        const view = [
            xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
            -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
            -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
            1
        ];

        // Mult
        const mvp = new Float32Array(16);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) sum += proj[k * 4 + i] * view[j * 4 + k]; // Note index order for this specific lookat/proj combo
                mvp[i * 4 + j] = sum;
            }
        }
        // Actually, just use standard column major mult if arrays are standard
        // Proj (Col Major) * View (Col Major)
        // out[col][row]
        const out = new Float32Array(16);
        for(let col=0; col<4; col++) {
            for(let row=0; row<4; row++) {
                let s = 0;
                for(let k=0; k<4; k++) s += proj[k*4+row] * view[col*4+k];
                out[col*4+row] = s;
            }
        }
        return out;
    }

    animate() {
        if (!this.isActive) return;

        this.frame++;
        const aspect = this.container.clientWidth / this.container.clientHeight;
        const mvp = this.getMVP(aspect);

        // 1. WebGL2 Render
        if (this.gl) {
            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT); // No depth clear needed if on top? Actually under.
            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_mvp'), false, mvp);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.LINES, 0, this.glVertexCount);
        }

        // 2. WebGPU Render
        if (this.device && this.renderPipeline) {
            // Update Uniforms
            const uniforms = new Float32Array(24);
            uniforms.set(mvp, 0);
            uniforms[16] = (Date.now() - this.startTime) * 0.001;
            uniforms[17] = this.gridSize;
            uniforms[18] = this.mouse.x;
            uniforms[19] = this.mouse.y;
            uniforms[20] = this.mouse.isPressed ? 1.0 : 0.0;

            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute Step (only every N frames to make it readable)
            if (this.frame % this.updateInterval === 0 || this.mouse.isPressed) {
                const computePass = commandEncoder.beginComputePass();
                computePass.setPipeline(this.computePipeline);
                // Use current state as input
                computePass.setBindGroup(0, this.useBufferA ? this.bindGroupA : this.bindGroupB);
                computePass.dispatchWorkgroups(Math.ceil(this.totalCells / 64));
                computePass.end();

                // Swap
                this.useBufferA = !this.useBufferA;
            }

            // Render Step
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
            // Render the *new* current state (which was just written to, or is static)
            // If we just swapped, 'useBufferA' points to the *next* input.
            // So we want to render the buffer that was just written to.
            // If useBufferA is true, it means A is input for next frame. So B was output of last frame.
            // So render B.
            renderPass.setBindGroup(0, this.useBufferA ? this.renderBindGroupB : this.renderBindGroupA);

            renderPass.draw(36, this.totalCells); // 36 vertices per instance, N instances
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}
