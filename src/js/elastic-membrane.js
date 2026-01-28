/**
 * Elastic Membrane Experiment
 * Hybrid WebGL2 (Container Frame) + WebGPU (Spring-Mass Simulation)
 */

export class ElasticMembraneExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };
        this.gridSize = 64; // 64x64 particles

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
        this.indexBuffer = null;
        this.indexCount = 0;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        // 1. WebGL2 (Static Frame)
        this.initWebGL2();

        // 2. WebGPU (Simulation)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("ElasticMembrane: WebGPU error", e);
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
        // Normalize to [-1, 1]
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1); // Flip Y
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Wireframe Container)
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

        // Create a simple wireframe box/frame
        const positions = new Float32Array([
            -0.85, -0.85,
             0.85, -0.85,
             0.85,  0.85,
            -0.85,  0.85
        ]);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec2 a_position;
            uniform vec2 u_resolution;
            void main() {
                vec2 pos = a_position;
                if (u_resolution.y > 0.0) {
                    pos.x *= u_resolution.y / u_resolution.x;
                }
                gl_Position = vec4(pos, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            out vec4 outColor;
            void main() {
                outColor = vec4(0.3, 0.3, 0.35, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 2, this.gl.FLOAT, false, 0, 0);
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
    // WebGPU IMPLEMENTATION (Spring-Mass)
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
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // ---------------------------------------------------------
        // WGSL Shaders
        // ---------------------------------------------------------

        const computeCode = `
            struct Particle {
                pos : vec2f,
                vel : vec2f,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Params {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                gridSize : f32,
            }
            @group(0) @binding(1) var<uniform> params : Params;

            fn getIndex(x: u32, y: u32) -> u32 {
                return y * u32(params.gridSize) + x;
            }

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let x = GlobalInvocationID.x;
                let y = GlobalInvocationID.y;
                let size = u32(params.gridSize);

                if (x >= size || y >= size) { return; }

                let index = getIndex(x, y);
                var p = particles[index];

                // 1. Spring Forces
                var force = vec2f(0.0);
                let k = 150.0; // Spring constant
                let restLen = 1.6 / params.gridSize; // Target spacing (covers -0.8 to 0.8)

                // Neighbors (Up, Down, Left, Right)
                // We access valid neighbors. Fixed boundary conditions?
                // Let's pin the edges.
                if (x == 0u || x == size - 1u || y == 0u || y == size - 1u) {
                    // Pinned edges - velocity zero, force zero
                    p.vel = vec2f(0.0);
                    // Reset position to grid default just in case? No, assume init is correct.
                    particles[index] = p;
                    return;
                }

                // Neighbor offsets
                let dirs = array<vec2i, 4>(
                    vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1)
                );

                for (var i = 0; i < 4; i++) {
                    let nx = i32(x) + dirs[i].x;
                    let ny = i32(y) + dirs[i].y;

                    let nIndex = getIndex(u32(nx), u32(ny));
                    let neighbor = particles[nIndex];

                    let delta = neighbor.pos - p.pos;
                    let dist = length(delta);
                    let dir = normalize(delta);

                    // Hooke's Law
                    let stretch = dist - restLen;
                    force = force + dir * (k * stretch);
                }

                // 2. Mouse Interaction
                let mPos = vec2f(params.mouseX, params.mouseY);
                // Adjust for aspect? The mouse is already aspect-corrected in JS usually,
                // but let's assume raw UV space [-1, 1].
                // The grid is in [-0.8, 0.8].
                let mouseDelta = p.pos - mPos;
                // Correct for aspect if needed, but for now simple circle
                let mDist = length(mouseDelta);
                if (mDist < 0.3) {
                    // Repulsion
                    let mDir = normalize(mouseDelta);
                    let mForce = (0.3 - mDist) * 20.0; // Strength
                    force = force + mDir * mForce;
                }

                // 3. Integration
                p.vel = p.vel + force * params.dt;
                p.vel = p.vel * 0.95; // Damping
                p.pos = p.pos + p.vel * params.dt;

                particles[index] = p;
            }
        `;

        const renderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct Particle {
                pos : vec2f,
                vel : vec2f,
            }
            @group(0) @binding(0) var<storage, read> particles : array<Particle>;

            struct Params {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
            }
            @group(0) @binding(1) var<uniform> params : Params;

            @vertex
            fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
                // accessing via index buffer logic implicitly handled by pipeline input
                // wait, if I use drawIndexed, I need to fetch the particle data manually using the index?
                // Actually, if I don't use vertex buffers and instead use storage buffers,
                // I need to map the vertexIndex (which comes from the Index Buffer) to the storage array.
                // But in WGSL @builtin(vertex_index) gives the index from the index buffer!

                let index = vertexIndex;
                let p = particles[index];

                var output : VertexOutput;

                // Aspect correction for rendering
                var pos = p.pos;
                if (params.aspect > 0.0) {
                   pos.x = pos.x / params.aspect; // Compress X if screen is wide?
                   // Usually projection matrix handles this. Here we work in raw NDC.
                   // If aspect = width/height.
                   // To keep square grid looking square:
                   pos.y = pos.y * params.aspect;
                }

                output.position = vec4f(pos, 0.0, 1.0);

                // Color based on velocity/stretch
                let speed = length(p.vel);
                let baseColor = vec3f(0.0, 0.5, 1.0);
                let hotColor = vec3f(1.0, 0.2, 0.5);

                let c = mix(baseColor, hotColor, min(speed * 2.0, 1.0));
                output.color = vec4f(c, 1.0);

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // ---------------------------------------------------------
        // Buffers
        // ---------------------------------------------------------
        const numParticles = this.gridSize * this.gridSize;
        const particleUnitSize = 16; // 4 floats (pos:2, vel:2)
        const initialData = new Float32Array(numParticles * 4);

        // Initialize grid in [-0.8, 0.8] range
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                const i = (y * this.gridSize + x) * 4;
                const u = x / (this.gridSize - 1);
                const v = y / (this.gridSize - 1);

                initialData[i + 0] = (u * 1.6) - 0.8; // x
                initialData[i + 1] = (v * 1.6) - 0.8; // y
                initialData[i + 2] = 0; // vx
                initialData[i + 3] = 0; // vy
            }
        }

        this.particleBuffer = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // Params struct size
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Index Buffer for Grid Lines
        const indices = [];
        // Horizontal lines
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize - 1; x++) {
                const i = y * this.gridSize + x;
                indices.push(i, i + 1);
            }
        }
        // Vertical lines
        for (let x = 0; x < this.gridSize; x++) {
            for (let y = 0; y < this.gridSize - 1; y++) {
                const i = y * this.gridSize + x;
                indices.push(i, i + this.gridSize);
            }
        }
        this.indexCount = indices.length;
        const indexArray = new Uint32Array(indices);

        this.indexBuffer = this.device.createBuffer({
            size: indexArray.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, indexArray);

        // ---------------------------------------------------------
        // Pipelines
        // ---------------------------------------------------------

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // Read-only for Vertex? No, needs read-write for compute
                // Actually, Storage buffer binding type defaults to 'read-only-storage' in vertex?
                // Wait, I need 'read-write-storage' for Compute, and 'read-only-storage' for Vertex (usually).
                // Or I can use one Layout for Compute and another for Render?
                // Let's use separate BindGroups or just compatible layout?
                // In WebGPU, 'buffer' type defaults: 'uniform'
                // For storage: type: 'storage' (read-only), 'read-only-storage' (explicitly read only).
                // 'storage' in GPUShaderStage.COMPUTE is usually read-write if declared as `var<storage, read_write>`.
                // In Vertex, we usually only read.
                // Let's try declaring binding 0 as generic storage and see if it works for both.
                // The issue is `read_write` in vertex shader is not allowed.
                // But in vertex shader I declared `var<storage, read>`.
                // So I can use the same buffer, but maybe I need different BindGroups if the Layout requires specific access mode?
                // Actually, it's safer to have one bind group layout compatible with both if possible, or two layouts.
                // Let's try separate layouts for safety.
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ]
        });

        // Actually, let's redefine entries for Compute specifically
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // Read-write
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ]
        });

        // And Render Layout
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // Read-only
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: renderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                // No vertex buffers, we use storage buffer + index buffer
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }],
            },
            primitive: {
                topology: 'line-list',
                cullMode: 'none',
            },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "WebGPU Not Available";
        msg.style.cssText = "position:absolute; bottom:20px; right:20px; color:red;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
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
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;

        // WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.clearColor(0, 0, 0, 0); // Transparent background
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.LINE_LOOP, 0, 4);
        }

        // WebGPU
        if (this.device && this.context && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, // dt
                time,
                this.mouse.x,
                this.mouse.y,
                aspect,
                this.gridSize,
                0, 0 // padding
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            // 8x8 workgroups. Grid is 64x64.
            // 64/8 = 8 workgroups per dimension.
            computePass.dispatchWorkgroups(8, 8, 1);
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
            renderPass.setBindGroup(0, this.renderBindGroup); // Use Render BindGroup
            renderPass.setIndexBuffer(this.indexBuffer, 'uint32');
            renderPass.drawIndexed(this.indexCount);
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
