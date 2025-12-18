/**
 * Neon City Experiment
 * Combines WebGL2 for procedural city rendering and WebGPU for compute-heavy effects.
 */

class NeonCityExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.speed = 0.5;
        this.rainDensity = 0.7;

        // WebGL2 State (City)
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.instanceCount = 2000;
        this.buildingBuffer = null; // Buffer for building properties (pos, size)

        // WebGPU State (Rain)
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.rainPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.rainBuffer = null;
        this.uniformBuffer = null;
        this.numRainDrops = 10000;

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        // Setup container
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050510';

        // Controls
        const speedInput = document.getElementById('speed-control');
        if (speedInput) {
            speedInput.addEventListener('input', (e) => {
                this.speed = parseInt(e.target.value) / 100;
            });
        }
        const rainInput = document.getElementById('rain-control');
        if (rainInput) {
            rainInput.addEventListener('input', (e) => {
                this.rainDensity = parseInt(e.target.value) / 100;
                this.updateRainParams();
            });
        }

        // 1. Init WebGL2 (City Layer)
        this.initWebGL2();

        // 2. Init WebGPU (Rain Layer)
        if (navigator.gpu) {
            try {
                await this.initWebGPU();
            } catch (e) {
                console.warn("NeonCity: WebGPU failed to init", e);
            }
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Cityscape)
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

        // Generate City Data (Instanced)
        // Position (x, z), Size (w, h, d), Color (r, g, b)
        // We pack this into attributes.
        // Let's keep it simple: Position X, Z, Scale Y (height), Random Seed
        const instanceData = new Float32Array(this.instanceCount * 4);
        const range = 200;

        for (let i = 0; i < this.instanceCount; i++) {
            const x = (Math.random() - 0.5) * range;
            const z = -Math.random() * range; // Extend into distance
            const h = Math.random() * 5.0 + 1.0; // Height
            const seed = Math.random();

            instanceData[i * 4 + 0] = x;
            instanceData[i * 4 + 1] = z;
            instanceData[i * 4 + 2] = h;
            instanceData[i * 4 + 3] = seed;
        }

        // Cube Vertices (Unit Cube)
        const vertices = new Float32Array([
            // Front face
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            // Back face
            -0.5, -0.5, -0.5,
            -0.5,  0.5, -0.5,
             0.5,  0.5, -0.5,
             0.5, -0.5, -0.5,
            // Top face
            -0.5,  0.5, -0.5,
            -0.5,  0.5,  0.5,
             0.5,  0.5,  0.5,
             0.5,  0.5, -0.5,
             // Bottom face
            -0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,
             0.5, -0.5,  0.5,
            -0.5, -0.5,  0.5,
            // Right face
             0.5, -0.5, -0.5,
             0.5,  0.5, -0.5,
             0.5,  0.5,  0.5,
             0.5, -0.5,  0.5,
             // Left face
            -0.5, -0.5, -0.5,
            -0.5, -0.5,  0.5,
            -0.5,  0.5,  0.5,
            -0.5,  0.5, -0.5,
        ]);

        const indices = new Uint16Array([
            0,  1,  2,      0,  2,  3,    // front
            4,  5,  6,      4,  6,  7,    // back
            8,  9,  10,     8,  10, 11,   // top
            12, 13, 14,     12, 14, 15,   // bottom
            16, 17, 18,     16, 18, 19,   // right
            20, 21, 22,     20, 22, 23    // left
        ]);

        // Create Program
        const vs = `#version 300 es
        layout(location=0) in vec3 a_position;
        layout(location=1) in vec4 a_instanceData; // x, z, height, seed

        uniform mat4 u_projection;
        uniform mat4 u_view;
        uniform float u_time;
        uniform float u_scrollSpeed;

        out vec3 v_color;
        out float v_dist;

        void main() {
            vec3 pos = a_position;

            // Instance data
            float ix = a_instanceData.x;
            float iz = a_instanceData.y;
            float ih = a_instanceData.z;
            float iseed = a_instanceData.w;

            // Scroll Logic
            float zPos = iz + u_time * u_scrollSpeed * 20.0;
            // Wrap around
            float range = 200.0;
            if (zPos > 10.0) {
                zPos -= range;
            }

            // Scale building
            pos.y *= ih;
            pos.y += ih * 0.5; // Move base to y=0

            // Apply World Position
            vec3 worldPos = vec3(pos.x + ix, pos.y, pos.z + zPos);

            gl_Position = u_projection * u_view * vec4(worldPos, 1.0);

            // Color based on height and randomness
            float glow = 0.2 + 0.8 * iseed;
            vec3 buildingColor = mix(vec3(0.1, 0.0, 0.2), vec3(0.0, 0.8, 1.0), iseed);

            // Windows effect?
            if (mod(worldPos.y * 2.0, 1.0) > 0.5 && mod(worldPos.x + worldPos.z, 2.0) > 1.0) {
                 buildingColor += vec3(0.8, 0.8, 0.5) * glow;
            }

            v_color = buildingColor;
            v_dist = length(worldPos.xz);
        }
        `;

        const fs = `#version 300 es
        precision highp float;

        in vec3 v_color;
        in float v_dist;
        out vec4 outColor;

        void main() {
            vec3 color = v_color;

            // Fog
            float fogFactor = smoothstep(10.0, 100.0, v_dist);
            vec3 fogColor = vec3(0.05, 0.05, 0.1);

            color = mix(color, fogColor, fogFactor);

            outColor = vec4(color, 1.0);
        }
        `;

        this.glProgram = this.createGLProgram(vs, fs);

        // VAO Setup
        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        // Geometry Buffer
        const geoBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, geoBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);

        // Instance Buffer
        const instBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, instBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, instanceData, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(1);
        this.gl.vertexAttribPointer(1, 4, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(1, 1); // Important: Per instance

        // Index Buffer
        const idxBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        this.gl.enable(this.gl.DEPTH_TEST);
        this.resizeGL();
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fs));
            return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Data Rain)
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

        // Compute Shader: Update rain positions
        const computeCode = `
            struct Particle {
                pos: vec2f,
                speed: f32,
                len: f32,
            }
            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Uniforms {
                dt: f32,
                density: f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let i = id.x;
                if (i >= ${this.numRainDrops}) { return; }

                var p = particles[i];

                // Fall down
                p.pos.y = p.pos.y - p.speed * uniforms.dt;

                // Reset if below screen
                if (p.pos.y < -1.2) {
                    p.pos.y = 1.2 + fract(p.speed * 123.45) * 0.5;
                    p.pos.x = (fract(p.pos.x * 67.89 + uniforms.dt) - 0.5) * 2.0;
                }

                particles[i] = p;
            }
        `;

        // Render Shader: Draw rain as lines
        const drawCode = `
            struct Particle {
                pos: vec2f,
                speed: f32,
                len: f32,
            }
            @group(0) @binding(0) var<storage, read> particles : array<Particle>;

            struct VertexOutput {
                @builtin(position) pos : vec4f,
                @location(0) speed : f32,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vIdx : u32, @builtin(instance_index) iIdx : u32) -> VertexOutput {
                let p = particles[iIdx];

                // Draw a vertical line (2 vertices)
                // Vertex 0: Top, Vertex 1: Bottom
                let yOffset = f32(vIdx) * p.len * 0.1;

                var out: VertexOutput;
                out.pos = vec4f(p.pos.x, p.pos.y + yOffset, 0.0, 1.0);
                out.speed = p.speed;
                return out;
            }

            @fragment
            fn fs_main(@location(0) speed : f32) -> @location(0) vec4f {
                let alpha = clamp(speed * 0.5, 0.2, 0.8);
                return vec4f(0.0, 1.0, 0.5, alpha);
            }
        `;

        // Init buffers
        const pSize = 4 * 4; // 4 floats (pos:vec2, speed:f32, len:f32) aligned to 16 bytes?
        // vec2f is 8 bytes, f32 is 4 bytes. Struct padding might be needed.
        // WGSL struct alignment: vec2f (0), speed (8), len (12). Size 16. Correct.

        const initData = new Float32Array(this.numRainDrops * 4);
        for(let i=0; i<this.numRainDrops; i++) {
            initData[i*4+0] = (Math.random() - 0.5) * 2.0; // x
            initData[i*4+1] = Math.random() * 2.0 - 1.0; // y
            initData[i*4+2] = 0.5 + Math.random(); // speed
            initData[i*4+3] = 0.5 + Math.random(); // len
        }

        this.rainBuffer = this.device.createBuffer({
            size: initData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.rainBuffer.getMappedRange()).set(initData);
        this.rainBuffer.unmap();

        this.uniformBuffer = this.device.createBuffer({
            size: 16, // dt(4), density(4), pad(8)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Layouts & Pipelines
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // Actually read-write in compute
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        // Note: For compute it needs to be storage (read_write). For vertex it is read-only.
        // WebGPU separates these. We need to be careful.
        // Actually, 'read-only-storage' in vertex and 'storage' in compute for the same buffer is tricky if using same bind group.
        // Better:
        // Compute BindGroup: Binding 0 (Storage), Binding 1 (Uniform)
        // Render BindGroup: Binding 0 (ReadOnlyStorage)

        // Let's try one BG layout with "buffer: { type: 'storage' }" which implies read-write in compute.
        // In vertex shader, we can define it as `var<storage, read>`.
        // However, standard says default storage is read-only. `read-write` needs explicit type.

        // Let's create two layouts if needed, or just one that is permissive?
        // Actually, `buffer: { type: 'storage' }` is fine for both if usage allows.

        const computeBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // read-write
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        const renderBGLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.rainBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.rainBuffer } }
            ]
        });

        // Compute Pipeline
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGLayout] }),
            compute: { module: this.device.createShaderModule({ code: computeCode }), entryPoint: 'main' }
        });

        // Render Pipeline
        this.rainPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBGLayout] }),
            vertex: {
                module: this.device.createShaderModule({ code: drawCode }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.device.createShaderModule({ code: drawCode }),
                entryPoint: 'fs_main',
                targets: [{ format, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                } }]
            },
            primitive: { topology: 'line-list' }
        });

        this.resizeGPU();
    }

    updateRainParams() {
        // We could map this to uniform density if we implemented logic for it in shader
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth * dpr;
        const h = this.container.clientHeight * dpr;

        this.resizeGL(w, h);
        this.resizeGPU(w, h);
    }

    resizeGL(w, h) {
        if (!this.glCanvas) return;
        this.glCanvas.width = w;
        this.glCanvas.height = h;
        this.gl.viewport(0, 0, w, h);
    }

    resizeGPU(w, h) {
        if (!this.gpuCanvas) return;
        this.gpuCanvas.width = w;
        this.gpuCanvas.height = h;
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016; // Fixed timestep for simplicity

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Matrices
            const aspect = this.glCanvas.width / this.glCanvas.height;
            const projection = this.createPerspectiveMatrix(60, aspect, 0.1, 500.0);
            const view = this.createLookAtMatrix(
                [0, 5, -20], // Eye
                [0, 0, 50],  // Target
                [0, 1, 0]    // Up
            );

            const projLoc = this.gl.getUniformLocation(this.glProgram, 'u_projection');
            const viewLoc = this.gl.getUniformLocation(this.glProgram, 'u_view');
            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const scrollLoc = this.gl.getUniformLocation(this.glProgram, 'u_scrollSpeed');

            this.gl.uniformMatrix4fv(projLoc, false, projection);
            this.gl.uniformMatrix4fv(viewLoc, false, view);
            this.gl.uniform1f(timeLoc, time);
            this.gl.uniform1f(scrollLoc, this.speed);

            this.gl.clearColor(0.02, 0.02, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            // Draw 36 vertices (cube) * instanceCount
            this.gl.drawElementsInstanced(this.gl.TRIANGLES, 36, this.gl.UNSIGNED_SHORT, 0, this.instanceCount);
        }

        // 2. WebGPU Render
        if (this.device && this.rainPipeline) {
            // Update Uniforms
            const uData = new Float32Array([dt * (1.0 + this.speed * 5.0), this.rainDensity]); // Speed up rain with scroll
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uData);

            const encoder = this.device.createCommandEncoder();

            // Compute
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numRainDrops / 64));
            cPass.end();

            // Render
            const textureView = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'load', // Load WebGL canvas content beneath? No, they are separate canvases.
                    storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.rainPipeline);
            rPass.setBindGroup(0, this.renderBindGroup);
            // Draw 2 vertices per instance * numRainDrops instances
            rPass.draw(2, this.numRainDrops);
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    // Matrix Helpers
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
        // y is normalized if z and x are

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
}

if (typeof window !== 'undefined') {
    window.NeonCityExperiment = NeonCityExperiment;
}
