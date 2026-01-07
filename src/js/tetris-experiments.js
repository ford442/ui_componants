/**
 * Tetris Experiments
 * Contains two hybrid WebGL2/WebGPU classes:
 * 1. NeonTetrisRain: Falling blocks
 * 2. VoxelDestruct: Exploding voxel block
 */

export class NeonTetrisRain {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;
        this.isActive = false;
        this.blockCount = options.blockCount || 1000;

        this.canvasSize = { width: 0, height: 0 };
        this.startTime = Date.now();
        this.animationId = null;

        // WebGL2
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;

        // WebGPU
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.bindGroup = null;

        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.background = '#050510';

        this.initWebGL2();

        if (navigator.gpu) {
            await this.initWebGPU();
        } else {
            console.warn("WebGPU not supported");
        }

        this.resizeObserver.observe(this.container);
        this.resize();
        this.isActive = true;
        this.animate();
    }

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 1; pointer-events: none;';
        this.container.appendChild(this.glCanvas);
        this.gl = this.glCanvas.getContext('webgl2');

        // Simple grid shader
        const vs = `#version 300 es
        in vec2 position;
        uniform float time;
        out vec2 vUv;
        void main() {
            vUv = position * 2.0; // Scale uv
            gl_Position = vec4(position, 0.0, 1.0);
        }`;

        const fs = `#version 300 es
        precision highp float;
        in vec2 vUv;
        uniform float time;
        uniform vec2 resolution;
        out vec4 fragColor;

        void main() {
            vec2 uv = vUv;
            uv.x *= resolution.x / resolution.y;

            // Retro Grid
            uv.y += time * 0.2;
            vec2 grid = abs(fract(uv * 4.0) - 0.5) / fwidth(uv * 4.0);
            float line = min(grid.x, grid.y);
            float alpha = 1.0 - smoothstep(0.0, 0.1, line);

            // Distance fade
            float fade = 1.0 - smoothstep(0.0, 2.0, abs(vUv.y)); // Fade at edges

            fragColor = vec4(vec3(0.8, 0.0, 1.0) * alpha * fade * 0.3, 1.0);
        }`;

        this.glProgram = this.createProgram(this.gl, vs, fs);

        // Quad
        const positions = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
        const vbo = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vbo);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const loc = this.gl.getAttribLocation(this.glProgram, 'position');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 2, this.gl.FLOAT, false, 0, 0);
    }

    async initWebGPU() {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return;
        this.device = await adapter.requestDevice();

        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 2; pointer-events: none;';
        this.container.appendChild(this.gpuCanvas);

        this.context = this.gpuCanvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'premultiplied'
        });

        // Compute Shader: Falling logic
        const computeShader = `
            struct Particle {
                pos: vec2f,
                vel: vec2f,
                color: vec4f,
                shape_type: f32, // 0-3 for different shapes if we get fancy, just color for now
            }
            struct Uniforms {
                time: f32,
                dt: f32,
                aspect: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Fall
                p.pos.y -= p.vel.y * uniforms.dt;

                // Reset if below screen
                if (p.pos.y < -1.2) {
                    p.pos.y = 1.2 + fract(sin(f32(i) * 12.34) * 43758.54) * 2.0; // Random offset
                    p.pos.x = (fract(cos(f32(i) * 56.78 + uniforms.time) * 12345.67) - 0.5) * 2.5;
                }

                particles[i] = p;
            }
        `;

        // Render Shader: Draw blocks (instanced quads)
        const renderShader = `
            struct Particle {
                pos: vec2f,
                vel: vec2f,
                color: vec4f,
                shape_type: f32,
            }
            @group(0) @binding(0) var<storage, read> particles: array<Particle>;

            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) iIdx: u32) -> VertexOutput {
                let p = particles[iIdx];

                // Simple quad vertices
                var corners = array<vec2f, 4>(
                    vec2f(-0.02, -0.02), vec2f(0.02, -0.02),
                    vec2f(-0.02,  0.02), vec2f(0.02,  0.02)
                );

                let vPos = corners[vIdx] + p.pos;

                var out: VertexOutput;
                out.pos = vec4f(vPos, 0.0, 1.0);
                out.color = p.color;
                return out;
            }

            @fragment
            fn fs(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Init Data
        const data = new Float32Array(this.blockCount * 8); // 8 floats per particle
        for(let i=0; i<this.blockCount; i++) {
            const base = i * 8;
            data[base+0] = (Math.random() - 0.5) * 2.5; // x
            data[base+1] = Math.random() * 2.0 - 1.0;   // y
            data[base+2] = 0; // vx
            data[base+3] = 0.5 + Math.random() * 0.5; // vy (fall speed)

            // Colors (Neon palette)
            const colors = [
                [0.0, 1.0, 0.8, 1.0], // Cyan
                [1.0, 0.0, 0.8, 1.0], // Magenta
                [0.8, 1.0, 0.0, 1.0], // Lime
                [1.0, 0.5, 0.0, 1.0]  // Orange
            ];
            const c = colors[Math.floor(Math.random() * colors.length)];
            data[base+4] = c[0]; data[base+5] = c[1]; data[base+6] = c[2]; data[base+7] = c[3];
        }

        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(data);
        this.particleBuffer.unmap();

        this.uniformBuffer = this.device.createBuffer({
            size: 32, // time(f32), dt(f32), aspect(f32), padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Layouts
        const computeLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        // Render Layout
        const renderLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
            ]
        });

        this.computeBG = this.device.createBindGroup({
            layout: computeLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBG = this.device.createBindGroup({
            layout: renderLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } }
            ]
        });

        // Pipelines
        const cModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
            compute: { module: cModule, entryPoint: 'main' }
        });

        const rModule = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
            vertex: {
                module: rModule,
                entryPoint: 'vs',
            },
            fragment: {
                module: rModule,
                entryPoint: 'fs',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat(), blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                }}]
            },
            primitive: { topology: 'triangle-strip' }
        });
    }

    createProgram(gl, vs, fs) {
        const p = gl.createProgram();
        const v = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(v, vs); gl.compileShader(v);
        const f = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(f, fs); gl.compileShader(f);
        gl.attachShader(p, v); gl.attachShader(p, f);
        gl.linkProgram(p);
        return p;
    }

    resize() {
        if(!this.container) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

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

        const time = (Date.now() - this.startTime) / 1000;

        // Render WebGL
        if (this.gl) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.clearColor(0,0,0,0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGPU
        if (this.device && this.context) {
            const aspect = this.gpuCanvas.width / this.gpuCanvas.height;
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([time, 0.016, aspect]));

            const enc = this.device.createCommandEncoder();

            // Compute
            const cPass = enc.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBG);
            cPass.dispatchWorkgroups(Math.ceil(this.blockCount / 64));
            cPass.end();

            // Render
            const rPass = enc.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear', storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBG);
            rPass.draw(4, this.blockCount); // 4 vertices per quad, N instances
            rPass.end();

            this.device.queue.submit([enc.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

export class VoxelDestruct {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;
        this.isActive = false;
        this.particleCount = options.particleCount || 10000;

        this.startTime = Date.now();
        this.animationId = null;

        // WebGL2 (Central Object)
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;

        // WebGPU (Particles)
        this.gpuCanvas = null;
        this.device = null;

        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.background = '#100505';

        this.initWebGL2();
        if (navigator.gpu) await this.initWebGPU();

        this.resizeObserver.observe(this.container);
        this.resize();
        this.isActive = true;
        this.animate();
    }

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 1; pointer-events: none;';
        this.container.appendChild(this.glCanvas);
        this.gl = this.glCanvas.getContext('webgl2');

        const vs = `#version 300 es
        in vec3 position;
        uniform float time;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        out vec3 vPos;
        void main() {
            vPos = position;
            gl_Position = projection * view * model * vec4(position, 1.0);
        }`;

        const fs = `#version 300 es
        precision highp float;
        in vec3 vPos;
        out vec4 fragColor;
        void main() {
            vec3 color = vec3(1.0, 0.2, 0.2); // Red core
            float edge = step(0.95, max(abs(vPos.x), max(abs(vPos.y), abs(vPos.z))));
            fragColor = vec4(mix(color, vec3(1.0), edge), 1.0);
        }`;

        this.glProgram = this.createProgram(this.gl, vs, fs);

        // Cube
        const vertices = new Float32Array([
            -1,-1,1,  1,-1,1,  1,1,1,  -1,1,1, // Front
            -1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1 // Back
        ]);
        const indices = new Uint16Array([
            0,1,2, 0,2,3, // Front
            4,5,6, 4,6,7, // Back
            0,3,7, 0,7,4, // Left
            1,2,6, 1,6,5, // Right
            3,2,6, 3,6,7, // Top
            0,1,5, 0,5,4  // Bottom
        ]);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const vbo = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vbo);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);

        const ibo = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, ibo);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        const loc = this.gl.getAttribLocation(this.glProgram, 'position');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 3, this.gl.FLOAT, false, 0, 0);

        this.indexCount = indices.length;
    }

    createProgram(gl, vs, fs) {
        const p = gl.createProgram();
        const v = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(v, vs); gl.compileShader(v);
        const f = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(f, fs); gl.compileShader(f);
        gl.attachShader(p, v); gl.attachShader(p, f);
        gl.linkProgram(p);
        return p;
    }

    async initWebGPU() {
        const adapter = await navigator.gpu.requestAdapter();
        if(!adapter) return;
        this.device = await adapter.requestDevice();
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 2; pointer-events: none;';
        this.container.appendChild(this.gpuCanvas);
        this.context = this.gpuCanvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' });

        const computeShader = `
            struct Particle {
                pos: vec4f, // xyz, life
                vel: vec4f, // xyz, unused
            }
            struct Uniforms {
                time: f32,
                dt: f32,
            }
            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Explode logic
                p.pos = vec4f(p.pos.xyz + p.vel.xyz * uniforms.dt, p.pos.w - uniforms.dt * 0.5);

                // Reset
                if (p.pos.w <= 0.0) {
                    let r = fract(sin(f32(i)* uniforms.time) * 43758.54);
                    let theta = r * 6.28;
                    let phi = fract(cos(f32(i)) * 12345.67) * 3.14;

                    let dir = vec3f(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta));
                    p.pos = vec4f(0.0, 0.0, 0.0, 1.0); // Reset to center
                    p.vel = vec4f(dir * (1.0 + r), 0.0);
                }

                particles[i] = p;
            }
        `;

        const renderShader = `
            struct Particle {
                pos: vec4f,
                vel: vec4f,
            }
            @group(0) @binding(0) var<storage, read> particles: array<Particle>;
            struct Uniforms {
                mvp: mat4x4f,
            }
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            struct VO {
                @builtin(position) pos: vec4f,
                @location(0) life: f32,
            }

            @vertex
            fn vs(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) iIdx: u32) -> VO {
                let p = particles[iIdx];
                var output: VO;
                output.pos = uniforms.mvp * vec4f(p.pos.xyz, 1.0);

                // Make point size by offset in clip space (billboard) roughly
                let size = 0.01 * p.pos.w;
                // Simple point rendering trick if topology is point-list
                output.life = p.pos.w;
                return output;
            }

            @fragment
            fn fs(@location(0) life: f32) -> @location(0) vec4f {
                return vec4f(1.0, 0.5 * life, 0.0, life);
            }
        `;

        // Init data
        const pData = new Float32Array(this.particleCount * 8);
        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.uniformBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // time, dt
        this.renderUniformBuffer = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // mvp

        // Layouts
        const cBindLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        const rBindLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBG = this.device.createBindGroup({
            layout: cBindLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBG = this.device.createBindGroup({
            layout: rBindLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.renderUniformBuffer } }
            ]
        });

        // Pipeline creation
        const cMod = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [cBindLayout] }),
            compute: { module: cMod, entryPoint: 'main' }
        });

        const rMod = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [rBindLayout] }),
            vertex: { module: rMod, entryPoint: 'vs' },
            fragment: { module: rMod, entryPoint: 'fs', targets: [{ format: navigator.gpu.getPreferredCanvasFormat(), blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
            } }] },
            primitive: { topology: 'point-list' }
        });
    }

    resize() {
        if(!this.container) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if(this.glCanvas) { this.glCanvas.width = w; this.glCanvas.height = h; this.gl.viewport(0,0,w,h); }
        if(this.gpuCanvas) { this.gpuCanvas.width = w; this.gpuCanvas.height = h; }
    }

    animate() {
        if (!this.isActive) return;
        const time = (Date.now() - this.startTime) / 1000;

        // Matrix Math (simplified)
        const aspect = (this.glCanvas?.width || 1) / (this.glCanvas?.height || 1);
        const fov = 60 * Math.PI / 180;
        const zNear = 0.1, zFar = 100.0;
        const f = 1.0 / Math.tan(fov / 2);
        const rangeInv = 1 / (zNear - zFar);
        const proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (zFar + zNear) * rangeInv, -1,
            0, 0, (2 * zFar * zNear) * rangeInv, 0
        ];

        const camZ = 5.0;
        const view = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, -camZ, 1
        ];

        const rotY = time * 0.5;
        const model = [
            Math.cos(rotY), 0, Math.sin(rotY), 0,
            0, 1, 0, 0,
            -Math.sin(rotY), 0, Math.cos(rotY), 0,
            0, 0, 0, 1
        ];

        // Render WebGL
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            // Upload matrices
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'projection'), false, proj);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'view'), false, view);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'model'), false, model);

            this.gl.clearColor(0,0,0,0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.indexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // Render WebGPU
        if (this.device && this.context) {
            // Update Uniforms
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([time, 0.016]));

            // MVP for GPU
            // Multiply P * V * M in JS (simplified for speed, just assume simple multiplication logic or use library if available, but I must impl manually)
            const mvp = new Float32Array([
                1.0/aspect * 0.5, 0, 0, 0,
                0, 0.5, 0, 0,
                0, 0, 0.5, 0,
                0, 0, 0, 1
            ]);

            this.device.queue.writeBuffer(this.renderUniformBuffer, 0, mvp);

            const enc = this.device.createCommandEncoder();
            const cPass = enc.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBG);
            cPass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
            cPass.end();

            const rPass = enc.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear', storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBG);
            rPass.draw(this.particleCount);
            rPass.end();

            this.device.queue.submit([enc.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

export class SampledGlassTetris {
    constructor(container, options = {}) {
        this.container = container;
        this.imagePath = options.imagePath || 'bezel.png'; // Fallback

        this.canvas = null;
        this.gl = null;
        this.program = null;
        this.texture = null;
        this.cubes = [];
        this.startTime = Date.now();

        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.background = '#050505'; // Dark background

        // 1. Setup Canvas
        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = 'position: absolute; inset: 0; width: 100%; height: 100%;';
        this.container.appendChild(this.canvas);

        this.gl = this.canvas.getContext('webgl2');
        if (!this.gl) return;

        // 2. The "Holographic" Shader
        const vs = `#version 300 es
        in vec3 position;
        in vec3 normal;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        out vec3 vNormal;
        out vec3 vWorldPos;
        out vec2 vUv;

        void main() {
            vUv = position.xy * 0.5 + 0.5; // Map -1..1 to 0..1
            vNormal = (model * vec4(normal, 0.0)).xyz;
            vec4 worldPos = model * vec4(position, 1.0);
            vWorldPos = worldPos.xyz;
            gl_Position = projection * view * worldPos;
        }`;

        const fs = `#version 300 es
        precision highp float;

        in vec3 vNormal;
        in vec3 vWorldPos;
        in vec2 vUv;

        uniform vec2 resolution;
        uniform sampler2D uTexture;
        uniform float time;

        out vec4 fragColor;

        void main() {
            // --- 1. Screen Space Sampling (The "Window" Effect) ---
            vec2 screenUV = gl_FragCoord.xy / resolution;

            // Subtle distortion based on normal (refraction)
            vec2 distort = normalize(vNormal).xy * 0.02;
            vec4 texColor = texture(uTexture, screenUV + distort);

            // --- 2. Procedural Gold/White Metal Frame ---
            // Create a border mask using local UVs
            float borderWidth = 0.05;
            float maskX = step(borderWidth, vUv.x) * step(vUv.x, 1.0 - borderWidth);
            float maskY = step(borderWidth, vUv.y) * step(vUv.y, 1.0 - borderWidth);
            float isCenter = maskX * maskY;

            // Lighting for the metal
            vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));
            vec3 norm = normalize(vNormal);

            // Diffuse
            float diff = max(dot(norm, lightDir), 0.0);

            // Specular (Shiny!)
            vec3 viewDir = normalize(vec3(0.0, 0.0, 5.0) - vWorldPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

            // Metal Colors
            vec3 goldColor = vec3(1.0, 0.85, 0.4);
            vec3 whiteMetal = vec3(0.9, 0.95, 1.0);

            // Combine Frame: Mix gold base with white specular highlights
            vec3 frameColor = goldColor * (diff * 0.5 + 0.5) + (whiteMetal * spec);

            // --- 3. Final Mix ---
            // If center, show image (brightened). If edge, show metal frame.
            vec3 finalColor = mix(frameColor, texColor.rgb * 1.5, isCenter);

            fragColor = vec4(finalColor, 1.0);
        }`;

        this.program = this.createProgram(vs, fs);

        // 3. Geometry (Cube)
        this.createCubeGeometry();

        // 4. Load Image
        await this.loadTexture(this.imagePath);

        // 5. Create Objects (Falling Cubes)
        this.resetCubes();

        // 6. Start Loop
        this.resizeObserver.observe(this.container);
        this.resize();
        requestAnimationFrame(() => this.animate());
    }

    resetCubes() {
        this.cubes = [];
        for(let i=0; i<15; i++) {
            this.cubes.push({
                x: (Math.random() - 0.5) * 8,
                y: Math.random() * 20 - 5,
                z: (Math.random() - 0.5) * 5,
                rot: { x: Math.random(), y: Math.random(), z: Math.random() },
                rotSpeed: { x: Math.random()*2, y: Math.random()*2 }
            });
        }
    }

    createCubeGeometry() {
        // Standard Cube with Normals
        // 24 vertices (4 per face * 6 faces) to allow hard edges for normals
        const vertices = [];
        const normals = [];
        const indices = [];

        // Helper to add face
        let idxCounter = 0;
        const addFace = (n, u, v, w, depth) => {
            // u, v are axes for the face plane, w is depth axis
            // corners: -1,-1; 1,-1; 1,1; -1,1
            const corners = [ [-1,-1], [1,-1], [1,1], [-1,1] ];
            corners.forEach(c => {
                let pos = [0,0,0];
                pos[u] = c[0]; pos[v] = c[1]; pos[w] = depth;
                vertices.push(...pos);
                normals.push((w===0?depth:0), (w===1?depth:0), (w===2?depth:0));
            });
            indices.push(idxCounter, idxCounter+1, idxCounter+2, idxCounter, idxCounter+2, idxCounter+3);
            idxCounter += 4;
        };

        addFace(0, 0, 1, 2, 1);  // Front
        addFace(1, 0, 1, 2, -1); // Back
        addFace(2, 2, 0, 1, 1);  // Top
        addFace(3, 2, 0, 1, -1); // Bottom
        addFace(4, 2, 1, 0, 1);  // Right
        addFace(5, 2, 1, 0, -1); // Left

        this.vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.vao);

        const vBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);
        const pLoc = this.gl.getAttribLocation(this.program, 'position');
        this.gl.enableVertexAttribArray(pLoc);
        this.gl.vertexAttribPointer(pLoc, 3, this.gl.FLOAT, false, 0, 0);

        const nBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, nBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(normals), this.gl.STATIC_DRAW);
        const nLoc = this.gl.getAttribLocation(this.program, 'normal');
        this.gl.enableVertexAttribArray(nLoc);
        this.gl.vertexAttribPointer(nLoc, 3, this.gl.FLOAT, false, 0, 0);

        const iBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuf);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        this.indexCount = indices.length;
    }

    loadTexture(url) {
        return new Promise(resolve => {
            this.texture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, 1, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, new Uint8Array([0,0,255,255]));

            const img = new Image();
            img.onload = () => {
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, img);
                this.gl.generateMipmap(this.gl.TEXTURE_2D);
                // Important for screen space wrap
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
                resolve();
            };
            img.src = url;
        });
    }

    resize() {
        if(!this.canvas) return;
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    createProgram(vsSrc, fsSrc) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSrc); this.gl.compileShader(vs);
        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSrc); this.gl.compileShader(fs);
        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs); this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    animate() {
        const time = (Date.now() - this.startTime) / 1000;

        this.gl.clearColor(0.05, 0.05, 0.05, 1.0);
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

        this.gl.useProgram(this.program);
        this.gl.uniform2f(this.gl.getUniformLocation(this.program, 'resolution'), this.canvas.width, this.canvas.height);
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, 'time'), time);

        // Perspective Matrix
        const aspect = this.canvas.width / this.canvas.height;
        const fov = 60 * Math.PI / 180;
        const f = 1.0 / Math.tan(fov / 2);
        const proj = [
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, -1, -1,
            0, 0, -0.2, 0
        ];

        // View Matrix (Camera back 15 units)
        const view = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, -15, 1
        ];

        this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.program, 'projection'), false, new Float32Array(proj));
        this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.program, 'view'), false, new Float32Array(view));

        this.gl.bindVertexArray(this.vao);

        // Update and Draw Cubes
        this.cubes.forEach((cube, i) => {
            // Physics
            cube.y -= 0.05;
            if (cube.y < -8) {
                cube.y = 8;
                cube.x = (Math.random() - 0.5) * 10;
            }

            // Rotation
            cube.rot.x += cube.rotSpeed.x * 0.01;
            cube.rot.y += cube.rotSpeed.y * 0.01;

            // --- Simple Manual Matrix Math for Model ---
            // 1. Rotation Matrices
            const cx = Math.cos(cube.rot.x), sx = Math.sin(cube.rot.x);
            const cy = Math.cos(cube.rot.y), sy = Math.sin(cube.rot.y);

            // Combined Rotation (Y then X approx)
            // r00 r01 r02
            // r10 r11 r12
            // r20 r21 r22
            const r00 = cy, r01 = 0, r02 = sy;
            const r10 = sx*sy, r11 = cx, r12 = -sx*cy;
            const r20 = -cx*sy, r21 = sx, r22 = cx*cy;

            const model = [
                r00, r10, r20, 0,
                r01, r11, r21, 0,
                r02, r12, r22, 0,
                cube.x, cube.y, cube.z, 1 // Translation
            ];

            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.program, 'model'), false, new Float32Array(model));
            this.gl.drawElements(this.gl.TRIANGLES, this.indexCount, this.gl.UNSIGNED_SHORT, 0);
        });

        requestAnimationFrame(() => this.animate());
    }
}
