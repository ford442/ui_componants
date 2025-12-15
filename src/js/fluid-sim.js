/**
 * Fluid Simulation Experiment
 * Hybrid Rendering:
 * - WebGL2: Renders the container tank and lighting environment.
 * - WebGPU: Simulates particle-based fluid dynamics and renders the fluid.
 */

class FluidSimulationExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            particleCount: options.particleCount || 4096,
            ...options
        };

        this.isActive = false;
        this.animationId = null;

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
        this.particleBuffer1 = null;
        this.particleBuffer2 = null;
        this.uniformBuffer = null;
        this.computeBindGroup1 = null;
        this.computeBindGroup2 = null;

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#111';

        // 1. WebGL2 (Tank)
        this.initWebGL2();

        // 2. WebGPU (Fluid)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("FluidSim: WebGPU error", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    // ========================================================================
    // WebGL2 (Tank Container)
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

        // Simple Cube Wireframe
        const vs = `#version 300 es
            in vec3 a_pos;
            uniform mat4 u_matrix;
            void main() {
                gl_Position = u_matrix * vec4(a_pos, 1.0);
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            out vec4 outColor;
            void main() {
                outColor = vec4(0.5, 0.6, 0.7, 0.3);
            }
        `;

        this.glProgram = this.createProgram(vs, fs);
        this.uMatrixLoc = this.gl.getUniformLocation(this.glProgram, 'u_matrix');

        // Cube Lines
        const s = 1.0; // box size (extends -1 to 1)
        const boxVerts = new Float32Array([
            -s,-s,-s,  s,-s,-s,
             s,-s,-s,  s, s,-s,
             s, s,-s, -s, s,-s,
            -s, s,-s, -s,-s,-s,

            -s,-s, s,  s,-s, s,
             s,-s, s,  s, s, s,
             s, s, s, -s, s, s,
            -s, s, s, -s,-s, s,

            -s,-s,-s, -s,-s, s,
             s,-s,-s,  s,-s, s,
             s, s,-s,  s, s, s,
            -s, s,-s, -s, s, s
        ]);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const buf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, boxVerts, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_pos');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

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
    // WebGPU (Fluid Simulation)
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

        // --- Data ---
        // Particle: [pos.x, pos.y, pos.z, padding, vel.x, vel.y, vel.z, padding]
        const pSize = 8 * 4;
        const initialData = new Float32Array(this.options.particleCount * 8);
        for(let i=0; i<this.options.particleCount; i++) {
            // Spawn in a column
            initialData[i*8+0] = (Math.random() - 0.5) * 0.5;
            initialData[i*8+1] = Math.random() * 1.5 - 0.5;
            initialData[i*8+2] = (Math.random() - 0.5) * 0.5;
            // Velocity 0
        }

        this.particleBuffer1 = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.particleBuffer1, 0, initialData);

        // Ping-pong buffer (needed if we were doing complex interactions, but for simple physics,
        // we can update in place or use one buffer if we don't need sync between particles perfectly)
        // For simplicity, we'll just update in place for this basic demo.

        this.uniformBuffer = this.device.createBuffer({
            size: 80, // MVP (64) + Time/DT (16)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // --- Compute Shader (Simple Gravity + Bounce) ---
        const computeShader = `
            struct Particle {
                pos : vec3f,
                pad1 : f32,
                vel : vec3f,
                pad2 : f32,
            }

            struct Uniforms {
                mvp : mat4x4f,
                dt : f32,
                time : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Gravity
                p.vel.y -= 9.8 * uniforms.dt;

                // Update Pos
                p.pos += p.vel * uniforms.dt;

                // Floor Collision
                if (p.pos.y < -1.0) {
                    p.pos.y = -1.0;
                    p.vel.y *= -0.6; // bounce
                }

                // Walls
                if (abs(p.pos.x) > 1.0) {
                    p.pos.x = sign(p.pos.x) * 1.0;
                    p.vel.x *= -0.5;
                }
                if (abs(p.pos.z) > 1.0) {
                    p.pos.z = sign(p.pos.z) * 1.0;
                    p.vel.z *= -0.5;
                }

                particles[i] = p;
            }
        `;

        const computeModule = this.device.createShaderModule({ code: computeShader });
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer1 } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // --- Render Shader ---
        const drawShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct Uniforms {
                mvp : mat4x4f,
                dt : f32,
                time : f32,
            }

            @group(0) @binding(0) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @location(0) pos : vec3f,
                @location(1) vel : vec3f // skipping pads in buffer layout definition
            ) -> VertexOutput {
                var output : VertexOutput;
                output.position = uniforms.mvp * vec4f(pos, 1.0);

                // Color based on velocity
                let speed = length(vel);
                output.color = mix(vec4f(0.0, 0.5, 1.0, 0.8), vec4f(0.5, 0.8, 1.0, 1.0), clamp(speed * 0.2, 0.0, 1.0));

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
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
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' }, // pos
                        { shaderLocation: 1, offset: 16, format: 'float32x3' } // vel (skip pad1 at offset 12)
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
        msg.innerHTML = "⚠️ WebGPU Not Available (Fluid Simulation requires WebGPU)";
        msg.style.cssText = "position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:white; background:rgba(200,0,0,0.8); padding:20px; border-radius:8px; text-align:center;";
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

        // Shared MVP
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

        const r = 3.5;
        const cx = Math.sin(time * 0.5) * r;
        const cz = Math.cos(time * 0.5) * r;
        const cy = 1.5;

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

        // --- Render WebGL2 ---
        if (this.gl) {
            this.gl.clearColor(0.1, 0.1, 0.1, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(this.uMatrixLoc, false, mvp);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.LINES, 0, 24);
        }

        // --- Render WebGPU ---
        if (this.device && this.context) {
            // Update Uniforms
            const uniformData = new Float32Array(20);
            uniformData.set(mvp, 0);
            uniformData[16] = 0.016; // dt
            uniformData[17] = time;

            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.options.particleCount / 64));
            computePass.end();

            // Render
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
            renderPass.setVertexBuffer(0, this.particleBuffer1);
            renderPass.draw(this.options.particleCount);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    // Math Helpers
    normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
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
    }
}

if (typeof window !== 'undefined') {
    window.FluidSimulationExperiment = FluidSimulationExperiment;
}
