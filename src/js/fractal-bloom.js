/**
 * Fractal Bloom Experiment
 * Demonstrates WebGL2 and WebGPU working in tandem.
 * - WebGL2: Renders a recursive fractal tree (L-System style).
 * - WebGPU: Renders a compute-driven particle blossom swarm.
 */

class FractalBloomExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glVertexCount = 0;
        this.treeTips = []; // Store tip positions for WebGPU

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.tipsBuffer = null;
        this.numParticles = options.numParticles || 20000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#051015'; // Dark teal/black background

        console.log("FractalBloom: Initializing...");

        // 1. Initialize WebGL2 Layer (Tree)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Blossoms)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("FractalBloom: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("FractalBloom: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("FractalBloom: WebGPU initialized successfully.");
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Fractal Tree)
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
        if (!this.gl) {
            console.warn("FractalBloom: WebGL2 not supported.");
            return;
        }

        // Generate Fractal Tree Geometry
        const positions = [];
        const colors = [];
        this.treeTips = [];

        // Recursive generation
        const generateBranch = (x, y, z, length, angleX, angleZ, depth) => {
            if (depth === 0) {
                this.treeTips.push(x, y, z, 0); // 4 floats for alignment
                return;
            }

            const endX = x + length * Math.sin(angleZ) * Math.cos(angleX);
            const endY = y + length * Math.cos(angleZ) * Math.cos(angleX);
            const endZ = z + length * Math.sin(angleX);

            // Line segment
            positions.push(x, y, z);
            positions.push(endX, endY, endZ);

            // Gradient Color (Brown to Green)
            const mixFactor = 1.0 - (depth / 10.0); // 0 (base) to 1 (tip) assuming max depth 10

            // Base Color (0.4, 0.3, 0.2)
            // Tip Color (0.2, 0.9, 0.4)
            colors.push(0.4 - mixFactor * 0.2, 0.3 + mixFactor * 0.6, 0.2 + mixFactor * 0.2);
            colors.push(0.4 - mixFactor * 0.2, 0.3 + mixFactor * 0.6, 0.2 + mixFactor * 0.2);

            const newLength = length * 0.75;
            const spread = 0.5;

            generateBranch(endX, endY, endZ, newLength, angleX + spread, angleZ + spread, depth - 1);
            generateBranch(endX, endY, endZ, newLength, angleX - spread, angleZ - spread, depth - 1);
        };

        // Start tree at bottom center
        generateBranch(0, -0.8, 0, 0.5, 0, 0, 9);

        this.glVertexCount = positions.length / 3;

        // Buffers
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const colorBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, colorBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(colors), this.gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
            in vec3 a_position;
            in vec3 a_color;

            uniform float u_time;
            uniform float u_aspect;

            out vec3 v_color;

            void main() {
                vec3 pos = a_position;

                // Wind Sway
                float wind = sin(u_time * 1.5 + pos.y * 2.0) * 0.05 * (pos.y + 0.8);
                pos.x += wind;

                // Simple 3D projection
                // Scale y slightly to fit
                pos.y *= 0.9;

                // Aspect ratio correction
                pos.x /= u_aspect;

                v_color = a_color;
                gl_Position = vec4(pos, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec3 v_color;
            out vec4 outColor;
            void main() {
                outColor = vec4(v_color, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        // Attributes
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, colorBuffer);
        const colorLoc = this.gl.getAttribLocation(this.glProgram, 'a_color');
        this.gl.enableVertexAttribArray(colorLoc);
        this.gl.vertexAttribPointer(colorLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
        this.resizeGL();
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('WebGL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('WebGL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Blossoms)
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
            background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        if (!navigator.gpu) return false;
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // Compute Shader
        const computeShaderCode = `
            struct Particle {
                pos : vec2f,
                vel : vec2f,
                life : f32,
                pad1 : f32,
                pad2 : vec2f,
            }

            struct SimParams {
                dt : f32,
                time : f32,
                tipCount : f32,
                aspect : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : SimParams;
            @group(0) @binding(2) var<storage, read> tips : array<vec4f>; // Using vec4 for alignment (x,y,z,pad)

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Respawn
                if (p.life <= 0.0) {
                    let r = rand(vec2f(params.time, f32(index)));
                    // Pick random tip
                    let tipIndex = u32(r * params.tipCount) % u32(params.tipCount);
                    let tipPos = tips[tipIndex];

                    p.pos = tipPos.xy; // We only use XY for now, assume 2D projection matches

                    // Add wind sway offset to spawn match the visual tree
                    let wind = sin(params.time * 1.5 + p.pos.y * 2.0) * 0.05 * (p.pos.y + 0.8);
                    p.pos.x += wind;

                    // Simple projection correction from GL
                    p.pos.y *= 0.9;

                    // Random velocity
                    p.vel = vec2f((rand(vec2f(r, r)) - 0.5) * 0.2, (rand(vec2f(r + 1.0, r)) * 0.1) + 0.05);
                    p.life = 1.0 + rand(vec2f(f32(index), params.time));
                }

                // Update
                // Wind force
                p.vel.x += sin(params.time + p.pos.y * 4.0) * 0.001;
                // Gravity/Lift
                p.vel.y += 0.0002;

                p.pos += p.vel * params.dt;
                p.life -= params.dt * 0.5;

                particles[index] = p;
            }
        `;

        // Render Shader
        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) size : f32,
            }

            struct Uniforms {
                dt : f32,
                time : f32,
                tipCount : f32,
                aspect : f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f,
                @location(2) life : f32
            ) -> VertexOutput {
                var output : VertexOutput;

                // Apply aspect ratio correction in vertex shader
                var pos = particlePos;
                pos.x /= uniforms.aspect;

                output.position = vec4f(pos, 0.0, 1.0);

                let alpha = smoothstep(0.0, 0.2, life) * smoothstep(1.0, 0.8, life);

                // Pink/Purple blossoms
                output.color = vec4f(1.0, 0.4, 0.8, alpha);
                output.size = 3.0;

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const particleUnitSize = 32; // 8 floats * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;

        // Initialize particles
        const particleData = new Float32Array(this.numParticles * 8);
        // Fill with dead particles so they respawn immediately
        for(let i=0; i<this.numParticles; i++) {
             particleData[i*8 + 4] = -1.0; // life
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);

        // Tips Buffer
        const tipsData = new Float32Array(this.treeTips);
        this.tipsBuffer = this.device.createBuffer({
            size: tipsData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.tipsBuffer, 0, tipsData);

        // Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Layouts & Pipelines
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
                { binding: 2, resource: { buffer: this.tipsBuffer } },
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' }, // pos
                        { shaderLocation: 1, offset: 8, format: 'float32x2' }, // vel
                        { shaderLocation: 2, offset: 16, format: 'float32' },  // life
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    }
                }],
            },
            primitive: { topology: 'point-list' },
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px;
            background: rgba(100, 20, 20, 0.9); color: white; padding: 8px 16px;
            border-radius: 8px; font-size: 13px; font-family: monospace; pointer-events: none;
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const displayWidth = Math.floor(width * dpr);
        const displayHeight = Math.floor(height * dpr);

        this.resizeGL(displayWidth, displayHeight);
        this.resizeGPU(displayWidth, displayHeight);
    }

    resizeGL(width, height) {
        if (!this.glCanvas) return;
        if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
            this.glCanvas.width = width;
            this.glCanvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    resizeGPU(width, height) {
        if (!this.gpuCanvas) return;
        if (this.gpuCanvas.width !== width || this.gpuCanvas.height !== height) {
            this.gpuCanvas.width = width;
            this.gpuCanvas.height = height;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const aspect = this.glCanvas ? this.glCanvas.width / this.glCanvas.height : 1.0;

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            this.gl.uniform1f(timeLoc, time);

            const aspectLoc = this.gl.getUniformLocation(this.glProgram, 'u_aspect');
            this.gl.uniform1f(aspectLoc, aspect);

            this.gl.clearColor(0.02, 0.05, 0.08, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.lineWidth(2.0);
            this.gl.drawArrays(this.gl.LINES, 0, this.glVertexCount);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            const tipCount = this.treeTips.length / 4;
            const params = new Float32Array([0.016, time, tipCount, aspect]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            const textureView = this.context.getCurrentTexture().createView();
            const renderPassDescriptor = {
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            };

            const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup); // Need bind group for aspect uniform
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

export { FractalBloomExperiment };
if (typeof window !== 'undefined') {
    window.FractalBloomExperiment = FractalBloomExperiment;
}
