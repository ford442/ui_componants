/**
 * Cymatic Frequency Plate Experiment
 * WebGL2 Metallic Plate + WebGPU Particle Physics (Chladni Patterns)
 */

export class CymaticPlate {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0.5, y: 0.5 }; // Normalized -1 to 1 effectively
        this.canvasSize = { width: 0, height: 0 };

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
        this.numParticles = options.numParticles || 60000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        console.log("CymaticPlate: Initializing...");

        // 1. Initialize WebGL2 Layer (Background Plate)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Sand Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CymaticPlate: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("CymaticPlate: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("CymaticPlate: WebGPU initialized successfully.");
        }

        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        // Map to [-1, 1] range, correcting Y for WebGPU coords
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Metallic Plate Background)
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

        // Quad
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_resolution;
            out vec4 outColor;

            // Brushed metal noise
            float rand(vec2 co) {
                return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
            }

            void main() {
                vec2 uv = v_uv;
                if (u_resolution.y > 0.0) {
                    uv.x *= u_resolution.x / u_resolution.y;
                }

                // Radial brushed metal effect
                float angle = atan(uv.y, uv.x);
                float dist = length(uv);

                float noise = rand(vec2(angle * 20.0, 0.0)) * 0.1;
                noise += rand(vec2(angle * 100.0, 0.0)) * 0.05;

                // Vignette / Lighting
                float light = 1.0 - dist * 0.5;

                vec3 metalColor = vec3(0.15, 0.16, 0.18);
                vec3 highlight = vec3(0.3, 0.32, 0.35);

                vec3 finalColor = mix(metalColor, highlight, noise + light * 0.2);

                // Subtle concentric rings (plate grooves)
                float rings = sin(dist * 100.0) * 0.02;
                finalColor += rings;

                outColor = vec4(finalColor, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
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
    // WebGPU IMPLEMENTATION (Sand Particles)
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

        // WGSL - Compute Physics
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
            }
            @group(0) @binding(1) var<uniform> params : Params;

            // Random hash
            fn hash(p: vec2f) -> f32 {
                return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            // Calculate vibrational amplitude at position p
            fn getAmplitude(p: vec2f) -> f32 {
                // Adjust for aspect to keep waves circular
                let p_adj = vec2f(p.x * params.aspect, p.y);
                let m_adj = vec2f(params.mouseX * params.aspect, params.mouseY);

                let freq = 15.0 + sin(params.time * 0.2) * 5.0; // Varying frequency

                // Source 1: Center
                let d1 = length(p_adj);
                let w1 = sin(d1 * freq - params.time * 2.0);

                // Source 2: Mouse
                let d2 = length(p_adj - m_adj);
                let w2 = sin(d2 * (freq * 1.2) - params.time * 2.0);

                // Source 3: Lissajous-like interference
                let w3 = sin(p_adj.x * 10.0 + params.time) * sin(p_adj.y * 10.0);

                return w1 + w2 + w3;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Physics: Move towards nodal lines (where Amplitude is 0)
                // Potential energy = |Amplitude|
                // Force = -Gradient(Potential)

                let eps = 0.01;
                let v = abs(getAmplitude(p.pos));
                let vx = abs(getAmplitude(p.pos + vec2f(eps, 0.0)));
                let vy = abs(getAmplitude(p.pos + vec2f(0.0, eps)));

                let grad = vec2f(vx - v, vy - v) / eps;

                // Update velocity
                // Force pushes towards lower potential (0 amplitude)
                p.vel = p.vel - grad * params.dt * 25.0;

                // Drag / Friction (simulating sand on a plate)
                p.vel = p.vel * 0.90;

                // Brownian motion (jitter) to prevent stacking perfectly
                let rnd = hash(p.pos + vec2f(params.time, f32(index)));
                let jitter = (vec2f(rnd, fract(rnd * 10.0)) - 0.5) * 0.05;

                // Add jitter primarily when potential is high (high vibration areas shake particles)
                p.vel = p.vel + jitter * (v + 0.1) * params.dt * 50.0;

                // Update Position
                p.pos = p.pos + p.vel * params.dt;

                // Boundary containment
                if (p.pos.x < -1.1) { p.pos.x = -1.1; p.vel.x = -p.vel.x; }
                if (p.pos.x > 1.1) { p.pos.x = 1.1; p.vel.x = -p.vel.x; }
                if (p.pos.y < -1.1) { p.pos.y = -1.1; p.vel.y = -p.vel.y; }
                if (p.pos.y > 1.1) { p.pos.y = 1.1; p.vel.y = -p.vel.y; }

                particles[index] = p;
            }
        `;

        // WGSL - Render Points
        const renderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) particlePos : vec2f) -> VertexOutput {
                var output : VertexOutput;
                output.position = vec4f(particlePos, 0.0, 1.0);

                // Sand color
                output.color = vec4f(1.0, 0.9, 0.6, 1.0);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const particleUnitSize = 16;
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialData = new Float32Array(this.numParticles * 4);
        for(let i=0; i<this.numParticles; i++){
            initialData[i*4+0] = (Math.random()*2-1);
            initialData[i*4+1] = (Math.random()*2-1);
            initialData[i*4+2] = 0;
            initialData[i*4+3] = 0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 5 floats = 20 bytes -> padded to 32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Pipeline Config
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: renderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }]
                }],
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
                    }
                }],
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.innerHTML = "WebGPU Not Available - Physics Disabled";
        msg.style.cssText = "position:absolute; bottom:20px; right:20px; color:white; background:rgba(255,0,0,0.5); padding:10px;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width === 0 || height === 0) return;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

        const dw = Math.floor(width * dpr);
        const dh = Math.floor(height * dpr);

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

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.clearColor(0,0,0,1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGPU
        if (this.device && this.context && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, // dt
                time,
                this.mouse.x,
                this.mouse.y,
                aspect,
                0,0,0 // padding
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

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
        window.removeEventListener('mousemove', this.handleMouseMove);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();

        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.CymaticPlate = CymaticPlate;
}
