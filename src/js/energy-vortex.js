/**
 * Energy Vortex Experiment
 * Combines WebGL2 for a wireframe torus vortex and WebGPU for energy spark particles.
 */

export class EnergyVortexExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.canvasSize = { width: 0, height: 0 };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.torusMajorSegments = 64;
        this.torusMinorSegments = 32;
        this.indexCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 40000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.mouse = { x: 0, y: 0 };

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        console.log("EnergyVortex: Initializing...");

        // 1. Initialize WebGL2 Layer (Background Geometry)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("EnergyVortex: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("EnergyVortex: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("EnergyVortex: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Torus Vortex)
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
            console.warn("EnergyVortex: WebGL2 not supported.");
            return;
        }

        // Generate Torus Geometry
        const positions = [];
        const indices = [];

        const R = 3.0; // Major radius
        const r = 1.0; // Minor radius

        for (let i = 0; i <= this.torusMajorSegments; i++) {
            const phi = (i / this.torusMajorSegments) * Math.PI * 2;
            const cosPhi = Math.cos(phi);
            const sinPhi = Math.sin(phi);

            for (let j = 0; j <= this.torusMinorSegments; j++) {
                const theta = (j / this.torusMinorSegments) * Math.PI * 2;
                const cosTheta = Math.cos(theta);
                const sinTheta = Math.sin(theta);

                // Torus formula
                const x = (R + r * cosTheta) * cosPhi;
                const y = (R + r * cosTheta) * sinPhi;
                const z = r * sinTheta;

                positions.push(x, y, z);
            }
        }

        for (let i = 0; i < this.torusMajorSegments; i++) {
            for (let j = 0; j < this.torusMinorSegments; j++) {
                const base = i * (this.torusMinorSegments + 1) + j;
                const next = base + (this.torusMinorSegments + 1);

                // Wireframe lines
                indices.push(base, base + 1);
                indices.push(base, next);
            }
        }
        this.indexCount = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;

            out float v_depth;

            void main() {
                vec3 pos = a_position;

                // Rotation matrices
                float c = cos(u_time * 0.5);
                float s = sin(u_time * 0.5);
                mat3 rotX = mat3(
                    1.0, 0.0, 0.0,
                    0.0, c, -s,
                    0.0, s, c
                );

                float c2 = cos(u_time * 0.3);
                float s2 = sin(u_time * 0.3);
                mat3 rotY = mat3(
                    c2, 0.0, s2,
                    0.0, 1.0, 0.0,
                    -s2, 0.0, c2
                );

                pos = rotX * rotY * pos;

                // Mouse interaction - twist
                float dist = length(pos.xy);
                float twist = u_mouse.x * 2.0 * exp(-dist * 0.5);
                float ct = cos(twist);
                float st = sin(twist);
                mat2 twistMat = mat2(ct, -st, st, ct);
                pos.xy = twistMat * pos.xy;

                // Camera
                pos.z -= 8.0;

                // Perspective projection
                float aspect = u_resolution.x / u_resolution.y;
                float zToDist = 1.0 / -pos.z;

                gl_Position = vec4(pos.x * zToDist / aspect, pos.y * zToDist, 0.0, 1.0);

                v_depth = pos.z;
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            uniform float u_time;
            out vec4 outColor;

            void main() {
                // Color pulse
                vec3 color = vec3(0.1, 0.5, 1.0); // Blue
                color += vec3(0.5, 0.8, 1.0) * sin(u_time * 3.0 + v_depth) * 0.5;

                // Distance fade
                float alpha = smoothstep(-15.0, -5.0, v_depth);

                outColor = vec4(color, alpha * 0.4);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('EnergyVortex WebGL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('EnergyVortex WebGL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Energy Sparks)
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

        let adapter;
        try {
            adapter = await navigator.gpu.requestAdapter();
        } catch (e) {
            console.warn("WebGPU Adapter request failed:", e);
            this.gpuCanvas.remove();
            return false;
        }

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

        const computeShaderCode = `
            struct Particle {
                pos : vec3f,
                vel : vec3f,
                life : f32,
                pad : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

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

                if (p.life <= 0.0) {
                    // Respawn near center
                    let r = rand(vec2f(params.time, f32(index))) * 0.5;
                    let theta = rand(vec2f(f32(index), params.time)) * 6.28;
                    p.pos = vec3f(cos(theta) * r, sin(theta) * r, 0.0);

                    // Outward velocity
                    p.vel = normalize(p.pos) * (2.0 + rand(vec2f(p.pos.x, p.pos.y)) * 3.0);
                    p.vel.z = (rand(vec2f(p.pos.y, p.pos.x)) - 0.5) * 2.0;

                    p.life = 1.0;
                } else {
                    // Update
                    p.pos = p.pos + p.vel * params.dt;

                    // Vortex force - rotate around Z axis
                    let angle = params.dt * 2.0;
                    let c = cos(angle);
                    let s = sin(angle);
                    let x = p.pos.x * c - p.pos.y * s;
                    let y = p.pos.x * s + p.pos.y * c;
                    p.pos.x = x;
                    p.pos.y = y;

                    // Mouse attraction
                    let mousePos = vec3f(params.mouseX * 5.0, params.mouseY * 5.0, 0.0);
                    let diff = mousePos - p.pos;
                    let dist = length(diff);
                    if (dist < 4.0) {
                        p.vel = p.vel + normalize(diff) * 5.0 * params.dt;
                    }

                    p.life = p.life - params.dt * 0.3;
                }

                particles[index] = p;
            }
        `;

        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            @vertex
            fn vs_main(
                @location(0) pos : vec3f,
                @location(1) vel : vec3f,
                @location(2) life : f32
            ) -> VertexOutput {
                var output : VertexOutput;

                // Simple camera setup matching WebGL
                let camZ = -8.0;
                let viewPos = pos;
                let zDist = viewPos.z - camZ;

                // Perspective
                let projX = viewPos.x / zDist / params.aspect * 2.0; // Adjusted scale
                let projY = viewPos.y / zDist * 2.0;

                output.position = vec4f(projX, projY, 0.0, 1.0);

                // Color
                let alpha = smoothstep(0.0, 0.1, life) * smoothstep(1.0, 0.5, life);
                let energy = length(vel);
                output.color = vec4f(1.0, 0.8, 0.5, alpha); // Gold/Orange sparks

                // Mix with blue based on velocity
                output.color = mix(output.color, vec4f(0.2, 0.8, 1.0, alpha), clamp(energy * 0.2, 0.0, 1.0));

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                if (color.a < 0.01) { discard; }
                return color;
            }
        `;

        const particleUnitSize = 32;
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 8 + 0] = 0; // x
            initialParticleData[i * 8 + 1] = 0; // y
            initialParticleData[i * 8 + 2] = 0; // z
            initialParticleData[i * 8 + 6] = 0; // life
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
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
                        { shaderLocation: 0, offset: 0, format: 'float32x3' }, // pos
                        { shaderLocation: 1, offset: 12, format: 'float32x3' }, // vel
                        { shaderLocation: 2, offset: 24, format: 'float32' },   // life
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{ format: presentationFormat, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                } }],
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;

        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(100, 20, 20, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: monospace;
            pointer-events: none;
        `;
        msg.innerHTML = "WebGPU Not Available (WebGL2 Only)";
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

        const displayWidth = Math.floor(width * dpr);
        const displayHeight = Math.floor(height * dpr);

        this.resizeGL(displayWidth, displayHeight);
        this.resizeGPU(displayWidth, displayHeight);
    }

    resizeGL(width, height) {
        if (!this.glCanvas) return;
        if (width <= 0 || height <= 0) return;

        if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
            this.glCanvas.width = width;
            this.glCanvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    resizeGPU(width, height) {
        if (!this.gpuCanvas) return;
        if (width <= 0 || height <= 0) return;

        if (this.gpuCanvas.width !== width || this.gpuCanvas.height !== height) {
            this.gpuCanvas.width = width;
            this.gpuCanvas.height = height;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            this.gl.uniform1f(timeLoc, time);

            const resLoc = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            const mouseLoc = this.gl.getUniformLocation(this.glProgram, 'u_mouse');
            this.gl.uniform2f(mouseLoc, this.mouse.x, this.mouse.y);

            this.gl.clearColor(0.0, 0.0, 0.0, 0.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.indexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0 && this.gpuCanvas.height > 0) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, // dt
                time,  // time
                this.mouse.x, // mouseX
                this.mouse.y, // mouseY
                aspect, // aspect
                0, 0, 0 // padding
            ]);

            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render
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
            renderPass.setBindGroup(0, this.computeBindGroup);
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
        this.container.removeEventListener('mousemove', this.handleMouseMove);

        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }

        if (this.device) {
            this.device.destroy();
        }

        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.EnergyVortexExperiment = EnergyVortexExperiment;
}
