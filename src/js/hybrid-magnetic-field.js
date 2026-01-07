/**
 * Hybrid Magnetic Field Experiment
 * Combines WebGL2 for field visualization and WebGPU for particle simulation.
 */

class HybridMagneticField {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
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
        this.numParticles = options.numParticles || 50000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#0a0a0f';

        console.log("HybridMagneticField: Initializing...");

        // 1. Initialize WebGL2 Layer (Background)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("HybridMagneticField: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("HybridMagneticField: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("HybridMagneticField: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        // Normalize mouse coordinates to [-1, 1] (UV space centered)
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1); // Flip Y to match WebGL/WebGPU coords
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Field Viz)
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
            console.warn("HybridMagneticField: WebGL2 not supported.");
            return;
        }

        // Setup simple quad
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        // Fragment Shader - Field Visualization
        const fsSource = `#version 300 es
            precision highp float;

            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_mouse;
            uniform vec2 u_resolution;

            out vec4 outColor;

            #define PI 3.14159265359

            // Simplex noise function (simplified)
            vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
            float snoise(vec2 v) {
                const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
                vec2 i  = floor(v + dot(v, C.yy) );
                vec2 x0 = v -   i + dot(i, C.xx);
                vec2 i1;
                i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                vec4 x12 = x0.xyxy + C.xxzz;
                x12.xy -= i1;
                i = mod(i, 289.0);
                vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                + i.x + vec3(0.0, i1.x, 1.0 ));
                vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
                m = m*m ;
                m = m*m ;
                vec3 x = 2.0 * fract(p * C.www) - 1.0;
                vec3 h = abs(x) - 0.5;
                vec3 ox = floor(x + 0.5);
                vec3 a0 = x - ox;
                m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
                vec3 g;
                g.x  = a0.x  * x0.x  + h.x  * x0.y;
                g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                return 130.0 * dot(m, g);
            }

            vec2 getField(vec2 p) {
                // Mouse influence
                vec2 mDir = p - u_mouse;
                float dist = length(mDir);
                vec2 force = normalize(mDir) / (dist + 0.1) * 0.5;

                // Rotational field around center
                vec2 center = vec2(0.0);
                vec2 cDir = p - center;
                float cDist = length(cDir);
                vec2 rotation = vec2(-cDir.y, cDir.x) * (1.0 / (cDist + 0.5));

                // Noise field
                float n = snoise(p * 2.0 + u_time * 0.1);
                vec2 noiseField = vec2(cos(n * PI), sin(n * PI));

                return rotation + force * -1.0 + noiseField * 0.3;
            }

            void main() {
                vec2 uv = v_uv;
                // Correct aspect ratio
                if (u_resolution.y > 0.0) {
                    uv.x *= u_resolution.x / u_resolution.y;
                }

                vec2 fieldVec = getField(uv);
                float fieldLen = length(fieldVec);
                float fieldAngle = atan(fieldVec.y, fieldVec.x);

                // Visualize field lines
                float lines = sin(fieldAngle * 10.0 + u_time * 2.0 + fieldLen * 5.0);
                lines = smoothstep(0.8, 1.0, lines);

                // Colorize
                vec3 baseColor = vec3(0.05, 0.05, 0.1);
                vec3 fieldColor = vec3(0.2, 0.4, 0.8);
                vec3 activeColor = vec3(0.0, 0.8, 0.5); // WebGPU particle color match

                vec3 color = baseColor;
                color += fieldColor * fieldLen * 0.5;
                color += activeColor * lines * 0.3;

                // Vignette
                float vig = 1.0 - smoothstep(0.5, 1.5, length(v_uv));
                color *= vig;

                outColor = vec4(color, 1.0);
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

        // Remove internal resize call
        // this.resizeGL();
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
    // WebGPU IMPLEMENTATION (Particle Simulation)
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

        // WGSL Helper functions need to match GLSL logic as close as possible
        const commonWGSL = `
            fn permute(x: vec3f) -> vec3f { return ((x * 34.0) + 1.0) * x % 289.0; }
            fn snoise(v: vec2f) -> f32 {
                let C = vec4f(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
                var i = floor(v + dot(v, C.yy));
                let x0 = v - i + dot(i, C.xx);
                var i1 = vec2f(0.0);
                if (x0.x > x0.y) { i1 = vec2f(1.0, 0.0); } else { i1 = vec2f(0.0, 1.0); }
                var x12 = x0.xyxy + C.xxzz;
                x12.x = x12.x - i1.x;
                x12.y = x12.y - i1.y;
                i = i % 289.0;
                let p = permute(permute(i.y + vec3f(0.0, i1.y, 1.0)) + i.x + vec3f(0.0, i1.x, 1.0));
                var m = max(0.5 - vec3f(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3f(0.0));
                m = m * m;
                m = m * m;
                let x = 2.0 * fract(p * C.www) - 1.0;
                let h = abs(x) - 0.5;
                let ox = floor(x + 0.5);
                let a0 = x - ox;
                m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));
                var g = vec3f(0.0);
                g.x = a0.x * x0.x + h.x * x0.y;
                g.y = a0.y * x12.x + h.y * x12.y;
                g.z = a0.z * x12.z + h.z * x12.w;
                return 130.0 * dot(m, g);
            }
        `;

        // COMPUTE SHADER - Cleanly defined
        const computeShaderCode = `
            ${commonWGSL}

            struct Particle {
                pos : vec2f,
                vel : vec2f,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                pad : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            fn getField(p: vec2f) -> vec2f {
                // Adjust P for aspect ratio in field calculation to match
                let p_aspect = vec2f(p.x * params.aspect, p.y);
                let m_aspect = vec2f(params.mouseX * params.aspect, params.mouseY);

                // Mouse influence
                let mDir = p_aspect - m_aspect;
                let dist = length(mDir);
                let force = normalize(mDir) / (dist + 0.1) * 0.5;

                // Rotational field
                let center = vec2f(0.0);
                let cDir = p_aspect - center;
                let cDist = length(cDir);
                let rotation = vec2f(-cDir.y, cDir.x) * (1.0 / (cDist + 0.5));

                // Noise
                let n = snoise(p_aspect * 2.0 + params.time * 0.1);
                let noiseField = vec2f(cos(n * 3.14159), sin(n * 3.14159));

                return rotation + force * -1.0 + noiseField * 0.3;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                let fieldForce = getField(p.pos);

                // Accelerate based on field
                p.vel = p.vel + fieldForce * params.dt * 2.0;

                // Damping
                p.vel = p.vel * 0.96;

                // Update pos
                p.pos = p.pos + p.vel * params.dt;

                // Bounds wrap
                if (p.pos.x < -1.2) { p.pos.x = 1.2; }
                if (p.pos.x > 1.2) { p.pos.x = -1.2; }
                if (p.pos.y < -1.2) { p.pos.y = 1.2; }
                if (p.pos.y > 1.2) { p.pos.y = -1.2; }

                particles[index] = p;
            }
        `;

        // RENDER SHADER - Cleanly defined
        const drawShaderCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) life : f32,
            }

            @vertex
            fn vs_main(
                @location(0) particlePos : vec2f,
                @location(1) particleVel : vec2f
            ) -> VertexOutput {
                var output : VertexOutput;
                output.position = vec4f(particlePos, 0.0, 1.0);

                let speed = length(particleVel);
                // Color ramp: Cyan -> White -> Pink based on speed
                let c1 = vec3f(0.0, 1.0, 0.8); // Cyan
                let c2 = vec3f(1.0, 1.0, 1.0); // White
                let c3 = vec3f(1.0, 0.2, 0.5); // Pink

                var col = mix(c1, c2, smoothstep(0.0, 0.5, speed));
                col = mix(col, c3, smoothstep(0.5, 1.5, speed));

                output.color = vec4f(col, 1.0);
                output.life = 1.0;
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleUnitSize = 16; // 4 floats * 4 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 4);

        for (let i = 0; i < this.numParticles; i++) {
            initialParticleData[i * 4 + 0] = (Math.random() * 2 - 1); // x
            initialParticleData[i * 4 + 1] = (Math.random() * 2 - 1); // y
            initialParticleData[i * 4 + 2] = 0; // vx
            initialParticleData[i * 4 + 3] = 0; // vy
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group
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

        // Pipelines
        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{ format: presentationFormat, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                } }], // Additive blending
            },
            primitive: { topology: 'point-list' },
        });

        // Remove internal resize call
        // this.resizeGPU();
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

            const mouseLoc = this.gl.getUniformLocation(this.glProgram, 'u_mouse');
            this.gl.uniform2f(mouseLoc, this.mouse.x, this.mouse.y);

            const resLoc = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0 && this.gpuCanvas.height > 0) {
            // Update simulation params
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
    window.HybridMagneticField = HybridMagneticField;
}
