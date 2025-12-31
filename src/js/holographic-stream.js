/**
 * Holographic Data Stream Experiment
 * Combining WebGL2 (Tunnel & Structures) + WebGPU (Data Particle Swarm)
 */
class HolographicStream {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            numParticles: options.numParticles || 20000,
            particleSpeed: options.particleSpeed || 1.0,
            ...options
        };

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glNumVertices = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.particleBuffer = null;
        this.simParamBuffer = null;

        // Bind resize handler
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000510'; // Dark blue-ish background

        console.log("HolographicStream: Initializing...");

        // 1. WebGL2 Layer (Background Structure)
        this.initWebGL2();

        // 2. WebGPU Layer (Particle Swarm)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("HolographicStream: WebGPU init failed:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
            console.log("HolographicStream: Running in WebGL2-only mode.");
        } else {
            console.log("HolographicStream: WebGPU initialized.");
        }

        this.resize();
        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Wireframe Tunnel)
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

        // Initial Resize
        const dpr = window.devicePixelRatio || 1;
        this.glCanvas.width = this.container.clientWidth * dpr;
        this.glCanvas.height = this.container.clientHeight * dpr;

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) {
            console.warn("HolographicStream: WebGL2 not supported.");
            return;
        }
        this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);

        // --- Generate Tunnel Geometry ---
        // A simple grid folded into a cylinder
        const segmentsX = 40; // Around
        const segmentsY = 40; // Length
        const vertices = [];

        for (let y = 0; y <= segmentsY; y++) {
            for (let x = 0; x <= segmentsX; x++) {
                const u = x / segmentsX;
                const v = y / segmentsY;

                // Cylinder coordinates
                const radius = 2.0;
                const angle = u * Math.PI * 2.0;

                const px = Math.cos(angle) * radius;
                const py = Math.sin(angle) * radius;
                const pz = v * 20.0 - 10.0; // Extend along Z

                vertices.push(px, py, pz);
            }
        }

        // Create lines (wireframe)
        const indices = [];
        for (let y = 0; y < segmentsY; y++) {
            for (let x = 0; x < segmentsX; x++) {
                const i0 = y * (segmentsX + 1) + x;
                const i1 = i0 + 1;
                const i2 = (y + 1) * (segmentsX + 1) + x;
                const i3 = i2 + 1;

                // Horizontal line
                indices.push(i0, i1);
                // Vertical line
                indices.push(i0, i2);
            }
        }

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);
        this.glNumVertices = indices.length;

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform mat4 u_projection;
            uniform mat4 u_view;

            out vec3 v_pos;
            out float v_depth;

            void main() {
                vec3 pos = a_position;

                // Move tunnel towards camera
                pos.z = mod(pos.z + u_time * 5.0, 20.0) - 10.0;

                // Add some twisting
                float twist = pos.z * 0.1 + u_time * 0.5;
                float c = cos(twist);
                float s = sin(twist);
                float x = pos.x * c - pos.y * s;
                float y = pos.x * s + pos.y * c;
                pos.x = x;
                pos.y = y;

                v_pos = pos;
                gl_Position = u_projection * u_view * vec4(pos, 1.0);
                v_depth = gl_Position.w;
            }
        `;

        // Fragment Shader
        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            in float v_depth;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Distance fade
                float alpha = 1.0 - smoothstep(5.0, 15.0, v_depth);

                // Pulse effect
                float pulse = 0.5 + 0.5 * sin(v_pos.z * 0.5 - u_time * 3.0);

                vec3 color = mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.8), pulse);

                outColor = vec4(color, alpha * 0.8 + 0.2);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer); // Bind index buffer to VAO
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error("GL VS Error:", this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error("GL FS Error:", this.gl.getShaderInfoLog(fs));
            return null;
        }

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vs);
        this.gl.attachShader(prog, fs);
        this.gl.linkProgram(prog);

        return prog;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Particle Swarm)
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

        // Initial Resize
        const dpr = window.devicePixelRatio || 1;
        this.gpuCanvas.width = this.container.clientWidth * dpr;
        this.gpuCanvas.height = this.container.clientHeight * dpr;

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // Compute Shader
        const computeCode = `
            struct Particle {
                pos : vec4f, // xyz, w=unused
                vel : vec4f, // xyz, w=life
            }

            struct SimParams {
                time : f32,
                dt : f32,
                speed : f32,
                pad : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : SimParams;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Move
                p.pos.z = p.pos.z + params.speed * 10.0 * params.dt;

                // Spiral motion
                let angle = params.time * 2.0 + f32(i) * 0.01;
                let radius = 1.5 + sin(params.time + f32(i) * 0.1) * 0.5;

                p.pos.x = cos(angle) * radius;
                p.pos.y = sin(angle) * radius;

                // Reset if too close
                if (p.pos.z > 5.0) {
                    p.pos.z = -15.0 - rand(vec2f(f32(i), params.time)) * 10.0;
                }

                particles[i] = p;
            }
        `;

        // Render Shader
        const drawCode = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @group(0) @binding(0) var<uniform> viewProj : mat4x4f;

            @vertex
            fn vs_main(@location(0) pos : vec4f) -> VertexOutput {
                var output : VertexOutput;
                output.position = viewProj * vec4f(pos.xyz, 1.0);

                // Size attenuation based on depth
                // WebGPU points are constant size in screen space usually,
                // but we can fake it or just let them be pixels.
                // For simplicity, we just project.

                let dist = output.position.w;
                let alpha = clamp(1.0 - (dist / 20.0), 0.0, 1.0);

                output.color = vec4f(0.0, 1.0, 1.0, alpha);

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const particleSize = 8 * 4; // 8 floats
        const initialData = new Float32Array(this.options.numParticles * 8);
        for(let i=0; i<this.options.numParticles; i++) {
            initialData[i*8+0] = (Math.random() - 0.5) * 4;
            initialData[i*8+1] = (Math.random() - 0.5) * 4;
            initialData[i*8+2] = -20.0 + Math.random() * 20.0; // z depth
            initialData[i*8+3] = 1.0;

            initialData[i*8+4] = 0; // vx
            initialData[i*8+5] = 0; // vy
            initialData[i*8+6] = 0; // vz
            initialData[i*8+7] = Math.random(); // life
        }

        this.particleBuffer = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.viewProjBuffer = this.device.createBuffer({
            size: 64, // mat4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Compute Pipeline
        const computeModule = this.device.createShaderModule({ code: computeCode });
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ]
        });

        // Render Pipeline
        const drawModule = this.device.createShaderModule({ code: drawCode });
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.viewProjBuffer } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                    ]
                }]
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' },
            depthStencil: undefined // No depth test for additive particles
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
            background: rgba(200, 50, 50, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            pointer-events: none;
        `;
        msg.innerText = "WebGPU not supported - Hybrid Mode inactive";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        const displayWidth = Math.floor(width * dpr);
        const displayHeight = Math.floor(height * dpr);

        this.resizeGL(displayWidth, displayHeight);
        this.resizeGPU(displayWidth, displayHeight);
    }

    resizeGL(width, height) {
        if (!this.glCanvas || !width || !height) return;
        if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
            this.glCanvas.width = width;
            this.glCanvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    resizeGPU(width, height) {
        if (!this.gpuCanvas || !width || !height) return;
        if (this.gpuCanvas.width !== width || this.gpuCanvas.height !== height) {
            this.gpuCanvas.width = width;
            this.gpuCanvas.height = height;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;
        const dt = 0.016; // Fixed step for simplicity

        // --- Render WebGL2 ---
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Matrix setup (Perspective)
            const aspect = this.glCanvas.width / this.glCanvas.height;
            const fov = 60 * Math.PI / 180;
            const zNear = 0.1;
            const zFar = 100.0;
            const f = 1.0 / Math.tan(fov / 2);

            // Projection Matrix (Column Major)
            const proj = new Float32Array([
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (zFar + zNear) / (zNear - zFar), -1,
                0, 0, (2 * zFar * zNear) / (zNear - zFar), 0
            ]);

            // View Matrix (Identity moved back)
            // Camera at 0,0,5 looking at 0,0,0
            const view = new Float32Array([
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -5, 1 // Translation (0, 0, -5)
            ]);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const projLoc = this.gl.getUniformLocation(this.glProgram, 'u_projection');
            const viewLoc = this.gl.getUniformLocation(this.glProgram, 'u_view');

            this.gl.uniform1f(timeLoc, time);
            this.gl.uniformMatrix4fv(projLoc, false, proj);
            this.gl.uniformMatrix4fv(viewLoc, false, view);

            // GL Blend (Additive-ish for wireframe)
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE); // Additive blending
            this.gl.clearColor(0.0, 0.05, 0.1, 1.0); // Dark blue background for visibility
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.glNumVertices, this.gl.UNSIGNED_SHORT, 0);
        }

        // --- Render WebGPU ---
        if (this.device && this.context && this.renderPipeline) {
            // Update Uniforms
            const params = new Float32Array([time, dt, this.options.particleSpeed, 0]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            // ViewProj for WebGPU
            const aspect = this.gpuCanvas.width / this.gpuCanvas.height;
            const fov = 60 * Math.PI / 180;
            const zNear = 0.1;
            const zFar = 100.0;
            const f = 1.0 / Math.tan(fov / 2);

            // WebGPU uses 0..1 depth
            // Projection
            const proj = [
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, zFar / (zNear - zFar), -1,
                0, 0, (zNear * zFar) / (zNear - zFar), 0
            ];

            const viewProj = new Float32Array(proj);

             // Compute col 3
             // V[3] is (0,0,-5,1)
             // Res[3] = P[0]*0 + P[4]*0 + P[8]*-5 + P[12]*1
             viewProj[12] = proj[8] * -5.0 + proj[12];
             viewProj[13] = proj[9] * -5.0 + proj[13];
             viewProj[14] = proj[10] * -5.0 + proj[14];
             viewProj[15] = proj[11] * -5.0 + proj[15];

            this.device.queue.writeBuffer(this.viewProjBuffer, 0, viewProj);


            const commandEncoder = this.device.createCommandEncoder();

            // Compute Pass
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.options.numParticles / 64));
            computePass.end();

            // Render Pass
            const textureView = this.context.getCurrentTexture().createView();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });

            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.renderBindGroup);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.options.numParticles);
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
            this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        }
        if (this.device) {
            this.device.destroy();
        }
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.HolographicStream = HolographicStream;
}
