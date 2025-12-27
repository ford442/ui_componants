/**
 * Cyber Crystal Experiment
 * A hybrid WebGL2 + WebGPU experiment.
 * - WebGL2: Renders a central "Cyber Crystal" (icosahedron) with wireframe glow.
 * - WebGPU: Renders orbiting energy shards using compute shaders.
 */

export class CyberCrystalExperiment {
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
        this.glNumIndices = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.particleBuffer = null;
        this.simParamBuffer = null;
        this.renderUniformBuffer = null;
        this.renderBindGroup = null;
        this.numParticles = options.numParticles || 10000;

        // Bind resize handler
        this.handleResize = this.resize.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050510';

        console.log("CyberCrystalExperiment: Initializing...");

        // 1. Initialize WebGL2 (The Core)
        this.initWebGL2();

        // 2. Initialize WebGPU (The Energy)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CyberCrystalExperiment: WebGPU init error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("CyberCrystalExperiment: WebGPU not available. WebGL2 only.");
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();
        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION
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

        this.gl = this.glCanvas.getContext('webgl2', { alpha: true }); // Alpha for transparency if needed
        if (!this.gl) return;

        // Generate Icosahedron
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;
        const v = [
            -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
             0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
             t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1
        ];

        // Scale vertices
        const scale = 1.5;
        for(let i=0; i<v.length; i++) v[i] *= scale;

        const indices = [
            0, 11, 5,   0, 5, 1,   0, 1, 7,   0, 7, 10,  0, 10, 11,
            1, 5, 9,   5, 11, 4,  11, 10, 2,  10, 7, 6,   7, 1, 8,
            3, 9, 4,   3, 4, 2,   3, 2, 6,   3, 6, 8,   3, 8, 9,
            4, 9, 5,   2, 4, 11,  6, 2, 10,  8, 6, 7,   9, 8, 1
        ];

        this.glNumIndices = indices.length;

        // Buffers
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(v), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform vec2 u_resolution;
            uniform mat4 u_model;

            out vec3 v_pos;
            out vec3 v_barycentric; // For wireframe effect

            void main() {
                v_pos = a_position;

                // Manual perspective matrix
                float fov = 1.0;
                float aspect = u_resolution.x / u_resolution.y;
                float f = 1.0 / tan(fov / 2.0);
                float zNear = 0.1;
                float zFar = 100.0;

                mat4 projection = mat4(
                    f / aspect, 0.0, 0.0, 0.0,
                    0.0, f, 0.0, 0.0,
                    0.0, 0.0, (zFar + zNear) / (zNear - zFar), -1.0,
                    0.0, 0.0, (2.0 * zFar * zNear) / (zNear - zFar), 0.0
                );

                // Model rotation handled in JS or here
                float c = cos(u_time * 0.5);
                float s = sin(u_time * 0.5);

                // Rotate Y
                vec3 pos = a_position;
                float x = pos.x * c - pos.z * s;
                float z = pos.x * s + pos.z * c;
                pos.x = x;
                pos.z = z;

                // Rotate X
                c = cos(u_time * 0.3);
                s = sin(u_time * 0.3);
                float y = pos.y * c - pos.z * s;
                z = pos.y * s + pos.z * c;
                pos.y = y;
                pos.z = z;

                // View translation
                pos.z -= 6.0;

                gl_Position = projection * vec4(pos, 1.0);

                // Pass some varying for coloring
                v_pos = a_position;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Pulse effect
                float pulse = 0.5 + 0.5 * sin(u_time * 2.0);

                // Crystal color (Cyan/Blueish)
                vec3 baseColor = vec3(0.0, 0.8, 1.0);
                vec3 glowColor = vec3(0.8, 0.0, 1.0);

                // Fresnel-like effect based on position
                float rim = 1.0 - abs(dot(normalize(v_pos), vec3(0.0, 0.0, 1.0)));
                rim = pow(rim, 3.0);

                vec3 finalColor = mix(baseColor, glowColor, rim * pulse);

                // Semi-transparent
                outColor = vec4(finalColor, 0.3 + 0.4 * rim);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);
    }

    createGLProgram(vs, fs) {
        const createShader = (type, src) => {
            const s = this.gl.createShader(type);
            this.gl.shaderSource(s, src);
            this.gl.compileShader(s);
            if (!this.gl.getShaderParameter(s, this.gl.COMPILE_STATUS)) {
                console.error('Shader Error:', this.gl.getShaderInfoLog(s));
                return null;
            }
            return s;
        };

        const v = createShader(this.gl.VERTEX_SHADER, vs);
        const f = createShader(this.gl.FRAGMENT_SHADER, fs);
        if (!v || !f) return null;

        const p = this.gl.createProgram();
        this.gl.attachShader(p, v);
        this.gl.attachShader(p, f);
        this.gl.linkProgram(p);

        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION
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
        if (!adapter) throw new Error("No adapter");

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // Compute Shader: Orbiting particles
        const computeCode = `
            struct Particle {
                pos : vec4f, // x, y, z, angle
                vel : vec4f, // vx, vy, vz, speed
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Params {
                dt : f32,
                time : f32,
            }
            @group(0) @binding(1) var<uniform> params : Params;

            fn rand(seed: vec2f) -> f32 {
                return fract(sin(dot(seed, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.numParticles}) { return; }

                var p = particles[index];

                // Initialization check (if vel.w == 0)
                if (p.vel.w == 0.0) {
                    let s = f32(index);
                    p.pos.w = rand(vec2f(s, params.time)) * 6.28; // angle
                    let dist = 2.0 + rand(vec2f(s + 1.0, params.time)) * 3.0; // radius
                    p.vel.w = 0.5 + rand(vec2f(s + 2.0, params.time)); // orbital speed

                    // Sphere position
                    let theta = rand(vec2f(s, s)) * 6.28;
                    let phi = acos(2.0 * rand(vec2f(s+3.0, s)) - 1.0);

                    p.pos.x = dist * sin(phi) * cos(theta);
                    p.pos.y = dist * sin(phi) * sin(theta);
                    p.pos.z = dist * cos(phi);
                }

                // Orbit logic
                // Rotate around Y axis
                let speed = p.vel.w * params.dt;
                let c = cos(speed);
                let s = sin(speed);

                let x = p.pos.x * c - p.pos.z * s;
                let z = p.pos.x * s + p.pos.z * c;
                p.pos.x = x;
                p.pos.z = z;

                // Vertical wave
                p.pos.y += sin(params.time * 2.0 + p.pos.x) * 0.01;

                particles[index] = p;
            }
        `;

        // Render Shader
        const renderCode = `
            struct Uniforms {
                modelViewProjection : mat4x4f,
                time : f32,
            }
            @group(0) @binding(2) var<uniform> uniforms : Uniforms;

            struct VertexOut {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) particlePos : vec4f) -> VertexOut {
                var output : VertexOut;

                // Manual projection matches WebGL2
                // We construct the matrix in JS and pass it via uniform

                var pos = particlePos.xyz;

                // Manual rotation to match crystal rotation
                let t = uniforms.time;
                let c = cos(t * 0.5);
                let s = sin(t * 0.5);

                // Rotate Y (Base rotation)
                let x = pos.x * c - pos.z * s;
                let z = pos.x * s + pos.z * c;
                pos.x = x;
                pos.z = z;

                // Rotate X
                let c2 = cos(t * 0.3);
                let s2 = sin(t * 0.3);
                let y = pos.y * c2 - pos.z * s2;
                let z2 = pos.y * s2 + pos.z * c2;
                pos.y = y;
                pos.z = z2;

                // Apply MVP (Projection * View)
                output.position = uniforms.modelViewProjection * vec4f(pos, 1.0);

                // Distance fade
                let dist = length(pos);
                let alpha = smoothstep(5.0, 2.0, dist);

                output.color = vec4f(0.5, 1.0, 0.8, alpha);
                output.position.w = output.position.w; // Ensure w is correct

                // Point size hack for point-list
                if (output.position.w > 0.0) {
                     output.position.z = output.position.z;
                }

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const pSize = 32; // 8 floats
        this.particleBuffer = this.device.createBuffer({
            size: this.numParticles * pSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        // Init buffer with zeros so vel.w starts at 0 for init logic
        this.device.queue.writeBuffer(this.particleBuffer, 0, new Float32Array(this.numParticles * 8));

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // dt, time, pad, pad
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.renderUniformBuffer = this.device.createBuffer({
            size: 64 + 16, // mat4 (64) + time (4) + pad
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Bind Groups
        const computeLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }},
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer }},
                { binding: 1, resource: { buffer: this.simParamBuffer }}
            ]
        });

        const renderLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' }}
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderLayout,
            entries: [
                { binding: 2, resource: { buffer: this.renderUniformBuffer }}
            ]
        });

        // Pipelines
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
            compute: { module: this.device.createShaderModule({ code: computeCode }), entryPoint: 'main' }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
            vertex: {
                module: this.device.createShaderModule({ code: renderCode }),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: pSize,
                    stepMode: 'vertex',
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] // just pos needed
                }]
            },
            fragment: {
                module: this.device.createShaderModule({ code: renderCode }),
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "WebGPU Not Available - Rendering WebGL2 Crystal Only";
        msg.style.cssText = "position:absolute; bottom:20px; right:20px; color:white; background:rgba(100,0,0,0.8); padding:10px; border-radius:5px;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // RUNTIME
    // ========================================================================

    resize() {
        if (!this.container) return;
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
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;

        // Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.0, 0.0, 0.0, 0.0); // Transparent to let background show if any
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE); // Additiveish

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glNumIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // Render WebGPU
        if (this.device && this.renderPipeline) {
            // Update Sim Params
            this.device.queue.writeBuffer(this.simParamBuffer, 0, new Float32Array([0.016, time, 0, 0]));

            // Update Render Matrix (Simulate MVP)
            // Aspect Ratio
            const aspect = this.gpuCanvas.width / this.gpuCanvas.height;
            const fov = 1.0;
            const f = 1.0 / Math.tan(fov/2);
            const rangeInv = 1.0 / (0.1 - 100.0);

            const C = (100.0 + 0.1) * rangeInv; // m10
            const D = (2.0 * 100.0 * 0.1) * rangeInv; // m14

            const mat = new Float32Array([
                f/aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, C, -1,
                0, 0, C * -6.0 + D, 6.0
            ]);

            this.device.queue.writeBuffer(this.renderUniformBuffer, 0, mat);
            this.device.queue.writeBuffer(this.renderUniformBuffer, 64, new Float32Array([time])); // time offset 64

            const encoder = this.device.createCommandEncoder();

            // Compute Pass
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            // Render Pass
            const view = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: view,
                    clearValue: { r:0, g:0, b:0, a:0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBindGroup);
            rPass.setVertexBuffer(0, this.particleBuffer);
            rPass.draw(this.numParticles);
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}
