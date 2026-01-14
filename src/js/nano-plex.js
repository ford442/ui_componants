/**
 * NanoPlex Assembly Experiment
 * - WebGL2: Raymarched Infinite Gyroid Structure (The "Scaffold")
 * - WebGPU: Swarm of "Assembler Bots" (Particles) that traverse the surface
 */

export class NanoPlexExperiment {
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
        this.numParticles = options.numParticles || 40000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        console.log("NanoPlex: Initializing...");

        // 1. WebGL2 Layer (Background Structure)
        this.initWebGL2();

        // 2. WebGPU Layer (Particle Swarm)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("NanoPlex: WebGPU initialization error:", e);
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
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Raymarched Gyroid)
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

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
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

            // Gyroid SDF
            float sdGyroid(vec3 p, float scale, float thickness, float bias) {
                p *= scale;
                float d = dot(sin(p), cos(p.yzx)) + bias;
                return abs(d) / scale - thickness;
            }

            float map(vec3 p) {
                // Rotated domain
                float t = u_time * 0.1;
                float c = cos(t), s = sin(t);
                mat2 m = mat2(c, -s, s, c);
                p.xz = m * p.xz;

                return sdGyroid(p, 2.0, 0.05, 0.0) * 0.8;
            }

            vec3 getNormal(vec3 p) {
                float d = map(p);
                vec2 e = vec2(0.001, 0.0);
                return normalize(vec3(
                    d - map(p - e.xyy),
                    d - map(p - e.yxy),
                    d - map(p - e.yyx)
                ));
            }

            void main() {
                vec2 uv = v_uv;
                uv.x *= u_resolution.x / u_resolution.y;

                vec3 ro = vec3(0.0, 0.0, -3.0); // Ray origin
                vec3 rd = normalize(vec3(uv, 1.5)); // Ray direction

                float t = 0.0;
                float tmax = 20.0;
                float d = 0.0;
                int i = 0;

                for(i=0; i<64; i++) {
                    vec3 p = ro + rd * t;
                    d = map(p);
                    if(d < 0.001 || t > tmax) break;
                    t += d;
                }

                vec3 col = vec3(0.0);

                if(t < tmax) {
                    vec3 p = ro + rd * t;
                    vec3 n = getNormal(p);

                    // Lighting
                    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
                    float diff = max(dot(n, lightDir), 0.0);
                    float amb = 0.1;

                    // Iridescent material
                    vec3 matCol = 0.5 + 0.5 * cos(u_time * 0.2 + p.xyx * 0.5 + vec3(0,2,4));

                    col = matCol * (diff + amb);

                    // Fog
                    col = mix(col, vec3(0.0), smoothstep(5.0, 15.0, t));
                }

                // Background gradient
                vec3 bg = mix(vec3(0.0), vec3(0.05, 0.05, 0.1), length(uv));
                col += bg * smoothstep(0.0, 1.0, d); // Add BG if missed

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const loc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 2, this.gl.FLOAT, false, 0, 0);
    }

    createGLProgram(vs, fs) {
        const p = this.gl.createProgram();
        const v = this.gl.createShader(this.gl.VERTEX_SHADER);
        const f = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(v, vs);
        this.gl.shaderSource(f, fs);
        this.gl.compileShader(v);
        this.gl.compileShader(f);
        if (!this.gl.getShaderParameter(v, this.gl.COMPILE_STATUS)) console.error(this.gl.getShaderInfoLog(v));
        if (!this.gl.getShaderParameter(f, this.gl.COMPILE_STATUS)) console.error(this.gl.getShaderInfoLog(f));
        this.gl.attachShader(p, v);
        this.gl.attachShader(p, f);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Particle Swarm)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2; pointer-events: none; background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) {
            this.gpuCanvas.remove();
            return false;
        }
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        const computeCode = `
            struct Particle {
                pos : vec4f, // xyz, w=life
                vel : vec4f, // xyz, w=padding
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct Params {
                time : f32,
                dt : f32,
                mouseX : f32,
                mouseY : f32,
            }
            @group(0) @binding(1) var<uniform> params : Params;

            // Replicate Gyroid SDF logic in WGSL
            fn sdGyroid(p: vec3f, scale: f32, thickness: f32, bias: f32) -> f32 {
                let ps = p * scale;
                let d = dot(sin(ps), cos(ps.yzx)) + bias;
                return abs(d) / scale - thickness;
            }

            fn map(p_in: vec3f) -> f32 {
                var p = p_in;
                // Rotated domain to match WebGL
                let t = params.time * 0.1;
                let c = cos(t);
                let s = sin(t);
                // Manual 2D rotation on XZ
                let x = p.x;
                let z = p.z;
                p.x = c * x - s * z;
                p.z = s * x + c * z;

                return sdGyroid(p, 2.0, 0.05, 0.0);
            }

            fn getNormal(p: vec3f) -> vec3f {
                let e = vec2f(0.01, 0.0);
                let d = map(p);
                return normalize(vec3f(
                    d - map(p - e.xyy),
                    d - map(p - e.yxy),
                    d - map(p - e.yyx)
                ));
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];

                // Attraction to surface
                let d = map(p.pos.xyz);
                let n = getNormal(p.pos.xyz);

                // Force: Push towards surface (negative gradient)
                let attraction = n * d * -5.0;

                // Flow: Tangent flow (cross normal with up or axis)
                let flowDir = normalize(cross(n, vec3f(0.0, 1.0, 0.0)));

                // Mouse Interaction (repel)
                // Project mouse 2D into 3D ray approximation (simple Z plane match)
                let mousePos3D = vec3f(params.mouseX * 2.0, params.mouseY * 2.0, 0.0);
                let mDir = p.pos.xyz - mousePos3D;
                let mDist = length(mDir);
                let mForce = normalize(mDir) / (mDist * mDist + 0.1) * 2.0;

                // Update Velocity
                p.vel = p.vel + vec4f(attraction + flowDir * 0.5 + mForce, 0.0) * params.dt;
                p.vel = p.vel * 0.95; // Damping

                // Update Position
                p.pos = p.pos + p.vel * params.dt;

                // Bounds Reset
                if (length(p.pos.xyz) > 5.0 || length(p.pos.xyz) < 0.1) {
                    p.pos = vec4f(
                        (fract(sin(f32(idx)*0.1)*43758.5453)-0.5)*4.0,
                        (fract(cos(f32(idx)*0.1)*23421.2312)-0.5)*4.0,
                        (fract(sin(f32(idx)*0.5)*12312.1231)-0.5)*4.0,
                        1.0
                    );
                    p.vel = vec4f(0.0);
                }

                particles[idx] = p;
            }
        `;

        const drawCode = `
            struct VertexOutput {
                @builtin(position) pos : vec4f,
                @location(0) color : vec4f,
            }

            @group(0) @binding(0) var<uniform> params : vec4f; // Aspect ratio stored here

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f) -> VertexOutput {
                var out : VertexOutput;

                // Simple perspective projection
                // Camera at (0,0,-3) looking at (0,0,0)
                // Map p.xyz to screen

                let camPos = vec3f(0.0, 0.0, -3.0);
                let p = pos.xyz - camPos; // relative to cam

                // Project
                let fov = 1.5;
                let z = p.z;

                // If behind camera, clip
                if (z <= 0.1) {
                    out.pos = vec4f(0.0, 0.0, 0.0, 0.0);
                    return out;
                }

                let aspect = params.x;
                out.pos = vec4f(p.x / (z * aspect) * fov, p.y / z * fov, 0.0, 1.0);

                let speed = length(vel.xyz);
                out.color = vec4f(0.0, 0.8, 1.0, 1.0) + vec4f(1.0, 0.2, 0.0, 0.0) * speed * 2.0;

                // Size attenuation
                out.pos.w = 1.0;

                return out;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const particleData = new Float32Array(this.numParticles * 8); // 8 floats per particle
        for (let i = 0; i < this.numParticles; i++) {
            const idx = i * 8;
            particleData[idx] = (Math.random() - 0.5) * 4.0;
            particleData[idx + 1] = (Math.random() - 0.5) * 4.0;
            particleData[idx + 2] = (Math.random() - 0.5) * 4.0;
            particleData[idx + 3] = 1.0; // life
            // vel = 0
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Pipeline Setup
        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'main' }
        });

        const drawModule = this.device.createShaderModule({ code: drawCode });

        // Use a separate buffer for draw uniforms (aspect)
        this.drawUniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const drawBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' }}]
        });

        this.drawBindGroup = this.device.createBindGroup({
            layout: drawBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.drawUniformBuffer } }]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [drawBindGroupLayout] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // vel
                    ]
                }]
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.innerText = "WebGPU Not Available (WebGL2 Only)";
        msg.style.cssText = "position:absolute; bottom:20px; right:20px; color:white; background:rgba(100,0,0,0.8); padding:10px;";
        this.container.appendChild(msg);
    }

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

        const dw = Math.floor(width * dpr);
        const dh = Math.floor(height * dpr);

        if (this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            if(this.gl) this.gl.viewport(0, 0, dw, dh);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = dw;
            this.gpuCanvas.height = dh;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. WebGPU Render
        if (this.device && this.context && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;

            // Update Sim Params
            this.device.queue.writeBuffer(this.simParamBuffer, 0, new Float32Array([
                time,
                0.016,
                this.mouse.x * aspect, // Scale mouse X by aspect for correct 3D mapping
                this.mouse.y
            ]));

            // Update Draw Uniforms (Aspect)
            this.device.queue.writeBuffer(this.drawUniformBuffer, 0, new Float32Array([aspect, 0, 0, 0]));

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.drawBindGroup);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);
        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        this.device?.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.NanoPlexExperiment = NanoPlexExperiment;
}
