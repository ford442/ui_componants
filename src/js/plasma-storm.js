/**
 * Plasma Storm Experiment
 * Hybrid WebGL2 (Volumetric Clouds) + WebGPU (Plasma Particles)
 */

class PlasmaStorm {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };

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
        this.computeBindGroup = null;
        this.uniformBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 60000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020205';

        // 1. Initialize WebGL2
        this.initWebGL2();

        // 2. Initialize WebGPU
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("PlasmaStorm: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);

        // Initial resize
        this.resize();

        this.animate();
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -(e.clientY - rect.top) / rect.height * 2 + 1;
    }

    // ========================================================================
    // WebGL2 (Volumetric Clouds)
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

        // Fullscreen Quad
        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vs = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;
            out vec4 outColor;

            // Noise functions
            float hash(float n) { return fract(sin(n) * 43758.5453123); }
            float noise(vec3 x) {
                vec3 p = floor(x);
                vec3 f = fract(x);
                f = f * f * (3.0 - 2.0 * f);
                float n = p.x + p.y * 57.0 + p.z * 113.0;
                return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                               mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
                           mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                               mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
            }

            float fbm(vec3 p) {
                float f = 0.0;
                f += 0.5000 * noise(p); p *= 2.02;
                f += 0.2500 * noise(p); p *= 2.03;
                f += 0.1250 * noise(p); p *= 2.01;
                f += 0.0625 * noise(p);
                return f;
            }

            // Volumetric Raymarching
            vec4 march(vec3 ro, vec3 rd) {
                float t = 0.0;
                vec4 sum = vec4(0.0);

                for(int i = 0; i < 60; i++) {
                    if(sum.a > 0.99 || t > 20.0) break;

                    vec3 pos = ro + rd * t;
                    // Cloud density
                    float d = fbm(pos * 0.5 + vec3(0.0, 0.0, u_time * 0.2));
                    // Sphere shaping
                    float sphere = length(pos) - 3.5;
                    float dens = clamp((d - sphere * 0.5) * 0.5, 0.0, 1.0);

                    if(dens > 0.01) {
                        float alpha = dens * 0.4;
                        // Color gradient: dark purple to bright electric blue
                        vec3 col = mix(vec3(0.1, 0.0, 0.2), vec3(0.4, 0.6, 1.0), d);
                        col += vec3(1.0) * smoothstep(0.8, 1.0, d) * 0.5; // Highlights

                        sum += vec4(col * alpha, alpha) * (1.0 - sum.a);
                    }
                    t += 0.1 + t * 0.02;
                }
                return sum;
            }

            void main() {
                vec2 uv = v_uv;
                uv.x *= u_resolution.x / u_resolution.y;

                vec3 ro = vec3(0.0, 0.0, 4.0);
                vec3 rd = normalize(vec3(uv, -1.0));

                // Mouse rotation
                float mx = u_mouse.x * 2.0;
                float my = u_mouse.y * 2.0;
                mat2 rotX = mat2(cos(mx), -sin(mx), sin(mx), cos(mx));
                mat2 rotY = mat2(cos(my), -sin(my), sin(my), cos(my));
                ro.xz *= rotX;
                rd.xz *= rotX;
                ro.yz *= rotY;
                rd.yz *= rotY;

                // Background gradient
                vec3 bg = mix(vec3(0.0), vec3(0.05, 0.02, 0.1), length(uv));

                vec4 clouds = march(ro, rd);

                vec3 finalColor = bg * (1.0 - clouds.a) + clouds.rgb;

                // Vignette
                finalColor *= 1.0 - smoothstep(0.5, 1.5, length(v_uv));

                outColor = vec4(finalColor, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('GL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('GL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Plasma Particles)
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
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // Compute Shader: Chaotic Attractor
        const computeCode = `
            struct Particle {
                pos: vec4f, // xyz, life
                vel: vec4f, // xyz, unused
            }

            struct Uniforms {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            // Aizawa Attractor parameters
            const a: f32 = 0.95;
            const b: f32 = 0.7;
            const c: f32 = 0.6;
            const d: f32 = 3.5;
            const e: f32 = 0.25;
            const f: f32 = 0.1;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];
                let x = p.pos.x;
                let y = p.pos.y;
                let z = p.pos.z;

                // Modulate params with mouse/time
                let mod_d = d + uniforms.mouseX * 2.0;
                let mod_e = e + sin(uniforms.time * 0.5) * 0.1;

                // Aizawa Attractor Equations
                // dx = (z - b) * x - d * y
                // dy = d * x + (z - b) * y
                // dz = c + a * z - (z^3 / 3) - (x^2 + y^2) * (1 + e * z) + f * z * x^3

                let dx = (z - b) * x - mod_d * y;
                let dy = mod_d * x + (z - b) * y;
                let dz = c + a * z - (z*z*z / 3.0) - (x*x + y*y) * (1.0 + mod_e * z) + f * z * x*x*x;

                let speed = 0.5 * uniforms.dt;
                p.pos.x += dx * speed;
                p.pos.y += dy * speed;
                p.pos.z += dz * speed;

                // Reset if lost or random reset for life
                p.pos.w -= uniforms.dt * 0.2; // Decay life

                // Bounds check or life check
                if (p.pos.w <= 0.0 || length(p.pos.xyz) > 20.0) {
                    // Respawn near center with some noise
                    let theta = fract(sin(f32(idx) * 0.1 + uniforms.time) * 43758.5) * 6.28;
                    let phi = fract(cos(f32(idx) * 0.1 + uniforms.time) * 23421.6) * 3.14;
                    let r = 0.1;

                    p.pos.x = r * sin(phi) * cos(theta);
                    p.pos.y = r * sin(phi) * sin(theta);
                    p.pos.z = r * cos(phi);
                    p.pos.w = 1.0; // Reset life
                }

                particles[idx] = p;
            }
        `;

        // Render Shader
        const renderCode = `
            struct Uniforms {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
            }
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs_main(@location(0) pPos: vec4f, @location(1) pVel: vec4f) -> VertexOut {
                var out: VertexOut;

                // Rotate camera
                let mx = uniforms.mouseX * 2.0;
                let my = uniforms.mouseY * 2.0;
                let c = cos(mx); let s = sin(mx);
                let cy = cos(my); let sy = sin(my);

                var x = pPos.x;
                var y = pPos.y;
                var z = pPos.z;

                // Rot Y
                let rx = x * c - z * s;
                let rz = x * s + z * c;
                x = rx; z = rz;

                // Rot X
                let ry = y * cy - z * sy;
                let rz2 = y * sy + z * cy;
                y = ry; z = rz2;

                // Perspective
                z = z - 4.0; // Push back
                let scale = 1.0 / -z; // simple perspective

                // Adjust aspect? Assuming square for simplicity in this snippet or relying on canvas style
                // Ideally pass aspect ratio uniform.

                out.pos = vec4f(x * scale, y * scale, 0.0, 1.0);

                // Color based on velocity/position and life
                let life = pPos.w;
                let col1 = vec3f(1.0, 0.2, 0.0); // Orange
                let col2 = vec3f(0.0, 0.8, 1.0); // Cyan

                let mixVal = smoothstep(-1.0, 1.0, pPos.z);
                let color = mix(col1, col2, mixVal);

                out.color = vec4f(color, life * 0.8);

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialize Buffers
        const data = new Float32Array(this.numParticles * 8);
        for (let i = 0; i < this.numParticles; i++) {
            data[i*8+0] = (Math.random() - 0.5) * 2.0; // x
            data[i*8+1] = (Math.random() - 0.5) * 2.0; // y
            data[i*8+2] = (Math.random() - 0.5) * 2.0; // z
            data[i*8+3] = Math.random(); // life
            // vel unused init
        }

        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, data);

        this.uniformBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: renderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos + life
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // vel
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    }
                }]
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = 'WebGPU Not Supported';
        msg.style.cssText = 'position:absolute;top:10px;right:10px;color:red;background:rgba(0,0,0,0.8);padding:5px;pointer-events:none;';
        this.container.appendChild(msg);
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        if (this.glCanvas) {
            this.glCanvas.width = w * dpr;
            this.glCanvas.height = h * dpr;
            this.gl.viewport(0, 0, w * dpr, h * dpr);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = w * dpr;
            this.gpuCanvas.height = h * dpr;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;

        // Render WebGL
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGPU
        if (this.device && this.renderPipeline) {
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([0.016, time, this.mouse.x, this.mouse.y]));

            const encoder = this.device.createCommandEncoder();

            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            const textureView = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.computeBindGroup); // Need bindgroup for uniforms in VS
            rPass.setVertexBuffer(0, this.particleBuffer);
            rPass.draw(this.numParticles);
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);
    }
}

if (typeof window !== 'undefined') {
    window.PlasmaStorm = PlasmaStorm;
}
