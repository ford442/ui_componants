/**
 * Cyber Rain Experiment
 * Hybrid WebGL2 (Cityscape) + WebGPU (Rain Particles)
 */

class CyberRain {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
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
        this.computeBindGroup = null;
        this.uniformBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 50000;

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050510';

        this.initWebGL2();

        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CyberRain: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        window.addEventListener('resize', this.handleResize);
        this.animate();
    }

    // ========================================================================
    // WebGL2 (Cityscape Grid)
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
                v_uv = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_resolution;
            out vec4 outColor;

            float random(vec2 st) {
                return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
            }

            void main() {
                vec2 uv = (v_uv * 2.0 - 1.0) * (u_resolution / u_resolution.y);

                // Camera movement
                vec3 ro = vec3(0.0, 5.0, -10.0 + u_time * 5.0);
                vec3 rd = normalize(vec3(uv, 1.0));

                // Rotate camera down slightly
                float angle = 0.4;
                float s = sin(angle);
                float c = cos(angle);
                mat3 rot = mat3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
                rd = rot * rd;

                // Floor plane (y = 0)
                float t = -ro.y / rd.y;

                vec3 col = vec3(0.05, 0.05, 0.1); // Sky color

                if (t > 0.0 && t < 100.0) {
                    vec3 pos = ro + rd * t;

                    // Grid
                    vec2 gridUV = pos.xz * 0.5;
                    vec2 grid = abs(fract(gridUV) - 0.5);
                    float line = 1.0 - smoothstep(0.02, 0.05, min(grid.x, grid.y));

                    // Distance fade
                    float fade = exp(-t * 0.05);

                    // Moving lights
                    float light = 0.0;
                    if (random(floor(gridUV + vec2(0.0, u_time))) > 0.9) {
                        light = 1.0;
                    }

                    vec3 gridColor = vec3(0.0, 1.0, 0.8) * line;
                    gridColor += vec3(1.0, 0.2, 0.5) * light * line; // Red lights

                    col = mix(col, gridColor, fade);
                }

                // Scanlines
                col *= 0.9 + 0.1 * sin(v_uv.y * 500.0 + u_time * 10.0);

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.resizeGL();
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

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Rain Particles)
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

        const computeCode = `
            struct Particle {
                pos: vec4f, // xyz, life
                vel: vec4f, // xyz, type (0=drop, 1=splash)
            }

            struct Uniforms {
                dt: f32,
                time: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            // Pseudo-random
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];

                // Gravity
                p.vel.y -= 9.8 * uniforms.dt;
                p.pos.xyz += p.vel.xyz * uniforms.dt;

                // Floor collision (approximate, since camera moves, we simulate rain relative to camera)
                // Let's say floor is at y = -2.0 relative to camera "screen" rain

                if (p.pos.y < -2.0) {
                    // Reset to top
                    p.pos.y = 5.0 + rand(vec2f(p.pos.x, uniforms.time)) * 5.0;
                    p.pos.x = (rand(vec2f(f32(idx), uniforms.time)) - 0.5) * 20.0;
                    p.pos.z = -2.0 + (rand(vec2f(p.pos.y, f32(idx))) * 10.0); // Depth
                    p.vel.y = -5.0 - rand(vec2f(p.pos.x, p.pos.z)) * 5.0; // Random speed
                    p.vel.x = 0.0;
                    p.vel.z = 0.0;
                }

                // Add "wind" relative to camera movement (camera moves +z)
                // So rain should appear to move -z
                p.pos.z -= 5.0 * uniforms.dt;

                if (p.pos.z < -10.0) {
                     p.pos.z = 5.0;
                }

                particles[idx] = p;
            }
        `;

        const renderCode = `
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs_main(@location(0) pPos: vec4f, @location(1) pVel: vec4f) -> VertexOut {
                var out: VertexOut;

                // Simple projection
                // We want rain to look like lines (motion blur)
                // But for now, points

                let x = pPos.x;
                let y = pPos.y;
                let z = pPos.z; // depth relative to camera

                // Camera at 0,0,0 looking at -z?
                // In compute we treated z as world z.
                // Let's do simple perspective

                // We want the rain to be "in front" of the camera.
                // WebGL camera was at (0, 5, -10 + t). Looking +z.
                // Let's make rain move in screen space for simplicity or just world space

                // Let's assume particles are in view space relative to camera
                // z goes from -1 (close) to -20 (far)

                // Adjust projection manually
                let depth = pPos.z + 10.0; // Offset to put in front
                let scale = 1.0 / max(depth, 0.1);

                out.pos = vec4f(x * scale, y * scale, 0.0, 1.0);

                // Color based on depth
                let alpha = clamp(1.0 - (depth * 0.1), 0.0, 1.0);
                out.color = vec4f(0.6, 0.8, 1.0, alpha);

                // Make point size depend on depth (gl_PointSize not in WGSL, but we can do it via quad expansion or just trust 1px points for now)

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const data = new Float32Array(this.numParticles * 8);
        for (let i = 0; i < this.numParticles; i++) {
            data[i*8+0] = (Math.random() - 0.5) * 20.0; // x
            data[i*8+1] = Math.random() * 10.0 - 2.0;   // y
            data[i*8+2] = Math.random() * 10.0 - 5.0;   // z
            data[i*8+3] = 1.0; // life
            data[i*8+4] = 0.0; // vx
            data[i*8+5] = -5.0 - Math.random() * 5.0; // vy
            data[i*8+6] = 0.0; // vz
            data[i*8+7] = 0.0; // type
        }

        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, data);

        this.uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
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
            layout: 'auto',
            vertex: {
                module: renderModule,
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

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = 'WebGPU Not Supported';
        msg.style.cssText = 'position:absolute;top:10px;right:10px;color:red;background:rgba(0,0,0,0.8);padding:5px;';
        this.container.appendChild(msg);
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        this.resizeGL(w*dpr, h*dpr);
        this.resizeGPU(w*dpr, h*dpr);
    }

    resizeGL(w, h) {
        if (this.glCanvas) {
            this.glCanvas.width = w;
            this.glCanvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
    }

    resizeGPU(w, h) {
        if (this.gpuCanvas) {
            this.gpuCanvas.width = w;
            this.gpuCanvas.height = h;
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
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGPU
        if (this.device && this.renderPipeline) {
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([0.016, time, 0, 0]));

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
    }
}

if (typeof window !== 'undefined') {
    window.CyberRain = CyberRain;
}
