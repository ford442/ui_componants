/**
 * Bioluminescent Abyss
 * Hybrid WebGL2 (Underwater Terrain) + WebGPU (Bio-luminescent Organisms)
 */

export class BioluminescentAbyss {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // Interaction State
        this.mouse = { x: 0, y: 0 };

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
        this.uniformBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 30000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleTouchMove = this.onTouchMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000005';

        // 1. Initialize WebGL2 Layer (Background Terrain)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground Creatures)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("BioluminescentAbyss: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("BioluminescentAbyss: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('touchmove', this.handleTouchMove, { passive: false });
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        // Normalize to -1 to 1
        this.mouse.x = x * 2.0 - 1.0;
        this.mouse.y = -(y * 2.0 - 1.0); // Invert Y
    }

    onTouchMove(e) {
        if (e.touches && e.touches.length > 0) {
            const rect = this.container.getBoundingClientRect();
            const x = (e.touches[0].clientX - rect.left) / rect.width;
            const y = (e.touches[0].clientY - rect.top) / rect.height;
            this.mouse.x = x * 2.0 - 1.0;
            this.mouse.y = -(y * 2.0 - 1.0);
        }
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Raymarched Underwater Terrain)
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
            console.warn("BioluminescentAbyss: WebGL2 not supported.");
            return;
        }

        // Fullscreen Quad
        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;
            out vec4 outColor;

            // 3D Noise for terrain
            float hash(float n) { return fract(sin(n) * 753.5453123); }
            float noise(vec3 x) {
                vec3 p = floor(x);
                vec3 f = fract(x);
                f = f * f * (3.0 - 2.0 * f);
                float n = p.x + p.y * 157.0 + 113.0 * p.z;
                return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                               mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
                           mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                               mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
            }

            float fbm(vec3 p) {
                float f = 0.0;
                float w = 0.5;
                for (int i = 0; i < 5; i++) {
                    f += w * noise(p);
                    p *= 2.0;
                    w *= 0.5;
                }
                return f;
            }

            float map(vec3 p) {
                // Canyon shape
                float ground = p.y + 10.0 + fbm(p * 0.1) * 8.0;
                // Add some walls
                float walls = abs(p.x) - 5.0 - fbm(p * 0.2 + vec3(0.0, 1.0, 0.0)) * 10.0;

                return min(ground, max(-p.y + 20.0, -walls)); // Open top, walls and ground
            }

            vec3 getNormal(vec3 p) {
                const vec2 e = vec2(0.01, 0.0);
                return normalize(vec3(
                    map(p + e.xyy) - map(p - e.xyy),
                    map(p + e.yxy) - map(p - e.yxy),
                    map(p + e.yyx) - map(p - e.yyx)
                ));
            }

            void main() {
                vec2 uv = (v_uv * 2.0 - 1.0) * (u_resolution / u_resolution.y);

                vec3 ro = vec3(0.0, 0.0, u_time * 2.0);
                vec3 rd = normalize(vec3(uv, 1.0));

                // Add some camera sway + mouse look
                rd.x += sin(u_time * 0.3) * 0.1 + u_mouse.x * 0.5;
                rd.y += cos(u_time * 0.2) * 0.1 + u_mouse.y * 0.5;
                rd = normalize(rd);

                float t = 0.0;
                float d = 0.0;
                int steps = 0;

                for (int i = 0; i < 64; i++) {
                    vec3 p = ro + rd * t;
                    d = map(p);
                    if (d < 0.01 || t > 50.0) break;
                    t += d * 0.5; // Slower march for better detail/fewer artifacts
                    steps = i;
                }

                vec3 col = vec3(0.0, 0.05, 0.1); // Deep blue background

                if (t < 50.0) {
                    vec3 p = ro + rd * t;
                    vec3 n = getNormal(p);
                    vec3 lightDir = normalize(vec3(0.0, 1.0, 0.5));

                    // Lighting
                    float diff = max(dot(n, lightDir), 0.0);
                    float amb = 0.1;

                    // Caustics approximation
                    float caustics = pow(noise(vec3(p.x * 2.0, p.z * 2.0, u_time * 2.0)), 3.0) * 0.5;
                    if (p.y > -5.0) caustics *= 2.0;

                    vec3 objColor = vec3(0.1, 0.2, 0.2);
                    col = objColor * (diff + amb) + vec3(0.2, 0.8, 1.0) * caustics;

                    // Fog
                    float fog = 1.0 - exp(-t * 0.05);
                    col = mix(col, vec3(0.0, 0.02, 0.05), fog);
                }

                // Vignette
                col *= 1.0 - length(uv) * 0.3;

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
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

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Bioluminescent Organisms)
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

        // Compute Shader: Flocking/Swirling behavior
        const computeCode = `
            struct Particle {
                pos: vec4f, // xyz, life/phase
                vel: vec4f, // xyz, type
            }

            struct Uniforms {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            // Simple noise function
            fn hash(p: vec3f) -> vec3f {
                var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
                p3 += dot(p3, p3.yxz + 33.33);
                return fract((p3.xxy + p3.yzz) * p3.zyx) * 2.0 - 1.0;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];

                // Add noise force for organic movement
                let noiseForce = hash(p.pos.xyz * 0.5 + uniforms.time * 0.2) * 2.0;

                // Mouse Attractor
                // Mouse is -1..1 range. Map to rough world space bounds.
                // Assuming view width ~40 units at standard depth.
                let mousePos = vec3f(uniforms.mouseX * 20.0, uniforms.mouseY * 10.0, p.pos.z);
                let dirToMouse = mousePos - p.pos.xyz;
                let distToMouse = length(dirToMouse);
                let mouseForce = normalize(dirToMouse) * (20.0 / (distToMouse + 1.0));

                // Attract to center somewhat to keep them in view
                let centerDir = -normalize(p.pos.xyz);
                let dist = length(p.pos.xyz);
                let centerAttract = centerDir * smoothstep(5.0, 20.0, dist) * 1.0;

                // Combine forces
                p.vel.x += (noiseForce.x + centerAttract.x + mouseForce.x) * uniforms.dt;
                p.vel.y += (noiseForce.y + centerAttract.y + mouseForce.y) * uniforms.dt;
                p.vel.z += (noiseForce.z + centerAttract.z) * uniforms.dt;

                // Dampen
                p.vel.x *= 0.98;
                p.vel.y *= 0.98;
                p.vel.z *= 0.98;

                // Update Position
                p.pos.x += p.vel.x * uniforms.dt;
                p.pos.y += p.vel.y * uniforms.dt;
                p.pos.z += p.vel.z * uniforms.dt;

                // "Camera" moves at speed 2.0 in Z (matching WebGL)
                // To keep particles relative to camera, we wrap them in Z

                p.pos.z -= 1.0 * uniforms.dt; // Swim slower than camera (so they pass by)

                if (p.pos.z < -10.0) {
                    p.pos.z += 40.0;
                    p.pos.x = (hash(vec3f(f32(idx), uniforms.time, 0.0)).x) * 20.0;
                    p.pos.y = (hash(vec3f(f32(idx), uniforms.time, 1.0)).y) * 10.0;
                }

                // Animate life/phase for pulsing
                p.pos.w += uniforms.dt * 2.0;

                particles[idx] = p;
            }
        `;

        // Render Shader: Soft glowing particles
        const renderCode = `
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
                @location(1) uv: vec2f,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vIdx: u32,
                       @location(0) pPos: vec4f,
                       @location(1) pVel: vec4f) -> VertexOut {
                var out: VertexOut;

                // Billboard quad expansion
                let corner = vIdx % 6;
                var offset = vec2f(0.0);
                if (corner == 0) { offset = vec2f(-1.0, -1.0); }
                else if (corner == 1) { offset = vec2f(1.0, -1.0); }
                else if (corner == 2) { offset = vec2f(-1.0, 1.0); }
                else if (corner == 3) { offset = vec2f(-1.0, 1.0); }
                else if (corner == 4) { offset = vec2f(1.0, -1.0); }
                else if (corner == 5) { offset = vec2f(1.0, 1.0); }

                out.uv = offset;

                // Perspective projection
                let x = pPos.x;
                let y = pPos.y;
                let z = pPos.z;

                let depth = z + 15.0; // Push forward to be visible

                if (depth < 0.1) {
                    out.pos = vec4f(0.0, 0.0, 0.0, 0.0); // Cull behind camera
                    return out;
                }

                let scale = 1.0 / depth;

                // Size depends on depth
                let size = 0.2 * scale;

                out.pos = vec4f(
                    x * scale + offset.x * size,
                    y * scale + offset.y * size * (16.0/9.0), // Aspect ratio fix approx
                    0.0,
                    1.0
                );

                // Color based on velocity/type
                let speed = length(pVel.xyz);
                var col = vec3f(0.2, 0.8, 1.0); // Cyan base
                if (pVel.w > 0.5) {
                    col = vec3f(0.8, 0.2, 1.0); // Purple variant
                }

                // Pulse
                let pulse = sin(pPos.w) * 0.5 + 0.5;
                col += pulse * 0.5;

                let alpha = clamp(1.0 - (abs(z)/20.0), 0.0, 1.0); // Fade distance

                out.color = vec4f(col, alpha);

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f, @location(1) uv: vec2f) -> @location(0) vec4f {
                let dist = length(uv);
                if (dist > 1.0) { discard; }

                // Soft glow falloff
                let glow = pow(1.0 - dist, 2.0);

                return vec4f(color.rgb, color.a * glow);
            }
        `;

        // Buffer Init
        const data = new Float32Array(this.numParticles * 8); // 8 floats per particle
        for (let i = 0; i < this.numParticles; i++) {
            data[i*8+0] = (Math.random() - 0.5) * 40.0; // x
            data[i*8+1] = (Math.random() - 0.5) * 20.0; // y
            data[i*8+2] = (Math.random() - 0.5) * 30.0; // z
            data[i*8+3] = Math.random() * 6.28;         // phase

            data[i*8+4] = (Math.random() - 0.5) * 0.1;  // vx
            data[i*8+5] = (Math.random() - 0.5) * 0.1;  // vy
            data[i*8+6] = (Math.random() - 0.5) * 0.1;  // vz
            data[i*8+7] = Math.random();                // type
        }

        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, data);

        this.uniformBuffer = this.device.createBuffer({
            size: 16, // dt (f32), time (f32), mouseX (f32), mouseY (f32)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group
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

        // Pipelines
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
                    arrayStride: 32, // 8 floats * 4 bytes
                    stepMode: 'instance', // Instanced rendering
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
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }, // Additive blend
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    }
                }]
            },
            primitive: { topology: 'triangle-list' },
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = 'WebGPU Not Supported - Creatures Hidden';
        msg.style.cssText = 'position:absolute;bottom:20px;right:20px;color:rgba(255,100,100,0.8);background:rgba(0,0,0,0.8);padding:10px;border-radius:4px;font-family:sans-serif;font-size:12px;pointer-events:none;';
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        if (!this.container) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.resizeGL(w * dpr, h * dpr);
        this.resizeGPU(w * dpr, h * dpr);
    }

    resizeGL(w, h) {
        if (this.glCanvas) {
            // Check if resize needed
            if (this.glCanvas.width !== w || this.glCanvas.height !== h) {
                this.glCanvas.width = w;
                this.glCanvas.height = h;
                this.gl.viewport(0, 0, w, h);
            }
        }
    }

    resizeGPU(w, h) {
        if (this.gpuCanvas) {
            if (this.gpuCanvas.width !== w || this.gpuCanvas.height !== h) {
                this.gpuCanvas.width = w;
                this.gpuCanvas.height = h;
            }
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const dt = 0.016; // Fixed step for now

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
            // Update Uniforms: dt, time, mouseX, mouseY
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([dt, time, this.mouse.x, this.mouse.y]));

            const encoder = this.device.createCommandEncoder();

            // Compute Pass
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            // Render Pass
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
            // Draw 6 vertices per instance (quad) * numParticles instances
            rPass.draw(6, this.numParticles);
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);

        window.removeEventListener('resize', this.handleResize);
        if (this.container) {
            this.container.removeEventListener('mousemove', this.handleMouseMove);
            this.container.removeEventListener('touchmove', this.handleTouchMove);
        }

        // WebGL Cleanup
        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
        if (this.glCanvas) this.glCanvas.remove();

        // WebGPU Cleanup
        if (this.device) this.device.destroy();
        if (this.gpuCanvas) this.gpuCanvas.remove();

        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.BioluminescentAbyss = BioluminescentAbyss;
}
