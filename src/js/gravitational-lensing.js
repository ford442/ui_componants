/**
 * Gravitational Lensing Experiment
 * Hybrid WebGL2 (Lensed Starfield) + WebGPU (Accretion Disk)
 */

class GravitationalLensing {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // Interaction
        this.mouse = { x: 0, y: 0 }; // Normalized -1 to 1

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
        this.numParticles = options.numParticles || 100000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        // 1. WebGL2 Layer
        this.initWebGL2();

        // 2. WebGPU Layer
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("GravitationalLensing: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            const errEl = document.getElementById('webgpu-error');
            if (errEl) errEl.style.display = 'block';
        }

        // Events
        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);

        // Start
        this.isActive = true;
        this.resize(); // Ensure correct size
        this.animate();
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        // Map to -1 to 1
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1); // Flip Y for GL conventions
    }

    // ========================================================================
    // WebGL2 (Background Starfield & Lensing)
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

            // Simple hash for stars
            float random(vec2 st) {
                return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
            }

            float noise(vec2 st) {
                vec2 i = floor(st);
                vec2 f = fract(st);
                float a = random(i);
                float b = random(i + vec2(1.0, 0.0));
                float c = random(i + vec2(0.0, 1.0));
                float d = random(i + vec2(1.0, 1.0));
                vec2 u = f*f*(3.0-2.0*f);
                return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
            }

            void main() {
                // Aspect corrected UV for calculations
                float aspect = u_resolution.x / u_resolution.y;
                vec2 uv = v_uv;
                uv.x *= aspect;

                vec2 mouse = u_mouse;
                mouse.x *= aspect;

                // Black hole center
                vec2 center = mouse;
                vec2 toCenter = uv - center;
                float dist = length(toCenter);
                float radius = 0.2; // Event horizon radius approximation in visual space
                float strength = 0.05; // Lensing strength

                // Lensing distortion
                // Light bends towards the mass. So we look "outwards" from the mass to see what's behind it.
                // The distortion vector points towards the center.
                // New UV = UV - Distortion

                // Schwarzschild lensing approximation
                // theta_deflection = 4GM / (c^2 * b)  ~ 1/dist

                float distortion = strength / max(dist, 0.01);

                // Don't distort inside the event horizon (it's black)
                // But we simulate the distortion of the background *around* it.

                vec2 distortedUV = v_uv - normalize(toCenter) * distortion * (1.0 / aspect); // Back to UV space for texture lookup if we had one

                // Starfield generation based on Distorted UVs
                // We generate stars procedurally

                vec2 starUV = distortedUV * 20.0; // Tiling
                float stars = 0.0;

                if (random(floor(starUV)) > 0.98) {
                    vec2 centerOfTile = floor(starUV) + 0.5;
                    float d = length(starUV - centerOfTile);
                    stars = smoothstep(0.1, 0.05, d) * (0.5 + 0.5 * sin(u_time * 2.0 + random(centerOfTile)*10.0));
                }

                // Accretion disk glow (background layer part)
                float glow = exp(-dist * 2.0) * 0.5;
                vec3 glowColor = vec3(1.0, 0.6, 0.2) * glow;

                // Event Horizon (Shadow)
                // In reality, the shadow is roughly 2.6 * Schwarzschild radius
                float shadow = smoothstep(radius, radius + 0.02, dist);

                vec3 col = vec3(stars);
                col += glowColor;
                col *= shadow; // Black hole eats light

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
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fs));
            return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Accretion Disk Particles)
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

        // WGSL
        const computeCode = `
            struct Particle {
                pos: vec2f,
                vel: vec2f,
                life: f32,
                pad: f32,
            }

            struct Uniforms {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
                aspect: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];

                // Mouse/Blackhole position (aspect corrected)
                let bhPos = vec2f(uniforms.mouseX * uniforms.aspect, uniforms.mouseY);
                let pPosAspect = vec2f(p.pos.x * uniforms.aspect, p.pos.y);

                let toBH = bhPos - pPosAspect;
                let distSq = dot(toBH, toBH);
                let dist = sqrt(distSq);

                // Gravity: F = G*M / r^2
                // Force direction is normalize(toBH)
                // Acceleration = Force (mass=1)

                let G = 0.5; // Strength
                let forceMag = G / (distSq + 0.01);
                let forceDir = normalize(toBH);

                // Tangential velocity for orbit
                // Ideally v = sqrt(GM/r) for circular orbit

                // Add gravity to velocity
                p.vel = p.vel + forceDir * forceMag * uniforms.dt;

                // Drag (simulating friction in accretion disk)
                // Increases as we get closer
                let drag = 0.0 + 1.0 / (dist + 0.1) * 0.05 * uniforms.dt;
                p.vel = p.vel * (1.0 - drag);

                // Update position
                p.pos = p.pos + p.vel * uniforms.dt;

                // Event Horizon consumption or too far
                // Event Horizon radius approx 0.15 (visual matches GL)

                if (dist < 0.15 || dist > 3.0) {
                    // Reset particle to outer rim
                    let angle = rand(vec2f(p.pos.x, uniforms.time)) * 6.28318;
                    let radius = 1.5 + rand(vec2f(p.pos.y, f32(idx))) * 0.5;

                    let startPosAspect = bhPos + vec2f(cos(angle), sin(angle)) * radius;

                    // Convert aspect back to normalized coords
                    p.pos.x = startPosAspect.x / uniforms.aspect;
                    p.pos.y = startPosAspect.y;

                    // Initial orbital velocity: perpendicular to radius
                    // v = sqrt(GM/r)
                    let vMag = sqrt(G / radius);
                    let vDir = vec2f(-sin(angle), cos(angle)); // Tangent

                    // Add some randomness to spread the disk
                    p.vel = vDir * vMag * (0.8 + 0.4 * rand(vec2f(f32(idx), uniforms.time)));

                    p.life = 1.0;
                }

                particles[idx] = p;
            }
        `;

        const renderCode = `
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            struct Uniforms {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
                aspect: f32,
            }

            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            @vertex
            fn vs_main(@location(0) pPos: vec2f, @location(1) pVel: vec2f, @location(2) life: f32) -> VertexOut {
                var out: VertexOut;
                out.pos = vec4f(pPos, 0.0, 1.0);

                let speed = length(pVel);

                // Doppler shift / Heat coloring
                // Faster = Hotter (Blue/White), Slower = Cooler (Red)
                // Accretion disk inner edge is hot

                let t = smoothstep(0.0, 2.0, speed);
                let colHot = vec3f(0.8, 0.9, 1.0); // Blue-white
                let colCold = vec3f(1.0, 0.2, 0.0); // Red-orange

                out.color = vec4f(mix(colCold, colHot, t), 1.0);

                // Point size handled by primitive topology (1px usually)
                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialization
        const data = new Float32Array(this.numParticles * 6); // pos(2), vel(2), life(1), pad(1)
        // Stride is 6 floats = 24 bytes. But we need alignment usually?
        // Wait, vec2f is 8 bytes.
        // struct Particle { pos: vec2f, vel: vec2f, life: f32, pad: f32 } -> 8+8+4+4 = 24 bytes?
        // WGSL struct alignment is usually 16 bytes for vec3/vec4.
        // vec2f is 8 byte align.
        // Total size 24 bytes is fine if array stride is 24?
        // Actually, let's check alignment rules.
        // struct Particle { pos: vec2f, vel: vec2f, life: f32, pad: f32 }
        // pos: offset 0
        // vel: offset 8
        // life: offset 16
        // pad: offset 20
        // Size 24.

        // However, vertex buffer stride must be multiple of 4. 24 is ok.

        for (let i = 0; i < this.numParticles; i++) {
            // Random start positions
            data[i*6+0] = (Math.random() * 2 - 1);
            data[i*6+1] = (Math.random() * 2 - 1);
            data[i*6+2] = 0; // vx
            data[i*6+3] = 0; // vy
            data[i*6+4] = 1; // life
            data[i*6+5] = 0; // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, data);

        this.uniformBuffer = this.device.createBuffer({
            size: 32, // 5 floats = 20 bytes -> round up to 32? Uniform buffer size must be multiple of 16 bytes? No, but binding size usually aligned.
            // struct Uniforms { dt, time, mouseX, mouseY, aspect } -> 5 floats.
            // WGSL Uniform struct alignment: 16 bytes.
            // Members: f32 (4), f32 (4), f32 (4), f32 (4), f32 (4).
            // It will be packed as:
            // vec4 (dt, time, mouseX, mouseY)
            // float (aspect)
            // padding to 32 bytes (16 aligned size? No, 16 aligned start).
            // Size 20 bytes.
            // createBuffer size must be multiple of 4.
            // But writing to it... let's allocate 32 bytes (8 floats) to be safe.
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
                    arrayStride: 24,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' }, // pos
                        { shaderLocation: 1, offset: 8, format: 'float32x2' }, // vel
                        { shaderLocation: 2, offset: 16, format: 'float32' },  // life
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

        this.resizeGPU(this.container.clientWidth, this.container.clientHeight);
        return true;
    }

    resize() {
        if (!this.isActive) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        const dw = Math.floor(w * dpr);
        const dh = Math.floor(h * dpr);

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

    // Explicit helper required if called internally with specific dims
    resizeGPU(w, h) {
       // Logic handled in resize() generally, but if called directly:
       if (this.gpuCanvas) {
           this.gpuCanvas.width = w;
           this.gpuCanvas.height = h;
       }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const dt = 0.016; // Fixed step for stability

        // WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);

            this.gl.clearColor(0, 0, 0, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // WebGPU
        if (this.device && this.renderPipeline) {
            const aspect = this.glCanvas ? (this.glCanvas.width / this.glCanvas.height) : 1.0;
            const uniforms = new Float32Array([dt, time, this.mouse.x, this.mouse.y, aspect, 0, 0, 0]); // Padding

            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

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
            rPass.setBindGroup(0, this.computeBindGroup); // Need bind group for uniforms in vertex shader too
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

        // Context lost handling not strictly necessary for simple examples but good practice
    }
}

if (typeof window !== 'undefined') {
    window.GravitationalLensing = GravitationalLensing;
}
