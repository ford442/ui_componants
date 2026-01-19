/**
 * Subatomic Collider Experiment
 * A hybrid WebGL2 + WebGPU visualization of a high-energy particle detector.
 */

export class SubatomicColliderExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.canvasSize = { width: 0, height: 0 };
        this.mouse = { x: 0, y: 0 };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.torusIndexCount = 0;
        this.numRings = 20;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 100000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020205';

        // 1. Initialize WebGL2 (Detector Structure)
        this.initWebGL2();

        // 2. Initialize WebGPU (Particle Physics)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("SubatomicCollider: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();
        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);

        // Handle touch for mobile
        this.container.addEventListener('touchmove', (e) => {
            if (e.touches && e.touches[0]) {
                const rect = this.container.getBoundingClientRect();
                this.mouse.x = (e.touches[0].clientX - rect.left) / rect.width * 2 - 1;
                this.mouse.y = -((e.touches[0].clientY - rect.top) / rect.height * 2 - 1);
            }
        }, { passive: true });
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Detector Rings)
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

        // Create Torus Geometry
        const { vertices, indices } = this.createTorus(4.0, 0.2, 32, 16);
        this.torusIndexCount = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const idxBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Shader
        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform vec2 u_resolution;
            uniform float u_zOffset; // Offset for this ring instance

            out float v_depth;
            out vec3 v_normal;

            mat4 rotateZ(float angle) {
                float s = sin(angle);
                float c = cos(angle);
                return mat4(c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1);
            }

            void main() {
                vec3 pos = a_position;

                // Position ring
                pos.z += u_zOffset;

                // Rotate ring slightly based on time and ID
                float angle = u_time * 0.5 + u_zOffset * 0.1;
                pos = (rotateZ(angle) * vec4(pos, 1.0)).xyz;

                // Simple perspective
                float zDist = 15.0 - pos.z;
                float fov = 1.0;
                float aspect = u_resolution.x / u_resolution.y;

                gl_Position = vec4(pos.x / aspect, pos.y, pos.z, zDist * 0.5);

                v_depth = pos.z;
                v_normal = normalize(a_position); // Approximate normal for torus
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            in vec3 v_normal;
            uniform float u_time;
            out vec4 outColor;

            void main() {
                // Fade distant rings
                float alpha = smoothstep(-20.0, 5.0, v_depth);

                // Lighting
                vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
                float diff = max(dot(v_normal, lightDir), 0.0);

                // Base metal color
                vec3 color = vec3(0.2, 0.2, 0.25) + vec3(0.1) * diff;

                // Glowing sensors
                float strip = step(0.9, sin(atan(v_normal.y, v_normal.x) * 10.0 + u_time * 5.0));
                vec3 glow = vec3(1.0, 0.6, 0.2) * strip;

                outColor = vec4(color + glow, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (this.glProgram) {
            const loc = this.gl.getAttribLocation(this.glProgram, 'a_position');
            this.gl.enableVertexAttribArray(loc);
            this.gl.vertexAttribPointer(loc, 3, this.gl.FLOAT, false, 0, 0);
        }
        this.glVao = vao;
    }

    createTorus(radius, tube, radialSegments, tubularSegments) {
        const vertices = [];
        const indices = [];

        for (let j = 0; j <= radialSegments; j++) {
            for (let i = 0; i <= tubularSegments; i++) {
                const u = (i / tubularSegments) * Math.PI * 2;
                const v = (j / radialSegments) * Math.PI * 2;

                const x = (radius + tube * Math.cos(v)) * Math.cos(u);
                const y = (radius + tube * Math.cos(v)) * Math.sin(u);
                const z = tube * Math.sin(v);

                vertices.push(x, y, z);
            }
        }

        for (let j = 1; j <= radialSegments; j++) {
            for (let i = 1; i <= tubularSegments; i++) {
                const a = (tubularSegments + 1) * j + i;
                const b = (tubularSegments + 1) * (j - 1) + i;
                const c = (tubularSegments + 1) * (j - 1) + i - 1;
                const d = (tubularSegments + 1) * j + i - 1;

                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }
        return { vertices, indices };
    }

    createGLProgram(vs, fs) {
        const p = this.gl.createProgram();
        const v = this.gl.createShader(this.gl.VERTEX_SHADER);
        const f = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(v, vs);
        this.gl.shaderSource(f, fs);
        this.gl.compileShader(v);
        this.gl.compileShader(f);
        this.gl.attachShader(p, v);
        this.gl.attachShader(p, f);
        this.gl.linkProgram(p);
        if (!this.gl.getProgramParameter(p, this.gl.LINK_STATUS)) {
            console.error(this.gl.getProgramInfoLog(p));
            return null;
        }
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Particle Physics)
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
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // Use vec4f for strict 16-byte alignment of struct members
        // Particle struct size = 32 bytes (16 + 16)
        const computeCode = `
            struct Particle {
                pos: vec4f, // xyz, w = pad
                vel: vec4f, // xyz, w = life
            }

            struct Params {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: Params;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];
                var life = p.vel.w;

                if (life <= 0.0) {
                    // Respawn burst from center
                    let theta = rand(vec2f(params.time, f32(i))) * 6.28;
                    let phi = rand(vec2f(f32(i), params.time)) * 3.14;
                    let speed = 20.0 + rand(vec2f(p.pos.x, p.pos.y)) * 30.0;

                    p.pos = vec4f(0.0, 0.0, 0.0, 0.0);
                    p.vel = vec4f(
                        sin(phi) * cos(theta) * speed,
                        sin(phi) * sin(theta) * speed,
                        cos(phi) * speed,
                        1.0 + rand(vec2f(f32(i), f32(i))) * 0.5 // New life
                    );
                } else {
                    // Lorentz force F = q(v x B)
                    // Magnetic field B controlled by mouse
                    let B = vec3f(params.mouseX * 5.0, params.mouseY * 5.0, 2.0);
                    let vel3 = p.vel.xyz;
                    let force = cross(vel3, B);

                    vel3 += force * params.dt * 2.0;

                    p.pos = vec4f(p.pos.xyz + vel3 * params.dt, p.pos.w);
                    p.vel = vec4f(vel3, life - params.dt * 0.5);
                }
                particles[i] = p;
            }
        `;

        const renderCode = `
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            struct Params {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
                aspect: f32, // Struct alignment note: 5 floats -> 20 bytes. Padded to 32 in JS.
            }

            @group(0) @binding(1) var<uniform> params: Params;

            @vertex
            fn vs_main(
                @location(0) pos: vec3f,
                @location(1) vel: vec3f,
                @location(2) life: f32
            ) -> VertexOut {
                var out: VertexOut;

                // Simple perspective
                let zDist = 15.0 - pos.z;
                let scale = 1.0 / max(0.1, zDist); // Avoid div by zero

                out.pos = vec4f(pos.x * scale / params.aspect, pos.y * scale, 0.0, 1.0);

                // Color based on velocity
                let speed = length(vel);
                let heat = smoothstep(20.0, 50.0, speed);
                out.color = mix(
                    vec4f(0.0, 0.5, 1.0, 1.0), // Blue (slow)
                    vec4f(1.0, 0.8, 0.5, 1.0), // Orange/White (fast)
                    heat
                );
                out.color.a *= life;

                // Clip if behind
                if (zDist < 1.0) { out.pos = vec4f(2.0, 2.0, 2.0, 1.0); }

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffer Setup
        // JS Data: [x, y, z, pad, vx, vy, vz, life] = 8 floats = 32 bytes
        const pSize = 32;
        const initialData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            // Init with zeroes/defaults
            initialData[i*8 + 7] = 0.0; // life starts at 0 to trigger respawn
        }

        this.particleBuffer = this.device.createBuffer({
            size: this.numParticles * pSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        // 5 floats needed (dt, time, mx, my, aspect) -> 20 bytes.
        // Round up to 32 bytes for alignment safety
        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ]
        });

        const cMod = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
            compute: { module: cMod, entryPoint: 'main' },
        });

        const rMod = this.device.createShaderModule({ code: renderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
            vertex: {
                module: rMod,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: pSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' },  // pos (x,y,z)
                        { shaderLocation: 1, offset: 16, format: 'float32x3' }, // vel (vx,vy,vz) - skip pad (4 bytes)
                        { shaderLocation: 2, offset: 28, format: 'float32' },   // life (in vel.w) - offset 16+12=28
                    ]
                }]
            },
            fragment: {
                module: rMod,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `position: absolute; bottom: 20px; right: 20px; color: red; font-family: monospace;`;
        msg.innerText = 'WebGPU Not Available';
        this.container.appendChild(msg);
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (w === 0 || h === 0) return;
        this.canvasSize = { width: w, height: h };

        if (this.glCanvas) {
            this.glCanvas.width = w;
            this.glCanvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = w;
            this.gpuCanvas.height = h;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const aspect = this.canvasSize.width / this.canvasSize.height;

        // Render WebGL2 (Tunnel)
        if (this.gl && this.glProgram) {
            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
            this.gl.useProgram(this.glProgram);

            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.canvasSize.width, this.canvasSize.height);

            const zLoc = this.gl.getUniformLocation(this.glProgram, 'u_zOffset');

            this.gl.bindVertexArray(this.glVao);

            // Draw multiple rings
            for (let i = 0; i < this.numRings; i++) {
                // Modulo math for infinite tunnel
                let z = (i * 2.0 - time * 5.0) % (this.numRings * 2.0);
                if (z > 0) z -= this.numRings * 2.0;

                this.gl.uniform1f(zLoc, z);
                this.gl.drawElements(this.gl.TRIANGLES, this.torusIndexCount, this.gl.UNSIGNED_SHORT, 0);
            }
        }

        // Render WebGPU (Particles)
        if (this.device && this.renderPipeline) {
            // Update Uniforms
            const params = new Float32Array([
                0.016, // dt
                time,
                this.mouse.x,
                this.mouse.y,
                aspect,
                0, 0, 0 // Padding to 32 bytes
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const cmd = this.device.createCommandEncoder();

            // Compute Pass
            const cp = cmd.beginComputePass();
            cp.setPipeline(this.computePipeline);
            cp.setBindGroup(0, this.computeBindGroup);
            cp.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cp.end();

            // Render Pass
            const rp = cmd.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            rp.setPipeline(this.renderPipeline);
            rp.setBindGroup(0, this.computeBindGroup);
            rp.setVertexBuffer(0, this.particleBuffer);
            rp.draw(this.numParticles);
            rp.end();

            this.device.queue.submit([cmd.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.SubatomicColliderExperiment = SubatomicColliderExperiment;
}
