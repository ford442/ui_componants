
export class HolographicText {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            text: options.text || 'HYBRID',
            particleSize: options.particleSize || 2.0,
            color: options.color || [0.0, 1.0, 1.0], // Cyan
            ...options
        };

        this.width = container.clientWidth;
        this.height = container.clientHeight;

        // State
        this.time = 0;
        this.mouse = { x: 0, y: 0, active: false };
        this.targets = null;

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.backgroundColor = '#050510';

        // 1. Generate Targets
        this.targets = this.generateTextTargets(this.options.text);
        this.numParticles = this.targets.length / 3;
        console.log(`HolographicText: Generated ${this.numParticles} target points.`);

        // 2. WebGL2 Layer (Projector Base)
        this.initWebGL2();

        // 3. WebGPU Layer (Particles)
        let gpuSuccess = false;
        if (navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.error("HolographicText: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.showError("WebGPU Not Supported - Rendering Base Only");
        }

        // Events
        this.container.addEventListener('mousemove', e => {
            const rect = this.container.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * 2 - 1;
            const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
            this.mouse = { x, y, active: true };
        });

        this.container.addEventListener('mouseleave', () => {
            this.mouse.active = false;
        });

        window.addEventListener('resize', () => this.resize());

        // Start Loop
        this.animate();
    }

    generateTextTargets(text) {
        const w = 512;
        const h = 256;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, w, h);

        ctx.font = 'bold 120px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#fff';
        ctx.fillText(text, w / 2, h / 2);

        const data = ctx.getImageData(0, 0, w, h).data;
        const targets = [];
        const stride = 3; // Density control

        for (let y = 0; y < h; y += stride) {
            for (let x = 0; x < w; x += stride) {
                const i = (y * w + x) * 4;
                if (data[i] > 128) {
                    // Map to -1.0 to 1.0
                    // Preserve aspect ratio of text block
                    const nx = (x / w) * 2 - 1;
                    const ny = -((y / h) * 2 - 1); // Flip Y
                    targets.push(nx * 1.5, ny * 0.75, 0); // Scale to fill screen nicely
                }
            }
        }
        return new Float32Array(targets);
    }

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;';
        this.container.appendChild(this.glCanvas);
        this.glCanvas.width = this.width;
        this.glCanvas.height = this.height;

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        const gl = this.gl;
        gl.viewport(0, 0, this.width, this.height);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // Additive

        // Simple Grid Shader
        const vs = `#version 300 es
        in vec3 a_pos;
        uniform float u_time;
        uniform mat4 u_proj;
        uniform mat4 u_view;
        out vec3 v_pos;
        void main() {
            vec3 pos = a_pos;
            // Rotate grid slowly
            float c = cos(u_time * 0.1);
            float s = sin(u_time * 0.1);
            float x = pos.x * c - pos.z * s;
            float z = pos.x * s + pos.z * c;
            pos.x = x; pos.z = z;

            v_pos = pos;
            gl_Position = u_proj * u_view * vec4(pos, 1.0);
        }`;

        const fs = `#version 300 es
        precision highp float;
        in vec3 v_pos;
        uniform float u_time;
        out vec4 color;
        void main() {
            // Distance fade
            float d = length(v_pos.xz);
            float alpha = smoothstep(2.0, 0.0, d);

            // Grid lines
            float grid = step(0.98, fract(v_pos.x * 5.0)) + step(0.98, fract(v_pos.z * 5.0));

            // Scanning line
            float scan = step(0.95, fract(v_pos.z * 0.5 + u_time * 0.5));

            vec3 c = vec3(0.0, 0.5, 0.5) * grid + vec3(0.0, 1.0, 1.0) * scan * 0.5;
            color = vec4(c, alpha * 0.5);
        }`;

        this.glProgram = this.createProgram(gl, vs, fs);

        // Grid Geometry (Plane)
        const vertices = [
            -2, -1, -2,  2, -1, -2,
            -2, -1,  2,  2, -1,  2
        ];
        // Triangle Strip

        this.glVao = gl.createVertexArray();
        gl.bindVertexArray(this.glVao);

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        const loc = gl.getAttribLocation(this.glProgram, 'a_pos');
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }

    createProgram(gl, vsSrc, fsSrc) {
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSrc);
        gl.compileShader(vs);
        if(!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vs));

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSrc);
        gl.compileShader(fs);
        if(!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fs));

        const p = gl.createProgram();
        gl.attachShader(p, vs);
        gl.attachShader(p, fs);
        gl.linkProgram(p);
        return p;
    }

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:2;pointer-events:none;';
        this.container.appendChild(this.gpuCanvas);
        this.gpuCanvas.width = this.width;
        this.gpuCanvas.height = this.height;

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.ctx = this.gpuCanvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();

        this.ctx.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied'
        });

        // 1. Particle Buffers
        // Data: [pos.xyz, life, vel.xyz, pad] -> 32 bytes
        // Target: [pos.xyz, pad] -> 16 bytes
        const pData = new Float32Array(this.numParticles * 8);
        const tData = new Float32Array(this.numParticles * 4);

        for(let i=0; i<this.numParticles; i++) {
            // Start at bottom
            pData[i*8+0] = (Math.random()-0.5)*0.5;
            pData[i*8+1] = -1.0;
            pData[i*8+2] = (Math.random()-0.5)*0.5;
            pData[i*8+3] = 1.0; // Life

            pData[i*8+4] = (Math.random()-0.5)*0.01; // vx
            pData[i*8+5] = (Math.random()*0.05);     // vy up
            pData[i*8+6] = (Math.random()-0.5)*0.01; // vz
            pData[i*8+7] = 0; // pad

            // Target
            tData[i*4+0] = this.targets[i*3+0];
            tData[i*4+1] = this.targets[i*3+1];
            tData[i*4+2] = this.targets[i*3+2];
            tData[i*4+3] = 0; // pad
        }

        this.particleBuffer = this.createBuffer(pData, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);
        this.targetBuffer = this.createBuffer(tData, GPUBufferUsage.STORAGE);

        // Uniforms
        this.uniformBuffer = this.device.createBuffer({
            size: 32, // time, dt, mouseX, mouseY ...
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Shader
        const computeCode = `
            struct Particle {
                pos: vec3f,
                life: f32,
                vel: vec3f,
                pad: f32
            }
            struct Uniforms {
                time: f32,
                dt: f32,
                mouseX: f32,
                mouseY: f32,
                active: f32,
                pad: f32,
                pad2: f32,
                pad3: f32
            }
            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<storage, read> targets: array<vec4f>; // xyz, pad
            @group(0) @binding(2) var<uniform> u: Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];
                let target = targets[i].xyz;

                // Spring force to target
                let diff = target - p.pos;
                let dist = length(diff);
                let dir = normalize(diff);

                // Attraction
                let force = dir * dist * 2.0;

                // Mouse Repulsion
                if (u.active > 0.5) {
                    let mPos = vec3f(u.mouseX, u.mouseY, 0.0);
                    let mDiff = p.pos - mPos;
                    let mDist = length(mDiff);
                    if (mDist < 0.5) {
                        let repel = normalize(mDiff) * (1.0 - mDist/0.5) * 10.0;
                        p.vel += repel * u.dt;
                    }
                }

                // Random Noise movement (idle jitter)
                let noise = sin(u.time * 10.0 + f32(i)) * 0.05;

                p.vel += force * u.dt;
                p.vel *= 0.92; // Damping

                p.pos += p.vel * u.dt;

                // Floor collision (reset if fell too far?)
                if(p.pos.y < -2.0) { p.pos.y = -2.0; p.vel.y = -p.vel.y * 0.5; }

                particles[i] = p;
            }
        `;

        // Render Shader
        const renderCode = `
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f
            }
            struct Uniforms {
                time: f32,
                dt: f32,
                mouseX: f32,
                mouseY: f32
            }
            @group(0) @binding(0) var<uniform> u: Uniforms;
            @group(0) @binding(1) var<uniform> proj: mat4x4f;

            @vertex
            fn vs(@location(0) pos: vec3f, @location(1) life: f32, @location(2) vel: vec3f) -> VertexOut {
                var out: VertexOut;
                out.pos = proj * vec4f(pos, 1.0);

                // Color based on velocity
                let speed = length(vel);
                let c = mix(vec3f(0.0, 1.0, 1.0), vec3f(1.0, 0.0, 1.0), min(speed * 2.0, 1.0));

                // Size
                let dist = out.pos.w;
                // Simple point size approximation not available in WGSL vertex stage for 'point-list'
                // But we are drawing point-list topology so RenderPipeline handles point size?
                // No, WebGPU doesn't support gl_PointSize equivalent directly in all implementations
                // without extension or using quads.
                // However, standard 'point-list' usually renders 1px points.
                // To get larger points, we must use instance rendering (quads).
                // But for simplicity in this prototype, let's assume 1px points are fine
                // OR upgrade to quads.
                // Given the 'Hologram' vibe, 1px points might be too faint.
                // Let's stick to 1px for now to keep it simple and ensure it compiles.
                // If invisible, I'll switch to quads.

                out.pos.w = max(out.pos.w, 0.001); // Prevent div by zero
                out.color = vec4f(c, 0.8);
                return out;
            }

            @fragment
            fn fs(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Wait, 1px points are hard to see. Let's use Triangle List Quads for proper billboarding.
        // Or actually, let's use the 'point-list' with a simpler shader first.
        // Actually, many WebGPU demos use instance rendering for particles.
        // I'll stick to 'point-list' for now. If it's too small, I'll update.
        // NOTE: WebGPU default point size is 1.0.

        const cMod = this.device.createShaderModule({ code: computeCode });
        const rMod = this.device.createShaderModule({ code: renderCode });

        // Pipelines
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: cMod, entryPoint: 'main' }
        });

        // Need ViewProj matrix buffer
        this.viewProjBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const renderLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
            vertex: {
                module: rMod,
                entryPoint: 'vs',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' }, // pos
                        { shaderLocation: 1, offset: 12, format: 'float32' },  // life
                        { shaderLocation: 2, offset: 16, format: 'float32x3' } // vel
                    ]
                }]
            },
            fragment: {
                module: rMod, entryPoint: 'fs',
                targets: [{
                    format: this.format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        // Bind Groups
        this.computeBG = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.targetBuffer } },
                { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBG = this.device.createBindGroup({
            layout: renderLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.viewProjBuffer } }
            ]
        });

        return true;
    }

    createBuffer(data, usage) {
        const buf = this.device.createBuffer({
            size: data.byteLength,
            usage: usage | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(buf.getMappedRange()).set(data);
        buf.unmap();
        return buf;
    }

    showError(msg) {
        const d = document.createElement('div');
        d.style.cssText = 'position:absolute;bottom:10px;right:10px;background:red;color:white;padding:5px;font-family:sans-serif;font-size:12px;';
        d.textContent = msg;
        this.container.appendChild(d);
    }

    resize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        if(this.glCanvas) {
            this.glCanvas.width = this.width;
            this.glCanvas.height = this.height;
            this.gl.viewport(0, 0, this.width, this.height);
        }
        if(this.gpuCanvas) {
            this.gpuCanvas.width = this.width;
            this.gpuCanvas.height = this.height;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        const now = performance.now();
        const dt = Math.min((now - this.time) * 0.001, 0.1); // Cap dt
        this.time = now;
        const timeSec = now * 0.001;

        // Render WebGL2
        if(this.gl) {
            const gl = this.gl;
            gl.clearColor(0,0,0,0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.useProgram(this.glProgram);

            // Projection
            const aspect = this.width / this.height;
            const fov = 60 * Math.PI / 180;
            const f = 1.0 / Math.tan(fov/2);
            const near = 0.1, far = 100.0;
            const proj = new Float32Array([
                f/aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (far+near)/(near-far), -1,
                0, 0, (2*far*near)/(near-far), 0
            ]);

            // View (Camera at 0, 0, 4)
            const view = new Float32Array([
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -4, 1
            ]);

            const uTime = gl.getUniformLocation(this.glProgram, 'u_time');
            const uProj = gl.getUniformLocation(this.glProgram, 'u_proj');
            const uView = gl.getUniformLocation(this.glProgram, 'u_view');

            gl.uniform1f(uTime, timeSec);
            gl.uniformMatrix4fv(uProj, false, proj);
            gl.uniformMatrix4fv(uView, false, view);

            gl.bindVertexArray(this.glVao);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            // Update WebGPU matrices
            if(this.device) {
                // Compute viewProj for WebGPU
                // Simple matrix mult for proj * view
                const vp = new Float32Array(16);
                // Since view is just translation z=-4, VP is just P with modified translation col
                // Col 2 (z) of view is unaffected. Col 3 (w) becomes modified.
                // P * V
                // [P00 0   0   0 ]   [1 0 0  0]
                // [0   P11 0   0 ] * [0 1 0  0]
                // [0   0   P22 P23]  [0 0 1  0]
                // [0   0   P32 0 ]   [0 0 -4 1]

                // Result col 3:
                // R03 = P00*0 + ... = 0
                // R13 = 0
                // R23 = P20*0 + P21*0 + P22*-4 + P23*1
                // R33 = P30*0 + ... + P32*-4 + 0

                vp.set(proj);
                vp[14] = proj[10] * -4.0 + proj[11]; // z
                vp[15] = proj[14] * -4.0 + 0.0;      // w - WAIT. proj[14] is P32.
                // Matrix indices:
                // 0  4  8  12
                // 1  5  9  13
                // 2  6  10 14
                // 3  7  11 15
                // P32 is index 14. P23 is index 11. P22 is index 10.

                // Correct math:
                // V is translation T(0,0,-4).
                // M[12] = 0, M[13] = 0, M[14] = -4.
                // Res[12] = P[0]*x + P[4]*y + P[8]*z + P[12]*1 -> P[12]
                // Res[13] = P[13]
                // Res[14] = P[10]*-4 + P[14]
                // Res[15] = P[11]*-4 + P[15]  -> P[11] is -1. P[15] is 0. -> 4.

                vp[12] = proj[12];
                vp[13] = proj[13];
                vp[14] = proj[10] * -4.0 + proj[14];
                vp[15] = proj[11] * -4.0 + proj[15];

                this.device.queue.writeBuffer(this.viewProjBuffer, 0, vp);
            }
        }

        // Render WebGPU
        if(this.device && this.renderPipeline) {
            // Update Uniforms
            const uData = new Float32Array([
                timeSec, dt, this.mouse.x, this.mouse.y,
                this.mouse.active ? 1.0 : 0.0, 0, 0, 0
            ]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uData);

            const enc = this.device.createCommandEncoder();

            // Compute
            const cPass = enc.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBG);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles/64));
            cPass.end();

            // Render
            const view = this.ctx.getCurrentTexture().createView();
            const rPass = enc.beginRenderPass({
                colorAttachments: [{
                    view: view,
                    clearValue: {r:0,g:0,b:0,a:0},
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBG);
            rPass.setVertexBuffer(0, this.particleBuffer);
            rPass.draw(this.numParticles);
            rPass.end();

            this.device.queue.submit([enc.finish()]);
        }
    }
}
