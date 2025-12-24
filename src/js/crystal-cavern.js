/**
 * Crystal Cavern Experiment
 * Hybrid Rendering:
 * - WebGL2: Raymarched Crystal Cluster (Background)
 * - WebGPU: Bio-luminescent Spores (Foreground)
 */

class CrystalCavern {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;
        this.numParticles = options.numParticles || 20000;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

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

        this.handleResize = this.resize.bind(this);
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050010'; // Deep dark purple

        // 1. WebGL2 Layer
        this.initWebGL2();

        // 2. WebGPU Layer
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CrystalCavern: WebGPU error:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();
        window.addEventListener('resize', this.handleResize);
    }

    // ========================================================================
    // WebGL2 (Raymarching Crystals)
    // ========================================================================
    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        const buf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const vs = `#version 300 es
            in vec2 a_pos;
            out vec2 v_uv;
            void main() {
                v_uv = a_pos;
                gl_Position = vec4(a_pos, 0.0, 1.0);
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_res;
            out vec4 outColor;

            // Rotation matrix
            mat2 rot(float a) {
                float s = sin(a), c = cos(a);
                return mat2(c, -s, s, c);
            }

            // SDF for an Octahedron (Crystal shape)
            float sdOctahedron(vec3 p, float s) {
                p = abs(p);
                return (p.x + p.y + p.z - s) * 0.57735027;
            }

            float map(vec3 p) {
                vec3 p1 = p;
                p1.xz *= rot(u_time * 0.2);
                p1.xy *= rot(u_time * 0.1);

                // Main crystal
                float d = sdOctahedron(p1, 1.5);

                // Clusters
                vec3 p2 = p;
                p2.y -= 2.0;
                p2.xz *= rot(u_time * -0.3);
                d = min(d, sdOctahedron(p2, 1.0));

                vec3 p3 = p;
                p3.y += 2.0;
                p3.xz *= rot(u_time * 0.4);
                d = min(d, sdOctahedron(p3, 1.0));

                return d;
            }

            vec3 getNormal(vec3 p) {
                vec2 e = vec2(0.001, 0.0);
                return normalize(vec3(
                    map(p + e.xyy) - map(p - e.xyy),
                    map(p + e.yxy) - map(p - e.yxy),
                    map(p + e.yyx) - map(p - e.yyx)
                ));
            }

            void main() {
                vec2 uv = v_uv;
                uv.x *= u_res.x / u_res.y;

                vec3 ro = vec3(0.0, 0.0, -5.0);
                vec3 rd = normalize(vec3(uv, 1.0));

                float t = 0.0;
                float d = 0.0;
                int i = 0;

                // Raymarching
                for(i=0; i<64; i++) {
                    vec3 p = ro + rd * t;
                    d = map(p);
                    t += d;
                    if(d < 0.001 || t > 20.0) break;
                }

                vec3 col = vec3(0.05, 0.0, 0.1); // Background

                if(d < 0.001) {
                    vec3 p = ro + rd * t;
                    vec3 n = getNormal(p);
                    vec3 light = normalize(vec3(1.0, 2.0, -2.0));

                    // Crystal material
                    float diff = max(dot(n, light), 0.0);
                    float spec = pow(max(dot(reflect(-light, n), -rd), 0.0), 32.0);
                    float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);

                    vec3 baseCol = vec3(0.4, 0.0, 0.8); // Purple
                    col = baseCol * (diff * 0.5 + 0.5) + vec3(1.0)*spec + vec3(0.5, 0.8, 1.0)*fresnel;
                }

                // Vignette
                col *= 1.0 - length(v_uv * 0.5);

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createProgram(vs, fs);
        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const loc = this.gl.getAttribLocation(this.glProgram, 'a_pos');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 2, this.gl.FLOAT, false, 0, 0);

        this.resizeGL();
    }

    createProgram(vsSrc, fsSrc) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSrc);
        this.gl.compileShader(vs);
        if(!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) console.error(this.gl.getShaderInfoLog(vs));

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSrc);
        this.gl.compileShader(fs);
        if(!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) console.error(this.gl.getShaderInfoLog(fs));

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Spores)
    // ========================================================================
    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2; pointer-events: none; background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) { this.gpuCanvas.remove(); return false; }

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        const computeShader = `
            struct Particle {
                pos: vec3f,
                vel: vec3f,
                life: f32,
            }
            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

            struct Params {
                dt: f32,
                time: f32,
            }
            @group(0) @binding(1) var<uniform> params: Params;

            // Simple hash for randomness
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= ${this.numParticles}) { return; }

                var p = particles[i];

                // Orbit logic
                let dist = length(p.pos.xyz);
                let center = vec3f(0.0);
                let dir = normalize(center - p.pos);

                // Tangent force (swirl)
                let tangent = normalize(cross(vec3f(0.0, 1.0, 0.0), p.pos));

                p.vel += (dir * 0.5 + tangent * 0.5) * params.dt * 2.0;
                p.pos += p.vel * params.dt;

                // Damping
                p.vel *= 0.98;

                // Reset if too close or far
                if (dist < 0.5 || dist > 4.0) {
                   let r = rand(vec2f(f32(i), params.time));
                   let theta = r * 6.28;
                   let phi = rand(vec2f(params.time, f32(i))) * 3.14;

                   p.pos = vec3f(
                       sin(phi) * cos(theta),
                       sin(phi) * sin(theta),
                       cos(phi)
                   ) * 3.5;
                   p.vel = vec3f(0.0);
                }

                particles[i] = p;
            }
        `;

        const drawShader = `
            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs(@location(0) pos: vec3f) -> VertexOutput {
                var out: VertexOutput;
                // Simple perspective projection (manual)
                let z = pos.z - 5.0; // Camera z offset
                let projX = pos.x / -z * 2.0; // FOV approx
                let projY = pos.y / -z * 2.0; // Aspect ratio assumed square-ish here but handled in CSS/canvas size

                out.pos = vec4f(projX, projY, 0.0, 1.0);

                // Distance fade
                let dist = length(pos);
                let alpha = smoothstep(4.0, 1.0, dist);
                out.color = vec4f(0.5, 1.0, 0.8, alpha);
                return out;
            }

            @fragment
            fn fs(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        // Particle struct: pos(3), vel(3), life(1), padding(1) -> 8 floats = 32 bytes
        const particleSize = 32;
        const totalSize = this.numParticles * particleSize;
        const initData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            // Random start pos on sphere surface
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 3.0 + Math.random();

            initData[i*8+0] = r * Math.sin(phi) * Math.cos(theta); // x
            initData[i*8+1] = r * Math.sin(phi) * Math.sin(theta); // y
            initData[i*8+2] = r * Math.cos(phi); // z
            // vel 0, life 0
        }

        this.particleBuffer = this.device.createBuffer({
            size: totalSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 2 floats + padding to 16 bytes alignment often safer
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const bgLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bgLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }),
            compute: { module: this.device.createShaderModule({ code: computeShader }), entryPoint: 'main' }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: drawShader }),
                entryPoint: 'vs',
                buffers: [{
                    arrayStride: particleSize,
                    stepMode: 'vertex',
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] // just pos
                }]
            },
            fragment: {
                module: this.device.createShaderModule({ code: drawShader }),
                entryPoint: 'fs',
                targets: [{ format, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                } }]
            },
            primitive: { topology: 'point-list' }
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px;
            background: rgba(100, 20, 20, 0.9); color: white;
            padding: 8px 16px; border-radius: 8px; pointer-events: none;
        `;
        msg.innerHTML = "⚠️ WebGPU Not Available (WebGL2 Only)";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Common
    // ========================================================================
    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = Math.floor(this.container.clientWidth * dpr);
        const h = Math.floor(this.container.clientHeight * dpr);
        this.resizeGL(w, h);
        this.resizeGPU(w, h);
    }

    resizeGL(w, h) {
        if (this.glCanvas) {
            this.glCanvas.width = w || this.glCanvas.width;
            this.glCanvas.height = h || this.glCanvas.height;
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
        }
    }

    resizeGPU(w, h) {
        if (this.gpuCanvas) {
            this.gpuCanvas.width = w || this.gpuCanvas.width;
            this.gpuCanvas.height = h || this.gpuCanvas.height;
        }
    }

    animate() {
        if (!this.isActive) return;
        const time = (Date.now() - this.startTime) * 0.001;

        // Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_res'), this.glCanvas.width, this.glCanvas.height);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGPU
        if (this.device && this.renderPipeline) {
            this.device.queue.writeBuffer(this.simParamBuffer, 0, new Float32Array([0.016, time]));
            const enc = this.device.createCommandEncoder();

            const cmp = enc.beginComputePass();
            cmp.setPipeline(this.computePipeline);
            cmp.setBindGroup(0, this.computeBindGroup);
            cmp.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cmp.end();

            const tex = this.context.getCurrentTexture().createView();
            const pass = enc.beginRenderPass({
                colorAttachments: [{
                    view: tex,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear', storeOp: 'store'
                }]
            });
            pass.setPipeline(this.renderPipeline);
            pass.setVertexBuffer(0, this.particleBuffer);
            pass.draw(this.numParticles);
            pass.end();

            this.device.queue.submit([enc.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.CrystalCavern = CrystalCavern;
}
