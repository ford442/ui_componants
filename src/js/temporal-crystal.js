/**
 * Temporal Crystal Experiment
 * Hybrid WebGL2 + WebGPU
 * - WebGL2: Renders a faceted crystal core with refraction-like shader.
 * - WebGPU: Computes and renders a temporal particle field orbiting the core.
 */

export class TemporalCrystal {
    constructor(container) {
        this.container = container;
        this.width = container.clientWidth;
        this.height = container.clientHeight;

        // State
        this.isActive = true;
        this.time = 0;
        this.rotation = { x: 0, y: 0 };
        this.targetRotation = { x: 0, y: 0 };
        this.mouse = { x: 0, y: 0, isDown: false };

        // WebGL2
        this.gl = null;
        this.glCanvas = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndicesCount = 0;

        // WebGPU
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.particleBuffer = null;
        this.simParamBuffer = null;
        this.computePipeline = null;
        this.renderPipeline = null;
        this.computeBindGroup = null;
        this.numParticles = 50000;

        this.init();
    }

    async init() {
        this.initWebGL2();

        if (navigator.gpu) {
            try {
                await this.initWebGPU();
            } catch (e) {
                console.warn('WebGPU init failed:', e);
                this.addWarning('WebGPU failed. Running WebGL2 only.');
            }
        } else {
            this.addWarning('WebGPU not supported. Running WebGL2 only.');
        }

        this.attachEvents();
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    addWarning(text) {
        const div = document.createElement('div');
        div.style.cssText = `
            position: absolute; bottom: 50px; right: 20px;
            color: #ff5555; font-family: monospace; font-size: 12px;
            background: rgba(0,0,0,0.8); padding: 5px 10px; border: 1px solid #ff5555;
            z-index: 100;
        `;
        div.textContent = text;
        this.container.appendChild(div);
    }

    // ========================================================================
    // WebGL2 (Crystal Core)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.width = this.width;
        this.glCanvas.height = this.height;
        this.glCanvas.style.cssText = 'position: absolute; top: 0; left: 0; z-index: 10;';
        this.container.appendChild(this.glCanvas);

        const gl = this.glCanvas.getContext('webgl2', { alpha: true });
        if (!gl) return;
        this.gl = gl;

        // Shader: Faceted Crystal
        const vs = `#version 300 es
            in vec3 a_position;
            in vec3 a_normal;

            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_projection;
            uniform float u_time;

            out vec3 v_normal;
            out vec3 v_pos;
            out vec3 v_worldPos;

            void main() {
                // Pulsate
                vec3 pos = a_position + a_normal * sin(u_time * 2.0 + a_position.y * 5.0) * 0.05;

                vec4 worldPos = u_model * vec4(pos, 1.0);
                v_worldPos = worldPos.xyz;
                v_normal = mat3(u_model) * a_normal;
                v_pos = pos;

                gl_Position = u_projection * u_view * worldPos;
            }
        `;

        const fs = `#version 300 es
            precision highp float;

            in vec3 v_normal;
            in vec3 v_pos;
            in vec3 v_worldPos;

            uniform vec3 u_cameraPos;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                vec3 N = normalize(v_normal);
                vec3 V = normalize(u_cameraPos - v_worldPos);
                vec3 L = normalize(vec3(2.0, 4.0, 5.0)); // Fixed light

                // Flat shading using fwidth
                vec3 flatN = normalize(cross(dFdx(v_worldPos), dFdy(v_worldPos)));

                // Diffuse
                float diff = max(dot(flatN, L), 0.0);

                // Specular
                vec3 R = reflect(-L, flatN);
                float spec = pow(max(dot(V, R), 0.0), 32.0);

                // Iridescence / Crystal color
                vec3 baseColor = vec3(0.1, 0.8, 0.9); // Cyan
                vec3 shimmer = 0.5 + 0.5 * cos(u_time + v_worldPos.xyx + vec3(0, 2, 4));

                vec3 color = baseColor * (diff * 0.5 + 0.2) + shimmer * spec * 0.8;

                // Rim light
                float rim = 1.0 - max(dot(V, flatN), 0.0);
                rim = pow(rim, 3.0);
                color += vec3(0.5, 0.8, 1.0) * rim;

                outColor = vec4(color, 0.7 + 0.3 * rim); // Semi-transparent
            }
        `;

        this.glProgram = this.createGLProgram(gl, vs, fs);

        // Geometry: Icosahedron (Low Poly Sphere)
        const geometry = this.createIcosahedron();

        this.glVao = gl.createVertexArray();
        gl.bindVertexArray(this.glVao);

        const posBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.vertices), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        const normBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.normals), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        const idxBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(geometry.indices), gl.STATIC_DRAW);

        this.glIndicesCount = geometry.indices.length;

        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }

    createGLProgram(gl, vsSource, fsSource) {
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSource);
        gl.compileShader(vs);
        if(!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vs));

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSource);
        gl.compileShader(fs);
        if(!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fs));

        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        return prog;
    }

    // ========================================================================
    // WebGPU (Particles)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.width = this.width;
        this.gpuCanvas.height = this.height;
        this.gpuCanvas.style.cssText = 'position: absolute; top: 0; left: 0; z-index: 5; pointer-events: none;';
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No adapter');

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // Compute Shader
        const computeShader = `
            struct Particle {
                pos: vec4f, // xyz, w=life
                vel: vec4f, // xyz, w=unused
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

            struct Uniforms {
                time: f32,
                dt: f32,
            }
            @group(0) @binding(1) var<uniform> u: Uniforms;

            // Simple noise
            fn hash(p: u32) -> f32 {
                var p_ = p;
                p_ = (p_ << 13u) ^ p_;
                return (1.0 - f32((p_ * (p_ * p_ * 15731u + 789221u) + 1376312589u) & 0x7fffffffu) / 1073741824.0);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];
                let dt = u.dt;

                // Orbit logic
                let dist = length(p.pos.xyz);
                let center = vec3f(0.0);

                // Tangent force for swirl
                let up = vec3f(0.0, 1.0, 0.0);
                let tangent = normalize(cross(p.pos.xyz, up));

                // Attraction to center but repelled at close range
                let dir = normalize(center - p.pos.xyz);
                let force = dir * (1.0 / (dist * dist + 0.1));

                if (dist < 1.5) {
                    p.vel = p.vel - dir * 5.0 * dt; // Repel
                } else {
                    p.vel = p.vel + force * 2.0 * dt; // Attract
                }

                // Swirl
                p.vel = p.vel + tangent * 2.0 * dt;

                // Damping
                p.vel = p.vel * 0.98;

                // Move
                p.pos = vec4f(p.pos.xyz + p.vel.xyz * dt, p.pos.w);

                // Reset if too far
                if (dist > 15.0 || abs(p.pos.y) > 10.0) {
                    let r = hash(i + u32(u.time * 100.0)) * 5.0 + 2.0;
                    let theta = hash(i * 2u) * 6.28;
                    p.pos = vec4f(r * cos(theta), (hash(i * 3u) - 0.5) * 4.0, r * sin(theta), 1.0);
                    p.vel = vec4f(0.0);
                }

                particles[i] = p;
            }
        `;

        // Render Shader
        const renderShader = `
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            struct Uniforms {
                viewProj: mat4x4f,
            }
            @group(0) @binding(0) var<uniform> u: Uniforms;

            @vertex
            fn vs_main(@location(0) pos: vec4f, @location(1) vel: vec4f) -> VertexOut {
                var out: VertexOut;
                out.pos = u.viewProj * vec4f(pos.xyz, 1.0);

                let speed = length(vel.xyz);
                out.color = vec4f(0.5, 0.8, 1.0, 0.5 + speed * 0.5);

                // Size attenuation approx via point size not supported in WGSL directly without extension?
                // Using point-list topology assumes 1px points usually.

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const pSize = 8 * 4; // 8 floats (pos4 + vel4)
        const pData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            const r = 3.0 + Math.random() * 2.0;
            const theta = Math.random() * Math.PI * 2;
            pData[i*8+0] = r * Math.cos(theta); // x
            pData[i*8+1] = (Math.random() - 0.5) * 4.0; // y
            pData[i*8+2] = r * Math.sin(theta); // z
            pData[i*8+3] = 1.0; // w
        }

        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(pData);
        this.particleBuffer.unmap();

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // time(f32), dt(f32), padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.viewProjBuffer = this.device.createBuffer({
            size: 64, // mat4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Pipeline
        const computeLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        const cMod = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
            compute: { module: cMod, entryPoint: 'main' }
        });

        // Render Pipeline
        const renderLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderLayout,
            entries: [
                { binding: 0, resource: { buffer: this.viewProjBuffer } }
            ]
        });

        const rMod = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
            vertex: {
                module: rMod,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: pSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: rMod,
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
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    createIcosahedron() {
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;

        const vertices = [
            -1,  t,  0,
             1,  t,  0,
            -1, -t,  0,
             1, -t,  0,
             0, -1,  t,
             0,  1,  t,
             0, -1, -t,
             0,  1, -t,
             t,  0, -1,
             t,  0,  1,
            -t,  0, -1,
            -t,  0,  1
        ];

        // Normalize vertices to project to sphere
        for(let i=0; i<vertices.length; i+=3) {
            const x=vertices[i], y=vertices[i+1], z=vertices[i+2];
            const len = Math.sqrt(x*x+y*y+z*z);
            vertices[i] /= len; vertices[i+1] /= len; vertices[i+2] /= len;
        }

        const indices = [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
        ];

        // Flat shade: duplicate vertices
        const flatVerts = [];
        const flatNorms = [];
        const flatIdx = [];

        for(let i=0; i<indices.length; i++) {
            const idx = indices[i];
            const x = vertices[idx*3], y = vertices[idx*3+1], z = vertices[idx*3+2];
            flatVerts.push(x, y, z);
            // Normal of sphere is position
            flatNorms.push(x, y, z);
            flatIdx.push(i);
        }

        return { vertices: flatVerts, normals: flatNorms, indices: flatIdx };
    }

    attachEvents() {
        this.container.addEventListener('mousedown', e => {
            this.mouse.isDown = true;
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });
        window.addEventListener('mouseup', () => this.mouse.isDown = false);
        window.addEventListener('mousemove', e => {
            if (this.mouse.isDown) {
                const dx = e.clientX - this.mouse.x;
                const dy = e.clientY - this.mouse.y;
                this.targetRotation.y += dx * 0.01;
                this.targetRotation.x += dy * 0.01;
                this.mouse.x = e.clientX;
                this.mouse.y = e.clientY;
            }
        });
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        if (this.glCanvas) {
            this.glCanvas.width = this.width;
            this.glCanvas.height = this.height;
            this.gl.viewport(0, 0, this.width, this.height);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = this.width;
            this.gpuCanvas.height = this.height;
        }
    }

    // ========================================================================
    // Math Utils
    // ========================================================================

    mat4Perspective(out, fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
        out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
        out[8] = 0; out[9] = 0; out[10] = (far + near) / (near - far); out[11] = -1;
        out[12] = 0; out[13] = 0; out[14] = (2 * far * near) / (near - far); out[15] = 0;
    }

    mat4Identity(out) {
        for(let i=0; i<16; i++) out[i] = (i%5 === 0) ? 1 : 0;
    }

    mat4RotateY(out, rad) {
        const s = Math.sin(rad), c = Math.cos(rad);
        const a00 = out[0], a01 = out[1], a02 = out[2], a03 = out[3];
        const a20 = out[8], a21 = out[9], a22 = out[10], a23 = out[11];
        out[0] = a00 * c - a20 * s;
        out[8] = a00 * s + a20 * c;
        out[1] = a01 * c - a21 * s;
        out[9] = a01 * s + a21 * c;
        out[2] = a02 * c - a22 * s;
        out[10] = a02 * s + a22 * c;
        out[3] = a03 * c - a23 * s;
        out[11] = a03 * s + a23 * c;
    }

    mat4RotateX(out, rad) {
        const s = Math.sin(rad), c = Math.cos(rad);
        const a10 = out[4], a11 = out[5], a12 = out[6], a13 = out[7];
        const a20 = out[8], a21 = out[9], a22 = out[10], a23 = out[11];
        out[4] = a10 * c + a20 * s;
        out[8] = -a10 * s + a20 * c;
        out[5] = a11 * c + a21 * s;
        out[9] = -a11 * s + a21 * c;
        out[6] = a12 * c + a22 * s;
        out[10] = -a12 * s + a22 * c;
        out[7] = a13 * c + a23 * s;
        out[11] = -a13 * s + a23 * c;
    }

    animate() {
        const now = performance.now();
        const dt = Math.min((now - this.time) * 0.001, 0.05);
        this.time = now;

        // Smooth rotation
        this.rotation.x += (this.targetRotation.x - this.rotation.x) * 0.1;
        this.rotation.y += (this.targetRotation.y - this.rotation.y) * 0.1;

        // Auto rotate
        this.targetRotation.y += 0.002;

        const projection = new Float32Array(16);
        this.mat4Perspective(projection, 45 * Math.PI / 180, this.width / this.height, 0.1, 100.0);

        const view = new Float32Array(16);
        this.mat4Identity(view);
        view[14] = -5.0; // Translate Z

        const model = new Float32Array(16);
        this.mat4Identity(model);
        this.mat4RotateX(model, this.rotation.x);
        this.mat4RotateY(model, this.rotation.y);

        // Matrix multiplication View * Model (simplified as Model is mostly rotation)
        // Actually we need to pass Model, View, Projection separate or combined.
        // My shader expects u_model, u_view, u_projection.

        // Render WebGL2
        if (this.gl) {
            this.gl.clearColor(0, 0, 0, 0); // Transparent background
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.useProgram(this.glProgram);

            const uModel = this.gl.getUniformLocation(this.glProgram, 'u_model');
            const uView = this.gl.getUniformLocation(this.glProgram, 'u_view');
            const uProj = this.gl.getUniformLocation(this.glProgram, 'u_projection');
            const uTime = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const uCam = this.gl.getUniformLocation(this.glProgram, 'u_cameraPos');

            this.gl.uniformMatrix4fv(uModel, false, model);
            this.gl.uniformMatrix4fv(uView, false, view);
            this.gl.uniformMatrix4fv(uProj, false, projection);
            this.gl.uniform1f(uTime, now * 0.001);
            this.gl.uniform3f(uCam, 0, 0, 5); // Approx

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glIndicesCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // Render WebGPU
        if (this.device && this.context) {
            // Update Uniforms
            this.device.queue.writeBuffer(this.simParamBuffer, 0, new Float32Array([now * 0.001, dt]));

            // VP Matrix calculation for WebGPU (needs to match WebGL)
            // Multiply Projection * View
            const vp = new Float32Array(16);
            // Simple matrix mult View * Proj (Column major: Proj * View)
            // Since View is just Translation(0,0,-5), we can just modify projection z term?
            // Better to do full multiply.
            // But let's cheat for simplicity: View is identity with z=-5.
            // Proj * View is just Proj but with z offset.
            // Let's implement mat4Multiply briefly.

            const mat4Multiply = (out, a, b) => {
                let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
                let a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
                let a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
                let a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
                let b0  = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
                out[0] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
                out[1] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
                out[2] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
                out[3] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
                b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
                out[4] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
                out[5] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
                out[6] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
                out[7] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
                b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
                out[8] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
                out[9] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
                out[10] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
                out[11] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
                b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
                out[12] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
                out[13] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
                out[14] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
                out[15] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
            };

            mat4Multiply(vp, projection, view);
            this.device.queue.writeBuffer(this.viewProjBuffer, 0, vp);

            const cmd = this.device.createCommandEncoder();

            const cmpPass = cmd.beginComputePass();
            cmpPass.setPipeline(this.computePipeline);
            cmpPass.setBindGroup(0, this.computeBindGroup);
            cmpPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cmpPass.end();

            const texView = this.context.getCurrentTexture().createView();
            const rndPass = cmd.beginRenderPass({
                colorAttachments: [{
                    view: texView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            rndPass.setPipeline(this.renderPipeline);
            rndPass.setBindGroup(0, this.renderBindGroup);
            rndPass.setVertexBuffer(0, this.particleBuffer);
            rndPass.draw(this.numParticles);
            rndPass.end();

            this.device.queue.submit([cmd.finish()]);
        }

        if (this.isActive) requestAnimationFrame(this.animate);
    }
}
