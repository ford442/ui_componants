/**
 * Primordial Soup Experiment
 * Combines WebGL2 (Petri Dish Rim) and WebGPU (Particle Life Simulation).
 */

export class PrimordialSoup {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // Configuration
        this.particleCount = options.particleCount || 2000; // Keep lower for N^2 interactions

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0, isDown: false };
        this.canvasSize = { width: 0, height: 0 };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndexCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.particleBuffer = null;
        this.paramBuffer = null;
        this.rulesBuffer = null;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = () => this.mouse.isDown = true;
        this.handleMouseUp = () => this.mouse.isDown = false;

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        // 1. Initialize WebGL2 (Dish Structure)
        this.initWebGL2();

        // 2. Initialize WebGPU (Particle Life)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("PrimordialSoup: WebGPU initialization error:", e);
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
        window.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (The Petri Dish)
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

        // Create Torus Mesh for the rim
        const { vertices, indices, normals } = this.createTorus(6.5, 0.2, 32, 64);
        this.glIndexCount = indices.length;

        // VAO
        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        // Position Buffer
        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);

        // Normal Buffer
        const normBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, normBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, normals, this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(1);
        this.gl.vertexAttribPointer(1, 3, this.gl.FLOAT, false, 0, 0);

        // Index Buffer
        const idxBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
            layout(location=0) in vec3 a_pos;
            layout(location=1) in vec3 a_normal;

            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_proj;

            out vec3 v_normal;
            out vec3 v_worldPos;

            void main() {
                v_normal = (u_model * vec4(a_normal, 0.0)).xyz;
                v_worldPos = (u_model * vec4(a_pos, 1.0)).xyz;
                gl_Position = u_proj * u_view * vec4(v_worldPos, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_normal;
            in vec3 v_worldPos;

            uniform vec3 u_viewPos;

            out vec4 outColor;

            void main() {
                vec3 N = normalize(v_normal);
                vec3 V = normalize(u_viewPos - v_worldPos);
                vec3 L = normalize(vec3(5.0, 10.0, 5.0));

                // Fresnel glass effect
                float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

                // Rim lighting
                float rim = smoothstep(0.4, 1.0, fresnel);

                vec3 glassColor = vec3(0.1, 0.2, 0.3);
                vec3 highlight = vec3(0.8, 0.9, 1.0);

                vec3 finalColor = mix(glassColor, highlight, rim);

                outColor = vec4(finalColor, 0.4 + rim * 0.6);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);

        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
        this.gl.enable(this.gl.DEPTH_TEST);
    }

    createTorus(radius, tube, radialSegments, tubularSegments) {
        const vertices = [];
        const normals = [];
        const indices = [];

        for (let j = 0; j <= radialSegments; j++) {
            for (let i = 0; i <= tubularSegments; i++) {
                const u = i / tubularSegments * Math.PI * 2;
                const v = j / radialSegments * Math.PI * 2;

                const cx = radius * Math.cos(u);
                const cy = radius * Math.sin(u);

                const x = (radius + tube * Math.cos(v)) * Math.cos(u);
                const z = (radius + tube * Math.cos(v)) * Math.sin(u);
                const y = tube * Math.sin(v);

                vertices.push(x, y, z);

                // Normal
                const nx = Math.cos(v) * Math.cos(u);
                const nz = Math.cos(v) * Math.sin(u);
                const ny = Math.sin(v);
                normals.push(nx, ny, nz);
            }
        }

        for (let j = 1; j <= radialSegments; j++) {
            for (let i = 1; i <= tubularSegments; i++) {
                const a = (tubularSegments + 1) * j + i - 1;
                const b = (tubularSegments + 1) * (j - 1) + i - 1;
                const c = (tubularSegments + 1) * (j - 1) + i;
                const d = (tubularSegments + 1) * j + i;

                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }

        return {
            vertices: new Float32Array(vertices),
            normals: new Float32Array(normals),
            indices: new Uint16Array(indices)
        };
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);
        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Particle Life)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2; pointer-events: none;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // --- Shader Code ---
        const wgslCode = `
            struct Particle {
                pos : vec4f, // x, y, z, type (0-3)
                vel : vec4f, // vx, vy, vz, padding
            }

            struct Params {
                matrix : mat4x4f, // ViewProj
                mouse : vec2f,
                time : f32,
                dt : f32,
                count : f32,
                friction : f32,
                padding1 : f32,
                padding2 : f32,
            }

            struct Rules {
                // 4x4 Interaction Matrix flattened or vec4s
                // Let's use 4 vec4s to represent rows
                row0 : vec4f,
                row1 : vec4f,
                row2 : vec4f,
                row3 : vec4f,
                radii : vec4f, // Interaction radii per type? or global? Let's use global for now.
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : Params;
            @group(0) @binding(2) var<uniform> rules : Rules;

            // Helper to get rule force from type A to B
            fn getRule(a: f32, b: f32) -> f32 {
                let ia = i32(a);
                let ib = i32(b);
                if (ia == 0) { return rules.row0[ib]; }
                if (ia == 1) { return rules.row1[ib]; }
                if (ia == 2) { return rules.row2[ib]; }
                return rules.row3[ib];
            }

            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id : vec3u) {
                let idx = id.x;
                let count = u32(params.count);
                if (idx >= count) { return; }

                var p = particles[idx];
                var force = vec3f(0.0);

                // Particle Life Interaction Loop
                // Optimally this uses shared memory or tiling, but for <3000 particles, brute force is acceptable on desktop GPU

                for (var i = 0u; i < count; i++) {
                    if (i == idx) { continue; }

                    let other = particles[i];
                    let delta = other.pos.xyz - p.pos.xyz;
                    var dist = length(delta);

                    // Wrap around? No, we are in a dish.
                    // If too close, repel strongly
                    if (dist > 0.0 && dist < 8.0) {
                        let f = getRule(p.pos.w, other.pos.w);
                        let dir = delta / dist;

                        // Interaction function:
                        // If dist < 0.3 (close), repel (-1.0)
                        // If 0.3 < dist < 1.0 (range), apply rule force * factor
                        // We scale distances: max range ~6.0

                        if (dist < 1.0) {
                            force -= dir * 10.0 * (1.0 - dist); // Repel
                        } else {
                            // Smooth falloff
                            // Factor based on distance
                            let range = 6.0;
                            if (dist < range) {
                                let val = f * (1.0 - abs(2.0 * dist - range - 1.0) / (range - 1.0)); // Peak at mid?
                                // Simpler: 1/r force
                                force += dir * f * 2.0 * (1.0 - dist/range);
                            }
                        }
                    }
                }

                // Boundary Force (Cylinder/Dish)
                let r = length(p.pos.xz);
                if (r > 6.0) {
                    let dir = normalize(vec3f(p.pos.x, 0.0, p.pos.z));
                    force -= dir * (r - 6.0) * 50.0; // Push back hard
                }

                // Mouse Interaction
                // Mouse coords are normalized -1 to 1. World scale is ~7.
                let mousePos = vec3f(params.mouse.x * 10.0, 0.0, params.mouse.y * 10.0); // Adjust scale approx
                let mDist = distance(p.pos.xyz, mousePos);
                if (mDist < 3.0) {
                     let dir = normalize(p.pos.xyz - mousePos);
                     force += dir * 50.0 * (1.0 - mDist/3.0); // Stir/Repel
                }

                // Update Physics
                p.vel = vec4f((p.vel.xyz + force * params.dt) * params.friction, 0.0);
                p.pos = vec4f(p.pos.xyz + p.vel.xyz * params.dt, p.pos.w);

                // Keep Y flatish
                p.pos.y = p.pos.y * 0.9;

                particles[idx] = p;
            }

            // --- Render Shader ---
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) uv : vec2f,
            }

            @vertex
            fn vs_main(
                @builtin(vertex_index) vIdx : u32,
                @builtin(instance_index) iIdx : u32
            ) -> VertexOutput {
                let p = particles[iIdx];
                var output : VertexOutput;

                // Billboard Quad generation
                let angle = f32(vIdx) * 1.57079; // 0, PI/2, PI, 3PI/2 approx (actually 0,1,2,3 mapped to corners)
                // Better: 0:(-1,-1), 1:(1,-1), 2:(-1,1), 3:(1,1) triangle strip?
                // Let's use array lookup for corners
                var corners = array<vec2f, 6>(
                    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
                );
                let corner = corners[vIdx] * 0.15; // Particle size

                // View-aligned (XZ plane mostly, but billboard to camera)
                // Simple ViewProj * (Pos + Offset)

                // Let's cheat and just add offset in screen space or view space?
                // For a soup, world space billboards facing up (y-axis) is fine if camera is high

                let worldPos = p.pos.xyz + vec3f(corner.x, 0.0, corner.y);
                output.position = params.matrix * vec4f(worldPos, 1.0);
                output.uv = corners[vIdx]; // -1 to 1

                // Color based on type
                let t = i32(p.pos.w);
                if (t == 0) { output.color = vec4f(1.0, 0.2, 0.2, 1.0); } // Red
                else if (t == 1) { output.color = vec4f(0.2, 1.0, 0.2, 1.0); } // Green
                else if (t == 2) { output.color = vec4f(0.2, 0.2, 1.0, 1.0); } // Blue
                else { output.color = vec4f(1.0, 1.0, 0.2, 1.0); } // Yellow

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f, @location(1) uv : vec2f) -> @location(0) vec4f {
                let dist = length(uv);
                if (dist > 1.0) { discard; }
                let alpha = smoothstep(1.0, 0.5, dist);
                return vec4f(color.rgb, color.a * alpha);
            }
        `;

        const module = this.device.createShaderModule({ code: wgslCode });

        // Buffer Setup
        const pSize = 32; // 8 floats
        this.particleBuffer = this.device.createBuffer({
            size: this.particleCount * pSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        // Init Particles
        const initData = new Float32Array(this.particleCount * 8);
        for(let i=0; i<this.particleCount; i++) {
            // Random pos inside radius 5
            const a = Math.random() * Math.PI * 2;
            const r = Math.sqrt(Math.random()) * 5.0;
            initData[i*8+0] = Math.cos(a) * r;
            initData[i*8+1] = (Math.random() - 0.5) * 0.1;
            initData[i*8+2] = Math.sin(a) * r;
            initData[i*8+3] = Math.floor(Math.random() * 4); // Type 0-3

            // Vel 0
            initData[i*8+4] = 0; initData[i*8+5] = 0; initData[i*8+6] = 0; initData[i*8+7] = 0;
        }
        this.device.queue.writeBuffer(this.particleBuffer, 0, initData);

        // Uniforms
        this.paramBuffer = this.device.createBuffer({
            size: 128, // Matrix (64) + floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.rulesBuffer = this.device.createBuffer({
            size: 80, // 4 vec4s + extra
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Set Interaction Rules
        // Row 0 (Red): Likes Red (+), Dislikes Green (-)
        // Values: -1 to 1
        const rulesData = new Float32Array([
            // R, G, B, Y (how R reacts to them)
            1.0, -0.5, 0.5, -0.2, // R
            // G (how G reacts to them)
            0.5, 1.0, -0.5, 0.2, // G
            // B (how B reacts to them)
            -0.5, 0.5, 1.0, 0.5, // B
            // Y (how Y reacts to them)
            0.2, -0.2, 0.5, 1.0, // Y
            // Radii/Pad
            0,0,0,0
        ]);
        this.device.queue.writeBuffer(this.rulesBuffer, 0, rulesData);

        // Pipelines
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.paramBuffer } },
                { binding: 2, resource: { buffer: this.rulesBuffer } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'cs_main' }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: { module, entryPoint: 'vs_main' },
            fragment: {
                module, entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list' },
            depthStencil: undefined // No depth for additive particles usually
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `position: absolute; bottom: 20px; left: 20px; color: #ff5555; font-family: monospace; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 4px; border: 1px solid #ff5555;`;
        msg.innerHTML = "⚠️ WebGPU Not Available<br>Simulation disabled. WebGL2 structure visible.";
        this.container.appendChild(msg);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.canvasSize.width = w;
        this.canvasSize.height = h;

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

    // Camera helpers
    perspective(fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        return [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ];
    }

    lookAt(eye, center, up) {
        const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
        const len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        const zx = z0 * len, zy = z1 * len, zz = z2 * len;
        const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
        const lenX = 1 / Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
        const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;
        const y0 = zy * xz - zz * xy, y1 = zz * xx - zx * xz, y2 = zx * xy - zy * xx;
        const lenY = 1 / Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
        const yx = y0 * lenY, yy = y1 * lenY, yz = y2 * lenY;
        return [
            xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
            -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
            -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
            1
        ];
    }

    multiply(a, b) { // Standard multiplication for uniforms
         const out = new Float32Array(16);
         for (let i = 0; i < 4; i++) {
             for (let j = 0; j < 4; j++) {
                 let sum = 0;
                 for (let k = 0; k < 4; k++) sum += a[k * 4 + i] * b[j * 4 + k]; // Adjusted for column-major
                 out[j * 4 + i] = sum;
             }
         }
         return out;
    }

    multiplyGL(a, b) { // For WebGL uniforms (Column Major)
        // A * B
        const out = new Float32Array(16);
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += a[k*4 + row] * b[col*4 + k];
                }
                out[col*4 + row] = sum;
            }
        }
        return out;
    }

    animate() {
        if (!this.isActive) return;
        this.animationId = requestAnimationFrame(() => this.animate());

        const time = (Date.now() - this.startTime) * 0.001;
        const aspect = this.canvasSize.width / this.canvasSize.height;

        // Camera Orbit
        const radius = 12.0;
        const camX = Math.sin(time * 0.1) * radius;
        const camZ = Math.cos(time * 0.1) * radius;
        const camY = 8.0;

        const view = this.lookAt([camX, camY, camZ], [0, 0, 0], [0, 1, 0]);
        const proj = this.perspective(Math.PI / 4, aspect, 0.1, 100.0);

        // 1. Render WebGL2 Background
        if (this.gl && this.glProgram) {
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
            this.gl.clearColor(0, 0, 0, 0); // Transparent, body bg is black
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.useProgram(this.glProgram);

            // Model: Scale up slightly
            const model = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];

            const loc = (n) => this.gl.getUniformLocation(this.glProgram, n);
            this.gl.uniformMatrix4fv(loc('u_model'), false, new Float32Array(model));
            this.gl.uniformMatrix4fv(loc('u_view'), false, new Float32Array(view));
            this.gl.uniformMatrix4fv(loc('u_proj'), false, new Float32Array(proj));
            this.gl.uniform3f(loc('u_viewPos'), camX, camY, camZ);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU Particles
        if (this.device && this.renderPipeline) {
            // Update Uniforms
            // Proj * View
            const vp = this.multiplyGL(proj, view);

            const paramsData = new Float32Array(32); // 128 bytes
            paramsData.set(vp, 0); // 0-15
            paramsData[16] = this.mouse.x; // Mouse X
            paramsData[17] = this.mouse.y; // Mouse Y
            paramsData[18] = time;
            paramsData[19] = 0.016; // dt
            paramsData[20] = this.particleCount;
            paramsData[21] = 0.95; // Friction

            this.device.queue.writeBuffer(this.paramBuffer, 0, paramsData);

            const cmd = this.device.createCommandEncoder();

            // Compute
            const cp = cmd.beginComputePass();
            cp.setPipeline(this.computePipeline);
            cp.setBindGroup(0, this.computeBindGroup);
            cp.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
            cp.end();

            // Render
            const rp = cmd.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            rp.setPipeline(this.renderPipeline);
            rp.setBindGroup(0, this.computeBindGroup);
            rp.draw(6, this.particleCount); // 6 vertices per quad
            rp.end();

            this.device.queue.submit([cmd.finish()]);
        }
    }
}
