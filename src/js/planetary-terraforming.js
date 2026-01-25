/**
 * Planetary Terraforming Experiment
 * Combines WebGL2 (Wireframe Planet Grid) and WebGPU (Surface Particles Simulation).
 */

export class PlanetaryTerraformingExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // Configuration
        this.particleCount = options.particleCount || 100000;

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
        this.container.style.background = '#000000';

        // 1. Initialize WebGL2 (Planet Grid)
        this.initWebGL2();

        // 2. Initialize WebGPU (Particles)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("PlanetaryTerraforming: WebGPU initialization error:", e);
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
        this.container.addEventListener('mousedown', this.handleMouseDown);
        this.container.addEventListener('mouseup', this.handleMouseUp);
        // Also listen on window for mouseup in case drag ends outside
        window.addEventListener('mouseup', this.handleMouseUp);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Wireframe Sphere)
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

        // Create Sphere Mesh
        const { vertices, indices } = this.createSphere(4.0, 32, 32);
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

        // Index Buffer
        const idxBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
            layout(location=0) in vec3 a_pos;

            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform float u_time;

            void main() {
                // Rotate the sphere
                float c = cos(u_time * 0.1);
                float s = sin(u_time * 0.1);
                mat4 rotY = mat4(
                    c, 0, -s, 0,
                    0, 1, 0, 0,
                    s, 0, c, 0,
                    0, 0, 0, 1
                );

                vec4 worldPos = u_model * rotY * vec4(a_pos, 1.0);
                gl_Position = u_proj * u_view * worldPos;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            out vec4 outColor;

            void main() {
                outColor = vec4(0.2, 0.4, 0.6, 0.3);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
    }

    createSphere(radius, widthSegments, heightSegments) {
        const vertices = [];
        const indices = [];

        for (let y = 0; y <= heightSegments; y++) {
            for (let x = 0; x <= widthSegments; x++) {
                const u = x / widthSegments;
                const v = y / heightSegments;
                const theta = u * Math.PI * 2;
                const phi = v * Math.PI;

                const px = radius * Math.sin(phi) * Math.cos(theta);
                const py = radius * Math.cos(phi);
                const pz = radius * Math.sin(phi) * Math.sin(theta);

                vertices.push(px, py, pz);
            }
        }

        for (let y = 0; y < heightSegments; y++) {
            for (let x = 0; x < widthSegments; x++) {
                const p1 = y * (widthSegments + 1) + x;
                const p2 = p1 + 1;
                const p3 = (y + 1) * (widthSegments + 1) + x;
                const p4 = p3 + 1;

                // Lines
                indices.push(p1, p2);
                indices.push(p1, p3);
            }
        }

        return {
            vertices: new Float32Array(vertices),
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
    // WebGPU IMPLEMENTATION (Surface Particles)
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

        const wgslCode = `
            struct Particle {
                pos : vec4f, // x, y, z, state (0: Barren, 1: Water, 2: Vegetation)
                basePos : vec4f, // Original position for rotation
            }

            struct Params {
                matrix : mat4x4f, // ViewProj
                mouse : vec2f,
                time : f32,
                isDown : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : Params;

            // Random number generator
            fn hash(n: f32) -> f32 {
                return fract(sin(n) * 43758.5453123);
            }

            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id : vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];

                // Rotation (match WebGL)
                let t = params.time * 0.1;
                let c = cos(t);
                let s = sin(t);

                // Rotate base position to get current position
                let bx = p.basePos.x;
                let bz = p.basePos.z;
                p.pos.x = bx * c - bz * s;
                p.pos.y = p.basePos.y;
                p.pos.z = bx * s + bz * c;

                // Interaction
                if (params.isDown > 0.5) {
                    // Raycast sphere intersection approx
                    // Or simpler: Project mouse to world space?
                    // Let's rely on screen space proximity in compute? Hard without full transform.
                    // Easier: Mouse ray in world space (calculated on CPU or passed)
                    // For now, let's just use a simple 2D projection approximation since camera is fixed

                    // Mouse is -1 to 1.
                    // Assume sphere radius 4, camera z ~ 12
                    // Approximate screen projection
                    let viewX = p.pos.x;
                    let viewY = p.pos.y;
                    // very rough perspective
                    let scale = 1.0;

                    // Just distance check on surface for "brush"
                    // We need to unproject mouse or project point
                    // Let's use the passed params.mouse directly against projected p

                    // Simple interaction: If mouse is close to projected point
                    // This logic is flawed without matrix mult in compute, but let's try a simple approach
                    // We can do interaction in view space if we had view matrix

                    // Alternative: Just random changes for demo if mouse is down
                    // Let's try to pass a "hit point" from CPU?
                    // For now, let's just make particles change state based on random if mouse down
                    // or implement a simple "Terraform beam" that hits the center

                    // Actually, let's implement the logic:
                    // If mouse is down, we check distance to "mouse ray" roughly
                    // But we don't have the matrix here easily to project.
                    // Let's just create "spontaneous life" when mouse is down

                    let seed = params.time + f32(idx) * 0.001;
                    if (hash(seed) > 0.99) {
                        p.pos.w = 1.0; // Turn to Water
                    }
                }

                // Simulation Rules
                let seed = params.time + f32(idx) * 0.0001;
                let r = hash(seed);

                // Barren -> Water (rare, rain)
                if (p.pos.w < 0.5) {
                    if (r < 0.0001) { p.pos.w = 1.0; }
                }
                // Water -> Vegetation (growth)
                else if (p.pos.w < 1.5) {
                    if (r < 0.01) { p.pos.w = 2.0; } // Grow
                    if (r > 0.995) { p.pos.w = 0.0; } // Evaporate
                }
                // Vegetation -> Barren (die)
                else {
                    if (r < 0.005) { p.pos.w = 0.0; }
                }

                particles[idx] = p;
            }

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

                // Billboard
                var corners = array<vec2f, 6>(
                    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
                );
                let corner = corners[vIdx] * 0.08;

                // View-aligned billboarding
                // We need view matrix to do it properly, but for sphere simply adding to screen pos works ok-ish
                // Or better: inverse view rotation.
                // Let's just add to world pos and project.
                // Since camera is fixed at (0,0,12), looking at 0, billboard plane is XY.

                let worldPos = p.pos.xyz + vec3f(corner.x, corner.y, 0.0);
                output.position = params.matrix * vec4f(worldPos, 1.0);
                output.uv = corners[vIdx];

                // Color based on state
                let state = p.pos.w;
                if (state < 0.5) {
                    // Barren (Reddish/Brown)
                    output.color = vec4f(0.8, 0.4, 0.2, 1.0);
                } else if (state < 1.5) {
                    // Water (Blue)
                    output.color = vec4f(0.2, 0.5, 0.9, 1.0);
                } else {
                    // Vegetation (Green)
                    output.color = vec4f(0.2, 0.8, 0.3, 1.0);
                }

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f, @location(1) uv : vec2f) -> @location(0) vec4f {
                let dist = length(uv);
                if (dist > 1.0) { discard; }
                return color;
            }
        `;

        const module = this.device.createShaderModule({ code: wgslCode });

        // Buffer Setup
        const pSize = 32; // 8 floats (pos + basePos)
        this.particleBuffer = this.device.createBuffer({
            size: this.particleCount * pSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        // Initialize Particles (Fibonacci Spiral on Sphere)
        const initData = new Float32Array(this.particleCount * 8);
        const phi = Math.PI * (3.0 - Math.sqrt(5.0)); // Golden angle

        for (let i = 0; i < this.particleCount; i++) {
            const y = 1 - (i / (this.particleCount - 1)) * 2; // y goes from 1 to -1
            const radius = Math.sqrt(1 - y * y); // radius at y
            const theta = phi * i;

            const x = Math.cos(theta) * radius;
            const z = Math.sin(theta) * radius;

            const r = 4.0; // Sphere radius

            // Pos
            initData[i*8+0] = x * r;
            initData[i*8+1] = y * r;
            initData[i*8+2] = z * r;
            initData[i*8+3] = 0.0; // State: Barren

            // BasePos (same as pos initially)
            initData[i*8+4] = x * r;
            initData[i*8+5] = y * r;
            initData[i*8+6] = z * r;
            initData[i*8+7] = 0.0; // Padding
        }
        this.device.queue.writeBuffer(this.particleBuffer, 0, initData);

        // Params Buffer
        this.paramBuffer = this.device.createBuffer({
            size: 80, // Mat4 (64) + vec2 + float + float
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Groups
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
                { binding: 1, resource: { buffer: this.paramBuffer } },
            ]
        });

        // Pipelines
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
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            }
        });

        // Add depth texture
        this.depthTexture = this.device.createTexture({
            size: [this.gpuCanvas.width, this.gpuCanvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `position: absolute; bottom: 10px; right: 10px; color: #ff5555; background: rgba(0,0,0,0.8); padding: 5px;`;
        msg.innerText = "WebGPU Not Available";
        this.container.appendChild(msg);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        if (w === 0 || h === 0) return;

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

            // Recreate depth texture
            if (this.device) {
                if (this.depthTexture) this.depthTexture.destroy();
                this.depthTexture = this.device.createTexture({
                    size: [this.gpuCanvas.width, this.gpuCanvas.height],
                    format: 'depth24plus',
                    usage: GPUTextureUsage.RENDER_ATTACHMENT,
                });
            }
        }
    }

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

    multiplyGL(a, b) {
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

        const time = (Date.now() - this.startTime) * 0.001;
        const aspect = this.canvasSize.width / this.canvasSize.height;

        // Camera
        const view = this.lookAt([0, 0, 12], [0, 0, 0], [0, 1, 0]);
        const proj = this.perspective(Math.PI / 4, aspect, 0.1, 100.0);

        // 1. WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.useProgram(this.glProgram);

            const loc = (n) => this.gl.getUniformLocation(this.glProgram, n);
            const model = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];

            this.gl.uniformMatrix4fv(loc('u_model'), false, new Float32Array(model));
            this.gl.uniformMatrix4fv(loc('u_view'), false, new Float32Array(view));
            this.gl.uniformMatrix4fv(loc('u_proj'), false, new Float32Array(proj));
            this.gl.uniform1f(loc('u_time'), time);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. WebGPU Render
        if (this.device && this.renderPipeline && this.gpuCanvas.width > 0) {
            const vp = this.multiplyGL(proj, view);

            const paramsData = new Float32Array(20);
            paramsData.set(vp, 0);
            paramsData[16] = this.mouse.x;
            paramsData[17] = this.mouse.y;
            paramsData[18] = time;
            paramsData[19] = this.mouse.isDown ? 1.0 : 0.0;

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
                }],
                depthStencilAttachment: {
                    view: this.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                }
            });
            rp.setPipeline(this.renderPipeline);
            rp.setBindGroup(0, this.computeBindGroup);
            rp.draw(6, this.particleCount);
            rp.end();

            this.device.queue.submit([cmd.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);

        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mouseup', this.handleMouseUp);

        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
        if (this.device) {
            this.device.destroy();
        }
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.PlanetaryTerraformingExperiment = PlanetaryTerraformingExperiment;
}
