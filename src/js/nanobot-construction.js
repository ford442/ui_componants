/**
 * Nanobot Construction Experiment
 * Combines WebGL2 (Blueprint Wireframe) and WebGPU (Assembler Swarm).
 */

export class NanobotConstruction {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.isMouseDown = false;
        this.canvasSize = { width: 0, height: 0 };
        this.numParticles = options.numParticles || 50000;

        // Geometry Data
        this.blueprintVertices = null; // Float32Array
        this.blueprintIndices = null; // Uint16Array
        this.targetPositions = null; // Float32Array for GPU

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
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.targetBuffer = null;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050505';

        // Generate Geometry (Icosahedron)
        this.generateGeometry();

        // 1. Initialize WebGL2 (Blueprint)
        this.initWebGL2();

        // 2. Initialize WebGPU (Swarm)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("NanobotConstruction: WebGPU initialization error:", e);
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
        // Touch support
        window.addEventListener('touchstart', this.handleMouseDown);
        window.addEventListener('touchend', this.handleMouseUp);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    onMouseDown() { this.isMouseDown = true; }
    onMouseUp() { this.isMouseDown = false; }

    generateGeometry() {
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;

        // 12 Vertices of Icosahedron
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

        // Normalize vertices
        for (let i = 0; i < vertices.length; i += 3) {
            const length = Math.sqrt(vertices[i]**2 + vertices[i+1]**2 + vertices[i+2]**2);
            vertices[i] /= length;
            vertices[i+1] /= length;
            vertices[i+2] /= length;
        }

        // 20 Faces (Indices) - needed for lines? No, we want Edges.
        // Actually, let's just use indices for TRIANGLES then draw LINES with barycentric logic or just draw outlines?
        // Simpler: Define edges manually or extract them.
        // An icosahedron has 30 edges.

        // Let's create indices for GL_LINES
        // Each vertex connects to 5 neighbors.
        // Distance check method to find edges (dist = 2.0 / sin(72/2)? approx 1.05 * R? No side length of unit icosahedron is ~1.05)

        const edges = [];
        const indices = [];

        // Find edges by distance
        const vCount = vertices.length / 3;
        for (let i = 0; i < vCount; i++) {
            for (let j = i + 1; j < vCount; j++) {
                const dx = vertices[i*3] - vertices[j*3];
                const dy = vertices[i*3+1] - vertices[j*3+1];
                const dz = vertices[i*3+2] - vertices[j*3+2];
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

                // Edge length of unit sphere icosahedron is approx 1.051
                // Threshold 1.1
                if (dist < 1.1) {
                    indices.push(i, j);
                    edges.push({ v1: i, v2: j });
                }
            }
        }

        this.blueprintVertices = new Float32Array(vertices);
        this.blueprintIndices = new Uint16Array(indices);
        this.glIndexCount = indices.length;

        // Generate Target Positions for Particles along these edges
        // We have 30 edges.
        // 50,000 particles.
        // ~1666 particles per edge.
        const targetPos = new Float32Array(this.numParticles * 4); // vec4 alignment
        const particlesPerEdge = Math.floor(this.numParticles / edges.length);

        let pIdx = 0;
        for (let e = 0; e < edges.length; e++) {
            const idx1 = edges[e].v1 * 3;
            const idx2 = edges[e].v2 * 3;
            const p1 = [vertices[idx1], vertices[idx1+1], vertices[idx1+2]];
            const p2 = [vertices[idx2], vertices[idx2+1], vertices[idx2+2]];

            for (let i = 0; i < particlesPerEdge; i++) {
                if (pIdx >= this.numParticles) break;

                const t = Math.random(); // Random point on edge
                targetPos[pIdx*4+0] = p1[0] + (p2[0] - p1[0]) * t;
                targetPos[pIdx*4+1] = p1[1] + (p2[1] - p1[1]) * t;
                targetPos[pIdx*4+2] = p1[2] + (p2[2] - p1[2]) * t;
                targetPos[pIdx*4+3] = 1.0; // padding/flag

                pIdx++;
            }
        }
        // Fill remainder
        while (pIdx < this.numParticles) {
            targetPos[pIdx*4+0] = 0;
            targetPos[pIdx*4+1] = 0;
            targetPos[pIdx*4+2] = 0;
            targetPos[pIdx*4+3] = 1.0;
            pIdx++;
        }

        this.targetPositions = targetPos;
    }

    // ========================================================================
    // WebGL2 (Blueprint)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;`;
        this.container.appendChild(this.glCanvas);
        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        const vs = `#version 300 es
            in vec3 a_position;
            uniform mat4 u_mvp;
            void main() {
                gl_Position = u_mvp * vec4(a_position, 1.0);
            }
        `;
        const fs = `#version 300 es
            precision highp float;
            uniform float u_time;
            out vec4 outColor;
            void main() {
                // Pulse opacity
                float alpha = 0.1 + 0.1 * sin(u_time * 2.0);
                outColor = vec4(0.0, 1.0, 0.5, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const vb = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vb);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.blueprintVertices, this.gl.STATIC_DRAW);
        const loc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 3, this.gl.FLOAT, false, 0, 0);

        const ib = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, ib);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, this.blueprintIndices, this.gl.STATIC_DRAW);
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Swarm)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none;`;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        const wgslCommon = `
            struct Particle {
                pos : vec4f,
                vel : vec4f,
            }
            struct Params {
                mvp : mat4x4f,
                mouse : vec2f,
                time : f32,
                isMouseDown : f32,
            }
        `;

        const computeShader = `
            ${wgslCommon}
            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<storage, read> targets : array<vec4f>;
            @group(0) @binding(2) var<uniform> params : Params;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];
                let target = targets[idx].xyz * 2.0; // Scale up geometry

                // Interaction
                var force = vec3f(0.0);

                // Seek Target
                let toTarget = target - p.pos.xyz;
                let dist = length(toTarget);
                let dir = normalize(toTarget);

                // Spring force
                force += dir * dist * 2.0;

                // Mouse Repulsion (Disruptor Field)
                if (params.isMouseDown > 0.5) {
                    // Project mouse ray? Approximate with sphere at z=2
                    // For simplicity, just repel from center if mouse is active?
                    // Or map mouse 2D to 3D roughly
                    let mPos = vec3f(params.mouse.x * 5.0, params.mouse.y * 5.0, 2.0);
                    let toMouse = p.pos.xyz - mPos;
                    let mDist = length(toMouse);
                    force += normalize(toMouse) * (10.0 / (mDist * mDist + 0.1));

                    // Add noise/chaos
                    force += vec3f(sin(params.time * 10.0 + f32(idx)), cos(params.time * 20.0 + f32(idx)), 0.0) * 5.0;
                }

                // Damping
                p.vel = p.vel * 0.9 + vec4f(force, 0.0) * 0.016;
                p.pos = p.pos + p.vel * 0.016;

                particles[idx] = p;
            }
        `;

        const renderShader = `
            ${wgslCommon}
            @group(0) @binding(2) var<uniform> params : Params;
            @group(0) @binding(1) var<storage, read> targets : array<vec4f>;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f, @builtin(vertex_index) vIdx : u32) -> VertexOutput {
                var out : VertexOutput;
                out.position = params.mvp * vec4f(pos.xyz, 1.0);

                // Color based on distance to target
                let target = targets[vIdx].xyz * 2.0;
                let dist = distance(pos.xyz, target);

                // Locked (close) = Cyan/Green. Far = Orange/Red.
                let t = smoothstep(0.0, 1.0, dist);
                let cLocked = vec3f(0.0, 1.0, 0.8);
                let cFar = vec3f(1.0, 0.2, 0.0);

                out.color = vec4f(mix(cLocked, cFar, t), 1.0);

                // Point size hack for WebGPU?
                // WebGPU renders 1px points by default.
                // To get larger points, we need a billboard approach, but for 50k particles, 1px is fine.
                out.position.z = out.position.z - 0.001; // Bias slightly in front of lines

                return out;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const pSize = 32; // 2 * vec4f
        const initP = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            // Random start pos
            initP[i*8+0] = (Math.random()-0.5) * 10.0;
            initP[i*8+1] = (Math.random()-0.5) * 10.0;
            initP[i*8+2] = (Math.random()-0.5) * 10.0;
            initP[i*8+3] = 1.0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: initP.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initP);

        this.targetBuffer = this.device.createBuffer({
            size: this.targetPositions.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.targetBuffer, 0, this.targetPositions);

        this.simParamBuffer = this.device.createBuffer({
            size: 80, // mat4 (64) + vec2 (8) + time (4) + isMouseDown (4)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Pipeline
        const moduleCompute = this.device.createShaderModule({ code: computeShader });
        const moduleRender = this.device.createShaderModule({ code: renderShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'storage' } }, // Particles
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // Targets
                { binding: 2, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }, // Params
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.targetBuffer } },
                { binding: 2, resource: { buffer: this.simParamBuffer } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: moduleCompute, entryPoint: 'main' }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: moduleRender,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: moduleRender,
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
            depthStencil: undefined // No depth for additive particles
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const el = document.createElement('div');
        el.className = 'webgpu-error';
        el.textContent = "WebGPU Not Available - Swarm Disabled";
        this.container.appendChild(el);
        const status = document.getElementById('status-indicator');
        if(status) status.textContent = "BLUEPRINT ONLY";
    }

    // ========================================================================
    // Loop
    // ========================================================================

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.canvasSize.width = w;
        this.canvasSize.height = h;

        const dw = Math.floor(w * dpr);
        const dh = Math.floor(h * dpr);

        if(this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            this.gl.viewport(0, 0, dw, dh);
        }
        if(this.gpuCanvas) {
            this.gpuCanvas.width = dw;
            this.gpuCanvas.height = dh;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const aspect = this.canvasSize.width / this.canvasSize.height;

        // Camera
        const fov = 45 * Math.PI / 180;
        const zNear = 0.1;
        const zFar = 100.0;
        const f = 1.0 / Math.tan(fov / 2);
        const rangeInv = 1 / (zNear - zFar);

        // Projection Matrix
        const p = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (zNear + zFar) * rangeInv, -1,
            0, 0, zNear * zFar * rangeInv * 2, 0
        ];

        // View Matrix (Orbit)
        const radius = 6.0;
        const cx = Math.sin(time * 0.2) * radius;
        const cz = Math.cos(time * 0.2) * radius;
        const cy = 2.0;

        const eye = [cx, cy, cz];
        const center = [0, 0, 0];
        const up = [0, 1, 0];

        // LookAt
        const z0 = eye[0]-center[0], z1 = eye[1]-center[1], z2 = eye[2]-center[2];
        const len = 1/Math.sqrt(z0*z0 + z1*z1 + z2*z2);
        const zx=z0*len, zy=z1*len, zz=z2*len;
        const x0=up[1]*zz - up[2]*zy, x1=up[2]*zx - up[0]*zz, x2=up[0]*zy - up[1]*zx;
        const lenX = 1/Math.sqrt(x0*x0+x1*x1+x2*x2);
        const xx=x0*lenX, xy=x1*lenX, xz=x2*lenX;
        const y0=zy*xz - zz*xy, y1=zz*xx - zx*xz, y2=zx*xy - zy*xx;
        const lenY = 1/Math.sqrt(y0*y0+y1*y1+y2*y2);
        const yx=y0*lenY, yy=y1*lenY, yz=y2*lenY;

        const v = [
            xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx*eye[0]+xy*eye[1]+xz*eye[2]),
            -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
            -(zx*eye[0]+zy*eye[1]+zz*eye[2]),
            1
        ];

        // Multiply P * V (Column Major)
        const mvp = new Float32Array(16);
        // Simple mult
        for(let row=0; row<4; row++) {
            for(let col=0; col<4; col++) {
                let sum = 0;
                for(let k=0; k<4; k++) sum += v[row + k*4] * p[k + row*4]; // Wait, logic error in mult indices
                // Standard: C[col][row] = sum(A[k][row] * B[col][k])
                // Indices are flat: [0..15]. idx = col*4 + row.
                // Output MVP = P * V.
                // out[col*4+row] = sum(P[k*4+row] * V[col*4+k])
                sum = 0;
                for(let k=0; k<4; k++) {
                    sum += p[k*4 + row] * v[col*4 + k];
                }
                mvp[col*4 + row] = sum;
            }
        }

        // Scale the model up for blueprint (2.0)
        // Apply Model Matrix (Scale 2.0)
        // mvp = mvp * scale
        const mvpScaled = new Float32Array(mvp);
        for(let i=0; i<12; i++) mvpScaled[i] *= 2.0; // Scale x,y,z columns

        // 1. WebGL Draw
        if (this.gl) {
            this.gl.useProgram(this.glProgram);
            const loc = this.gl.getUniformLocation(this.glProgram, 'u_mvp');
            this.gl.uniformMatrix4fv(loc, false, mvpScaled);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);

            this.gl.clearColor(0,0,0,0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT); // Transparent clear
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. WebGPU Draw
        if (this.device && this.renderPipeline) {
            // Write Uniforms
            const params = new Float32Array(20);
            params.set(mvpScaled, 0); // Use same MVP
            params[16] = this.mouse.x;
            params[17] = this.mouse.y;
            params[18] = time;
            params[19] = this.isMouseDown ? 1.0 : 0.0;

            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const cPass = commandEncoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            // Render
            const rPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.computeBindGroup);
            rPass.draw(this.numParticles);
            rPass.end();

            this.device.queue.submit([commandEncoder.finish()]);

            // UI Status Update
            const status = document.getElementById('status-indicator');
            if(status) {
                if (this.isMouseDown) {
                    status.textContent = "DISRUPTING";
                    status.style.color = "#ff4444";
                } else {
                    status.textContent = "ASSEMBLING";
                    status.style.color = "#00ff88";
                }
            }
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);
        window.removeEventListener('mousedown', this.handleMouseDown);
        window.removeEventListener('mouseup', this.handleMouseUp);
        window.removeEventListener('touchstart', this.handleMouseDown);
        window.removeEventListener('touchend', this.handleMouseUp);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.NanobotConstruction = NanobotConstruction;
}
