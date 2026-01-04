/**
 * Singularity Reactor
 * Hybrid WebGL2 (Structure) + WebGPU (Core Particle Simulation)
 */

export class SingularityReactor {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };
        this.numParticles = options.numParticles || 100000;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndicesCount = 0;

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
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050508';

        console.log("SingularityReactor: Initializing...");

        // 1. WebGL2 (Background Structure)
        this.initWebGL2();

        // 2. WebGPU (Foreground Core)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("SingularityReactor: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("SingularityReactor: WebGPU not available.");
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();
        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Reactor Ring)
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
        const torus = this.createTorus(2.2, 0.1, 30, 60);

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(torus.vertices), this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);

        const normalBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, normalBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(torus.normals), this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(1);
        this.gl.vertexAttribPointer(1, 3, this.gl.FLOAT, false, 0, 0);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(torus.indices), this.gl.STATIC_DRAW);

        this.glVao = vao;
        this.glIndicesCount = torus.indices.length;

        // Shaders
        const vs = `#version 300 es
            in vec3 a_pos;
            in vec3 a_normal;

            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_projection;

            out vec3 v_normal;
            out vec3 v_pos;

            void main() {
                v_pos = vec3(u_model * vec4(a_pos, 1.0));
                v_normal = mat3(transpose(inverse(u_model))) * a_normal;
                gl_Position = u_projection * u_view * vec4(v_pos, 1.0);
            }
        `;

        const fs = `#version 300 es
            precision highp float;

            in vec3 v_normal;
            in vec3 v_pos;

            uniform float u_time;

            out vec4 outColor;

            void main() {
                vec3 norm = normalize(v_normal);
                vec3 viewDir = normalize(-v_pos); // Approx view pos at 0,0,0 relative

                // Sci-fi metallic material
                vec3 lightDir = normalize(vec3(sin(u_time), 1.0, cos(u_time)));
                float diff = max(dot(norm, lightDir), 0.0);

                // Rim light
                float rim = 1.0 - max(dot(viewDir, norm), 0.0);
                rim = pow(rim, 3.0);

                // Pulse effect
                float pulse = sin(v_pos.y * 10.0 + u_time * 5.0) * 0.5 + 0.5;

                vec3 baseColor = vec3(0.1, 0.1, 0.15);
                vec3 glowColor = vec3(0.0, 0.5, 1.0);

                vec3 finalColor = baseColor + diff * 0.2 + rim * glowColor * (0.5 + pulse * 0.5);

                outColor = vec4(finalColor, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);
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

                const centerX = radius * Math.cos(u);
                const centerZ = radius * Math.sin(u);

                const x = (radius + tube * Math.cos(v)) * Math.cos(u);
                const y = tube * Math.sin(v);
                const z = (radius + tube * Math.cos(v)) * Math.sin(u);

                vertices.push(x, y, z);

                const nx = x - centerX;
                const ny = y;
                const nz = z - centerZ;
                const len = Math.sqrt(nx*nx + ny*ny + nz*nz);
                normals.push(nx/len, ny/len, nz/len);
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

        return { vertices, normals, indices };
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vs);
        this.gl.attachShader(prog, fs);
        this.gl.linkProgram(prog);
        return prog;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Particle Core)
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

        // WGSL Shaders
        const shaderCode = `
            struct Particle {
                pos: vec2f,
                vel: vec2f,
                life: f32,
                dummy: f32,
            }

            struct SimParams {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: SimParams;

            // Compute Shader
            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Attraction to center (Singularity)
                let center = vec2f(0.0, 0.0);
                let diff = center - p.pos;
                let distSq = dot(diff, diff);
                let dist = sqrt(distSq);

                // Spiral force
                let dir = normalize(diff);
                let tangent = vec2f(-dir.y, dir.x);

                // Mouse influence (Repel/Attract)
                let mousePos = vec2f(params.mouseX, params.mouseY);
                let mDiff = mousePos - p.pos;
                let mDist = length(mDiff);
                let mForce = normalize(mDiff) / (mDist + 0.1) * 2.0;

                // Physics
                let force = dir * (1.0 / (distSq + 0.01)) * 0.05 + tangent * 0.5 + mForce;

                p.vel = p.vel + force * params.dt;
                p.vel = p.vel * 0.98; // Drag
                p.pos = p.pos + p.vel * params.dt;

                // Reset if too close to center
                if (dist < 0.05 || abs(p.pos.x) > 2.0 || abs(p.pos.y) > 2.0) {
                    // Respawn on ring
                    let angle = params.time * 2.0 + f32(i) * 0.01;
                    let r = 1.5 + sin(f32(i) * 0.1) * 0.2;
                    p.pos = vec2f(cos(angle) * r, sin(angle) * r);
                    p.vel = -normalize(p.pos) * 0.5; // Initial inward velocity
                }

                particles[i] = p;
            }

            // Vertex Shader
            struct VSOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs_main(
                @location(0) pos: vec2f,
                @location(1) vel: vec2f
            ) -> VSOut {
                var out: VSOut;
                out.pos = vec4f(pos, 0.0, 1.0);

                let speed = length(vel);
                let energy = smoothstep(0.0, 2.0, speed);

                // Color gradient based on energy: Orange -> Purple -> White
                let c1 = vec3f(1.0, 0.3, 0.0);
                let c2 = vec3f(0.8, 0.0, 1.0);
                let c3 = vec3f(1.0, 1.0, 1.0);

                let col = mix(c1, c2, energy);
                let finalCol = mix(col, c3, max(0.0, energy - 0.5) * 2.0);

                out.color = vec4f(finalCol, 1.0); // Points are alpha 1.0, blend handles add
                return out;
            }

            // Fragment Shader
            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const module = this.device.createShaderModule({ code: shaderCode });

        // Buffers
        const pSize = 4 * 4; // 4 floats
        const initialData = new Float32Array(this.numParticles * 4);
        for(let i=0; i<this.numParticles; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = 1.0 + Math.random();
            initialData[i*4+0] = Math.cos(angle) * r; // x
            initialData[i*4+1] = Math.sin(angle) * r; // y
            initialData[i*4+2] = 0; // vx
            initialData[i*4+3] = 0; // vy
        }

        this.particleBuffer = this.device.createBuffer({
            size: initialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Layouts & Pipelines
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.computePipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_main' }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: pSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' }
                    ]
                }]
            },
            fragment: {
                module,
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

        return true;
    }

    addWebGPUNotSupportedMessage() {
        // Implementation similar to hybrid-magnetic-field
        const msg = document.createElement('div');
        msg.innerText = "WebGPU Not Supported - Core Simulation Disabled";
        msg.style.cssText = "position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); color:red; font-family:monospace; background:rgba(0,0,0,0.8); padding:1em;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        if (this.glCanvas) {
            this.glCanvas.width = width * dpr;
            this.glCanvas.height = height * dpr;
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = width * dpr;
            this.gpuCanvas.height = height * dpr;
        }

        this.canvasSize = { width, height };
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // 1. Render WebGL2 Background
        if (this.gl && this.glProgram) {
            this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.useProgram(this.glProgram);

            // Matrices
            const aspect = this.glCanvas.width / this.glCanvas.height;
            const projection = this.perspective(45 * Math.PI/180, aspect, 0.1, 100.0);

            // Camera (View)
            const camZ = 5.0;
            const view = new Float32Array([
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -camZ, 1
            ]);

            // Model (Rotation)
            const c = Math.cos(time * 0.2);
            const s = Math.sin(time * 0.2);
            const model = new Float32Array([
                c, 0, -s, 0,
                0, 1, 0, 0,
                s, 0, c, 0,
                0, 0, 0, 1
            ]);
            // Also tilt it a bit
            // ... (Simplified for brevity, just Y rotation)

            const locModel = this.gl.getUniformLocation(this.glProgram, 'u_model');
            const locView = this.gl.getUniformLocation(this.glProgram, 'u_view');
            const locProj = this.gl.getUniformLocation(this.glProgram, 'u_projection');
            const locTime = this.gl.getUniformLocation(this.glProgram, 'u_time');

            this.gl.uniformMatrix4fv(locModel, false, model);
            this.gl.uniformMatrix4fv(locView, false, view);
            this.gl.uniformMatrix4fv(locProj, false, projection);
            this.gl.uniform1f(locTime, time);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glIndicesCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU Foreground
        if (this.device && this.context && this.renderPipeline) {
            // Update Uniforms
            const params = new Float32Array([0.016, time, this.mouse.x, this.mouse.y]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const encoder = this.device.createCommandEncoder();

            // Compute Pass
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            // Render Pass
            const view = this.context.getCurrentTexture().createView();
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: view,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
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
        cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);

        // Clean up GL
        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
        // Clean up GPU
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }

    // Helper: Matrix Perspective
    perspective(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ]);
    }
}
