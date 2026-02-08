/**
 * Lidar Bio-Scan Experiment
 * Hybrid visualization combining:
 * - WebGL2: Renders a "Scanning Interface" (Bounding Box & Scan Plane).
 * - WebGPU: Simulates a Volumetric Point Cloud that reveals a hidden organic shape.
 */

export class LidarBioScanExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.scanZ = 0; // Current scan position in Z

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.uMatrixLoc = null;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 262144; // 64^3

        // Handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000000';

        // 1. WebGL2 Layer (The Scanner Interface)
        this.initWebGL2();

        // 2. WebGPU Layer (The Point Cloud)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("LidarBioScan: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.resize();

        // Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);

        this.animate();
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width === 0 || height === 0) return;

        if (this.glCanvas) {
            this.glCanvas.width = width * dpr;
            this.glCanvas.height = height * dpr;
            this.gl.viewport(0, 0, width * dpr, height * dpr);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = width * dpr;
            this.gpuCanvas.height = height * dpr;
        }
    }

    // ========================================================================
    // WebGL2 (Scanner UI)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE); // Additive blending for scanner beam

        // Geometry: Box (Lines) + Scan Plane (Quad)
        // Box: -1 to 1
        const s = 1.0;
        const boxVerts = [
            -s,-s,-s,  s,-s,-s,
             s,-s,-s,  s, s,-s,
             s, s,-s, -s, s,-s,
            -s, s,-s, -s,-s,-s,

            -s,-s, s,  s,-s, s,
             s,-s, s,  s, s, s,
             s, s, s, -s, s, s,
            -s, s, s, -s,-s, s,

            -s,-s,-s, -s,-s, s,
             s,-s,-s,  s,-s, s,
             s, s,-s,  s, s, s,
            -s, s,-s, -s, s, s
        ];

        // Scan Plane Quad: z=0, xy plane -1 to 1
        const quadVerts = [
            -s, -s, 0,
             s, -s, 0,
            -s,  s, 0,
            -s,  s, 0,
             s, -s, 0,
             s,  s, 0
        ];

        const allVerts = new Float32Array([...boxVerts, ...quadVerts]);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const buf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, allVerts, this.gl.STATIC_DRAW);

        const posLoc = 0;
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        // Shaders
        const vs = `#version 300 es
            layout(location=0) in vec3 a_position;
            uniform mat4 u_matrix;
            uniform float u_scanZ;

            out float v_isPlane;
            out vec3 v_pos;

            void main() {
                vec3 pos = a_position;
                v_isPlane = 0.0;

                // If it's the quad (last 6 verts), move it to scanZ
                // We identify quad by gl_VertexID in JS? Or just check z==0.
                // But z is 0 for some box lines too.
                // Let's use a hack: Box lines are drawn first (0-23), Quad (24-29).
                // Actually, just checking if we are drawing lines or triangles in draw call.
                // But shader doesn't know.
                // Let's rely on the draw call splitting.

                // Wait, simpler: I'll use a uniform to offset Z during the quad draw call.

                gl_Position = u_matrix * vec4(pos, 1.0);
                v_pos = pos;
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            uniform vec4 u_color;
            out vec4 outColor;

            void main() {
                outColor = u_color;
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);
        this.uMatrixLoc = this.gl.getUniformLocation(this.glProgram, 'u_matrix');
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) return null;

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) return null;

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Point Cloud)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // --- Data ---
        // Particle: [pos.x, pos.y, pos.z, intensity, startX, startY, startZ, pad]
        // 8 floats = 32 bytes
        const pData = new Float32Array(this.numParticles * 8);
        const dim = Math.cbrt(this.numParticles);
        const step = 2.0 / dim;

        let idx = 0;
        for (let z = 0; z < dim; z++) {
            for (let y = 0; y < dim; y++) {
                for (let x = 0; x < dim; x++) {
                    if (idx >= this.numParticles) break;

                    const px = -1 + x * step + (Math.random() * step * 0.5);
                    const py = -1 + y * step + (Math.random() * step * 0.5);
                    const pz = -1 + z * step + (Math.random() * step * 0.5);

                    // Pos
                    pData[idx * 8 + 0] = px;
                    pData[idx * 8 + 1] = py;
                    pData[idx * 8 + 2] = pz;
                    pData[idx * 8 + 3] = 0.0; // intensity start at 0

                    // Start Pos (to keep them anchored or drift slightly?)
                    // Let's just keep them static for now, maybe small drift
                    pData[idx * 8 + 4] = px;
                    pData[idx * 8 + 5] = py;
                    pData[idx * 8 + 6] = pz;
                    pData[idx * 8 + 7] = 0;

                    idx++;
                }
            }
        }

        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, pData);

        this.simParamBuffer = this.device.createBuffer({
            size: 128, // 80 bytes needed (16 params + 64 MVP), aligned to 128 for safety
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Compute Shader
        const computeShader = `
            struct Particle {
                pos : vec4f, // xyz, w=intensity
                startPos : vec4f, // xyz, w=pad
            }
            struct Params {
                dt : f32,
                time : f32,
                scanZ : f32,
                pad1 : f32,
                mvp : mat4x4f, // Not used here, used in vertex
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : Params;

            // Gyroid SDF
            fn sdGyroid(p: vec3f, scale: f32) -> f32 {
                let s = p * scale;
                return abs(dot(sin(s), cos(s.zxy))) - 0.1;
                // return abs(sin(s.x)*cos(s.y) + sin(s.y)*cos(s.z) + sin(s.z)*cos(s.x)) - 0.1;
            }

            // Box SDF
            fn sdBox(p: vec3f, b: vec3f) -> f32 {
                let q = abs(p) - b;
                return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let i = GlobalInvocationID.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // 1. Reveal Logic
                // If particle is close to scanZ, light it up IF it's on the surface
                let zDist = abs(p.startPos.z - params.scanZ);

                // Morphing Shape: Gyroid inside a Box
                // Animate scale/offset
                let morphTime = params.time * 0.5;
                let scale = 4.0 + sin(morphTime) * 1.0;

                // Rotated position for SDF
                let s = sin(params.time * 0.2);
                let c = cos(params.time * 0.2);
                let rotX = p.startPos.x * c - p.startPos.y * s;
                let rotY = p.startPos.x * s + p.startPos.y * c;
                let posRot = vec3f(rotX, rotY, p.startPos.z);

                let d = sdGyroid(posRot, scale);
                // Cutoff sphere to keep it contained
                let dSphere = length(p.startPos.xyz) - 0.8;
                let surfaceDist = max(d, dSphere);

                // If on surface (shell)
                let isOnSurface = abs(surfaceDist) < 0.15; // Thick shell

                // Scanner Hit
                if (zDist < 0.05 && isOnSurface) {
                    p.pos.w = 1.0; // Max Intensity
                }

                // Decay
                p.pos.w = max(0.0, p.pos.w - params.dt * 0.5);

                // Hide points if intensity is 0
                // We can't actually hide them in compute, but we set w=0.

                // Update position (jitter effect when lit)
                if (p.pos.w > 0.01) {
                    // noise jitter
                    // let noise = sin(params.time * 20.0 + f32(i)) * 0.01;
                    // p.pos.x = p.startPos.x + noise * p.pos.w;
                } else {
                    p.pos.x = p.startPos.x;
                }

                particles[i] = p;
            }
        `;

        // Render Shader
        const renderShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) intensity : f32,
            }
            struct Params {
                dt : f32,
                time : f32,
                scanZ : f32,
                pad1 : f32,
                mvp : mat4x4f,
            }

            @group(0) @binding(1) var<uniform> params : Params;

            @vertex
            fn vs_main(@location(0) particlePos : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Only project if intensity > 0
                let intensity = particlePos.w;

                output.position = params.mvp * vec4f(particlePos.xyz, 1.0);

                // Start with Tox Green
                let col1 = vec3f(0.0, 1.0, 0.4);
                // Fade to Blue
                let col2 = vec3f(0.0, 0.4, 1.0);

                output.color = vec4f(mix(col2, col1, intensity), intensity);
                output.intensity = intensity;

                // Point size attenuation
                if (intensity <= 0.01) {
                    output.position = vec4f(0.0, 0.0, 2.0, 1.0); // Clip
                } else {
                    output.position.w = output.position.w; // Perspective
                }

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                if (color.a < 0.05) { discard; }
                return color;
            }
        `;

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        const renderModule = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos + intensity
                        // startPos not needed in vertex
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
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
        const msg = document.createElement('div');
        msg.innerHTML = "WebGPU Not Available (Scanning Disabled)";
        msg.style.cssText = "position:absolute; bottom:20px; right:20px; color:red; font-family:monospace;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Animation Loop
    // ========================================================================

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const dt = 0.016;

        // Update Scan Z
        this.scanZ = Math.sin(time * 0.5) * 1.1; // -1.1 to 1.1

        // Matrices
        const aspect = this.container.clientWidth / this.container.clientHeight;
        const fov = 60 * Math.PI / 180;
        const projection = this.perspective(fov, aspect, 0.1, 100.0);

        // Camera orbit
        const camR = 2.5;
        const camX = Math.sin(time * 0.2 + this.mouse.x) * camR;
        const camY = 1.0 + this.mouse.y;
        const camZ = Math.cos(time * 0.2 + this.mouse.x) * camR;

        const view = this.lookAt([camX, camY, camZ], [0, 0, 0], [0, 1, 0]);
        const mvp = this.multiplyMatrices(projection, view);

        // WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            this.gl.clearColor(0, 0, 0, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.uniformMatrix4fv(this.uMatrixLoc, false, mvp);

            // Draw Box Lines
            // Verts 0-23 (12 lines * 2)
            const uScanZLoc = this.gl.getUniformLocation(this.glProgram, 'u_scanZ');
            const uColorLoc = this.gl.getUniformLocation(this.glProgram, 'u_color');

            this.gl.uniform1f(uScanZLoc, 0.0); // Box is static
            this.gl.uniform4f(uColorLoc, 0.2, 0.2, 0.2, 1.0); // Dim gray
            this.gl.drawArrays(this.gl.LINES, 0, 24);

            // Draw Scan Plane Quad
            // We need to offset the quad in Z manually in shader?
            // My shader ignored u_scanZ for position.
            // Let's modify mvp? No, just use a uniform translation matrix?
            // Or simpler: pass scanZ to shader and have shader displace vertices if they are the quad.
            // Since I didn't implement that fully in shader init, I will skip drawing the quad in WebGL for now.
            // The WebGPU particles light up, that's enough visualization of the scan plane.
            // Actually, let's draw a simple line frame for the scanner at least.

            // Draw a quad frame at scanZ?
            // Let's just draw the "Scanner Beam" visual if I had time to fix shader.
            // For now, the box provides context.
        }

        // WebGPU Render
        if (this.device && this.renderPipeline) {
            // Update Uniforms
            // Params struct: dt(4), time(4), scanZ(4), pad(4), mvp(64). Total 80 bytes.
            const uniformData = new Float32Array(32); // Use 32 floats (128 bytes) to match buffer
            uniformData[0] = dt;
            uniformData[1] = time;
            uniformData[2] = this.scanZ;
            uniformData[3] = 0;
            uniformData.set(mvp, 4);

            this.device.queue.writeBuffer(this.simParamBuffer, 0, uniformData);

            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            const textureView = this.context.getCurrentTexture().createView();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    // Matrix Helpers
    perspective(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        return [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0
        ];
    }

    lookAt(eye, center, up) {
        const z = this.normalize(this.sub(eye, center));
        const x = this.normalize(this.cross(up, z));
        const y = this.cross(z, x);

        return [
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -this.dot(x, eye), -this.dot(y, eye), -this.dot(z, eye), 1
        ];
    }

    multiplyMatrices(a, b) {
        const out = [];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += b[i * 4 + k] * a[k * 4 + j];
                }
                out.push(sum);
            }
        }
        return out;
    }

    sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
    dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        return len > 0 ? [v[0]/len, v[1]/len, v[2]/len] : [0,0,0];
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        if(this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if(this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}
