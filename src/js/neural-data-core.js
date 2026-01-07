/**
 * Neural Data Core Experiment
 * Combines WebGL2 for the central core visualization (Raymarching) and WebGPU for particle data flow.
 */

class NeuralDataCore {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };

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
        this.numParticles = options.numParticles || 100000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000000';

        console.log("NeuralDataCore: Initializing...");

        // 1. Initialize WebGL2 Layer (Background)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("NeuralDataCore: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("NeuralDataCore: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("NeuralDataCore: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        // Normalize mouse coordinates to [-1, 1] (UV space centered)
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1); // Flip Y to match WebGL/WebGPU coords
        this.mouse.x = x;
        this.mouse.y = y;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Central Core)
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
        if (!this.gl) {
            console.warn("NeuralDataCore: WebGL2 not supported.");
            return;
        }

        // Setup simple quad
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        // Vertex Shader
        const vsSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        // Fragment Shader - Raymarched Sphere with Tech Patterns
        const fsSource = `#version 300 es
            precision highp float;

            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_mouse;
            uniform vec2 u_resolution;

            out vec4 outColor;

            #define PI 3.14159265359

            mat2 rotate2D(float angle) {
                float s = sin(angle);
                float c = cos(angle);
                return mat2(c, -s, s, c);
            }

            // SDF for a sphere
            float sdSphere(vec3 p, float r) {
                return length(p) - r;
            }

            // Grid pattern logic
            float gridPattern(vec3 p) {
                vec3 grid = abs(fract(p * 2.0) - 0.5);
                float d = min(min(grid.x, grid.y), grid.z);
                return smoothstep(0.02, 0.05, d);
            }

            // Map the scene
            float map(vec3 p) {
                vec3 pSphere = p;
                // Rotate the sphere slowly
                pSphere.xz *= rotate2D(u_time * 0.2);
                pSphere.xy *= rotate2D(u_time * 0.1);

                // Base sphere
                float d = sdSphere(pSphere, 1.0);

                // Add surface detail (displacement)
                float pattern = sin(pSphere.x * 10.0) * sin(pSphere.y * 10.0) * sin(pSphere.z * 10.0);
                d += pattern * 0.02;

                return d;
            }

            // Raymarching
            float rayMarch(vec3 ro, vec3 rd) {
                float dO = 0.0;
                for (int i = 0; i < 100; i++) {
                    vec3 p = ro + rd * dO;
                    float dS = map(p);
                    dO += dS;
                    if (dO > 100.0 || dS < 0.001) break;
                }
                return dO;
            }

            // Calculate normal
            vec3 getNormal(vec3 p) {
                float d = map(p);
                vec2 e = vec2(0.001, 0);
                vec3 n = d - vec3(
                    map(p - e.xyy),
                    map(p - e.yxy),
                    map(p - e.yyx)
                );
                return normalize(n);
            }

            void main() {
                vec2 uv = v_uv;
                if (u_resolution.y > 0.0) {
                    uv.x *= u_resolution.x / u_resolution.y;
                }

                // Camera setup
                vec3 ro = vec3(0.0, 0.0, -3.0); // Ray origin
                vec3 rd = normalize(vec3(uv, 1.0)); // Ray direction

                float d = rayMarch(ro, rd);

                vec3 col = vec3(0.0);

                if (d < 100.0) {
                    vec3 p = ro + rd * d;
                    vec3 n = getNormal(p);

                    // Simple lighting
                    vec3 lightPos = vec3(2.0, 2.0, -3.0);
                    vec3 l = normalize(lightPos - p);
                    float diff = max(dot(n, l), 0.0);

                    // Tech texture effect based on position
                    vec3 pRot = p;
                    pRot.xz *= rotate2D(u_time * 0.2);
                    pRot.xy *= rotate2D(u_time * 0.1);

                    float grid = gridPattern(pRot);

                    // Color scheme: Dark blue/purple core with cyan grid
                    vec3 baseColor = vec3(0.1, 0.0, 0.2);
                    vec3 gridColor = vec3(0.0, 0.8, 1.0);

                    // Glow based on normal
                    float rim = 1.0 - max(dot(n, -rd), 0.0);
                    rim = pow(rim, 3.0);

                    col = baseColor * diff;
                    col += gridColor * (1.0 - grid) * 0.8; // Grid lines are 0 in gridPattern
                    col += vec3(0.5, 0.8, 1.0) * rim;
                } else {
                    // Background glow
                    float dCenter = length(uv);
                    col = vec3(0.05, 0.0, 0.1) * (1.0 - dCenter * 0.5);
                }

                // Vignette
                float vig = 1.0 - smoothstep(0.5, 1.5, length(v_uv));
                col *= vig;

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('WebGL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('WebGL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        return program;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Orbital Particles)
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
            background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        let adapter;
        try {
            adapter = await navigator.gpu.requestAdapter();
        } catch (e) {
            console.warn("WebGPU Adapter request failed:", e);
            this.gpuCanvas.remove();
            return false;
        }

        if (!adapter) {
            this.gpuCanvas.remove();
            return false;
        }

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // WGSL Helper functions
        const commonWGSL = `
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }
        `;

        // COMPUTE SHADER
        const computeShaderCode = `
            ${commonWGSL}

            struct Particle {
                pos : vec4f, // x, y, z, w (unused)
                vel : vec4f, // vx, vy, vz, life
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                pad : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Gravity towards center
                let center = vec3f(0.0, 0.0, 0.0);
                let diff = center - p.pos.xyz;
                let dist = length(diff);
                let dir = normalize(diff);

                // Keep particles in a shell around radius 1.2
                let targetRadius = 1.2;
                let radiusDiff = dist - targetRadius;

                // Force to correct radius
                let radiusForce = dir * radiusDiff * 2.0; // Push towards shell

                // Orbital velocity
                // Cross product with Up vector to get tangent
                let tangent = normalize(cross(p.pos.xyz, vec3f(0.0, 1.0, 0.0)));
                // But add some variation so they aren't all in one plane
                let tangent2 = normalize(cross(p.pos.xyz, vec3f(1.0, 0.0, 0.0)));

                let orbitalForce = (tangent * 0.8 + tangent2 * 0.2) * 0.5;

                // Mouse Interaction (Repulsion)
                // Project 3D pos to 2D screen space roughly
                let screenX = p.pos.x; // Simplified projection
                let screenY = p.pos.y;
                // Correct for aspect in mouse calc
                let mX = params.mouseX * params.aspect; // Map -1..1 to aspect
                // But actually, params.mouseX is already normalized -1..1
                // Let's just do simple distance check in 2D projection

                let dx = p.pos.x - params.mouseX * 2.0; // * 2.0 to widen range
                let dy = p.pos.y - params.mouseY * 2.0;
                let mDist = sqrt(dx*dx + dy*dy);

                var mouseForce = vec3f(0.0);
                if (mDist < 0.5) {
                    mouseForce = normalize(p.pos.xyz - vec3f(params.mouseX * 2.0, params.mouseY * 2.0, 0.0)) * (1.0 - mDist/0.5) * 5.0;
                }

                // Update Velocity
                p.vel = p.vel + vec4f((radiusForce + orbitalForce + mouseForce) * params.dt, 0.0);

                // Damping
                p.vel = p.vel * 0.98;

                // Update Position
                p.pos = p.pos + p.vel * params.dt;

                // Reset if too far or NaN
                if (dist > 5.0 || dist < 0.1) {
                    // Reset to random shell pos
                    let theta = rand(vec2f(p.pos.x, params.time)) * 6.28;
                    let phi = rand(vec2f(p.pos.y, params.time)) * 3.14;
                    let r = 1.2;

                    p.pos = vec4f(
                        r * sin(phi) * cos(theta),
                        r * sin(phi) * sin(theta),
                        r * cos(phi),
                        1.0
                    );
                    p.vel = vec4f(0.0);
                }

                particles[index] = p;
            }
        `;

        // RENDER SHADER
        const drawShaderCode = `
            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                pad : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(
                @location(0) particlePos : vec4f,
                @location(1) particleVel : vec4f
            ) -> VertexOutput {
                var output : VertexOutput;

                // Simple perspective projection
                let pos = particlePos.xyz;

                // Camera at (0, 0, -3) looking at (0, 0, 0)
                let camPos = vec3f(0.0, 0.0, -3.0);

                // View Transform (Inverse of Camera Transform)
                let viewPos = pos - camPos;

                // Projection (Perspective)
                // fov = 60 degrees
                let fov = 1.0;
                let f = 1.0 / tan(fov / 2.0);
                let zNear = 0.1;
                let zFar = 100.0;

                // We need to handle aspect ratio here
                let x = viewPos.x * f / params.aspect; // Fix aspect distortion
                let y = viewPos.y * f;
                let z = viewPos.z;

                // W component for perspective division
                // Since we view along +Z relative to camera (viewPos.z > 0), w should be +z for standard perspective division
                let w = z;

                // Manual projection matrix multiplication result for Z
                // Z maps to 0..1 range for WebGPU
                // let z_ndc = (zFar / (zNear - zFar)) * z + (zNear * zFar / (zNear - zFar));
                // Simplified since we just want it on screen

                output.position = vec4f(x, y, viewPos.z * 0.1, w);

                // Color based on speed and depth
                let speed = length(particleVel.xyz);
                var col = vec3f(0.0, 0.8, 1.0); // Cyan base

                // Fade out if behind core (roughly, core radius is 1.0)
                // pos.z is world space Z. Camera is at -3.
                // Core is at 0.
                // If pos.z > 0.0 (behind core relative to camera?), wait, camera is at -3 looking +Z?
                // Code said camera at -3 looking at 0. So +Z is away from camera.
                // Objects at z > 0 are behind the core center.
                // If core is solid sphere r=1, check if ray blocked.

                // Simple transparency based on Z
                let alpha = smoothstep(1.5, -1.5, pos.z);

                output.color = vec4f(col, alpha);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleUnitSize = 32; // 8 floats * 4 bytes (vec4 pos + vec4 vel)
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            // Sphere distribution
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 1.2 + (Math.random() - 0.5) * 0.2;

            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);

            initialParticleData[i * 8 + 0] = x;
            initialParticleData[i * 8 + 1] = y;
            initialParticleData[i * 8 + 2] = z;
            initialParticleData[i * 8 + 3] = 1.0; // w

            initialParticleData[i * 8 + 4] = 0; // vx
            initialParticleData[i * 8 + 5] = 0; // vy
            initialParticleData[i * 8 + 6] = 0; // vz
            initialParticleData[i * 8 + 7] = 0; // life
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        // Uniform Buffer
        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        // Pipelines
        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }), // Re-use layout for params access in VS
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    ],
                }],
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{ format: presentationFormat, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                } }], // Additive blending
            },
            primitive: { topology: 'point-list' },
            // Enable depth testing if we had a depth buffer, but for additive particles, maybe not needed
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;

        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(100, 20, 20, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: monospace;
            pointer-events: none;
        `;
        msg.innerHTML = "WebGPU Not Available (WebGL2 Only)";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // COMMON
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width === 0 || height === 0) return;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

        const displayWidth = Math.floor(width * dpr);
        const displayHeight = Math.floor(height * dpr);

        this.resizeGL(displayWidth, displayHeight);
        this.resizeGPU(displayWidth, displayHeight);
    }

    resizeGL(width, height) {
        if (!this.glCanvas) return;
        if (width <= 0 || height <= 0) return;

        if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
            this.glCanvas.width = width;
            this.glCanvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    resizeGPU(width, height) {
        if (!this.gpuCanvas) return;
        if (width <= 0 || height <= 0) return;

        if (this.gpuCanvas.width !== width || this.gpuCanvas.height !== height) {
            this.gpuCanvas.width = width;
            this.gpuCanvas.height = height;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const timeLoc = this.gl.getUniformLocation(this.glProgram, 'u_time');
            this.gl.uniform1f(timeLoc, time);

            const mouseLoc = this.gl.getUniformLocation(this.glProgram, 'u_mouse');
            this.gl.uniform2f(mouseLoc, this.mouse.x, this.mouse.y);

            const resLoc = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
            this.gl.uniform2f(resLoc, this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0 && this.gpuCanvas.height > 0) {
            // Update simulation params
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, // dt
                time,  // time
                this.mouse.x, // mouseX
                this.mouse.y, // mouseY
                aspect, // aspect
                0, 0, 0 // padding
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render
            const textureView = this.context.getCurrentTexture().createView();
            const renderPassDescriptor = {
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            };

            const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup); // Bind for Uniform params in VS
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);

        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);

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
    window.NeuralDataCore = NeuralDataCore;
}
