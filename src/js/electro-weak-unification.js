/**
 * Electro-Weak Unification Experiment
 * Demonstrates Hybrid WebGL2 + WebGPU implementation.
 * - WebGL2: Renders a "Weinberg Angle" Torus Knot wireframe.
 * - WebGPU: Renders a particle swarm representing W/Z bosons and photons.
 */

class ElectroWeakUnification {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0, isPressed: false };

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndexBuffer = null;
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
        this.numParticles = options.numParticles || 30000;

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
        this.container.style.background = '#020005';

        console.log("ElectroWeakUnification: Initializing...");

        // 1. Initialize WebGL2 Layer
        this.initWebGL2();

        // 2. Initialize WebGPU Layer
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("ElectroWeakUnification: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("ElectroWeakUnification: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("ElectroWeakUnification: WebGPU initialized successfully.");
        }

        this.isActive = true;

        // Event Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);

        this.resize();
        this.animate();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Torus Knot)
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
            console.warn("ElectroWeakUnification: WebGL2 not supported.");
            return;
        }

        // Generate Torus Knot Geometry
        const positions = [];
        const indices = [];

        const tubularSegments = 128;
        const radialSegments = 16;
        const p = 2; // loops around axis
        const q = 3; // loops around torus core
        const tubeRadius = 0.2;
        const torusRadius = 0.6;

        for (let i = 0; i <= tubularSegments; i++) {
            const u = (i / tubularSegments) * Math.PI * 2 * p;
            const cu = Math.cos(u);
            const su = Math.sin(u);

            const quOverP = (u * q) / p;
            const cs = Math.cos(quOverP);

            // Center of the tube at this segment
            const r = torusRadius + tubeRadius * Math.cos(quOverP);
            const centerPos = {
                x: r * Math.cos(u/p), // Parametric equation adjustment
                y: r * Math.sin(u/p),
                z: tubeRadius * Math.sin(quOverP)
            };

            // Simplified knot for wireframe
            // Let's use a standard torus knot parametric equation
            // x = (R + r * cos(q*phi)) * cos(p*phi)
            // y = (R + r * cos(q*phi)) * sin(p*phi)
            // z = r * sin(q*phi)

            const phi = (i / tubularSegments) * Math.PI * 2;
            const R = 0.6;
            const r0 = 0.25;

            // Tangent and Normal vectors for tube generation would be complex here.
            // Let's just draw the core line first, or a simple wire mesh.
            // Actually, let's just generate points on the surface.

            for (let j = 0; j <= radialSegments; j++) {
                const theta = (j / radialSegments) * Math.PI * 2;

                // Position on the cross-section circle
                // We need a frenet frame to do this properly, but let's approximate
                // by adding offset relative to center position.
                // This is a bit cheap, let's use a simpler geometry or just the line strip if lazy.
                // Better: Just vertices for a sphere? No, let's do a simple Torus.

                // Standard Torus for stability if Knot is too complex to code from scratch in one shot without library
                // x = (R + r*cos(theta)) * cos(phi)
                // y = (R + r*cos(theta)) * sin(phi)
                // z = r * sin(theta)

                const u_torus = (i / tubularSegments) * Math.PI * 2;
                const v_torus = (j / radialSegments) * Math.PI * 2;
                const R_torus = 0.6;
                const r_torus = 0.25;

                const x = (R_torus + r_torus * Math.cos(v_torus)) * Math.cos(u_torus);
                const y = (R_torus + r_torus * Math.cos(v_torus)) * Math.sin(u_torus);
                const z = r_torus * Math.sin(v_torus);

                positions.push(x, y, z);
            }
        }

        for (let i = 0; i < tubularSegments; i++) {
            for (let j = 0; j < radialSegments; j++) {
                const a = (radialSegments + 1) * i + j;
                const b = (radialSegments + 1) * (i + 1) + j;
                const c = (radialSegments + 1) * (i + 1) + (j + 1);
                const d = (radialSegments + 1) * i + (j + 1);

                indices.push(a, b);
                indices.push(b, c);
                // indices.push(c, d); // Reduce density
                // indices.push(d, a);
            }
        }

        this.glIndexCount = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.glIndexBuffer = indexBuffer;
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec3 a_position;

            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec2 u_mouse;

            out vec3 v_pos;
            out float v_depth;

            mat4 rotationX(float angle) {
                return mat4(1.0, 0.0, 0.0, 0.0,
                            0.0, cos(angle), -sin(angle), 0.0,
                            0.0, sin(angle), cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            mat4 rotationY(float angle) {
                return mat4(cos(angle), 0.0, sin(angle), 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            -sin(angle), 0.0, cos(angle), 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            mat4 rotationZ(float angle) {
                return mat4(cos(angle), -sin(angle), 0.0, 0.0,
                            sin(angle), cos(angle), 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0);
            }

            void main() {
                v_pos = a_position;

                // Complex rotation to simulate mixing angle
                mat4 rot = rotationY(u_time * 0.2) * rotationX(u_time * 0.15 + u_mouse.y) * rotationZ(u_time * 0.05 + u_mouse.x);
                vec4 pos = rot * vec4(a_position, 1.0);

                // Perspective
                float aspect = u_resolution.x / u_resolution.y;
                pos.x /= aspect;

                // Pseudo-perspective scale based on Z
                float zScale = 1.0 / (2.0 - pos.z);
                pos.xy *= zScale;
                v_depth = pos.z;

                gl_Position = vec4(pos.xy, pos.z, 1.0);
                gl_PointSize = 2.0;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_pos;
            in float v_depth;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                // Gold/Blue mixing colors
                vec3 colA = vec3(1.0, 0.8, 0.2); // Weak (Gold)
                vec3 colB = vec3(0.2, 0.5, 1.0); // Electro (Blue)

                float mixFactor = sin(v_pos.x * 3.0 + u_time) * 0.5 + 0.5;
                vec3 color = mix(colA, colB, mixFactor);

                // Depth fade
                float alpha = smoothstep(-1.0, 1.0, v_depth) * 0.5 + 0.2;

                outColor = vec4(color, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        const positionLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLoc);
        this.gl.vertexAttribPointer(positionLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
        this.resizeGL();
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
    // WebGPU IMPLEMENTATION
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
            return false;
        }
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // WGSL Shaders
        const computeShaderCode = `
            struct Particle {
                pos : vec4f, // xyz, life
                vel : vec4f, // xyz, type
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                isPressed : f32,
                dummy1 : f32,
                dummy2 : f32,
                dummy3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.numParticles}) { return; }

                var p = particles[index];

                // Initialize if dead
                if (p.pos.w <= 0.0) {
                    let r = rand(vec2f(params.time, f32(index)));
                    let theta = r * 6.28;
                    let phi = rand(vec2f(f32(index), params.time)) * 3.14;
                    let radius = 1.5;

                    p.pos = vec4f(
                        radius * sin(phi) * cos(theta),
                        radius * sin(phi) * sin(theta),
                        radius * cos(phi),
                        1.0 // Life
                    );

                    // Velocity towards center with swirl
                    p.vel = vec4f(
                        -p.pos.x * 0.5 + p.pos.y * 1.0,
                        -p.pos.y * 0.5 - p.pos.x * 1.0,
                        -p.pos.z * 0.5,
                        r // Type: 0..1 (determines color)
                    );
                }

                // Physics
                let dist = length(p.pos.xyz);
                let dir = normalize(p.pos.xyz);

                // Spiral towards center
                let tangent = cross(dir, vec3f(0.0, 0.0, 1.0));

                // Forces
                var force = -dir * 1.5; // Attraction
                force += tangent * 2.0; // Swirl

                // Mouse interaction (Repulsion)
                let mousePos = vec3f(params.mouseX * 2.0, params.mouseY * 2.0, 0.0);
                let mouseDiff = p.pos.xyz - mousePos;
                let mouseDist = length(mouseDiff);

                if (mouseDist < 0.5) {
                    force += normalize(mouseDiff) * 10.0;
                    if (params.isPressed > 0.5) {
                        force += normalize(mouseDiff) * 20.0;
                    }
                }

                p.vel.x += force.x * params.dt;
                p.vel.y += force.y * params.dt;
                p.vel.z += force.z * params.dt;

                // Damping
                p.vel.x *= 0.98;
                p.vel.y *= 0.98;
                p.vel.z *= 0.98;

                p.pos.x += p.vel.x * params.dt;
                p.pos.y += p.vel.y * params.dt;
                p.pos.z += p.vel.z * params.dt;

                // Decay life
                p.pos.w -= params.dt * 0.5;

                // Reset if too close to center
                if (dist < 0.1) {
                    p.pos.w = 0.0;
                }

                particles[index] = p;
            }
        `;

        const renderShaderCode = `
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
                var pos = particlePos.xyz;
                let z = pos.z;

                // Scale based on Z
                let scale = 1.0 / (2.0 - z);

                output.position = vec4f(pos.xy * scale, 0.0, 1.0);

                // Color based on Type (vel.w) and Speed
                let type = particleVel.w;
                let speed = length(particleVel.xyz);

                var col = vec3f(0.0);
                if (type < 0.33) {
                    col = vec3f(1.0, 0.2, 0.2); // W+ (Red)
                } else if (type < 0.66) {
                    col = vec3f(0.2, 0.5, 1.0); // W- (Blue)
                } else {
                    col = vec3f(1.0, 0.8, 0.2); // Z0 (Gold)
                }

                // Intensity by speed
                col *= (0.5 + speed * 0.5);

                // Fade by life
                let alpha = smoothstep(0.0, 0.2, particlePos.w);

                output.color = vec4f(col, alpha);

                // Point size emulation not available in pure WGSL unless point-list topology supports it or we use quads.
                // Defaulting to 1px points.
                output.position.w = 1.0;

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const particleUnitSize = 32; // 2x vec4f = 32 bytes
        const particleBufferSize = this.numParticles * particleUnitSize;

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // aligned
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Layouts
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: renderShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: particleUnitSize,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos + life
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // vel + type
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    }
                }]
            },
            primitive: { topology: 'point-list' },
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px; color: white; background: rgba(200,50,50,0.8);
            padding: 8px 12px; border-radius: 4px; font-family: monospace; font-size: 12px; pointer-events: none;
        `;
        msg.innerText = "WebGPU Not Available - Visuals Reduced";
        this.container.appendChild(msg);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        const displayWidth = Math.floor(width * dpr);
        const displayHeight = Math.floor(height * dpr);

        if (this.glCanvas) {
            this.glCanvas.width = displayWidth;
            this.glCanvas.height = displayHeight;
            if (this.gl) this.gl.viewport(0, 0, displayWidth, displayHeight);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = displayWidth;
            this.gpuCanvas.height = displayHeight;
        }
    }

    resizeGPU() {
        // Redundant with resize(), but kept for consistency
        if (this.gpuCanvas) {
             const dpr = window.devicePixelRatio || 1;
             this.gpuCanvas.width = Math.floor(this.container.clientWidth * dpr);
             this.gpuCanvas.height = Math.floor(this.container.clientHeight * dpr);
        }
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    }

    onMouseDown() { this.mouse.isPressed = true; }
    onMouseUp() { this.mouse.isPressed = false; }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const dt = 0.016;

        // WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            const uTime = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const uRes = this.gl.getUniformLocation(this.glProgram, 'u_resolution');
            const uMouse = this.gl.getUniformLocation(this.glProgram, 'u_mouse');

            this.gl.uniform1f(uTime, time);
            this.gl.uniform2f(uRes, this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(uMouse, this.mouse.x, this.mouse.y);

            this.gl.clearColor(0.0, 0.0, 0.02, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // WebGPU Render
        if (this.device && this.context && this.renderPipeline) {
            // Write Uniforms
            const params = new Float32Array([
                dt, time, this.mouse.x, this.mouse.y,
                this.mouse.isPressed ? 1.0 : 0.0,
                0, 0, 0
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
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
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
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('mousedown', this.handleMouseDown);
        window.removeEventListener('mouseup', this.handleMouseUp);

        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }

        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.ElectroWeakUnification = ElectroWeakUnification;
}

export { ElectroWeakUnification };
