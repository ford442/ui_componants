/**
 * Chrono Excavation Experiment
 * Combines WebGL2 for a buried wireframe artifact and WebGPU for an interactive sand/dust layer.
 */

export class ChronoExcavation {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

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
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 60000;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);
        this.animate = this.animate.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#0a0500'; // Dark brown/black background

        console.log("ChronoExcavation: Initializing...");

        // 1. Initialize WebGL2 Layer (Background - Buried Artifact)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground - Sand/Dust)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("ChronoExcavation: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("ChronoExcavation: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("ChronoExcavation: WebGPU initialized successfully.");
        }

        // Ensure resizing happens before animation starts
        this.resize();

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);

        // Touch support
        this.container.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            this.updateMousePos(touch.clientX, touch.clientY);
        }, { passive: false });
    }

    updateMousePos(clientX, clientY) {
        const rect = this.container.getBoundingClientRect();
        const x = (clientX - rect.left) / rect.width * 2 - 1;
        const y = -((clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    onMouseMove(e) {
        this.updateMousePos(e.clientX, e.clientY);
    }

    onMouseDown() {
        this.mouse.isDown = true;
    }

    onMouseUp() {
        this.mouse.isDown = false;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Buried Artifact - Wireframe Cube)
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

        // Create a Cube
        const positions = new Float32Array([
            // Front face
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            // Back face
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,
            // Top face
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0, -1.0,
            // Bottom face
            -1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0,  1.0,
            -1.0, -1.0,  1.0,
            // Right face
             1.0, -1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0,  1.0,
             1.0, -1.0,  1.0,
            // Left face
            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0,  1.0, -1.0,
        ]);

        const indices = new Uint16Array([
            0,  1,  2,      0,  2,  3,    // front
            4,  5,  6,      4,  6,  7,    // back
            8,  9, 10,      8, 10, 11,    // top
            12, 13, 14,     12, 14, 15,   // bottom
            16, 17, 18,     16, 18, 19,   // right
            20, 21, 22,     20, 22, 23    // left
        ]);

        this.glIndexCount = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const posBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const idxBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;

        // Shaders
        const vsSource = `#version 300 es
            layout(location = 0) in vec3 a_position;

            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_projection;

            out vec3 v_pos;

            void main() {
                v_pos = a_position;
                gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec3 v_pos;
            out vec4 outColor;

            uniform float u_time;

            void main() {
                // Golden/Ancient artifact look
                float pulse = 0.5 + 0.5 * sin(u_time * 2.0 + v_pos.y);
                vec3 baseColor = vec3(1.0, 0.8, 0.2); // Gold
                vec3 glowColor = vec3(1.0, 0.4, 0.1); // Amber

                // Wireframe effect (barycentric coords would be better, but we use lines mostly)
                outColor = vec4(mix(baseColor, glowColor, pulse), 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
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
    // WebGPU IMPLEMENTATION (Sand/Dust)
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

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        const computeShaderCode = `
            struct Particle {
                pos : vec4f,
                vel : vec4f,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

            struct SimParams {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                isDown : f32,
                pad2 : f32,
                pad3 : f32,
            }
            @group(0) @binding(1) var<uniform> params : SimParams;

            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Mouse interaction - "Excavation" Brush
                // Mouse is in normalized coords [-1, 1], z=0 (screen plane)
                // Particles are in 3D world space.
                // We project mouse into 3D roughly at z=0 plane.

                let mousePos = vec3f(params.mouseX * 3.0 * params.aspect, params.mouseY * 3.0, 0.0);

                let dist = distance(p.pos.xyz, mousePos);
                let brushRadius = 1.2;

                var force = vec3f(0.0);

                if (dist < brushRadius) {
                    // Push away
                    let pushDir = normalize(p.pos.xyz - mousePos);
                    let strength = (1.0 - dist / brushRadius) * 20.0;
                    force += pushDir * strength;
                }

                // Gravity / Settle
                // Particles tend to settle towards a shape or just stay put with friction
                // Let's make them settle into a "buried" state (random noise field)

                // Friction
                p.vel *= 0.90;

                // Apply force
                p.vel += vec4f(force * params.dt, 0.0);

                // Small random turbulence
                let noise = vec3f(
                    rand(vec2f(p.pos.y, params.time)) - 0.5,
                    rand(vec2f(p.pos.z, params.time)) - 0.5,
                    rand(vec2f(p.pos.x, params.time)) - 0.5
                );
                p.vel += vec4f(noise * 0.5 * params.dt, 0.0);

                // Update position
                p.pos += p.vel * params.dt;

                // Bounds check (Reset if too far)
                if (length(p.pos.xyz) > 10.0) {
                     p.pos = vec4f(mousePos.x + (rand(vec2f(params.time, 1.0)) - 0.5), mousePos.y + (rand(vec2f(params.time, 2.0)) - 0.5), 0.0, 1.0);
                     p.vel = vec4f(0.0);
                }

                particles[index] = p;
            }
        `;

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

                let pos = particlePos.xyz;
                let camPos = vec3f(0.0, 0.0, 5.0); // Camera backed out
                let viewPos = pos - camPos;

                let fov = 1.0;
                let f = 1.0 / tan(fov / 2.0);

                let x = viewPos.x * f / params.aspect;
                let y = viewPos.y * f;
                let z = viewPos.z;

                let w = -z; // Looking down -Z

                output.position = vec4f(x, y, z * 0.1, w);

                // Color based on velocity
                let speed = length(particleVel.xyz);
                // Sand color
                let sand = vec3f(0.8, 0.7, 0.5);
                let dust = vec3f(0.9, 0.8, 0.7);

                let col = mix(sand, dust, clamp(speed, 0.0, 1.0));

                // Alpha fade based on if it's moving (dust cloud)
                let alpha = 0.6 + clamp(speed * 0.5, 0.0, 0.4);

                output.color = vec4f(col, alpha);
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleUnitSize = 32;
        const particleBufferSize = this.numParticles * particleUnitSize;
        const initialParticleData = new Float32Array(this.numParticles * 8);

        // Initial distribution: Cloud covering the center
        for (let i = 0; i < this.numParticles; i++) {
            const r = Math.pow(Math.random(), 1.0/3.0) * 3.0; // Uniform sphere
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);

            initialParticleData[i * 8 + 0] = r * Math.sin(phi) * Math.cos(theta);
            initialParticleData[i * 8 + 1] = r * Math.sin(phi) * Math.sin(theta);
            initialParticleData[i * 8 + 2] = r * Math.cos(phi);
            initialParticleData[i * 8 + 3] = 1.0;

            initialParticleData[i * 8 + 4] = 0; // vx
            initialParticleData[i * 8 + 5] = 0; // vy
            initialParticleData[i * 8 + 6] = 0; // vz
            initialParticleData[i * 8 + 7] = 0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initialParticleData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

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

        const computeModule = this.device.createShaderModule({ code: computeShaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const drawModule = this.device.createShaderModule({ code: drawShaderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
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
                } }],
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.innerHTML = "WebGPU Not Available (WebGL2 Only)";
        msg.style.cssText = "position: absolute; bottom: 10px; right: 10px; color: red;";
        this.container.appendChild(msg);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width === 0 || height === 0) return;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

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

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const time = (now - this.startTime) * 0.001;

        // ----------------- WebGL2 Render -----------------
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Matrices
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const projection = this.perspective(45 * Math.PI / 180, aspect, 0.1, 100.0);

            // View: Orbit
            const camRadius = 5.0;
            // Static camera for artifact
            const view = this.lookAt(
                [0, 0, camRadius], // Eye
                [0, 0, 0],         // Target
                [0, 1, 0]          // Up
            );

            // Model: Rotate object
            const model = this.identity();
            this.rotateY(model, time * 0.5);
            this.rotateX(model, time * 0.2);

            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_projection'), false, projection);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_view'), false, view);
            this.gl.uniformMatrix4fv(this.gl.getUniformLocation(this.glProgram, 'u_model'), false, model);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);

            this.gl.clearColor(0.05, 0.02, 0.01, 1.0); // Deep dusty background
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
            this.gl.enable(this.gl.DEPTH_TEST);

            this.gl.bindVertexArray(this.glVao);
            // Draw lines (wireframe)
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // ----------------- WebGPU Render -----------------
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, time, this.mouse.x, this.mouse.y, aspect, this.mouse.isDown ? 1.0 : 0.0, 0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

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
                    storeOp: 'store',
                }],
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(this.animate);
    }

    // Matrix helpers
    perspective(fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ]);
    }

    lookAt(eye, center, up) {
        let z = [eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]];
        let len = Math.sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
        z = [z[0]/len, z[1]/len, z[2]/len];

        let x = [up[1]*z[2] - up[2]*z[1], up[2]*z[0] - up[0]*z[2], up[0]*z[1] - up[1]*z[0]];
        len = Math.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        if (len === 0) x = [0, 0, 0]; else x = [x[0]/len, x[1]/len, x[2]/len];

        let y = [z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2], z[0]*x[1] - z[1]*x[0]];
        len = Math.sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
        if (len === 0) y = [0, 0, 0]; else y = [y[0]/len, y[1]/len, y[2]/len];

        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]),
            -(y[0]*eye[0] + y[1]*eye[1] + y[2]*eye[2]),
            -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]),
            1
        ]);
    }

    identity() {
        return new Float32Array([
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1
        ]);
    }

    rotateY(m, angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const mv0 = m[0], mv4 = m[4], mv8 = m[8];
        const mv2 = m[2], mv6 = m[6], mv10 = m[10];

        m[0] = c * mv0 + s * mv2;
        m[4] = c * mv4 + s * mv6;
        m[8] = c * mv8 + s * mv10;

        m[2] = c * mv2 - s * mv0;
        m[6] = c * mv6 - s * mv4;
        m[10] = c * mv10 - s * mv8;
    }

    rotateX(m, angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const mv1 = m[1], mv5 = m[5], mv9 = m[9];
        const mv2 = m[2], mv6 = m[6], mv10 = m[10];

        m[1] = c * mv1 - s * mv2;
        m[5] = c * mv5 - s * mv6;
        m[9] = c * mv9 - s * mv10;

        m[2] = c * mv2 + s * mv1;
        m[6] = c * mv6 + s * mv5;
        m[10] = c * mv10 + s * mv9;
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mouseup', this.handleMouseUp);

        if (this.container) {
            this.container.removeEventListener('mousemove', this.handleMouseMove);
            this.container.removeEventListener('mousedown', this.handleMouseDown);
            this.container.innerHTML = '';
        }

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
    }
}

if (typeof window !== 'undefined') {
    window.ChronoExcavation = ChronoExcavation;
}
