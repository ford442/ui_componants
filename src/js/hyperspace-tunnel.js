/**
 * Hyperspace Tunnel Experiment
 * Combines WebGL2 (Wireframe Warp Tunnel) and WebGPU (Star Streak Particles).
 *
 * Refiner Upgrade: "Hyperboost" Interaction
 * - Click/Hold to engage Warp Speed.
 * - Accelerates tunnel scroll and star particles.
 * - Shifts color spectrum and warps geometry.
 */

export class HyperspaceTunnelExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.lastTime = Date.now();
        this.animationId = null;
        this.canvasSize = { width: 0, height: 0 };
        this.mouse = { x: 0, y: 0 };
        this.targetMouse = { x: 0, y: 0 };

        // Hyperboost State
        this.isBoosting = false;
        this.boostFactor = 0.0; // 0.0 to 1.0
        this.scrollProgress = 0.0;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.numIndices = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 20000;

        this.handleResize = this.handleResize.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';
        this.container.style.cursor = 'pointer'; // Indicate interaction

        // 1. Initialize WebGL2 (The Tunnel)
        this.initWebGL2();

        // 2. Initialize WebGPU (The Stars)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("HyperspaceTunnel: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.resize();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);

        // Interaction Listeners
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);
        this.container.addEventListener('touchstart', this.handleMouseDown, {passive: false});
        window.addEventListener('touchend', this.handleMouseUp);

        // Add instruction overlay
        this.addOverlay();

        this.animate();
    }

    addOverlay() {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: rgba(255, 255, 255, 0.7);
            font-family: 'Courier New', Courier, monospace;
            font-size: 14px;
            pointer-events: none;
            text-align: center;
            z-index: 10;
            text-shadow: 0 0 5px cyan;
        `;
        overlay.innerHTML = "CLICK & HOLD to ENGAGE HYPERDRIVE";
        this.container.appendChild(overlay);
    }

    handleResize() {
        this.resize();
    }

    handleMouseMove(e) {
        const x = (e.clientX / window.innerWidth) * 2 - 1;
        const y = -(e.clientY / window.innerHeight) * 2 + 1;
        this.targetMouse.x = x * 2.0; // Amplify for effect
        this.targetMouse.y = y * 2.0;
    }

    handleMouseDown(e) {
        if(e.type === 'touchstart') e.preventDefault();
        this.isBoosting = true;
    }

    handleMouseUp() {
        this.isBoosting = false;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Wireframe Tunnel)
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

        // Generate Tunnel Geometry
        const vertices = [];
        const indices = [];

        const rings = 40;
        const segments = 24;
        const radius = 5.0;
        const length = 100.0;

        // Create rings along negative Z
        for (let i = 0; i <= rings; i++) {
            const z = - (i / rings) * length;
            for (let j = 0; j < segments; j++) {
                const angle = (j / segments) * Math.PI * 2;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;
                vertices.push(x, y, z);
            }
        }

        // Generate lines
        // Ring connections
        for (let i = 0; i <= rings; i++) {
            for (let j = 0; j < segments; j++) {
                const current = i * segments + j;
                const next = i * segments + ((j + 1) % segments);
                indices.push(current, next);
            }
        }
        // Longitudinal connections
        for (let i = 0; i < rings; i++) {
            for (let j = 0; j < segments; j++) {
                const current = i * segments + j;
                const next = (i + 1) * segments + j;
                indices.push(current, next);
            }
        }

        this.numIndices = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);
        this.glVao = vao;

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const posLoc = 0; // Standardize location
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec3 a_position;
            uniform float u_time;
            uniform float u_scrollProgress;
            uniform float u_boostFactor;
            uniform vec2 u_mouse;
            uniform vec2 u_resolution;

            out float v_depth;
            out float v_boost;

            void main() {
                vec3 p = a_position;

                // Infinite Scroll: Shift Z based on scrollProgress
                // The geometry is from 0 to -100

                p.z += mod(u_scrollProgress, 100.0);
                if (p.z > 0.0) p.z -= 100.0;

                // Warp/Bend based on mouse
                // The further away (more negative Z), the more it bends
                // Boost increases the "warp" feeling (less bend, more tunnel vision? Or more chaotic?)
                // Let's make it more chaotic with boost.

                float baseBend = 0.05;
                float boostBend = 0.15; // More intense warp
                float bendAmt = baseBend + u_boostFactor * boostBend;

                float bendFactor = p.z * bendAmt; // Negative value
                p.x += u_mouse.x * bendFactor * bendFactor * 0.1;
                p.y += u_mouse.y * bendFactor * bendFactor * 0.1;

                // Spiral twist
                // Spin faster when boosting
                float spinSpeed = 0.5 + u_boostFactor * 2.0;
                float angle = p.z * 0.02 + u_time * spinSpeed;
                float c = cos(angle);
                float s = sin(angle);
                float x = p.x * c - p.y * s;
                float y = p.x * s + p.y * c;
                p.x = x;
                p.y = y;

                // Camera Projection
                // Widen FOV when boosting (Dolly Zoom effect)
                float fov = 1.2 + u_boostFactor * 0.5;
                float scale = 1.0 / tan(fov * 0.5);
                float aspect = u_resolution.x / u_resolution.y;

                float z = p.z;
                float px = p.x * scale / aspect;
                float py = p.y * scale;

                // WebGL clip space: -w to w
                // We use z as w
                gl_Position = vec4(px, py, z * 0.01 - 1.0, -z);
                v_depth = -z;
                v_boost = u_boostFactor;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            in float v_boost;
            out vec4 outColor;

            void main() {
                // Fade based on depth (fog)
                float alpha = smoothstep(90.0, 10.0, v_depth);

                // Color Shift
                vec3 cyan = vec3(0.0, 1.0, 1.0);
                vec3 magenta = vec3(1.0, 0.0, 1.0);
                vec3 white = vec3(1.0);

                // Mix: Normal -> Boost (Magenta) -> Max Boost (White tips)
                vec3 col = mix(cyan, magenta, v_boost);
                col = mix(col, white, v_boost * 0.5);

                outColor = vec4(col, alpha * (0.3 + v_boost * 0.4)); // Brighter when boosting
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
    }

    createGLProgram(vs, fs) {
        const vShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vShader, vs);
        this.gl.compileShader(vShader);

        if (!this.gl.getShaderParameter(vShader, this.gl.COMPILE_STATUS)) {
            console.error('Vertex Shader Error:', this.gl.getShaderInfoLog(vShader));
            return null;
        }

        const fShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fShader, fs);
        this.gl.compileShader(fShader);

        if (!this.gl.getShaderParameter(fShader, this.gl.COMPILE_STATUS)) {
            console.error('Fragment Shader Error:', this.gl.getShaderInfoLog(fShader));
            return null;
        }

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vShader);
        this.gl.attachShader(prog, fShader);
        this.gl.linkProgram(prog);

        if (!this.gl.getProgramParameter(prog, this.gl.LINK_STATUS)) {
             console.error('Program Link Error:', this.gl.getProgramInfoLog(prog));
             return null;
        }
        return prog;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Star Particles)
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

        // Compute Shader: Update Particle Positions
        const computeShader = `
            struct Particle {
                pos : vec4f, // x, y, z, baseZ (original z offset)
                vel : vec4f, // vx, vy, vz, size
            }

            struct Uniforms {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                boostFactor : f32, // New uniform
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            fn hash(n: f32) -> f32 {
                return fract(sin(n) * 43758.5453123);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Initial setup if time is small
                if (uniforms.time < 0.1) {
                    let seed = f32(index) * 0.001;
                    p.pos.x = (hash(seed) - 0.5) * 50.0;
                    p.pos.y = (hash(seed + 1.0) - 0.5) * 50.0;
                    p.pos.z = -hash(seed + 2.0) * 100.0;
                    p.vel.z = 50.0 + hash(seed + 3.0) * 50.0; // Speed
                    p.vel.w = 0.5 + hash(seed + 4.0); // Size
                }

                // Calculate Speed with Boost
                let baseSpeed = p.vel.z;
                let boostMult = 1.0 + uniforms.boostFactor * 4.0; // Up to 5x speed
                let currentSpeed = baseSpeed * boostMult;

                // Move stars towards camera (+Z)
                p.pos.z += currentSpeed * uniforms.dt;

                // Reset if past camera (z > 0)
                if (p.pos.z > 0.0) {
                    p.pos.z -= 100.0;
                    // Scramble XY slightly
                    let seed = uniforms.time + f32(index);
                    p.pos.x = (hash(seed) - 0.5) * 50.0;
                    p.pos.y = (hash(seed + 1.0) - 0.5) * 50.0;
                }

                // Warp/Steer based on mouse
                let turnSpeed = 20.0 * (1.0 + uniforms.boostFactor); // Steer faster too
                p.pos.x -= uniforms.mouseX * turnSpeed * uniforms.dt;
                p.pos.y -= uniforms.mouseY * turnSpeed * uniforms.dt;

                // Wrap XY to keep them in the tunnel
                if (p.pos.x > 25.0) { p.pos.x -= 50.0; }
                if (p.pos.x < -25.0) { p.pos.x += 50.0; }
                if (p.pos.y > 25.0) { p.pos.y -= 50.0; }
                if (p.pos.y < -25.0) { p.pos.y += 50.0; }

                particles[index] = p;
            }
        `;

        // Render Shader: Trails
        const renderShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) z : f32,
            }

            struct Uniforms {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                boostFactor : f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @builtin(vertex_index) vertexIndex : u32,
                @location(0) pos : vec4f,
                @location(1) vel : vec4f
            ) -> VertexOutput {
                var output : VertexOutput;

                let p = pos.xyz;
                let viewPos = p;

                // Perspective
                let fov = 1.2 + uniforms.boostFactor * 0.5; // Match WebGL FOV
                let scale = 1.0 / tan(fov * 0.5);

                let z_dist = -p.z; // since p.z is negative

                // Don't render if behind camera
                if (p.z >= -1.0) {
                     output.position = vec4f(0.0, 0.0, 0.0, 0.0);
                     return output;
                }

                let px = p.x * scale / uniforms.aspect;
                let py = p.y * scale;

                output.position = vec4f(px, py, p.z * 0.01, z_dist);
                output.z = z_dist;

                // Color based on Z (fade distant)
                let alpha = smoothstep(100.0, 20.0, z_dist);

                // Whiter/Brighter when boosting
                let boostColor = vec4f(1.0, 1.0, 1.0, alpha);
                let baseColor = vec4f(1.0, 0.8, 1.0, alpha);

                output.color = mix(baseColor, boostColor, uniforms.boostFactor);

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f, @location(1) z : f32) -> @location(0) vec4f {
                return color;
            }
        `;

        const particleData = new Float32Array(this.numParticles * 8); // 8 floats per particle
        // Initialize with zeros, compute shader will seed them

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.simParamBuffer = this.device.createBuffer({
            size: 48, // aligned: dt(4), time(4), mx(4), my(4), aspect(4), boost(4), pad(8) -> 32? No.
            // dt, time, mx, my = 16 bytes
            // aspect, boost, pad, pad = 16 bytes. Total 32 bytes should suffice.
            // Let's verify alignment.
            // vec4 chunks.
            // Chunk 1: dt, time, mouseX, mouseY.
            // Chunk 2: aspect, boostFactor, pad, pad.
            // Total 32 bytes.
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
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
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // vel
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    }
                }]
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `
            position: absolute; bottom: 10px; right: 10px;
            color: #ff5555; font-family: sans-serif; font-size: 12px;
            background: rgba(0,0,0,0.8); padding: 5px 10px; border-radius: 4px;
        `;
        msg.innerText = "WebGPU Not Available";
        this.container.appendChild(msg);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width === 0 || height === 0) return;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

        const dw = Math.floor(width * dpr);
        const dh = Math.floor(height * dpr);

        if (this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            this.gl.viewport(0, 0, dw, dh);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = dw;
            this.gpuCanvas.height = dh;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const dt = Math.min((now - this.lastTime) * 0.001, 0.1); // Cap at 0.1s
        this.lastTime = now;
        const time = (now - this.startTime) * 0.001;

        // Smooth mouse
        this.mouse.x += (this.targetMouse.x - this.mouse.x) * 0.1;
        this.mouse.y += (this.targetMouse.y - this.mouse.y) * 0.1;

        // Smooth Boost Factor
        const targetBoost = this.isBoosting ? 1.0 : 0.0;
        this.boostFactor += (targetBoost - this.boostFactor) * 0.1; // Smooth transition

        // Update Scroll
        const baseSpeed = 20.0;
        const currentSpeed = baseSpeed * (1.0 + this.boostFactor * 4.0); // 1x to 5x speed
        this.scrollProgress += currentSpeed * dt;

        // WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_scrollProgress'), this.scrollProgress);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_boostFactor'), this.boostFactor);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0, 0, 0, 0); // Transparent to see BG if any
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT); // Clear depth too?
            // Actually wireframe usually doesn't need depth test if additive, but let's enable it for proper obscuration if needed
            // For now, disable depth test for additive blending look
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE); // Additive

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // WebGPU Render
        if (this.device && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            // Align: dt(4), time(4), mx(4), my(4), aspect(4), boost(4), pad(4), pad(4)
            const uniforms = new Float32Array([
                dt, time, this.mouse.x, this.mouse.y, aspect, this.boostFactor, 0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, uniforms);

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

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);

        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);

        window.removeEventListener('mouseup', this.handleMouseUp);
        window.removeEventListener('touchend', this.handleMouseUp);

        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.HyperspaceTunnelExperiment = HyperspaceTunnelExperiment;
}
