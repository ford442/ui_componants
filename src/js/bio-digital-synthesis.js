/**
 * Bio-Digital Synthesis Experiment
 * Combines WebGL2 for an organic, pulsating biological core and WebGPU for a digital synthetic swarm.
 */

export class BioDigitalSynthesis {
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
        this.numParticles = options.numParticles || 50000;

        // Bind resize handler for cleanup
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.animate = this.animate.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050005';

        console.log("BioDigitalSynthesis: Initializing...");

        // 1. Initialize WebGL2 Layer (Background - Organic Core)
        this.initWebGL2();

        // 2. Initialize WebGPU Layer (Foreground - Digital Swarm)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("BioDigitalSynthesis: WebGPU initialization error:", e);
            }
        }

        if (!gpuSuccess) {
            console.log("BioDigitalSynthesis: WebGPU not enabled/supported. Running in WebGL2-only mode.");
            this.addWebGPUNotSupportedMessage();
        } else {
            console.log("BioDigitalSynthesis: WebGPU initialized successfully.");
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
    // WebGL2 IMPLEMENTATION (Organic Core)
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
            console.warn("BioDigitalSynthesis: WebGL2 not supported.");
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

        // Fragment Shader - Raymarched Organic Blob
        const fsSource = `#version 300 es
            precision highp float;

            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_mouse;
            uniform vec2 u_resolution;

            out vec4 outColor;

            // Smooth minimum for organic blending
            float smin(float a, float b, float k) {
                float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
                return mix(b, a, h) - k * h * (1.0 - h);
            }

            // Rotation matrix
            mat2 rotate2D(float angle) {
                float s = sin(angle);
                float c = cos(angle);
                return mat2(c, -s, s, c);
            }

            // SDF
            float map(vec3 p) {
                // Main pulsing body
                vec3 p1 = p;
                p1.y -= sin(u_time * 0.5) * 0.1;
                p1.xz *= rotate2D(u_time * 0.2);

                // Distortion
                float displacement = sin(p.x * 3.0 + u_time) * sin(p.y * 3.0 + u_time) * 0.1;

                // Three spheres merging
                vec3 q1 = p1 - vec3(0.5 * sin(u_time * 0.7), 0.3 * cos(u_time * 0.6), 0.0);
                vec3 q2 = p1 - vec3(-0.5 * sin(u_time * 0.8), -0.2 * cos(u_time * 0.5), 0.3 * sin(u_time));
                vec3 q3 = p1 - vec3(0.0, 0.5 * sin(u_time * 0.4), -0.3);

                float d1 = length(q1) - 0.6;
                float d2 = length(q2) - 0.5;
                float d3 = length(q3) - 0.5;

                float d = smin(d1, d2, 0.4);
                d = smin(d, d3, 0.4);

                return d + displacement;
            }

            // Normal calculation
            vec3 getNormal(vec3 p) {
                vec2 e = vec2(0.001, 0.0);
                float d = map(p);
                return normalize(vec3(
                    d - map(p - e.xyy),
                    d - map(p - e.yxy),
                    d - map(p - e.yyx)
                ));
            }

            void main() {
                vec2 uv = v_uv;
                if (u_resolution.y > 0.0) {
                    uv.x *= u_resolution.x / u_resolution.y;
                }

                vec3 ro = vec3(0.0, 0.0, -3.0);
                vec3 rd = normalize(vec3(uv, 1.2));

                float t = 0.0;
                float d = 0.0;
                bool hit = false;

                // Raymarching
                for(int i = 0; i < 80; i++) {
                    vec3 p = ro + rd * t;
                    d = map(p);
                    if(d < 0.001) {
                        hit = true;
                        break;
                    }
                    if(t > 10.0) break;
                    t += d;
                }

                vec3 col = vec3(0.05, 0.0, 0.05); // Dark purple background

                if(hit) {
                    vec3 p = ro + rd * t;
                    vec3 n = getNormal(p);
                    vec3 l = normalize(vec3(1.0, 1.0, -1.0));

                    // Flesh material
                    float diff = max(dot(n, l), 0.0);
                    float amb = 0.2;
                    float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);

                    // Subsurface scattering fake (wrap lighting)
                    float sss = max(0.0, dot(n, -l) + 0.5) * 0.5;

                    vec3 fleshColor = vec3(0.8, 0.2, 0.3);
                    vec3 highlightColor = vec3(1.0, 0.6, 0.6);

                    col = fleshColor * (diff + amb);
                    col += vec3(0.5, 0.1, 0.1) * sss; // Red glow inside
                    col += highlightColor * fresnel;
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
    // WebGPU IMPLEMENTATION (Digital Swarm)
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

        // WGSL
        const commonWGSL = `
            fn rand(co: vec2f) -> f32 {
                return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
            }
        `;

        const computeShaderCode = `
            ${commonWGSL}

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

                // Swarm Logic
                // 1. Attraction to center (Core)
                let center = vec3f(0.0, 0.0, 0.0);
                let diff = center - p.pos.xyz;
                let dist = length(diff);
                let dir = normalize(diff);

                // They should orbit around radius 1.5, interacting with the core
                let targetRadius = 1.3 + sin(params.time * 2.0 + p.pos.y * 2.0) * 0.2;
                let radiusForce = dir * (dist - targetRadius) * 2.0;

                // 2. Spiral motion (Orbit)
                let tangent = normalize(cross(p.pos.xyz, vec3f(0.0, 1.0, 0.0)));

                // 3. Chaos/Noise simulation (simplified)
                let noise = vec3f(
                    sin(p.pos.y * 4.0 + params.time),
                    sin(p.pos.z * 4.0 + params.time),
                    sin(p.pos.x * 4.0 + params.time)
                );

                // Mouse interaction - Attraction (Feeding mechanism)
                let mPos = vec3f(params.mouseX * 2.0 * params.aspect, params.mouseY * 2.0, 0.0); // Rough approximation
                let mDist = distance(p.pos.xyz, mPos);
                var mouseForce = vec3f(0.0);
                if (mDist < 0.8) {
                    mouseForce = normalize(mPos - p.pos.xyz) * 5.0; // Attract
                }

                let accel = radiusForce * 1.5 + tangent * 2.0 + noise * 0.5 + mouseForce;

                // Update velocity
                p.vel = p.vel * 0.95 + vec4f(accel * params.dt, 0.0);

                // Update position
                p.pos = p.pos + p.vel * params.dt;

                // Reset bounds
                if (dist > 6.0 || dist < 0.1) {
                    let r = 2.0 + rand(vec2f(p.pos.x, params.time)) * 0.5;
                    let theta = rand(vec2f(p.pos.y, params.time)) * 6.28;
                    let phi = rand(vec2f(p.pos.z, params.time)) * 3.14;

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
                let camPos = vec3f(0.0, 0.0, -3.0);
                let viewPos = pos - camPos;

                let fov = 1.0;
                let f = 1.0 / tan(fov / 2.0);

                let x = viewPos.x * f / params.aspect;
                let y = viewPos.y * f;
                let z = viewPos.z;

                // W component for perspective division (+z because looking towards +Z relative to cam)
                let w = z;

                output.position = vec4f(x, y, z * 0.1, w);

                // Color: Digital Green/Cyan
                // Change color based on speed
                let speed = length(particleVel.xyz);
                var col = mix(vec3f(0.0, 0.5, 0.2), vec3f(0.0, 1.0, 0.8), clamp(speed * 0.5, 0.0, 1.0));

                // Add some white for high energy
                if (speed > 2.0) {
                    col = vec3f(1.0, 1.0, 1.0);
                }

                output.color = vec4f(col, 0.6); // Semi-transparent
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

        for (let i = 0; i < this.numParticles; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 2.0 + Math.random();

            initialParticleData[i * 8 + 0] = r * Math.sin(phi) * Math.cos(theta);
            initialParticleData[i * 8 + 1] = r * Math.sin(phi) * Math.sin(theta);
            initialParticleData[i * 8 + 2] = r * Math.cos(phi);
            initialParticleData[i * 8 + 3] = 1.0;

            initialParticleData[i * 8 + 4] = 0;
            initialParticleData[i * 8 + 5] = 0;
            initialParticleData[i * 8 + 6] = 0;
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
        if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
            this.glCanvas.width = width;
            this.glCanvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    resizeGPU(width, height) {
        if (!this.gpuCanvas) return;
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
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        // 2. Render WebGPU
        if (this.device && this.context && this.renderPipeline && this.gpuCanvas.width > 0) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                0.016, time, this.mouse.x, this.mouse.y, aspect, 0, 0, 0
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

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);
        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.BioDigitalSynthesis = BioDigitalSynthesis;
}
