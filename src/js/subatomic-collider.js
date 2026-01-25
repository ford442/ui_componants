/**
 * Subatomic Collider Experiment
 * Combines WebGL2 (Wireframe Detector Geometry) and WebGPU (Particle Collision Physics).
 */

export class SubatomicColliderExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.canvasSize = { width: 0, height: 0 };
        this.mouse = { x: 0, y: 0 };
        this.isClicked = false;

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
        this.numParticles = options.numParticles || 100000;

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
        this.container.style.background = '#020205';

        // 1. Initialize WebGL2 (The Detector Structure)
        this.initWebGL2();

        // 2. Initialize WebGPU (The Particle Collision)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("SubatomicCollider: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.resize();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp); // Window for drag release
        this.animate();
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    onMouseDown(e) {
        this.isClicked = true;
    }

    onMouseUp(e) {
        this.isClicked = false;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Wireframe Detector)
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

        // Generate Detector Geometry (Octagonal Prism Tunnel)
        const vertices = [];
        const indices = [];

        const rings = 12;
        const segments = 8;
        const radius = 2.5;
        const length = 12.0;

        for (let i = 0; i <= rings; i++) {
            const z = -length / 2 + (i / rings) * length;
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

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec3 a_position;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform float u_click;
            uniform vec2 u_mouse;

            out float v_depth;

            void main() {
                vec3 p = a_position;

                // Rotate entire detector slowly
                float t = u_time * 0.2;
                float c = cos(t);
                float s = sin(t);
                float x = p.x * c - p.y * s;
                float y = p.x * s + p.y * c;
                p.x = x;
                p.y = y;

                // Interaction Pulse: Deform when clicked
                if (u_click > 0.0) {
                    float dist = length(p.xy);
                    // Create a wave that moves based on time
                    float wave = sin(p.z * 2.0 + u_time * 10.0);
                    // Distort radius
                    p.x += p.x * wave * 0.1 * u_click;
                    p.y += p.y * wave * 0.1 * u_click;
                }

                // Camera Projection
                vec3 camPos = vec3(0.0, 0.0, -8.0);
                vec3 viewPos = p - camPos;

                float fov = 1.0;
                float scale = 1.0 / tan(fov * 0.5);
                float aspect = u_resolution.x / u_resolution.y;

                float px = viewPos.x * scale / aspect;
                float py = viewPos.y * scale;
                float pz = viewPos.z;

                gl_Position = vec4(px, py, pz * 0.01, pz); // Perspective divide
                v_depth = pz;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            uniform float u_click;
            out vec4 outColor;

            void main() {
                // Fade based on depth
                float alpha = 1.0 - smoothstep(5.0, 20.0, v_depth);

                // Base color
                vec3 color = vec3(0.2, 0.3, 0.4);

                // Click boost
                if (u_click > 0.0) {
                    color = mix(color, vec3(0.5, 0.8, 1.0), u_click * 0.5);
                }

                outColor = vec4(color, alpha * 0.4);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);

        if (!this.glProgram) return;

        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
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
    // WebGPU IMPLEMENTATION (Particle Physics)
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

        // Compute Shader: Lorentz Force Simulation
        const computeShader = `
            struct Particle {
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, charge
            }

            struct Uniforms {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                click : f32, // New interaction
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            // Random number generator
            fn hash(n: f32) -> f32 {
                return fract(sin(n) * 43758.5453123);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Update Life
                p.pos.w -= uniforms.dt * 0.5; // decay

                // Respawn
                if (p.pos.w <= 0.0) {
                    let seed = uniforms.time + f32(index) * 0.001;

                    // Spawn at center with high velocity outwards
                    p.pos.x = (hash(seed) - 0.5) * 0.2;
                    p.pos.y = (hash(seed + 1.0) - 0.5) * 0.2;
                    p.pos.z = (hash(seed + 2.0) - 0.5) * 0.2;
                    p.pos.w = 1.0 + hash(seed + 3.0); // Random lifespan

                    // Random direction spherical
                    let theta = hash(seed + 4.0) * 6.28;
                    let phi = acos(2.0 * hash(seed + 5.0) - 1.0);
                    let speed = 5.0 + hash(seed + 6.0) * 10.0; // High speed

                    p.vel.x = sin(phi) * cos(theta) * speed;
                    p.vel.y = sin(phi) * sin(theta) * speed;
                    p.vel.z = cos(phi) * speed;

                    // Random Charge (-1, 0, 1)
                    let r = hash(seed + 7.0);
                    if (r < 0.33) { p.vel.w = -1.0; } // Electron
                    else if (r < 0.66) { p.vel.w = 1.0; } // Positron
                    else { p.vel.w = 0.0; } // Neutral
                }

                // Physics Step

                // Magnetic Field B (Along Z axis)
                // Strength affected by mouse
                let B_mag = 2.0 + uniforms.mouseX * 5.0;
                let B = vec3f(0.0, 0.0, B_mag);

                // Lorentz Force: F = q(v x B)
                // Assuming mass = 1
                let charge = p.vel.w;

                if (abs(charge) > 0.1) {
                    let v = p.vel.xyz;
                    let F = cross(v, B) * charge;

                    // Apply force
                    p.vel.x += F.x * uniforms.dt;
                    p.vel.y += F.y * uniforms.dt;
                    p.vel.z += F.z * uniforms.dt;
                }

                // Interaction Pulse: Repel from mouse ray
                if (uniforms.click > 0.5) {
                    // Project mouse to approximate world space at particle Z
                    // This is rough approximation for effect
                    let worldMouseX = uniforms.mouseX * 8.0 * uniforms.aspect;
                    let worldMouseY = uniforms.mouseY * 8.0;

                    let dx = p.pos.x - worldMouseX;
                    let dy = p.pos.y - worldMouseY;
                    let distSq = dx*dx + dy*dy;

                    if (distSq < 25.0) {
                        let force = 50.0 / (distSq + 1.0);
                        p.vel.x += dx * force * uniforms.dt;
                        p.vel.y += dy * force * uniforms.dt;
                        // Add some Z chaos
                        p.vel.z += force * uniforms.dt * 2.0;
                    }
                }

                // Position Update
                p.pos.x += p.vel.x * uniforms.dt;
                p.pos.y += p.vel.y * uniforms.dt;
                p.pos.z += p.vel.z * uniforms.dt;

                particles[index] = p;
            }
        `;

        // Render Shader: Point Sprites / Lines
        const renderShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) life : f32,
            }

            struct Uniforms {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                click : f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Camera Transform (Match WebGL2)
                // camPos = (0,0,-8)
                let p = pos.xyz;
                let viewPos = p - vec3f(0.0, 0.0, -8.0);

                let fov = 1.0;
                let scale = 1.0 / tan(fov * 0.5);

                output.position = vec4f(
                    viewPos.x * scale / uniforms.aspect,
                    viewPos.y * scale,
                    viewPos.z * 0.01,
                    viewPos.z
                );

                let charge = vel.w;
                var col = vec3f(1.0, 1.0, 1.0);

                if (charge < -0.5) { col = vec3f(0.2, 0.5, 1.0); } // Blue Electron
                else if (charge > 0.5) { col = vec3f(1.0, 0.2, 0.2); } // Red Positron
                else { col = vec3f(1.0, 1.0, 0.5); } // Yellow Neutral

                // Intensify color on click
                if (uniforms.click > 0.5) {
                    col = col + vec3f(0.5, 0.5, 0.5);
                }

                output.color = vec4f(col, 1.0);
                output.life = pos.w;

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f, @location(1) life : f32) -> @location(0) vec4f {
                // Fade out based on life
                let alpha = smoothstep(0.0, 0.2, life);
                return vec4f(color.rgb, alpha);
            }
        `;

        const particleData = new Float32Array(this.numParticles * 8);
        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 8 floats * 4 = 32 bytes
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
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos + life
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // vel + charge
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

        const time = (Date.now() - this.startTime) * 0.001;
        const clickVal = this.isClicked ? 1.0 : 0.0;

        // WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_click'), clickVal);

            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // WebGPU Render
        if (this.device && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            // Align: dt(4), time(4), mx(4), my(4), aspect(4), click(4), pad(4), pad(4)
            const uniforms = new Float32Array([
                0.016, time, this.mouse.x, this.mouse.y, aspect, clickVal, 0, 0
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
        if (this.container) {
            this.container.removeEventListener('mousemove', this.handleMouseMove);
            this.container.removeEventListener('mousedown', this.handleMouseDown);
        }
        window.removeEventListener('mouseup', this.handleMouseUp);

        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
        if (this.device) this.device.destroy();
        if (this.container) this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.SubatomicColliderExperiment = SubatomicColliderExperiment;
}
