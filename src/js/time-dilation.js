
export class TimeDilationChamber {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.getElementById(container) : container;
        if (!this.container) throw new Error('Container not found');

        this.particleCount = options.particleCount || 50000;
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        // State
        this.isActive = true;
        this.time = 0;
        this.mouse = { x: 0.0, y: 0.0 };
        this.gravityCenter = { x: 0, y: 0, z: 0 };

        // Init
        this.init();
    }

    async init() {
        // Create Canvases
        this.createCanvases();

        // Initialize WebGL2 (Chronometric Cage)
        this.initWebGL2();

        // Initialize WebGPU (Temporal Particles)
        const gpuSuccess = await this.initWebGPU();
        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        // Events
        this.setupEvents();

        // Start Loop
        this.animate();
    }

    createCanvases() {
        this.container.style.position = 'relative';
        this.container.style.backgroundColor = '#050005'; // Deep dark purple/black
        this.container.style.overflow = 'hidden';

        // WebGL2 Canvas (Bottom - Structure)
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.position = 'absolute';
        this.glCanvas.style.top = '0';
        this.glCanvas.style.left = '0';
        this.glCanvas.style.width = '100%';
        this.glCanvas.style.height = '100%';
        this.glCanvas.style.zIndex = '1';
        this.container.appendChild(this.glCanvas);

        // WebGPU Canvas (Top - Particles)
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.position = 'absolute';
        this.gpuCanvas.style.top = '0';
        this.gpuCanvas.style.left = '0';
        this.gpuCanvas.style.width = '100%';
        this.gpuCanvas.style.height = '100%';
        this.gpuCanvas.style.zIndex = '2';
        this.gpuCanvas.style.pointerEvents = 'none'; // Let mouse pass through to container
        this.container.appendChild(this.gpuCanvas);

        this.resize();
    }

    resize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        this.glCanvas.width = this.width;
        this.glCanvas.height = this.height;

        this.gpuCanvas.width = this.width;
        this.gpuCanvas.height = this.height;

        if (this.gl) this.gl.viewport(0, 0, this.width, this.height);
    }

    // ========================================================================
    // WebGL2: Wireframe Sphere (Chronometric Cage)
    // ========================================================================
    initWebGL2() {
        this.gl = this.glCanvas.getContext('webgl2', { alpha: true });
        if (!this.gl) return;

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.clearColor(0, 0, 0, 0);

        // Sphere Geometry
        const vertices = [];
        const radius = 2.5;
        const latSegments = 16;
        const longSegments = 24;

        // Latitude rings
        for (let i = 1; i < latSegments; i++) {
            const theta = (i / latSegments) * Math.PI;
            const y = Math.cos(theta) * radius;
            const r = Math.sin(theta) * radius;

            for (let j = 0; j <= longSegments; j++) {
                const phi = (j / longSegments) * Math.PI * 2;
                const x = Math.cos(phi) * r;
                const z = Math.sin(phi) * r;
                vertices.push(x, y, z);
            }
        }

        // Longitude lines
        const longStartIndex = vertices.length / 3;
        for (let j = 0; j < longSegments; j++) {
            const phi = (j / longSegments) * Math.PI * 2;
            const cosPhi = Math.cos(phi);
            const sinPhi = Math.sin(phi);

            for (let i = 0; i <= latSegments; i++) {
                const theta = (i / latSegments) * Math.PI;
                const y = Math.cos(theta) * radius;
                const r = Math.sin(theta) * radius;
                const x = cosPhi * r;
                const z = sinPhi * r;
                vertices.push(x, y, z);
            }
        }

        const vertexBuffer = new Float32Array(vertices);

        // VAO
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        this.positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertexBuffer, gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform float u_aspect;
        out float v_dist;

        void main() {
            vec3 pos = a_position;

            // Rotation (Slow tumble)
            float t = u_time * 0.2;
            float c = cos(t);
            float s = sin(t);

            // Rotate Y
            float x = pos.x * c - pos.z * s;
            float z = pos.x * s + pos.z * c;
            pos.x = x;
            pos.z = z;

            // Rotate Z
            float cz = cos(t * 0.5);
            float sz = sin(t * 0.5);
            float nx = pos.x * cz - pos.y * sz;
            float ny = pos.x * sz + pos.y * cz;
            pos.x = nx;
            pos.y = ny;

            v_dist = length(pos);

            // Perspective
            float scale = 1.0 / (pos.z + 8.0);
            gl_Position = vec4(pos.x * scale / u_aspect, pos.y * scale, pos.z, 1.0);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in float v_dist;
        uniform float u_time;
        out vec4 outColor;

        void main() {
            // Pulse alpha
            float alpha = 0.2 + 0.1 * sin(u_time + v_dist * 2.0);
            // Purple/Red gradient
            outColor = vec4(0.8, 0.1, 0.5, alpha);
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.locations = {
            position: gl.getAttribLocation(this.program, 'a_position'),
            time: gl.getUniformLocation(this.program, 'u_time'),
            aspect: gl.getUniformLocation(this.program, 'u_aspect')
        };

        gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.locations.position);

        // Counts
        // Lat rings: (latSegments - 1) * (longSegments + 1) vertices
        this.ringVertexCount = (longSegments + 1);
        this.ringCount = latSegments - 1;

        // Long lines: longSegments * (latSegments + 1)
        this.lineVertexCount = (latSegments + 1);
        this.lineCount = longSegments;
    }

    createProgram(gl, vsSource, fsSource) {
        const vs = this.createShader(gl, gl.VERTEX_SHADER, vsSource);
        const fs = this.createShader(gl, gl.FRAGMENT_SHADER, fsSource);
        const program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        return program;
    }

    createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px;
            color: #ff5555; font-family: sans-serif; font-size: 14px;
            background: rgba(0,0,0,0.8); padding: 10px 15px; border-radius: 8px;
            border: 1px solid #ff5555; pointer-events: none;
        `;
        msg.innerText = "WebGPU Not Available - Simulation Disabled";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // WebGPU: Time Dilation Simulation
    // ========================================================================
    async initWebGPU() {
        if (!navigator.gpu) return false;

        try {
            this.adapter = await navigator.gpu.requestAdapter();
            if (!this.adapter) return false;

            this.device = await this.adapter.requestDevice();
            this.context = this.gpuCanvas.getContext('webgpu');
            this.format = navigator.gpu.getPreferredCanvasFormat();

            this.context.configure({
                device: this.device,
                format: this.format,
                alphaMode: 'premultiplied'
            });

            // Particles: [x, y, z, mass, vx, vy, vz, life]
            const particleData = new Float32Array(this.particleCount * 8);
            for (let i = 0; i < this.particleCount; i++) {
               this.resetParticle(particleData, i);
            }

            this.particleBuffer = this.device.createBuffer({
                size: particleData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
            this.particleBuffer.unmap();

            // Uniforms: time (f32), pad (3x f32), gravityCenter (vec3), pad (f32) -> 32 bytes aligned?
            // Struct:
            // time: f32
            // _pad1: f32
            // _pad2: f32
            // _pad3: f32
            // gravityCenter: vec3<f32>
            // _pad4: f32
            // Total 32 bytes

            this.uniformBuffer = this.device.createBuffer({
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            const computeShader = `
                struct Particle {
                    pos: vec4<f32>, // w = mass/id
                    vel: vec4<f32>, // w = life
                }

                struct SimParams {
                    time: f32,
                    _pad1: f32,
                    _pad2: f32,
                    _pad3: f32,
                    gravityCenter: vec3<f32>,
                    _pad4: f32,
                }

                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                fn rand(n: f32) -> f32 {
                    return fract(sin(n) * 43758.5453123);
                }

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    if (idx >= arrayLength(&particles)) { return; }

                    var p = particles[idx];
                    var pos = p.pos.xyz;
                    var mass = p.pos.w;
                    var vel = p.vel.xyz;
                    var life = p.vel.w;

                    // Calculate distance to gravity center
                    let delta = params.gravityCenter - pos;
                    let dist = length(delta);
                    let dir = normalize(delta);

                    // Time Dilation Logic:
                    // Closer to center = Slower Time (smaller dt)
                    // Radius of effect approx 2.0
                    let timeScale = smoothstep(0.0, 2.0, dist);

                    // Gravity force (increases when close, but clamped to avoid exploding)
                    let force = 0.05 / (dist * dist + 0.1);

                    // Update Velocity
                    // Add gravity
                    vel += dir * force * 0.01;

                    // Add swirl (cross product with up vector)
                    let swirl = cross(dir, vec3<f32>(0.0, 1.0, 0.0));
                    vel += swirl * 0.02 * (1.0 - timeScale); // More swirl near center

                    // Update Position with Time Scale
                    // If time stops (scale 0), pos doesn't change
                    // Base speed + variable time
                    let dt = 0.01 + 0.99 * timeScale;
                    pos += vel * dt;

                    // Friction
                    vel *= 0.99;

                    // Reset if swallowed (too close) or out of bounds
                    if (dist < 0.2 || dist > 5.0) {
                        let seed = params.time + f32(idx) * 0.001;
                        // Respawn at random edge
                        let theta = rand(seed) * 6.28;
                        let phi = rand(seed + 1.0) * 3.14;
                        let r = 4.0;

                        pos = vec3<f32>(
                            r * sin(phi) * cos(theta),
                            r * cos(phi),
                            r * sin(phi) * sin(theta)
                        );

                        // Initial velocity towards centerish
                        vel = -normalize(pos) * 0.02;

                        // life reset
                        life = 1.0;
                    }

                    p.pos = vec4<f32>(pos, mass);
                    p.vel = vec4<f32>(vel, life);
                    particles[idx] = p;
                }
            `;

            const renderShader = `
                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) color: vec4<f32>,
                }

                struct Particle {
                    pos: vec4<f32>,
                    vel: vec4<f32>,
                }

                struct SimParams {
                    time: f32,
                    _pad1: f32,
                    _pad2: f32,
                    _pad3: f32,
                    gravityCenter: vec3<f32>,
                    _pad4: f32,
                }

                @group(0) @binding(0) var<storage, read> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                @vertex
                fn vs_main(@builtin(vertex_index) v_index: u32, @builtin(instance_index) i_index: u32) -> VertexOutput {
                    let p = particles[i_index];
                    let pos = p.pos.xyz;

                    // Recalculate time dilation factor for coloring
                    let delta = params.gravityCenter - pos;
                    let dist = length(delta);
                    let timeScale = smoothstep(0.0, 2.0, dist);

                    var output: VertexOutput;

                    // Rotate Scene (match WebGL roughly)
                    let t = params.time * 0.2;
                    let c = cos(t);
                    let s = sin(t);

                    // Rotate Y
                    var x = pos.x * c - pos.z * s;
                    var z = pos.x * s + pos.z * c;

                    // Rotate Z
                    let cz = cos(t * 0.5);
                    let sz = sin(t * 0.5);
                    let nx = x * cz - pos.y * sz;
                    let ny = x * sz + pos.y * cz;

                    let viewPos = vec3<f32>(nx, ny, z);

                    // Billboard
                    let corner = vec2<f32>(f32(v_index & 1u), f32((v_index >> 1u) & 1u)) * 2.0 - 1.0;
                    let size = 0.03 * (2.0 - timeScale); // Bigger when slow/close

                    let scale = 1.0 / (viewPos.z + 8.0);
                    let screenPos = vec2<f32>(viewPos.x, viewPos.y) * scale;

                    output.position = vec4<f32>(screenPos + corner * size * scale, 0.0, 1.0);

                    // Color:
                    // Fast (Far) = Blue/Cyan
                    // Slow (Near) = Red/Orange
                    let farColor = vec3<f32>(0.0, 1.0, 1.0); // Cyan
                    let nearColor = vec3<f32>(1.0, 0.2, 0.0); // RedOrange

                    let color = mix(nearColor, farColor, timeScale);
                    let alpha = 0.6;

                    output.color = vec4<f32>(color, alpha);

                    return output;
                }

                @fragment
                fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                    // Circular soft particle
                    // Ideally we'd pass UVs but we can just output color for now or compute distance from center if we passed UV
                    return color;
                }
            `;

            const computeModule = this.device.createShaderModule({ code: computeShader });
            const renderModule = this.device.createShaderModule({ code: renderShader });

            this.computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: computeModule, entryPoint: 'main' }
            });

            this.renderPipeline = this.device.createRenderPipeline({
                layout: 'auto',
                vertex: { module: renderModule, entryPoint: 'vs_main' },
                fragment: {
                    module: renderModule,
                    entryPoint: 'fs_main',
                    targets: [{
                        format: this.format,
                        blend: {
                            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                        }
                    }]
                },
                primitive: { topology: 'triangle-strip' }
            });

            this.bindGroup = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer } },
                    { binding: 1, resource: { buffer: this.uniformBuffer } }
                ]
            });

            // Render bind group needs same layout usually if sharing binding 1?
            // Actually pipeline layouts are different (compute vs render).
            // But we can reuse the bind group if the layouts match at index 0 and 1.
            // Let's create a separate one for render if needed, but often we can share if binding indices align.
            // However, render shader binds Group 0, bindings 0 & 1 same as compute.
            // The pipeline layouts are auto, so they might differ slightly if usage differs.
            // To be safe and simple, let's just use one bind group since both pipelines see the same buffers at same bindings.
            // Wait, createRenderPipeline auto layout might infer differently.
            // Let's explicitly create a second bind group for render if the first fails, but typically 'auto' creates compatible layouts if shaders match.
            // Actually, best practice is to create it based on the specific pipeline.

            this.renderBindGroup = this.device.createBindGroup({
                layout: this.renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer } },
                    { binding: 1, resource: { buffer: this.uniformBuffer } }
                ]
            });

        } catch (e) {
            console.error('WebGPU Init Failed:', e);
            return false;
        }
        return true;
    }

    resetParticle(data, i) {
        // Random sphere distribution
        const r = 3.0 + Math.random();
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;

        data[i * 8] = r * Math.sin(phi) * Math.cos(theta);
        data[i * 8 + 1] = r * Math.cos(phi);
        data[i * 8 + 2] = r * Math.sin(phi) * Math.sin(theta);
        data[i * 8 + 3] = Math.random(); // mass
        data[i * 8 + 4] = 0.0;
        data[i * 8 + 5] = 0.0;
        data[i * 8 + 6] = 0.0;
        data[i * 8 + 7] = Math.random(); // life
    }

    setupEvents() {
        this.mouseHandler = (e) => {
            const rect = this.container.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            // Map mouse to gravity center (roughly -2 to 2)
            this.gravityCenter.x = (x - 0.5) * 4.0;
            this.gravityCenter.y = -(y - 0.5) * 4.0; // Invert Y
            this.gravityCenter.z = 0.0;
        };
        this.container.addEventListener('mousemove', this.mouseHandler);

        this.resizeHandler = () => this.resize();
        window.addEventListener('resize', this.resizeHandler);
    }

    animate() {
        if (!this.isActive) return;
        this.time += 0.01;

        // Render WebGL2
        if (this.gl) {
            const gl = this.gl;
            gl.viewport(0, 0, this.width, this.height);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.useProgram(this.program);

            gl.uniform1f(this.locations.time, this.time);
            gl.uniform1f(this.locations.aspect, this.width / this.height);

            gl.bindVertexArray(this.vao);

            // Draw Rings
            let offset = 0;
            for (let i = 0; i < this.ringCount; i++) {
                gl.drawArrays(gl.LINE_STRIP, offset, this.ringVertexCount);
                offset += this.ringVertexCount;
            }
            // Draw Lines
            // Reuse logic roughly, or recalculate offset?
            // The vertex buffer was pushed sequentially.
            // Rings first, then Lines.
            // Ring verts: this.ringCount * this.ringVertexCount
            // Wait, my buffer logic pushed rings then lines.

            for (let i = 0; i < this.lineCount; i++) {
                gl.drawArrays(gl.LINE_STRIP, offset, this.lineVertexCount);
                offset += this.lineVertexCount;
            }
        }

        // Render WebGPU
        if (this.device && this.context) {
            // Update Uniforms
            const uniformData = new Float32Array([
                this.time, 0, 0, 0, // time + padding
                this.gravityCenter.x, this.gravityCenter.y, this.gravityCenter.z, 0 // center + padding
            ]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.bindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
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
            renderPass.setBindGroup(0, this.renderBindGroup);
            renderPass.draw(4, this.particleCount);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.mouseHandler) {
            this.container.removeEventListener('mousemove', this.mouseHandler);
        }
        if (this.resizeHandler) {
            window.removeEventListener('resize', this.resizeHandler);
        }

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
