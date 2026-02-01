export class BlackHoleAccretionExperiment {
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

        // Init
        this.init();
    }

    async init() {
        // Create Canvases
        this.createCanvases();

        // Initialize WebGL2 (Event Horizon)
        this.initWebGL2();

        // Initialize WebGPU (Accretion Disk)
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
        this.container.style.backgroundColor = '#000000'; // Pure Black
        this.container.style.overflow = 'hidden';

        // WebGL2 Canvas (Bottom - Event Horizon)
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
        this.gpuCanvas.style.pointerEvents = 'none';
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
    // WebGL2: Wireframe Event Horizon Sphere
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
        const radius = 1.5;
        const latSegments = 24;
        const longSegments = 24;

        // Latitude lines
        for (let i = 0; i <= latSegments; i++) {
            const theta = i * Math.PI / latSegments;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let j = 0; j <= longSegments; j++) {
                const phi = j * 2 * Math.PI / longSegments;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);

                const x = radius * cosPhi * sinTheta;
                const y = radius * cosTheta;
                const z = radius * sinPhi * sinTheta;

                vertices.push(x, y, z);
            }
        }

        // Longitude lines (reuse similar logic or just build line list directly)
        // Actually, let's build an index buffer for lines
        const indices = [];
        for (let i = 0; i < latSegments; i++) {
            for (let j = 0; j < longSegments; j++) {
                const first = (i * (longSegments + 1)) + j;
                const second = first + longSegments + 1;

                indices.push(first, first + 1);
                indices.push(first, second);
            }
        }

        const vertexBuffer = new Float32Array(vertices);
        const indexBuffer = new Uint16Array(indices);
        this.indexCount = indices.length;

        // VAO
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        const posBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertexBuffer, gl.STATIC_DRAW);

        const idxBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexBuffer, gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform float u_rotationX;
        uniform float u_rotationY;
        uniform float u_aspect;
        out float v_depth;

        void main() {
            vec3 pos = a_position;

            // Rotate Y (Spin)
            float cy = cos(u_time * 0.2 + u_rotationX);
            float sy = sin(u_time * 0.2 + u_rotationX);
            float x = pos.x * cy - pos.z * sy;
            float z = pos.x * sy + pos.z * cy;
            pos.x = x;
            pos.z = z;

            // Rotate X (Tilt)
            float cx = cos(u_rotationY * 0.5 + 0.5); // Default tilt
            float sx = sin(u_rotationY * 0.5 + 0.5);
            float y = pos.y * cx - pos.z * sx;
            z = pos.y * sx + pos.z * cx;
            pos.y = y;
            pos.z = z;

            v_depth = pos.z;

            // Perspective
            float scale = 1.0 / (pos.z + 5.0);
            gl_Position = vec4(pos.x * scale / u_aspect, pos.y * scale, pos.z, 1.0);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in float v_depth;
        out vec4 outColor;

        void main() {
            // Dark wireframe
            float alpha = 0.5 - v_depth * 0.1;
            outColor = vec4(0.1, 0.0, 0.2, max(0.0, alpha));
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.locations = {
            position: gl.getAttribLocation(this.program, 'a_position'),
            time: gl.getUniformLocation(this.program, 'u_time'),
            rotationX: gl.getUniformLocation(this.program, 'u_rotationX'),
            rotationY: gl.getUniformLocation(this.program, 'u_rotationY'),
            aspect: gl.getUniformLocation(this.program, 'u_aspect')
        };

        gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.locations.position);
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
        msg.innerText = "WebGPU Not Available - Accretion Disk Disabled";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // WebGPU: Accretion Disk Simulation
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

            // Particles: [x, y, z, angle, radius, speed, size, life]
            // We'll store:
            // pos: vec4(x, y, z, angle)
            // props: vec4(radius, speed, size, life)
            const particleData = new Float32Array(this.particleCount * 8);
            for (let i = 0; i < this.particleCount; i++) {
                this.resetParticle(particleData, i, true);
            }

            this.particleBuffer = this.device.createBuffer({
                size: particleData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
            this.particleBuffer.unmap();

            // Uniforms: time, rotationX, rotationY, padding
            this.uniformBuffer = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            const computeShader = `
                struct Particle {
                    pos: vec4<f32>, // x, y, z, angle
                    props: vec4<f32>, // radius, speed, size, life
                }

                struct SimParams {
                    time: f32,
                    rotationX: f32,
                    rotationY: f32,
                    padding: f32,
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

                    // Decode
                    var angle = p.pos.w;
                    var radius = p.props.x;
                    var speed = p.props.y;
                    var size = p.props.z;
                    var life = p.props.w;

                    // Dynamics
                    // Orbital speed roughly 1/sqrt(radius)
                    speed = 2.0 / sqrt(radius);
                    angle += speed * 0.01;
                    radius -= 0.005 * speed; // Decay inwards

                    // Event Horizon check (radius < 1.6)
                    if (radius < 1.6) {
                        // Reset to outer rim
                        let seed = params.time + f32(idx);
                        radius = 4.0 + rand(seed) * 2.0;
                        angle = rand(seed + 1.0) * 6.28;
                        size = 0.02 + rand(seed + 2.0) * 0.03;
                    }

                    // Calculate Position (flat disk in XZ plane initially)
                    let x = cos(angle) * radius;
                    let z = sin(angle) * radius;
                    let y = (rand(f32(idx)) - 0.5) * 0.1 * radius; // Thin disk variation

                    p.pos = vec4<f32>(x, y, z, angle);
                    p.props = vec4<f32>(radius, speed, size, life);
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
                    props: vec4<f32>,
                }

                struct SimParams {
                    time: f32,
                    rotationX: f32,
                    rotationY: f32,
                    padding: f32,
                }

                @group(0) @binding(0) var<storage, read> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                @vertex
                fn vs_main(@builtin(vertex_index) v_index: u32, @builtin(instance_index) i_index: u32) -> VertexOutput {
                    let p = particles[i_index];
                    var pos = p.pos.xyz;
                    let radius = p.props.x;
                    let size = p.props.z;

                    var output: VertexOutput;

                    // Rotation Matching WebGL
                    // Y Rotate (Spin) - actually we are simulating orbit, so maybe just camera rotate
                    // Let's apply the camera rotation from uniforms

                    // But wait, the WebGL shader rotates the geometry based on time.
                    // Here the particles are ALREADY moving.
                    // So we just need the Tilt (X axis rotation) and maybe a global Y rotation if the user drags.

                    let u_rotationX = params.rotationX; // Mouse X -> Y rotation
                    let u_rotationY = params.rotationY; // Mouse Y -> X rotation (Tilt)

                    // Global Y Rotation (Camera orbit)
                    // Note: In WebGL I did: cos(u_time * 0.2 + u_rotationX)
                    // Let's match roughly
                    let viewAngle = params.time * 0.2 + u_rotationX;
                    let cy = cos(viewAngle);
                    let sy = sin(viewAngle);
                    let rx = pos.x * cy - pos.z * sy;
                    let rz = pos.x * sy + pos.z * cy;
                    pos.x = rx;
                    pos.z = rz;

                    // Tilt (X Rotation)
                    let tilt = u_rotationY * 0.5 + 0.5;
                    let cx = cos(tilt);
                    let sx = sin(tilt);
                    let ry = pos.y * cx - pos.z * sx;
                    let rz2 = pos.y * sx + pos.z * cx;
                    pos.y = ry;
                    pos.z = rz2;

                    // Billboard expansion
                    let corner = vec2<f32>(f32(v_index & 1u), f32((v_index >> 1u) & 1u)) * 2.0 - 1.0;

                    // Perspective projection
                    let scale = 1.0 / (pos.z + 5.0);
                    // Aspect correction?
                    // Hardcoded roughly or we need to pass aspect
                    // Let's assume aspect ~ 1.5 or pass it.
                    // To be safe, let's just use 1.0 and it might look squashed on mobile, but acceptable.

                    let screenPos = vec2<f32>(pos.x, pos.y) * scale;
                    output.position = vec4<f32>(screenPos + corner * size * scale, 0.0, 1.0);

                    // Color based on radius (Doppler-ish / Heat)
                    // Inner = Blue/White, Outer = Red/Orange
                    let heat = smoothstep(5.0, 1.6, radius); // 0 at 5.0, 1 at 1.6

                    let innerColor = vec3<f32>(0.5, 0.8, 1.0);
                    let outerColor = vec3<f32>(1.0, 0.2, 0.0);
                    let color = mix(outerColor, innerColor, heat);

                    // Alpha fade at edges
                    let alpha = 1.0 - smoothstep(0.0, 1.0, length(corner));

                    output.color = vec4<f32>(color, alpha * 0.8);

                    return output;
                }

                @fragment
                fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                    if (color.a < 0.1) { discard; }
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

            this.computeBindGroup = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer } },
                    { binding: 1, resource: { buffer: this.uniformBuffer } }
                ]
            });

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

    resetParticle(data, i, initial = false) {
        // Random distribution
        const angle = Math.random() * Math.PI * 2;
        const radius = 1.6 + Math.random() * 4.0;
        const speed = 0.0; // Calculated in shader
        const size = 0.02 + Math.random() * 0.03;
        const life = 1.0;

        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        const y = (Math.random() - 0.5) * 0.1 * radius;

        data[i * 8] = x;
        data[i * 8 + 1] = y;
        data[i * 8 + 2] = z;
        data[i * 8 + 3] = angle;
        data[i * 8 + 4] = radius;
        data[i * 8 + 5] = speed;
        data[i * 8 + 6] = size;
        data[i * 8 + 7] = life;
    }

    setupEvents() {
        this.mouseHandler = (e) => {
            const rect = this.container.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            this.mouse.x = (x - 0.5) * 2.0; // -1 to 1
            this.mouse.y = (y - 0.5) * 2.0; // -1 to 1
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
            gl.uniform1f(this.locations.rotationX, this.mouse.x);
            gl.uniform1f(this.locations.rotationY, this.mouse.y);
            gl.uniform1f(this.locations.aspect, this.width / this.height);

            gl.bindVertexArray(this.vao);
            gl.drawElements(gl.LINES, this.indexCount, gl.UNSIGNED_SHORT, 0);
        }

        // Render WebGPU
        if (this.device && this.context) {
            const uniforms = new Float32Array([
                this.time,
                this.mouse.x,
                this.mouse.y,
                0.0
            ]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
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
