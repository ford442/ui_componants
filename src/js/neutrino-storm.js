
export class NeutrinoStormExperiment {
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
        this.fluxVector = { x: 0, y: -1, z: 0 }; // Initial downward flux

        // Init
        this.init();
    }

    async init() {
        // Create Canvases
        this.createCanvases();

        // Initialize WebGL2 (Detector Tank)
        this.initWebGL2();

        // Initialize WebGPU (Neutrinos)
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
        this.container.style.backgroundColor = '#000005'; // Deep dark blue/black
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
    // WebGL2: Wireframe Detector Tank
    // ========================================================================
    initWebGL2() {
        this.gl = this.glCanvas.getContext('webgl2', { alpha: true });
        if (!this.gl) return;

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.clearColor(0, 0, 0, 0);

        // Cylinder Geometry
        const vertices = [];
        const radius = 2.0;
        const height = 4.0;
        const segments = 24;
        const rings = 10;

        // Vertical lines
        for (let i = 0; i < segments; i++) {
            const theta = (i / segments) * Math.PI * 2;
            const x = Math.cos(theta) * radius;
            const z = Math.sin(theta) * radius;
            // Bottom
            vertices.push(x, -height/2, z);
            // Top
            vertices.push(x, height/2, z);
        }

        // Horizontal rings
        for (let j = 0; j <= rings; j++) {
            const y = -height/2 + (j / rings) * height;
            for (let i = 0; i <= segments; i++) { // <= to close loop
                const theta = (i / segments) * Math.PI * 2;
                const x = Math.cos(theta) * radius;
                const z = Math.sin(theta) * radius;
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
        uniform float u_rotation;
        uniform float u_aspect;
        out float v_y;

        void main() {
            vec3 pos = a_position;

            // Rotation
            float c = cos(u_rotation);
            float s = sin(u_rotation);
            float x = pos.x * c - pos.z * s;
            float z = pos.x * s + pos.z * c;
            pos.x = x;
            pos.z = z;

            // Tilt
            float cx = cos(0.2);
            float sx = sin(0.2);
            float y = pos.y * cx - pos.z * sx;
            z = pos.y * sx + pos.z * cx;
            pos.y = y;
            pos.z = z;

            v_y = pos.y;

            // Perspective
            float scale = 1.0 / (pos.z + 8.0);
            gl_Position = vec4(pos.x * scale / u_aspect, pos.y * scale, pos.z, 1.0);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in float v_y;
        uniform float u_time;
        out vec4 outColor;

        void main() {
            float alpha = 0.3 + 0.1 * sin(u_time * 2.0 + v_y);
            outColor = vec4(0.1, 0.3, 0.5, alpha); // Steel Blue
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.locations = {
            position: gl.getAttribLocation(this.program, 'a_position'),
            time: gl.getUniformLocation(this.program, 'u_time'),
            rotation: gl.getUniformLocation(this.program, 'u_rotation'),
            aspect: gl.getUniformLocation(this.program, 'u_aspect')
        };

        gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.locations.position);

        // Calculate counts
        // Verticals: segments * 2 vertices
        this.lineCount1 = segments * 2;
        // Rings: (rings + 1) * (segments + 1) vertices
        this.ringVertexCount = (segments + 1);
        this.ringCount = rings + 1;
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
        msg.innerText = "WebGPU Not Available - Neutrinos Disabled";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // WebGPU: Neutrino Simulation
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

            // Particles: [x, y, z, state, vx, vy, vz, life]
            // State: 0 = neutrino (invisible/faint), 1 = cherenkov burst (blue/bright)
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

            // Uniforms: time, rotation, fluxX, fluxY (16 bytes)
            this.uniformBuffer = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            const computeShader = `
                struct Particle {
                    pos: vec4<f32>, // w = state
                    vel: vec4<f32>, // w = life
                }

                struct SimParams {
                    time: f32,
                    rotation: f32,
                    fluxX: f32,
                    fluxY: f32,
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
                    var state = p.pos.w;
                    var vel = p.vel.xyz;
                    var life = p.vel.w;

                    // Update
                    life -= 0.01;
                    pos += vel;

                    // Interaction Probability
                    // Small chance to turn into Cherenkov radiation if inside tank
                    // Tank is roughly radius 2.0, height 4.0
                    // Just simple box check for now
                    if (state < 0.5 && life > 0.2) {
                        let seed = params.time + f32(idx) * 0.001;
                        if (rand(seed) > 0.9995) { // Rare event
                            // Boom! Cherenkov
                            state = 1.0;
                            life = 1.0; // Reset life for burst
                            vel = vel * 0.1; // Slow down
                        }
                    }

                    if (life <= 0.0 || pos.y < -3.0 || pos.y > 3.0) {
                        // Reset
                        // Emit from top or based on flux
                        let seed = params.time + f32(idx);
                        pos = vec3<f32>(
                            (rand(seed) - 0.5) * 4.0,
                            3.0, // Top
                            (rand(seed + 1.0) - 0.5) * 4.0
                        );

                        // Flux direction based on mouse
                        vel = vec3<f32>(
                            params.fluxX * 0.5 + (rand(seed+2.0)-0.5)*0.1,
                            -0.2 - rand(seed+3.0)*0.1, // Downwards
                            params.fluxY * 0.5 + (rand(seed+4.0)-0.5)*0.1
                        );

                        state = 0.0;
                        life = rand(seed+5.0);
                    }

                    p.pos = vec4<f32>(pos, state);
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

                @group(0) @binding(0) var<storage, read> particles: array<Particle>;

                // Hardcoded projection for now, matching WebGL roughly

                @vertex
                fn vs_main(@builtin(vertex_index) v_index: u32, @builtin(instance_index) i_index: u32) -> VertexOutput {
                    let p = particles[i_index];
                    let pos = p.pos.xyz;
                    let state = p.pos.w;
                    let life = p.vel.w;

                    var output: VertexOutput;

                    // Rotate same as WebGL
                    let t = 0.0; // Assume 0 for uniform simple match or pass uniform
                    // Actually we need the rotation uniform here too to match strictly
                    // But let's just do projection

                    // Simple Billboard
                    let corner = vec2<f32>(f32(v_index & 1u), f32((v_index >> 1u) & 1u)) * 2.0 - 1.0;

                    var size = 0.02;
                    if (state > 0.5) { size = 0.08 * life; } // Big burst

                    // Tilt (manual match to WebGL shader: x-axis 0.2 rad)
                    let cx = cos(0.2);
                    let sx = sin(0.2);
                    let y = pos.y * cx - pos.z * sx;
                    let z = pos.y * sx + pos.z * cx;

                    let scale = 1.0 / (z + 8.0);
                    let screenPos = vec2<f32>(pos.x, y) * scale;

                    output.position = vec4<f32>(screenPos + corner * size * scale, 0.0, 1.0);

                    if (state > 0.5) {
                        // Cherenkov Blue
                        output.color = vec4<f32>(0.2, 0.6, 1.0, life);
                    } else {
                        // Faint trace
                        output.color = vec4<f32>(0.5, 0.5, 0.5, 0.05);
                    }

                    return output;
                }

                @fragment
                fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
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
                    { binding: 0, resource: { buffer: this.particleBuffer } }
                ]
            });

        } catch (e) {
            console.error('WebGPU Init Failed:', e);
            return false;
        }
        return true;
    }

    resetParticle(data, i) {
        data[i * 8] = (Math.random() - 0.5) * 4.0;
        data[i * 8 + 1] = 4.0; // Top
        data[i * 8 + 2] = (Math.random() - 0.5) * 4.0;
        data[i * 8 + 3] = 0.0; // state
        data[i * 8 + 4] = 0.0;
        data[i * 8 + 5] = -0.2;
        data[i * 8 + 6] = 0.0;
        data[i * 8 + 7] = Math.random();
    }

    setupEvents() {
        this.mouseHandler = (e) => {
            const rect = this.container.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            // Map to flux (-1 to 1)
            this.fluxVector.x = (x - 0.5) * 2.0;
            this.fluxVector.z = (y - 0.5) * 2.0;
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
            gl.uniform1f(this.locations.rotation, this.time * 0.1);
            gl.uniform1f(this.locations.aspect, this.width / this.height);

            gl.bindVertexArray(this.vao);

            // Draw verticals (LINES)
            gl.drawArrays(gl.LINES, 0, this.lineCount1);

            // Draw rings (LINE_STRIP loop for each ring)
            let offset = this.lineCount1;
            for (let i = 0; i < this.ringCount; i++) {
                gl.drawArrays(gl.LINE_STRIP, offset, this.ringVertexCount);
                offset += this.ringVertexCount;
            }
        }

        // Render WebGPU
        if (this.device && this.context) {
            const uniforms = new Float32Array([
                this.time,
                this.time * 0.1,
                this.fluxVector.x,
                this.fluxVector.z
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
