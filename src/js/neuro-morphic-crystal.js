
export class NeuroMorphicCrystal {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.getElementById(container) : container;
        if (!this.container) throw new Error('Container not found');

        this.particleCount = options.particleCount || 30000;
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        // State
        this.time = 0;
        this.mouse = { x: 0.5, y: 0.5 };
        this.rotation = { x: 0, y: 0 };
        this.targetRotation = { x: 0, y: 0 };
        this.isRunning = true;

        // Init
        this.init();
    }

    async init() {
        // Create Canvases
        this.createCanvases();

        // Initialize WebGL2 (Crystal Geometry)
        this.initWebGL2();

        // Initialize WebGPU (Synaptic Particles)
        await this.initWebGPU();

        // Events
        this.setupEvents();

        // Start Loop
        this.animate();
    }

    createCanvases() {
        this.container.style.position = 'relative';
        this.container.style.backgroundColor = '#050510'; // Deep neural blue/black
        this.container.style.overflow = 'hidden';

        // WebGL2 Canvas (Bottom - The Structure)
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.position = 'absolute';
        this.glCanvas.style.top = '0';
        this.glCanvas.style.left = '0';
        this.glCanvas.style.width = '100%';
        this.glCanvas.style.height = '100%';
        this.glCanvas.style.zIndex = '1';
        this.container.appendChild(this.glCanvas);

        // WebGPU Canvas (Top - The Energy)
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.position = 'absolute';
        this.gpuCanvas.style.top = '0';
        this.gpuCanvas.style.left = '0';
        this.gpuCanvas.style.width = '100%';
        this.gpuCanvas.style.height = '100%';
        this.gpuCanvas.style.zIndex = '2';
        this.container.appendChild(this.gpuCanvas);

        this.resize();
    }

    resize() {
        if (!this.container) return;
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        if (this.glCanvas) {
            this.glCanvas.width = this.width;
            this.glCanvas.height = this.height;
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = this.width;
            this.gpuCanvas.height = this.height;
        }

        if (this.gl) this.gl.viewport(0, 0, this.width, this.height);
    }

    initWebGL2() {
        this.gl = this.glCanvas.getContext('webgl2', { alpha: true });
        if (!this.gl) return;

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.enable(gl.DEPTH_TEST);
        gl.clearColor(0, 0, 0, 0);

        // Icosahedron Geometry
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;
        const v = [
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
        ];

        // Normalize vertices to project onto sphere (r=1)
        const vertices = [];
        v.forEach(p => {
            const len = Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
            vertices.push(p[0]/len, p[1]/len, p[2]/len);
        });

        // Indices for lines (Icosahedron edges)
        // 12 vertices, 20 faces, 30 edges.
        // We will draw LINES.
        // Connection map is complex, manually listing triangles and converting to lines
        const indices = [
            0, 11, 0, 5, 0, 1, 0, 7, 0, 10,
            11, 5, 5, 1, 1, 7, 7, 10, 10, 11,
            11, 4, 5, 9, 1, 6, 7, 8, 10, 2,
            4, 9, 9, 6, 6, 8, 8, 2, 2, 4,
            3, 4, 3, 9, 3, 6, 3, 8, 3, 2,
            4, 5, 9, 1, 6, 7, 8, 10, 2, 11
        ];
        // Note: The above indices list is approximate for edges. Let's use a simpler known set or just all pairs?
        // Actually, let's use the standard face definitions and draw lines for each face edge.
        const faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 8], [7, 1, 6],
            [3, 9, 4], [3, 4, 2], [3, 2, 8], [3, 8, 6], [3, 6, 9],
            [4, 9, 5], [2, 4, 11], [8, 2, 10], [6, 8, 7], [9, 6, 1]
        ];

        const lineIndices = [];
        faces.forEach(f => {
            lineIndices.push(f[0], f[1], f[1], f[2], f[2], f[0]);
        });

        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        this.indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(lineIndices), gl.STATIC_DRAW);

        this.indexCount = lineIndices.length;

        // Shaders
        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform vec2 u_rotation;
        uniform float u_aspect;
        out vec3 v_pos;
        out float v_pulse;

        void main() {
            vec3 pos = a_position;

            // Rotation
            float cx = cos(u_rotation.x);
            float sx = sin(u_rotation.x);
            float cy = cos(u_rotation.y);
            float sy = sin(u_rotation.y);

            // Rotate Y
            float x = pos.x * cy - pos.z * sy;
            float z = pos.x * sy + pos.z * cy;
            pos.x = x;
            pos.z = z;

            // Rotate X
            float y = pos.y * cx - pos.z * sx;
            z = pos.y * sx + pos.z * cx;
            pos.y = y;
            pos.z = z;

            // Pulse effect: distort vertices based on sine wave
            float pulse = sin(u_time * 3.0 + pos.y * 5.0) * 0.5 + 0.5;
            pos *= 1.0 + pulse * 0.1;

            v_pos = pos;
            v_pulse = pulse;

            // Perspective
            float scale = 1.0 / (pos.z + 3.0);
            gl_Position = vec4(pos.x * scale / u_aspect, pos.y * scale, pos.z * 0.1, 1.0);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in vec3 v_pos;
        in float v_pulse;
        uniform float u_time;
        out vec4 outColor;

        void main() {
            // Neural blue/cyan glow
            vec3 color = mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.8), v_pulse);
            float alpha = 0.3 + v_pulse * 0.5;

            // Rim darkening
            alpha *= smoothstep(0.0, 1.0, 1.0 - abs(v_pos.z));

            outColor = vec4(color, alpha);
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.locations = {
            position: gl.getAttribLocation(this.program, 'a_position'),
            time: gl.getUniformLocation(this.program, 'u_time'),
            rotation: gl.getUniformLocation(this.program, 'u_rotation'),
            aspect: gl.getUniformLocation(this.program, 'u_aspect')
        };
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

    async initWebGPU() {
        if (!navigator.gpu) return;

        try {
            this.adapter = await navigator.gpu.requestAdapter();
            if (!this.adapter) return;

            this.device = await this.adapter.requestDevice();
            this.context = this.gpuCanvas.getContext('webgpu');
            this.format = navigator.gpu.getPreferredCanvasFormat();

            this.context.configure({
                device: this.device,
                format: this.format,
                alphaMode: 'premultiplied'
            });

            // Particle Data: [x, y, z, pad, vx, vy, vz, life] (32 bytes)
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

            // Uniforms: time, mouseX, mouseY, pad (16 bytes)
            this.uniformBuffer = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            // Compute Shader
            const computeShader = `
                struct Particle {
                    pos: vec4<f32>,
                    vel: vec4<f32>, // w is life
                }

                struct SimParams {
                    time: f32,
                    mouseX: f32,
                    mouseY: f32,
                    padding: f32,
                }

                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                // Hash function for randomness
                fn hash(n: f32) -> f32 {
                    return fract(sin(n) * 43758.5453);
                }

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    if (idx >= arrayLength(&particles)) { return; }

                    var p = particles[idx];
                    var pos = p.pos.xyz;
                    var vel = p.vel.xyz;
                    var life = p.vel.w;

                    // Neural impulse behavior:
                    // Orbit center, occasionally dart in/out

                    // Update life
                    life -= 0.005 + (0.01 * params.mouseX); // Mouse interaction speeds up decay
                    if (life < 0.0) {
                        life = 1.0;
                        // Respawn
                        let theta = hash(params.time + f32(idx)) * 6.28;
                        let phi = hash(params.time * 2.0 + f32(idx)) * 3.14;
                        let r = 2.0;
                        pos = vec3<f32>(r * sin(phi) * cos(theta), r * cos(phi), r * sin(phi) * sin(theta));

                        // Velocity towards center
                        vel = -normalize(pos) * (0.02 + hash(f32(idx)) * 0.05);
                    }

                    // Attraction to center (Neural Core)
                    let center = vec3<f32>(0.0, 0.0, 0.0);
                    let toCenter = center - pos;
                    let dist = length(toCenter);

                    // Spiral force
                    let tangent = cross(normalize(pos), vec3<f32>(0.0, 1.0, 0.0));

                    // Interaction: Mouse pushes particles out
                    // Mouse is 0..1, map to -1..1 range approx for screen space
                    // But here we are in 3D world space.
                    // Let's assume mouse influences a "disturbance" field

                    if (dist < 0.2) {
                        // Reached "synapse", fire out
                         vel = normalize(pos) * 0.1;
                    } else {
                        // Orbit + Attract
                        vel += normalize(toCenter) * 0.001;
                        vel += tangent * 0.002;
                    }

                    pos += vel;

                    p.pos = vec4<f32>(pos, 1.0);
                    p.vel = vec4<f32>(vel, life);
                    particles[idx] = p;
                }
            `;

            // Render Shader
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
                    mouseX: f32,
                    mouseY: f32,
                    padding: f32,
                }

                @group(0) @binding(0) var<storage, read> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                @vertex
                fn vs_main(@builtin(vertex_index) v_index: u32, @builtin(instance_index) i_index: u32) -> VertexOutput {
                    let p = particles[i_index];
                    let pos = p.pos.xyz;
                    let life = p.vel.w;

                    var output: VertexOutput;

                    // Quad generation
                    let corner = vec2<f32>(f32(v_index & 1u), f32((v_index >> 1u) & 1u)) * 2.0 - 1.0;
                    let size = 0.01 * (1.0 + params.mouseX * 2.0); // Mouse increases size

                    // Perspective
                    let zDist = pos.z + 3.0; // Same offset as WebGL
                    let scale = 1.0 / zDist;

                    // Apply rotation (must match WebGL rotation manually or via uniform)
                    // For simplicity, we keep particles static frame for now or duplicate rotation math
                    // Let's duplicate basic Y rotation to match camera orbit feel if we had one
                    // But here we rely on the visual "swarm" effect.

                    // Actually, let's just project simply.
                    let screenPos = vec2<f32>(pos.x, pos.y) * scale;
                    output.position = vec4<f32>(screenPos + corner * size * scale, 0.0, 1.0);

                    // Color based on life and speed
                    let speed = length(p.vel.xyz);
                    let intensity = smoothstep(0.0, 0.1, speed);

                    output.color = vec4<f32>(0.2, 0.8, 1.0, life * intensity);

                    return output;
                }

                @fragment
                fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                    return color;
                }
            `;

            // Modules
            const computeModule = this.device.createShaderModule({ code: computeShader });
            const renderModule = this.device.createShaderModule({ code: renderShader });

            // Pipelines
            this.computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: computeModule, entryPoint: 'main' }
            });

            this.renderPipeline = this.device.createRenderPipeline({
                layout: 'auto',
                vertex: {
                    module: renderModule,
                    entryPoint: 'vs_main'
                },
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
                primitive: {
                    topology: 'triangle-strip'
                }
            });

            // Bind Groups
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
        }
    }

    resetParticle(data, i, initial = false) {
        // Random spherical position
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);
        const r = 2.0;

        data[i * 8] = r * Math.sin(phi) * Math.cos(theta);
        data[i * 8 + 1] = r * Math.sin(phi) * Math.sin(theta);
        data[i * 8 + 2] = r * Math.cos(phi);
        data[i * 8 + 3] = 0; // pad

        data[i * 8 + 4] = 0; // vx
        data[i * 8 + 5] = 0; // vy
        data[i * 8 + 6] = 0; // vz
        data[i * 8 + 7] = Math.random(); // life
    }

    setupEvents() {
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            this.mouse.x = (e.clientX - rect.left) / rect.width;
            this.mouse.y = (e.clientY - rect.top) / rect.height;

            // Target rotation based on mouse
            this.targetRotation.x = (this.mouse.y - 0.5) * 1.0;
            this.targetRotation.y = (this.mouse.x - 0.5) * 2.0;
        });

        window.addEventListener('resize', () => this.resize());
    }

    animate() {
        if (!this.isRunning) return;

        this.time += 0.01;

        // Smooth rotation
        this.rotation.x += (this.targetRotation.x - this.rotation.x) * 0.1;
        this.rotation.y += (this.targetRotation.y - this.rotation.y) * 0.1;

        // Render WebGL2
        this.renderWebGL();

        // Render WebGPU
        this.renderWebGPU();

        requestAnimationFrame(() => this.animate());
    }

    renderWebGL() {
        if (!this.gl) return;
        const gl = this.gl;

        gl.viewport(0, 0, this.width, this.height);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(this.program);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.locations.position);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);

        gl.uniform1f(this.locations.time, this.time);
        gl.uniform2f(this.locations.rotation, this.rotation.x, this.rotation.y);
        gl.uniform1f(this.locations.aspect, this.width / this.height);

        gl.drawElements(gl.LINES, this.indexCount, gl.UNSIGNED_SHORT, 0);
    }

    renderWebGPU() {
        if (!this.device || !this.context) return;

        // Update Uniforms
        const uniforms = new Float32Array([
            this.time,
            this.mouse.x,
            this.mouse.y,
            0.0 // padding
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

        const commandEncoder = this.device.createCommandEncoder();

        // Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
        computePass.end();

        // Render Pass
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

    destroy() {
        this.isRunning = false;
        // Cleanup resources if needed
        if (this.glCanvas) this.glCanvas.remove();
        if (this.gpuCanvas) this.gpuCanvas.remove();
    }
}
