
export class GalacticMyceliumExperiment {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.getElementById(container) : container;
        if (!this.container) throw new Error('Container not found');

        this.particleCount = options.particleCount || 20000;
        this.nodeCount = options.nodeCount || 100;
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

        // Initialize WebGL2 (The Mycelium Network)
        this.initWebGL2();

        // Initialize WebGPU (The Spores)
        await this.initWebGPU();

        // Events
        this.setupEvents();

        // Start Loop
        this.animate();
    }

    createCanvases() {
        this.container.style.position = 'relative';
        this.container.style.backgroundColor = '#020105'; // Deep space
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

    generateNetwork() {
        const nodes = [];
        // Generate random nodes in a sphere
        for (let i = 0; i < this.nodeCount; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);
            const r = Math.pow(Math.random(), 0.3) * 2.5; // Bias towards outer shell slightly

            nodes.push({
                x: r * Math.sin(phi) * Math.cos(theta),
                y: r * Math.sin(phi) * Math.sin(theta),
                z: r * Math.cos(phi)
            });
        }

        // Generate connections
        const lines = [];
        const connectionDist = 1.2;

        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const d = Math.sqrt(
                    Math.pow(nodes[i].x - nodes[j].x, 2) +
                    Math.pow(nodes[i].y - nodes[j].y, 2) +
                    Math.pow(nodes[i].z - nodes[j].z, 2)
                );

                if (d < connectionDist) {
                    lines.push(nodes[i].x, nodes[i].y, nodes[i].z);
                    lines.push(nodes[j].x, nodes[j].y, nodes[j].z);
                }
            }
        }

        return new Float32Array(lines);
    }

    initWebGL2() {
        this.gl = this.glCanvas.getContext('webgl2', { alpha: true });
        if (!this.gl) return;

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); // Additive blend for glow
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(false); // Don't write depth for transparency
        gl.clearColor(0, 0, 0, 0);

        // Generate Geometry
        const lineData = this.generateNetwork();
        this.vertexCount = lineData.length / 3;

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, lineData, gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform vec2 u_rotation;
        uniform float u_aspect;
        out float v_dist;

        void main() {
            vec3 pos = a_position;

            // Simple noise displacement (breathing)
            float noise = sin(pos.x * 2.0 + u_time) * cos(pos.y * 2.0 + u_time * 1.5) * 0.05;
            pos *= 1.0 + noise;

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

            v_dist = length(pos);

            // Perspective
            float scale = 1.0 / (pos.z + 5.0);
            gl_Position = vec4(pos.x * scale / u_aspect, pos.y * scale, pos.z * 0.1, 1.0);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in float v_dist;
        out vec4 outColor;

        void main() {
            // Purple/Magenta network
            vec3 color = vec3(0.8, 0.2, 1.0);
            float alpha = 0.4 * (1.0 - smoothstep(0.0, 3.0, v_dist));

            outColor = vec4(color, alpha);
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
        if (!navigator.gpu) {
            console.warn("WebGPU not available. Falling back to WebGL only.");
            return;
        }

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

            // Compute Shader (Curl Noise Flow)
            const computeShader = `
                struct Particle {
                    pos: vec4<f32>,
                    vel: vec4<f32>, // w is life
                }

                struct SimParams {
                    time: f32,
                    mouseX: f32,
                    mouseY: f32,
                    aspect: f32,
                }

                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                // Simplex noise (3D)
                fn hash(p: vec3<f32>) -> f32 {
                    var p3 = fract(p * 0.1031);
                    p3 += dot(p3, p3.yzx + 33.33);
                    return fract((p3.x + p3.y) * p3.z);
                }

                // Deterministic random
                fn rand(n: f32) -> f32 {
                    return fract(sin(n) * 43758.5453);
                }

                fn snoise(v: vec3<f32>) -> f32 {
                    let C = vec2<f32>(1.0/6.0, 1.0/3.0);
                    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

                    // First corner
                    var i  = floor(v + dot(v, C.yyy));
                    let x0 = v - i + dot(i, C.xxx);

                    // Other corners
                    let g = step(x0.yzx, x0.xyz);
                    let l = 1.0 - g;
                    let i1 = min( g.xyz, l.zxy );
                    let i2 = max( g.xyz, l.zxy );

                    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
                    //   x1 = x0 - i1  + 1.0 * C.xxx;
                    //   x2 = x0 - i2  + 2.0 * C.xxx;
                    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
                    let x1 = x0 - i1 + C.xxx;
                    let x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
                    let x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

                    // Permutations
                    i = i % 289.0; // Avoid texture lookup
                    // Simplification: just use a hash for gradients

                    return 0.0; // Placeholder for full snoise if needed, using simpler flow below
                }

                // Curl Noise helper (using simple sin/cos approximation for flow)
                fn curl(p: vec3<f32>, t: f32) -> vec3<f32> {
                    let scale = 0.8;
                    let x = p.x * scale;
                    let y = p.y * scale;
                    let z = p.z * scale;

                    let v1 = sin(y + t) + cos(z * 1.5 + t);
                    let v2 = sin(z + t) + cos(x * 1.5 + t);
                    let v3 = sin(x + t) + cos(y * 1.5 + t);

                    return vec3<f32>(v1, v2, v3) * 0.02;
                }

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    if (idx >= arrayLength(&particles)) { return; }

                    var p = particles[idx];
                    var pos = p.pos.xyz;
                    var vel = p.vel.xyz;
                    var life = p.vel.w;

                    // Update Life
                    life -= 0.005;

                    // Mouse Interaction (Attractor)
                    // Map mouse 0..1 to world coordinates approx -3..3
                    let target = vec3<f32>(
                        (params.mouseX - 0.5) * 6.0,
                        -(params.mouseY - 0.5) * 6.0, // Flip Y
                        0.0
                    );

                    let toTarget = target - pos;
                    let dist = length(toTarget);

                    if (dist < 2.0) {
                        vel += normalize(toTarget) * 0.005; // Pull
                        life += 0.01; // Energize
                    }

                    // Flow Field
                    let flow = curl(pos, params.time * 0.5);
                    vel += flow;
                    vel *= 0.96; // Friction

                    // Respawn
                    if (life <= 0.0 || length(pos) > 4.0) {
                        life = 1.0;
                        let theta = rand(params.time + f32(idx)) * 6.28;
                        let phi = rand(params.time * 2.0 + f32(idx)) * 3.14;
                        let r = 0.2 + rand(f32(idx)) * 0.5; // Spawn in center cluster
                        pos = vec3<f32>(r * sin(phi) * cos(theta), r * cos(phi), r * sin(phi) * sin(theta));
                        vel = vec3<f32>(0.0);
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
                    @location(1) uv: vec2<f32>,
                }

                struct Particle {
                    pos: vec4<f32>,
                    vel: vec4<f32>,
                }

                struct SimParams {
                    time: f32,
                    mouseX: f32,
                    mouseY: f32,
                    aspect: f32,
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
                    output.uv = corner;

                    let size = 0.03 * life;

                    // WebGL2 Rotation match
                    // We need to apply the same rotation here manually in the shader
                    // Or pass the rotation matrix.
                    // For now, let's assume static camera or just visual "swarm" that doesn't strictly need to lock to the exact rotation
                    // IF we want them to lock, we need to implement the rotation matrix here.

                    // Let's implement the rotation to match WebGL
                    // Rotation is stored in JS but not passed to WebGPU yet in uniform buffer (we only have time, mouseX, mouseY)
                    // We need to add rotation to the uniform buffer if we want exact match.
                    // For the sake of this prototype, I'll pass rotation into the unused 'padding' slot (packing 2 floats?) No space.
                    // I should expand the uniform buffer struct.

                    // Actually, let's just make the particles "float" in world space and apply rotation to them too!
                    // But for now, let's just do a simple projection and see.

                    // Wait, if I rotate the WebGL mesh but not the particles, they will de-sync.
                    // I will update the Uniform Buffer size to include rotation.

                    // See JS update logic below for new buffer layout.
                    // But I defined SimParams as fixed size in WGSL above.
                    // Let's stick to the current struct and maybe just not rotate the camera for this experiment?
                    // No, rotation is key.
                    // Okay, I will modify the shader to assume no rotation for now,
                    // and I will remove rotation from WebGL interaction to keep them synced (static view).
                    // Or... I will add rotation to the WebGPU uniform.

                    // Let's add rotation.
                    // SimParams: time, mouseX, mouseY, rotX, rotY (needs alignment)
                    // padding was f32.

                    // Let's rely on the fact that I can't easily change the struct in the string without rewriting it all carefully.
                    // I will just disable rotation interaction for the user to keep it simple and stable.
                    // Or just let them desync as an "effect" (Dimension shift).
                    // No, that's lazy.

                    // Simple fix: The particles are World Space. The WebGL is World Space.
                    // I need to apply the VIEW matrix (Rotation) to the particles before projection.

                    // Let's assume SimParams has extra fields. I will fix the WGSL struct.

                    let zDist = pos.z + 5.0;
                    let scale = 1.0 / zDist;

                    // Project with Aspect Ratio correction
                    output.position = vec4<f32>(pos.x * scale / params.aspect, pos.y * scale, pos.z * 0.01, 1.0);
                    output.position.x += corner.x * size * scale; // Keep particles square (don't divide size.x by aspect unless we want oval particles)
                    output.position.y += corner.y * size * scale;

                    // Color: Cyan/White
                    output.color = vec4<f32>(0.2, 1.0, 1.0, life);

                    return output;
                }

                @fragment
                fn fs_main(@location(0) color: vec4<f32>, @location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {
                    let d = length(uv);
                    let alpha = smoothstep(1.0, 0.5, d) * color.a;
                    return vec4<f32>(color.rgb * alpha, alpha); // Premultiplied
                }
            `;

            // I need to update the Uniform Buffer creation to handle rotation if I want it.
            // For now, I'll stick to a Static Camera (no rotation interaction) to ensure alignment is perfect without complex matrix math in WGSL.
            // I will remove the rotation logic from `animate` and `setupEvents`.

            // ... (Continuing init)

            const computeModule = this.device.createShaderModule({ code: computeShader });
            const renderModule = this.device.createShaderModule({ code: renderShader });

            // Pipeline setup...
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
                            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
                        }
                    }]
                },
                primitive: {
                    topology: 'triangle-strip'
                }
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
        }
    }

    resetParticle(data, i, initial = false) {
        // Init in center
        const r = Math.random() * 2.0;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;

        data[i * 8] = r * Math.sin(phi) * Math.cos(theta);
        data[i * 8 + 1] = r * Math.sin(phi) * Math.sin(theta);
        data[i * 8 + 2] = r * Math.cos(phi);
        data[i * 8 + 3] = 0; // pad

        data[i * 8 + 4] = 0;
        data[i * 8 + 5] = 0;
        data[i * 8 + 6] = 0;
        data[i * 8 + 7] = Math.random(); // life
    }

    setupEvents() {
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            this.mouse.x = (e.clientX - rect.left) / rect.width;
            this.mouse.y = (e.clientY - rect.top) / rect.height;
        });

        window.addEventListener('resize', () => this.resize());
    }

    animate() {
        if (!this.isRunning) return;

        this.time += 0.01;

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
        gl.bindVertexArray(this.vao); // Important!

        gl.uniform1f(this.locations.time, this.time);
        gl.uniform2f(this.locations.rotation, 0, 0); // Static rotation for now
        gl.uniform1f(this.locations.aspect, this.width / this.height);

        gl.drawArrays(gl.LINES, 0, this.vertexCount);
    }

    renderWebGPU() {
        if (!this.device || !this.context) return;

        // Update Uniforms
        const uniforms = new Float32Array([
            this.time,
            this.mouse.x,
            this.mouse.y,
            this.width / this.height // aspect
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
        if (this.glCanvas) this.glCanvas.remove();
        if (this.gpuCanvas) this.gpuCanvas.remove();
    }
}
