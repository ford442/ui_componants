
export class DarkEnergyPrism {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.getElementById(container) : container;
        if (!this.container) throw new Error('Container not found');

        this.particleCount = options.particleCount || 20000;
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        // State
        this.time = 0;
        this.mouse = { x: 0.5, y: 0.5 };
        this.prismRotation = 0;
        this.targetRotation = 0;

        // Init
        this.init();
    }

    async init() {
        // Create Canvases
        this.createCanvases();

        // Initialize WebGL2 (Prism)
        this.initWebGL2();

        // Initialize WebGPU (Particles)
        await this.initWebGPU();

        // Events
        this.setupEvents();

        // Start Loop
        this.animate();
    }

    createCanvases() {
        this.container.style.position = 'relative';
        this.container.style.backgroundColor = '#000';
        this.container.style.overflow = 'hidden';

        // WebGL2 Canvas (Bottom)
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.position = 'absolute';
        this.glCanvas.style.top = '0';
        this.glCanvas.style.left = '0';
        this.glCanvas.style.width = '100%';
        this.glCanvas.style.height = '100%';
        this.glCanvas.style.zIndex = '1';
        this.container.appendChild(this.glCanvas);

        // WebGPU Canvas (Top)
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
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        this.glCanvas.width = this.width;
        this.glCanvas.height = this.height;

        this.gpuCanvas.width = this.width;
        this.gpuCanvas.height = this.height;

        if (this.gl) this.gl.viewport(0, 0, this.width, this.height);
    }

    initWebGL2() {
        this.gl = this.glCanvas.getContext('webgl2', { alpha: true });
        if (!this.gl) return;

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); // Standard transparent blending
        gl.clearColor(0, 0, 0, 0); // Transparent clear

        // Prism Geometry (Triangular Prism)
        // Vertices (x, y, z)
        const r = 0.5;
        const len = 0.8;
        const h = r * Math.sin(Math.PI / 3);
        const ho = r * Math.cos(Math.PI / 3);

        // Triangle vertices (Front)
        const v1 = [0, r, len];
        const v2 = [-h, -ho, len];
        const v3 = [h, -ho, len];

        // Triangle vertices (Back)
        const v4 = [0, r, -len];
        const v5 = [-h, -ho, -len];
        const v6 = [h, -ho, -len];

        const vertices = new Float32Array([
            // Edges
            ...v1, ...v2,  ...v2, ...v3,  ...v3, ...v1, // Front Face
            ...v4, ...v5,  ...v5, ...v6,  ...v6, ...v4, // Back Face
            ...v1, ...v4,  ...v2, ...v5,  ...v3, ...v6  // Connecting Edges
        ]);

        // Shaders
        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform float u_rotation;
        uniform float u_aspect;
        out vec3 v_pos;

        void main() {
            vec3 pos = a_position;

            // Rotation Y
            float c = cos(u_rotation);
            float s = sin(u_rotation);
            float x = pos.x * c - pos.z * s;
            float z = pos.x * s + pos.z * c;
            pos.x = x;
            pos.z = z;

            // Rotation X (tilt slightly)
            float cx = cos(0.3);
            float sx = sin(0.3);
            float y = pos.y * cx - pos.z * sx;
            z = pos.y * sx + pos.z * cx;
            pos.y = y;
            pos.z = z;

            v_pos = pos;

            // Perspective
            float fov = 1.5;
            float scale = 1.0 / (pos.z + 2.5);
            gl_Position = vec4(pos.x * scale / u_aspect, pos.y * scale, pos.z, 1.0);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in vec3 v_pos;
        uniform float u_time;
        out vec4 outColor;

        void main() {
            // Animated neon glow
            float alpha = 0.4 + 0.2 * sin(u_time * 2.0 + v_pos.y * 5.0);
            outColor = vec4(0.0, 0.8, 1.0, alpha);
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

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
                this.resetParticle(particleData, i);
                // Randomize initial life to stagger emission
                particleData[i * 8 + 7] = Math.random();
            }

            this.particleBuffer = this.device.createBuffer({
                size: particleData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
            this.particleBuffer.unmap();

            // Uniforms: time, rotation, mouseX, mouseY (16 bytes)
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
                    rotation: f32,
                    mouseX: f32,
                    mouseY: f32,
                }

                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                // Prism SDF (Triangular Extrusion)
                fn sdTriPrism(p: vec3<f32>, h: vec2<f32>) -> f32 {
                    let q = abs(p);
                    return max(q.z - h.y, max(q.x * 0.866025 + p.y * 0.5, -p.y) - h.x * 0.5);
                }

                // Rotate vector
                fn rotateY(p: vec3<f32>, angle: f32) -> vec3<f32> {
                    let c = cos(angle);
                    let s = sin(angle);
                    return vec3<f32>(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
                }

                 // Rotate vector X (tilt)
                fn rotateX(p: vec3<f32>, angle: f32) -> vec3<f32> {
                    let c = cos(angle);
                    let s = sin(angle);
                    return vec3<f32>(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
                }

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    if (idx >= arrayLength(&particles)) { return; }

                    var p = particles[idx];
                    var pos = p.pos.xyz;
                    var vel = p.vel.xyz;
                    var life = p.vel.w;

                    // Update life
                    life -= 0.005;

                    if (life <= 0.0) {
                        // Reset
                        // Emit from left side (-X) moving right (+X)
                        // Or back to front? Let's do Back (-Z) to Front (+Z)
                        pos = vec3<f32>((fract(sin(params.time * 100.0 + f32(idx)) * 43758.54) - 0.5) * 2.0,
                                        (fract(sin(params.time * 50.0 + f32(idx)) * 12345.67) - 0.5) * 2.0,
                                        -3.0);
                        vel = vec3<f32>(0.0, 0.0, 0.05 + fract(sin(f32(idx)) * 100.0) * 0.05);
                        life = 1.0;
                    } else {
                        // Move
                        pos += vel;

                        // Interaction with Prism
                        // 1. Transform particle into prism local space (Inverse rotation)
                        // Prism rotates Y by params.rotation, X by 0.3
                        var localPos = rotateX(pos, -0.3); // Inverse X
                        localPos = rotateY(localPos, -params.rotation); // Inverse Y

                        // 2. Check SDF
                        // Prism size: r=0.5 (width), len=0.8 (depth)
                        // sdTriPrism h parameter: x=width, y=depth
                        let dist = sdTriPrism(localPos, vec2<f32>(1.0, 0.8));

                        if (dist < 0.0) {
                            // Hit inside!
                            // Deflect (Refract)
                            // Simple scattering
                            vel = normalize(localPos) * 0.1; // Explode outwards based on local position
                            vel = rotateY(vel, params.rotation); // Rotate back to world

                            // Change color logic is handled in fragment shader based on velocity/position
                            // Here we could perhaps encode state in w component of pos?
                            // Let's use pos.w as a "hit" flag: 0 = stream, 1 = refracted
                            p.pos.w = 1.0;
                        }
                    }

                    p.pos = vec4<f32>(pos, p.pos.w);
                    p.vel = vec4<f32>(vel, life);
                    particles[idx] = p;
                }
            `;

            // Vertex/Fragment Shader
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

                @vertex
                fn vs_main(@builtin(vertex_index) v_index: u32, @builtin(instance_index) i_index: u32) -> VertexOutput {
                    let p = particles[i_index];
                    let pos = p.pos.xyz;
                    let life = p.vel.w;
                    let isHit = p.pos.w; // 0 or 1

                    var output: VertexOutput;

                    // Simple point sprite expansion
                    let corner = vec2<f32>(f32(v_index & 1u), f32((v_index >> 1u) & 1u)) * 2.0 - 1.0;
                    let size = 0.02 * (1.0 - isHit * 0.5); // Smaller if hit

                    // Perspective projection (Simple manual)
                    let zDist = pos.z + 2.5;
                    let scale = 1.0 / zDist;

                    let screenPos = vec2<f32>(pos.x, pos.y) * scale;

                    output.position = vec4<f32>(screenPos + corner * size * scale, 0.0, 1.0);

                    // Color
                    // Stream: Cyan
                    // Refracted: Purple/Pink
                    if (isHit > 0.5) {
                        output.color = vec4<f32>(1.0, 0.2, 0.8, life);
                    } else {
                        output.color = vec4<f32>(0.2, 1.0, 1.0, life * 0.5);
                    }

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
                    { binding: 0, resource: { buffer: this.particleBuffer } }
                ]
            });

        } catch (e) {
            console.error('WebGPU Init Failed:', e);
        }
    }

    resetParticle(data, i) {
        // Init logic moved to shader for reset, but needed here for initial filling
        data[i * 8] = (Math.random() - 0.5) * 2; // x
        data[i * 8 + 1] = (Math.random() - 0.5) * 2; // y
        data[i * 8 + 2] = -3.0 - Math.random(); // z (start behind)
        data[i * 8 + 3] = 0; // padding/flag
        data[i * 8 + 4] = 0; // vx
        data[i * 8 + 5] = 0; // vy
        data[i * 8 + 6] = 0.1; // vz
        data[i * 8 + 7] = Math.random(); // life
    }

    setupEvents() {
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            this.mouse = { x, y };

            // Map mouse X to rotation (-PI to PI)
            this.targetRotation = (x - 0.5) * Math.PI * 2;
        });

        window.addEventListener('resize', () => this.resize());
    }

    animate() {
        this.time += 0.01;

        // Smooth rotation
        this.prismRotation += (this.targetRotation - this.prismRotation) * 0.1;

        // Render WebGL2 (Background Prism)
        this.renderWebGL();

        // Render WebGPU (Particles)
        this.renderWebGPU();

        requestAnimationFrame(() => this.animate());
    }

    renderWebGL() {
        if (!this.gl) return;
        const gl = this.gl;

        gl.viewport(0, 0, this.width, this.height);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);

        gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.locations.position);

        gl.uniform1f(this.locations.time, this.time);
        gl.uniform1f(this.locations.rotation, this.prismRotation);
        gl.uniform1f(this.locations.aspect, this.width / this.height);

        // Draw Lines (Wireframe)
        gl.drawArrays(gl.LINES, 0, 36);
    }

    renderWebGPU() {
        if (!this.device || !this.context) return;

        // Update Uniforms
        const uniforms = new Float32Array([
            this.time,
            this.prismRotation,
            this.mouse.x,
            this.mouse.y
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
        renderPass.draw(4, this.particleCount); // 4 vertices per particle (quad)
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}
