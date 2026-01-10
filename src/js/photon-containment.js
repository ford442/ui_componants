
export class PhotonContainmentExperiment {
    constructor(container) {
        this.container = container;
        this.canvasGL = document.createElement('canvas');
        this.canvasGPU = document.createElement('canvas');

        // Setup styles for layering
        this.container.style.position = 'relative';
        this.container.style.width = '100%';
        this.container.style.height = '100%';
        this.container.style.overflow = 'hidden';
        this.container.style.backgroundColor = '#050505';

        // Add canvases
        [this.canvasGL, this.canvasGPU].forEach(canvas => {
            canvas.style.position = 'absolute';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            this.container.appendChild(canvas);
        });

        this.canvasGL.style.zIndex = '1'; // Background structure
        this.canvasGPU.style.zIndex = '2'; // Particles on top

        this.isPlaying = true;
        this.time = 0;
        this.mouse = { x: 0, y: 0 };
        this.rect = this.container.getBoundingClientRect();

        // Bind methods
        this.resize = this.resize.bind(this);
        this.render = this.render.bind(this);
        this.onMouseMove = this.onMouseMove.bind(this);

        window.addEventListener('resize', this.resize);
        this.container.addEventListener('mousemove', this.onMouseMove);

        this.init();
    }

    onMouseMove(e) {
        this.rect = this.container.getBoundingClientRect();
        // Normalize mouse to [-1, 1]
        const x = ((e.clientX - this.rect.left) / this.rect.width) * 2 - 1;
        const y = -(((e.clientY - this.rect.top) / this.rect.height) * 2 - 1);
        this.mouse.x = x;
        this.mouse.y = y;
    }

    async init() {
        this.initWebGL();
        await this.initWebGPU();

        this.resize();
        requestAnimationFrame(this.render);
    }

    // --- WebGL2 Setup (The Container Structure) ---
    initWebGL() {
        this.gl = this.canvasGL.getContext('webgl2');
        if (!this.gl) {
            console.error('WebGL2 not supported');
            return;
        }

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        // Simple shader for a glowing box/container
        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform vec2 u_resolution;

        void main() {
            // Slight oscillation
            vec3 pos = a_position;

            // Basic perspective
            float aspect = u_resolution.x / u_resolution.y;
            pos.z -= 2.5;
            pos.x /= aspect;

            gl_Position = vec4(pos, 1.0);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        uniform float u_time;
        out vec4 outColor;

        void main() {
            // Pulsing blue/purple
            float pulse = 0.5 + 0.5 * sin(u_time * 2.0);
            vec3 color = mix(vec3(0.0, 0.5, 1.0), vec3(0.5, 0.0, 1.0), pulse);
            outColor = vec4(color, 0.3);
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.positionLoc = gl.getAttribLocation(this.program, 'a_position');
        this.timeLoc = gl.getUniformLocation(this.program, 'u_time');
        this.resLoc = gl.getUniformLocation(this.program, 'u_resolution');

        // Cube lines
        const vertices = [
            // Front
            -0.8, -0.8,  0.8,   0.8, -0.8,  0.8,
             0.8, -0.8,  0.8,   0.8,  0.8,  0.8,
             0.8,  0.8,  0.8,  -0.8,  0.8,  0.8,
            -0.8,  0.8,  0.8,  -0.8, -0.8,  0.8,
            // Back
            -0.8, -0.8, -0.8,   0.8, -0.8, -0.8,
             0.8, -0.8, -0.8,   0.8,  0.8, -0.8,
             0.8,  0.8, -0.8,  -0.8,  0.8, -0.8,
            -0.8,  0.8, -0.8,  -0.8, -0.8, -0.8,
            // Connectors
            -0.8, -0.8,  0.8,  -0.8, -0.8, -0.8,
             0.8, -0.8,  0.8,   0.8, -0.8, -0.8,
             0.8,  0.8,  0.8,   0.8,  0.8, -0.8,
            -0.8,  0.8,  0.8,  -0.8,  0.8, -0.8
        ];

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        gl.enableVertexAttribArray(this.positionLoc);
        gl.vertexAttribPointer(this.positionLoc, 3, gl.FLOAT, false, 0, 0);

        this.vertexCount = vertices.length / 3;
    }

    createProgram(gl, vsSource, fsSource) {
        const vs = this.compileShader(gl, gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
        const program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        return program;
    }

    compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(shader));
        }
        return shader;
    }

    // --- WebGPU Setup (The Particles) ---
    async initWebGPU() {
        if (!navigator.gpu) {
            console.warn("WebGPU not supported");
            return;
        }

        try {
            this.adapter = await navigator.gpu.requestAdapter();
            if (!this.adapter) return;

            this.device = await this.adapter.requestDevice();
            this.contextGPU = this.canvasGPU.getContext('webgpu');
            this.format = navigator.gpu.getPreferredCanvasFormat();

            this.contextGPU.configure({
                device: this.device,
                format: this.format,
                alphaMode: 'premultiplied',
            });

            const particleCount = 50000;
            this.particleCount = particleCount;

            // Data: pos(2), vel(2) => 4 floats
            const particleData = new Float32Array(particleCount * 4);
            for (let i = 0; i < particleCount; i++) {
                particleData[i*4] = (Math.random() * 2 - 1) * 0.8;   // x
                particleData[i*4+1] = (Math.random() * 2 - 1) * 0.8; // y
                particleData[i*4+2] = (Math.random() - 0.5) * 0.01;  // vx
                particleData[i*4+3] = (Math.random() - 0.5) * 0.01;  // vy
            }

            this.particleBuffer = this.device.createBuffer({
                size: particleData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
            this.particleBuffer.unmap();

            // Sim Params: time(f32), mouse(vec2f), padding(f32) => 16 bytes
            this.simParamsBuffer = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // Compute Shader
            const computeShader = `
                struct Particle {
                    pos : vec2f,
                    vel : vec2f,
                }
                @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

                struct SimParams {
                    time : f32,
                    mouse : vec2f,
                    padding : f32,
                }
                @group(0) @binding(1) var<uniform> params : SimParams;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                    let index = GlobalInvocationID.x;
                    if (index >= arrayLength(&particles)) { return; }

                    var p = particles[index];

                    // Mouse Interaction (Repel)
                    let d = p.pos - params.mouse;
                    let distSq = dot(d, d);
                    if (distSq < 0.2) {
                        let force = normalize(d) * (0.001 / (distSq + 0.01));
                        p.vel += force;
                    }

                    // Movement
                    p.pos += p.vel;

                    // Bounds Bounce [-0.8, 0.8]
                    if (abs(p.pos.x) > 0.8) {
                        p.vel.x *= -0.9;
                        p.pos.x = sign(p.pos.x) * 0.8;
                    }
                    if (abs(p.pos.y) > 0.8) {
                        p.vel.y *= -0.9;
                        p.pos.y = sign(p.pos.y) * 0.8;
                    }

                    // Damping
                    p.vel *= 0.99;

                    particles[index] = p;
                }
            `;

            this.computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: this.device.createShaderModule({ code: computeShader }),
                    entryPoint: 'main',
                },
            });

            this.computeBindGroup = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer } },
                    { binding: 1, resource: { buffer: this.simParamsBuffer } },
                ],
            });

            // Render Shader
            const renderShader = `
                struct Particle {
                    pos : vec2f,
                    vel : vec2f,
                }
                @group(0) @binding(0) var<storage, read> particles : array<Particle>;

                struct VertexOutput {
                    @builtin(position) position : vec4f,
                    @location(0) color : vec4f,
                }

                @vertex
                fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
                    let p = particles[vertexIndex];
                    var out : VertexOutput;
                    out.position = vec4f(p.pos, 0.0, 1.0);

                    let speed = length(p.vel);
                    // Color based on speed: Blue -> White
                    out.color = mix(vec4f(0.0, 0.5, 1.0, 0.8), vec4f(1.0, 1.0, 1.0, 1.0), speed * 50.0);
                    return out;
                }

                @fragment
                fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                    return color;
                }
            `;

            this.renderPipeline = this.device.createRenderPipeline({
                layout: 'auto',
                vertex: {
                    module: this.device.createShaderModule({ code: renderShader }),
                    entryPoint: 'vs_main',
                },
                fragment: {
                    module: this.device.createShaderModule({ code: renderShader }),
                    entryPoint: 'fs_main',
                    targets: [{
                        format: this.format,
                        blend: {
                            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                            alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                        }
                    }],
                },
                primitive: {
                    topology: 'point-list',
                },
            });

            this.renderBindGroup = this.device.createBindGroup({
                layout: this.renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer } },
                ],
            });

        } catch (e) {
            console.error("WebGPU init failed:", e);
        }
    }

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (this.gl) {
            this.canvasGL.width = width;
            this.canvasGL.height = height;
            this.gl.viewport(0, 0, width, height);
        }

        if (this.contextGPU) {
            this.canvasGPU.width = width;
            this.canvasGPU.height = height;
        }
    }

    render(t) {
        if (!this.isPlaying) return;

        // Time in seconds
        this.time = t * 0.001;

        // --- WebGL2 Render ---
        if (this.gl) {
            const gl = this.gl;
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(this.program);
            gl.uniform1f(this.timeLoc, this.time);
            gl.uniform2f(this.resLoc, this.canvasGL.width, this.canvasGL.height);

            gl.bindVertexArray(this.vao);
            gl.drawArrays(gl.LINES, 0, this.vertexCount);
        }

        // --- WebGPU Render ---
        if (this.device && this.contextGPU && this.computePipeline && this.renderPipeline) {
            // Update Uniforms: time, mouseX, mouseY, padding
            const paramsData = new Float32Array([this.time, this.mouse.x, this.mouse.y, 0.0]);
            this.device.queue.writeBuffer(this.simParamsBuffer, 0, paramsData);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
            computePass.end();

            // Render
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.contextGPU.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.renderBindGroup);
            renderPass.draw(this.particleCount);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        requestAnimationFrame(this.render);
    }
}
