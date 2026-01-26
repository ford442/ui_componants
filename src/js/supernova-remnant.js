export class SupernovaRemnantExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.numParticles = options.numParticles || 30000;

        this.canvasGL = document.createElement('canvas');
        this.canvasGPU = document.createElement('canvas');

        this.container.style.position = 'relative';
        this.container.style.width = '100%';
        this.container.style.height = '100%';
        this.container.style.backgroundColor = '#000000';
        this.container.style.overflow = 'hidden';

        [this.canvasGL, this.canvasGPU].forEach(canvas => {
            canvas.style.position = 'absolute';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            this.container.appendChild(canvas);
        });

        this.canvasGL.style.zIndex = '1';
        this.canvasGPU.style.zIndex = '2';

        this.time = 0;
        this.isPlaying = true;
        this.mouse = { x: 0, y: 0 };

        this.resize = this.resize.bind(this);
        this.render = this.render.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.initWebGL();
        await this.initWebGPU();

        window.addEventListener('resize', this.resize);
        this.container.addEventListener('mousemove', this.handleMouseMove);

        this.resize();
        requestAnimationFrame(this.render);
    }

    handleMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        this.mouse.x = (x / rect.width) * 2 - 1;
        this.mouse.y = -((y / rect.height) * 2 - 1);
    }

    initWebGL() {
        this.gl = this.canvasGL.getContext('webgl2');
        if (!this.gl) return;

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.enable(gl.DEPTH_TEST);

        const vsSource = `#version 300 es
        in vec3 a_position;
        in vec3 a_normal;

        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec2 u_mouse;

        out float v_displacement;

        // Simple pseudo-random noise
        float hash(vec3 p) {
            p = fract(p * 0.3183099 + .1);
            p *= 17.0;
            return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
        }

        float noise(vec3 x) {
            vec3 i = floor(x);
            vec3 f = fract(x);
            f = f * f * (3.0 - 2.0 * f);
            return mix(mix(mix(hash(i + vec3(0,0,0)),
                               hash(i + vec3(1,0,0)), f.x),
                           mix(hash(i + vec3(0,1,0)),
                               hash(i + vec3(1,1,0)), f.x), f.y),
                       mix(mix(hash(i + vec3(0,0,1)),
                               hash(i + vec3(1,0,1)), f.x),
                           mix(hash(i + vec3(0,1,1)),
                               hash(i + vec3(1,1,1)), f.x), f.y), f.z);
        }

        void main() {
            // Rotation based on time and mouse
            float rotSpeed = u_time * 0.2;
            float rotX = rotSpeed + u_mouse.y;
            float rotY = rotSpeed + u_mouse.x;

            mat3 rot = mat3(
                cos(rotY), 0.0, sin(rotY),
                0.0, 1.0, 0.0,
                -sin(rotY), 0.0, cos(rotY)
            ) * mat3(
                1.0, 0.0, 0.0,
                0.0, cos(rotX), -sin(rotX),
                0.0, sin(rotX), cos(rotX)
            );

            vec3 pos = a_position;

            // Pulsating / Exploding effect
            float pulse = sin(u_time * 2.0) * 0.5 + 0.5;
            float n = noise(pos * 3.0 + u_time * 0.5);

            float displacement = n * (0.2 + pulse * 0.3);
            v_displacement = displacement;

            pos += a_normal * displacement;

            pos = rot * pos;

            // Perspective
            float aspect = u_resolution.x / u_resolution.y;
            pos.z -= 4.0;
            pos.x /= aspect;

            gl_Position = vec4(pos.x, pos.y, pos.z / (pos.z * -1.0), 1.0);
            gl_PointSize = 2.0;
        }`;

        const fsSource = `#version 300 es
        precision highp float;

        in float v_displacement;
        out vec4 outColor;

        void main() {
            // Color based on displacement (heat map)
            vec3 coreColor = vec3(1.0, 0.2, 0.1); // Red/Orange
            vec3 hotColor = vec3(1.0, 0.9, 0.5);  // Yellow/White

            vec3 color = mix(coreColor, hotColor, v_displacement * 2.0);

            // Add some transparency/glow feel
            outColor = vec4(color, 0.6);
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.timeLoc = gl.getUniformLocation(this.program, 'u_time');
        this.resLoc = gl.getUniformLocation(this.program, 'u_resolution');
        this.mouseLoc = gl.getUniformLocation(this.program, 'u_mouse');
        this.posLoc = gl.getAttribLocation(this.program, 'a_position');
        this.normLoc = gl.getAttribLocation(this.program, 'a_normal');

        // Create Sphere Geometry (IcoSphere or UV Sphere)
        // Simple UV sphere for now
        const positions = [];
        const normals = [];
        const latBands = 30;
        const longBands = 30;
        const radius = 1.0;

        for (let lat = 0; lat <= latBands; lat++) {
            const theta = lat * Math.PI / latBands;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let long = 0; long <= longBands; long++) {
                const phi = long * 2 * Math.PI / longBands;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);

                const x = cosPhi * sinTheta;
                const y = cosTheta;
                const z = sinPhi * sinTheta;

                normals.push(x, y, z);
                positions.push(radius * x, radius * y, radius * z);
            }
        }

        // Indices for lines (wireframe)
        const indices = [];
        for (let lat = 0; lat < latBands; lat++) {
            for (let long = 0; long < longBands; long++) {
                const first = (lat * (longBands + 1)) + long;
                const second = first + longBands + 1;

                indices.push(first, second);
                indices.push(first, first + 1);
            }
        }

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        const posBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(this.posLoc);
        gl.vertexAttribPointer(this.posLoc, 3, gl.FLOAT, false, 0, 0);

        const normBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(this.normLoc);
        gl.vertexAttribPointer(this.normLoc, 3, gl.FLOAT, false, 0, 0);

        const indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

        this.indexCount = indices.length;
    }

    createProgram(gl, vsSource, fsSource) {
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSource);
        gl.compileShader(vs);

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSource);
        gl.compileShader(fs);

        const program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);

        return program;
    }

    async initWebGPU() {
        if (!navigator.gpu) return;

        try {
            this.adapter = await navigator.gpu.requestAdapter();
            if (!this.adapter) return;
            this.device = await this.adapter.requestDevice();
        } catch (e) {
            console.warn("WebGPU init failed", e);
            return;
        }

        this.contextGPU = this.canvasGPU.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();

        this.contextGPU.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied'
        });

        // Particles: pos(4), vel(4) -> using vec4 for alignment padding
        // Structure: [x, y, z, life, vx, vy, vz, padding] -> 32 bytes per particle
        const particleData = new Float32Array(this.numParticles * 8);
        for (let i = 0; i < this.numParticles; i++) {
            // Initial position at center
            particleData[i * 8] = 0; // x
            particleData[i * 8 + 1] = 0; // y
            particleData[i * 8 + 2] = 0; // z
            particleData[i * 8 + 3] = Math.random(); // life (0-1) phase offset

            // Random direction velocity
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            const speed = 0.01 + Math.random() * 0.05;

            particleData[i * 8 + 4] = Math.sin(phi) * Math.cos(theta) * speed;
            particleData[i * 8 + 5] = Math.sin(phi) * Math.sin(theta) * speed;
            particleData[i * 8 + 6] = Math.cos(phi) * speed;
            particleData[i * 8 + 7] = 0; // padding
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
        this.particleBuffer.unmap();

        // Uniforms: time, mouseX, mouseY, padding
        this.uniformBuffer = this.device.createBuffer({
            size: 32, // 16 bytes min
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const computeShader = `
        struct Particle {
            pos : vec4f, // xyz, life
            vel : vec4f, // xyz, padding
        }

        @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

        struct Uniforms {
            time : f32,
            mouseX : f32,
            mouseY : f32,
            padding : f32,
        }
        @group(0) @binding(1) var<uniform> u : Uniforms;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
            let index = GlobalInvocationID.x;
            if (index >= arrayLength(&particles)) { return; }

            var p = particles[index];

            // Update position
            p.pos.x += p.vel.x;
            p.pos.y += p.vel.y;
            p.pos.z += p.vel.z;

            // Mouse interaction: gravity/repel
            let mousePos = vec3f(u.mouseX * 2.0, u.mouseY * 2.0, 0.0);
            let diff = mousePos - p.pos.xyz;
            let dist = length(diff);

            if (dist < 1.0) {
                let dir = normalize(diff);
                p.vel.x -= dir.x * 0.001;
                p.vel.y -= dir.y * 0.001;
                p.vel.z -= dir.z * 0.001;
            }

            // Life cycle
            // Use fractional part of time + initial phase to determine "reset"
            // Actually, let's just reset if they go too far
            if (length(p.pos.xyz) > 3.0) {
                p.pos.x = 0.0;
                p.pos.y = 0.0;
                p.pos.z = 0.0;

                // Random new velocity logic (pseudo random based on index and time)
                let t = u.time + f32(index);
                p.vel.x = sin(t) * 0.02;
                p.vel.y = cos(t * 1.3) * 0.02;
                p.vel.z = sin(t * 0.7) * 0.02;
            }

            particles[index] = p;
        }
        `;

        const renderShader = `
        struct Particle {
            pos : vec4f,
            vel : vec4f,
        }
        @group(0) @binding(0) var<storage, read> particles : array<Particle>;

        struct Uniforms {
            time : f32,
            mouseX : f32,
            mouseY : f32,
            padding : f32,
        }
        @group(0) @binding(1) var<uniform> u : Uniforms;

        struct VertexOutput {
            @builtin(position) position : vec4f,
            @location(0) color : vec4f,
        }

        @vertex
        fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
            let p = particles[vertexIndex];
            var out : VertexOutput;

            // Manual rotation matrix to match WebGL camera
            let rotSpeed = u.time * 0.2;
            let rotX = rotSpeed + u.mouseY;
            let rotY = rotSpeed + u.mouseX;

            let cx = cos(rotX); let sx = sin(rotX);
            let cy = cos(rotY); let sy = sin(rotY);

            var pos = p.pos.xyz;

            // Rotate Y
            let rx = pos.x * cy + pos.z * sy;
            let rz = -pos.x * sy + pos.z * cy;
            pos.x = rx; pos.z = rz;

            // Rotate X
            let ry = pos.y * cx - pos.z * sx;
            let rz2 = pos.y * sx + pos.z * cx;
            pos.y = ry; pos.z = rz2;

            // Project
            // Aspect ratio assumption roughly 1.0 or handled by canvas stretch
            // But let's apply simple perspective
            pos.z -= 4.0;

            out.position = vec4f(pos.x, pos.y, pos.z, -pos.z); // Perspective divide

            let dist = length(p.pos.xyz);
            out.color = vec4f(1.0, 0.5 + dist * 0.2, 0.1, 0.8);

            return out;
        }

        @fragment
        fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
            return color;
        }
        `;

        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: computeShader }),
                entryPoint: 'main'
            }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        this.bindGroup = this.device.createBindGroup({
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
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.canvasGL.width = w;
        this.canvasGL.height = h;
        if(this.gl) this.gl.viewport(0, 0, w, h);

        this.canvasGPU.width = w;
        this.canvasGPU.height = h;
    }

    render(t) {
        if (!this.isPlaying) return;
        this.time = t * 0.001;

        // GL Render
        if (this.gl) {
            const gl = this.gl;
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            gl.useProgram(this.program);
            gl.uniform1f(this.timeLoc, this.time);
            gl.uniform2f(this.resLoc, this.canvasGL.width, this.canvasGL.height);
            gl.uniform2f(this.mouseLoc, this.mouse.x, this.mouse.y);

            gl.bindVertexArray(this.vao);
            // Draw lines for wireframe
            gl.drawElements(gl.LINES, this.indexCount, gl.UNSIGNED_SHORT, 0);
        }

        // GPU Render
        if (this.device && this.contextGPU) {
            const uniforms = new Float32Array([this.time, this.mouse.x, this.mouse.y, 0]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

            const encoder = this.device.createCommandEncoder();

            const pass = encoder.beginComputePass();
            pass.setPipeline(this.computePipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            pass.end();

            const renderPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.contextGPU.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.renderBindGroup);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        requestAnimationFrame(this.render);
    }

    destroy() {
        this.isPlaying = false;
        window.removeEventListener('resize', this.resize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        // Clean up GL/GPU resources if needed
    }
}
