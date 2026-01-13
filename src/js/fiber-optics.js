
export class FiberOpticExperiment {
    constructor(container) {
        this.container = container;
        this.width = container.clientWidth || 300;
        this.height = container.clientHeight || 200;

        // Ensure container relative positioning
        if (window.getComputedStyle(this.container).position === 'static') {
            this.container.style.position = 'relative';
        }

        // Try to find existing canvases (standalone page) or create them (dashboard)
        this.glCanvas = this.container.querySelector('#webgl-canvas') || this.createCanvas(1);
        this.gpuCanvas = this.container.querySelector('#webgpu-canvas') || this.createCanvas(2);

        this.gl = this.glCanvas.getContext('webgl2', { alpha: true, antialias: true });

        this.adapter = null;
        this.device = null;
        this.gpuContext = null;

        this.animationId = null;
        this.time = 0;
        this.isPlaying = true;

        // Camera
        this.camera = {
            eye: [0, 0, 15],
            target: [0, 0, 0],
            up: [0, 1, 0],
            fov: 60 * Math.PI / 180,
            aspect: this.width / this.height,
            near: 0.1,
            far: 100.0,
            viewMatrix: new Float32Array(16),
            projectionMatrix: new Float32Array(16)
        };

        // Fiber Data
        this.numFibers = 200;
        this.segmentsPerFiber = 100;
        this.fiberRadius = 4.0;
        this.fiberLength = 30.0;
        this.twistFactor = 2.0;

        this.init().catch(console.error);
    }

    createCanvas(zIndex) {
        const canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.zIndex = zIndex;
        // WebGPU canvas (top layer) usually needs blend mode to see through
        if (zIndex === 2) {
            canvas.style.mixBlendMode = 'screen';
        }
        this.container.appendChild(canvas);
        return canvas;
    }

    async init() {
        this.resize();

        // 1. Initialize WebGPU
        if (navigator.gpu) {
            this.adapter = await navigator.gpu.requestAdapter();
            if (this.adapter) {
                this.device = await this.adapter.requestDevice();
                this.gpuContext = this.gpuCanvas.getContext('webgpu');
                this.gpuContext.configure({
                    device: this.device,
                    format: navigator.gpu.getPreferredCanvasFormat(),
                    alphaMode: 'premultiplied'
                });
                await this.initWebGPU();
            }
        }

        // 2. Initialize WebGL
        if (this.gl) {
            this.initWebGL();
        }

        // 3. Start Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    resize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        this.glCanvas.width = this.width;
        this.glCanvas.height = this.height;
        this.gpuCanvas.width = this.width;
        this.gpuCanvas.height = this.height;

        this.camera.aspect = this.width / this.height;
        this.updateProjectionMatrix();

        if (this.gl) this.gl.viewport(0, 0, this.width, this.height);
    }

    updateProjectionMatrix() {
        const f = 1.0 / Math.tan(this.camera.fov / 2);
        const rangeInv = 1 / (this.camera.near - this.camera.far);

        // WebGL projection (standard)
        this.camera.projectionMatrix = new Float32Array([
            f / this.camera.aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (this.camera.near + this.camera.far) * rangeInv, -1,
            0, 0, this.camera.near * this.camera.far * rangeInv * 2, 0
        ]);
    }

    updateViewMatrix() {
        // Simple orbiting camera
        const t = this.time * 0.2;
        const radius = 25;
        this.camera.eye[0] = Math.sin(t) * radius;
        this.camera.eye[2] = Math.cos(t) * radius;
        this.camera.eye[1] = Math.sin(t * 0.5) * 5;

        const zAxis = this.normalize(this.subtract(this.camera.eye, this.camera.target));
        const xAxis = this.normalize(this.cross(this.camera.up, zAxis));
        const yAxis = this.cross(zAxis, xAxis);

        this.camera.viewMatrix.set([
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -this.dot(xAxis, this.camera.eye), -this.dot(yAxis, this.camera.eye), -this.dot(zAxis, this.camera.eye), 1
        ]);
    }

    // --- WebGL2 Implementation (Static Fibers) ---

    initWebGL() {
        const gl = this.gl;

        // Generate Fiber Geometry (Lines)
        // Bundle of twisted fibers
        const vertices = [];
        const colors = []; // Use distinct colors per fiber

        for (let i = 0; i < this.numFibers; i++) {
            // Fiber parameters
            const angleOffset = (i / this.numFibers) * Math.PI * 2;
            const r = (Math.random() * 0.8 + 0.2) * this.fiberRadius;

            // Randomize twist slightly per fiber
            const localTwist = this.twistFactor + (Math.random() - 0.5) * 0.5;

            // Color: Cyberpunk palette (Cyan, Magenta, Blue)
            const colorType = Math.random();
            const rCol = colorType < 0.33 ? 0.0 : (colorType < 0.66 ? 1.0 : 0.0);
            const gCol = colorType < 0.33 ? 1.0 : (colorType < 0.66 ? 0.0 : 0.5);
            const bCol = 1.0;
            const alpha = 0.15; // Semi-transparent

            for (let j = 0; j <= this.segmentsPerFiber; j++) {
                const t = (j / this.segmentsPerFiber) * 2 - 1; // -1 to 1
                const z = t * this.fiberLength * 0.5;

                const theta = z * localTwist + angleOffset;
                const x = Math.cos(theta) * r;
                const y = Math.sin(theta) * r;

                vertices.push(x, y, z);
                colors.push(rCol, gCol, bCol, alpha);
            }
        }

        // Create Buffers
        this.fiberVAO = gl.createVertexArray();
        gl.bindVertexArray(this.fiberVAO);

        const vBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        const cBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);

        // Shaders
        const vsSource = `#version 300 es
            layout(location = 0) in vec3 a_position;
            layout(location = 1) in vec4 a_color;

            uniform mat4 u_view;
            uniform mat4 u_projection;

            out vec4 v_color;

            void main() {
                gl_Position = u_projection * u_view * vec4(a_position, 1.0);
                v_color = a_color;
            }
        `;

        const fsSource = `#version 300 es
            precision mediump float;
            in vec4 v_color;
            out vec4 outColor;

            void main() {
                outColor = v_color;
            }
        `;

        this.glProgram = this.createGLProgram(gl, vsSource, fsSource);
        this.uViewLoc = gl.getUniformLocation(this.glProgram, 'u_view');
        this.uProjLoc = gl.getUniformLocation(this.glProgram, 'u_projection');

        // GL State
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // Additive blending for glow look
        gl.disable(gl.DEPTH_TEST); // No depth test for transparency overlap look
    }

    createGLProgram(gl, vs, fs) {
        const createShader = (type, source) => {
            const s = gl.createShader(type);
            gl.shaderSource(s, source);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
                console.error(gl.getShaderInfoLog(s));
                return null;
            }
            return s;
        };
        const p = gl.createProgram();
        gl.attachShader(p, createShader(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, createShader(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p);
        return p;
    }

    // --- WebGPU Implementation (Dynamic Pulses) ---

    async initWebGPU() {
        const device = this.device;

        // Particle System Params
        this.numParticles = 10000;

        // Particle Buffer (Struct: { pos: vec3, life: float, fiberId: float, speed: float, pad: vec2 })
        // We just need to track t (position along fiber) and which fiber it belongs to.
        // Let's compute actual 3D pos in Vertex Shader to save bandwidth, but wait,
        // we can't easily share the "fiber generation logic" (randomness) between JS and WGSL.
        // Solution: Pre-calculate fiber params (radius, angleOffset, twist) and store in a storage buffer.

        // Fiber Params Buffer
        const fiberData = new Float32Array(this.numFibers * 4); // r, angleOffset, twist, pad
        for (let i = 0; i < this.numFibers; i++) {
            const angleOffset = (i / this.numFibers) * Math.PI * 2;
            const r = (Math.random() * 0.8 + 0.2) * this.fiberRadius;
            const twist = this.twistFactor + (Math.random() - 0.5) * 0.5;

            fiberData[i*4 + 0] = r;
            fiberData[i*4 + 1] = angleOffset;
            fiberData[i*4 + 2] = twist;
            fiberData[i*4 + 3] = 0; // pad
        }

        this.fiberBuffer = device.createBuffer({
            size: fiberData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.fiberBuffer.getMappedRange()).set(fiberData);
        this.fiberBuffer.unmap();

        // Particle State Buffer (t, fiberIndex, speed, isActive)
        const particleData = new Float32Array(this.numParticles * 4);
        for (let i = 0; i < this.numParticles; i++) {
            particleData[i*4 + 0] = Math.random() * 2 - 1; // t: -1 to 1
            particleData[i*4 + 1] = Math.floor(Math.random() * this.numFibers); // fiberIndex
            particleData[i*4 + 2] = 0.2 + Math.random() * 0.5; // speed
            particleData[i*4 + 3] = 1.0; // isActive
        }

        this.particleBuffer = device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
        this.particleBuffer.unmap();

        // Uniform Buffer (Time, MVP)
        this.uniformBufferSize = 16 * 4 + 16 * 4 + 4 * 4; // View(16), Proj(16), Params(4: dt, time, fiberLength, pad)
        this.uniformBuffer = device.createBuffer({
            size: this.uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Shader
        const computeSource = `
            struct Particle {
                t: f32,
                fiberIndex: f32,
                speed: f32,
                isActive: f32,
            }

            struct Uniforms {
                view: mat4x4<f32>,
                projection: mat4x4<f32>,
                dt: f32,
                time: f32,
                fiberLength: f32,
                pad: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) {
                    return;
                }

                var p = particles[index];

                // Move particle
                p.t += p.speed * uniforms.dt * 0.5;

                // Wrap around
                if (p.t > 1.0) {
                    p.t = -1.0;
                }

                particles[index] = p;
            }
        `;

        this.computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: computeSource }), entryPoint: 'main' }
        });

        this.computeBindGroup = device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // Render Shader
        const renderSource = `
            struct Uniforms {
                view: mat4x4<f32>,
                projection: mat4x4<f32>,
                dt: f32,
                time: f32,
                fiberLength: f32,
                pad: f32,
            }

            struct FiberParams {
                r: f32,
                angleOffset: f32,
                twist: f32,
                pad: f32,
            }

            struct Particle {
                t: f32,
                fiberIndex: f32,
                speed: f32,
                isActive: f32,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> fibers: array<FiberParams>;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>,
                @location(1) uv: vec2<f32>,
            }

            @vertex
            fn vs_main(
                @location(0) p_t: f32,
                @location(1) p_fiberIndex: f32,
                @location(2) p_speed: f32,
                @location(3) p_isActive: f32,
                @builtin(vertex_index) v_index: u32
            ) -> VertexOutput {
                // Billboarding quad (6 vertices)
                var pos = vec2<f32>(0.0, 0.0);
                if (v_index == 0u) { pos = vec2<f32>(-1.0, -1.0); }
                else if (v_index == 1u) { pos = vec2<f32>(1.0, -1.0); }
                else if (v_index == 2u) { pos = vec2<f32>(-1.0, 1.0); }
                else if (v_index == 3u) { pos = vec2<f32>(-1.0, 1.0); }
                else if (v_index == 4u) { pos = vec2<f32>(1.0, -1.0); }
                else if (v_index == 5u) { pos = vec2<f32>(1.0, 1.0); }

                // Calculate 3D position
                let fiber = fibers[u32(p_fiberIndex)];

                let z = p_t * uniforms.fiberLength * 0.5;
                let theta = z * fiber.twist + fiber.angleOffset;
                let worldPos = vec3<f32>(
                    cos(theta) * fiber.r,
                    sin(theta) * fiber.r,
                    z
                );

                // View Space Billboarding
                let viewPos = uniforms.view * vec4<f32>(worldPos, 1.0);
                let particleSize = 0.15;
                let finalPos = viewPos + vec4<f32>(pos * particleSize, 0.0, 0.0);

                var out: VertexOutput;
                out.position = uniforms.projection * finalPos;
                out.uv = pos;

                // Color based on speed and t
                let intensity = smoothstep(-1.0, -0.8, p_t) * smoothstep(1.0, 0.8, p_t);
                out.color = vec4<f32>(0.5, 0.8, 1.0, intensity);

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4<f32>, @location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let dist = length(uv);
                if (dist > 1.0) { discard; }
                let alpha = (1.0 - dist) * color.a;
                return vec4<f32>(color.rgb, alpha);
            }
        `;

        this.renderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: device.createShaderModule({ code: renderSource }),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 16,
                    stepMode: 'instance',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32' }, // t
                        { shaderLocation: 1, offset: 4, format: 'float32' }, // fiberIndex
                        { shaderLocation: 2, offset: 8, format: 'float32' }, // speed
                        { shaderLocation: 3, offset: 12, format: 'float32' }  // active
                    ]
                }]
            },
            fragment: {
                module: device.createShaderModule({ code: renderSource }),
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list' }
        });

        this.renderBindGroup = device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.fiberBuffer } }
            ]
        });
    }

    animate(timestamp) {
        if (!this.isPlaying) return;

        const dt = (timestamp - this.time) * 0.001;
        this.time = timestamp;

        this.updateViewMatrix();

        if (this.gl) {
            this.renderWebGL();
        }

        if (this.device) {
            this.renderWebGPU(dt);
        }

        this.animationId = requestAnimationFrame(this.animate);
    }

    destroy() {
        this.isPlaying = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);

        // Clean up resources if needed
        if (this.fiberBuffer) this.fiberBuffer.destroy();
        if (this.particleBuffer) this.particleBuffer.destroy();
        if (this.uniformBuffer) this.uniformBuffer.destroy();
        // this.device.destroy(); // Optional, depending on lifecycle
    }

    renderWebGL() {
        const gl = this.gl;
        gl.clearColor(0, 0, 0, 0); // Transparent!
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.glProgram);
        gl.uniformMatrix4fv(this.uViewLoc, false, this.camera.viewMatrix);
        gl.uniformMatrix4fv(this.uProjLoc, false, this.camera.projectionMatrix);

        gl.bindVertexArray(this.fiberVAO);

        // Draw each fiber as a line strip
        // Each fiber has segmentsPerFiber + 1 vertices
        const pointsPerFiber = this.segmentsPerFiber + 1;
        for (let i = 0; i < this.numFibers; i++) {
            gl.drawArrays(gl.LINE_STRIP, i * pointsPerFiber, pointsPerFiber);
        }
    }

    renderWebGPU(dt) {
        const device = this.device;
        const commandEncoder = device.createCommandEncoder();

        // Update Uniforms
        const uniforms = new Float32Array([...this.camera.viewMatrix, ...this.camera.projectionMatrix, dt, this.time * 0.001, this.fiberLength, 0]);
        device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

        // Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
        computePass.end();

        // Render Pass
        const textureView = this.gpuContext.getCurrentTexture().createView();
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
        renderPass.setVertexBuffer(0, this.particleBuffer);
        // Draw 6 vertices (quad) per instance (particle)
        renderPass.draw(6, this.numParticles);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    // Math Helpers
    normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (len > 0.00001) return [v[0]/len, v[1]/len, v[2]/len];
        return [0,0,0];
    }
    subtract(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    cross(a, b) { return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]; }
    dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
}
