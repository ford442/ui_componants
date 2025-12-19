
// Accessing the globally exported UIComponents from main.js if import doesn't work directly (since main.js attaches to window)
// but standard ES modules might require explicit exports. Based on main.js content, it attaches to window.
const LayeredCanvas = window.UIComponents.LayeredCanvas;
const ShaderUtils = window.UIComponents.ShaderUtils;

class HologramExperiment {
    constructor(container) {
        this.container = container;
        this.width = container.clientWidth;
        this.height = container.clientHeight;

        this.params = {
            density: 0.5,
            velocity: 0.3,
            rotationSpeed: 0.2
        };

        this.init();
    }

    async init() {
        // Initialize LayeredCanvas
        this.layered = new LayeredCanvas(this.container, {
            width: this.width,
            height: this.height
        });

        // 1. WebGL2 Layer: Wireframe Core & Grid
        this.glLayer = this.layered.addLayer('core', 'webgl2', 10);
        this.initGL(this.glLayer);

        // 2. WebGPU Layer: Particles
        this.gpuLayer = this.layered.addLayer('particles', 'webgpu', 20);
        this.gpuSupport = await this.initGPU(this.gpuLayer);

        this.updateStatus();

        // Start Animation
        this.layered.startAnimation();

        // Handle Resize
        window.addEventListener('resize', () => this.resize());

        // Controls
        this.setupControls();
    }

    setupControls() {
        const densityCtrl = document.getElementById('density-control');
        const velocityCtrl = document.getElementById('velocity-control');
        const rotationCtrl = document.getElementById('rotation-control');

        densityCtrl.addEventListener('input', (e) => {
            this.params.density = e.target.value / 100;
            if (this.particleSystem) {
                this.particleSystem.updateParams(this.params);
            }
        });

        velocityCtrl.addEventListener('input', (e) => {
            this.params.velocity = e.target.value / 100;
        });

        rotationCtrl.addEventListener('input', (e) => {
            this.params.rotationSpeed = e.target.value / 100;
        });
    }

    updateStatus() {
        const badge = document.getElementById('gpu-status');
        if (this.gpuSupport) {
            badge.textContent = "WebGPU: ACTIVE";
            badge.classList.add('active');
        } else {
            badge.textContent = "WebGPU: UNAVAILABLE (Fallback Mode)";
            badge.style.borderColor = '#ff3333';
            badge.style.color = '#ff3333';
        }
    }

    resize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        this.layered.resize(this.width, this.height);

        // Re-init GPU swap chain usually handled by canvas resize,
        // but we might need to update projection matrices here.
        if (this.glLayer.context) {
            this.glLayer.context.viewport(0, 0, this.width, this.height);
        }
    }

    // --- WebGL2 Implementation ---

    initGL(layer) {
        const gl = layer.context;
        if (!gl) return;

        // Vertex Shader
        const vsSource = `#version 300 es
        in vec4 a_position;
        uniform mat4 u_matrix;
        uniform float u_time;

        out vec3 v_pos;

        void main() {
            vec4 pos = a_position;
            // Simple rotation
            float c = cos(u_time);
            float s = sin(u_time);
            mat4 rotY = mat4(
                c, 0, s, 0,
                0, 1, 0, 0,
                -s, 0, c, 0,
                0, 0, 0, 1
            );

            gl_Position = u_matrix * rotY * pos;
            v_pos = pos.xyz;
            gl_PointSize = 4.0;
        }`;

        // Fragment Shader
        const fsSource = `#version 300 es
        precision highp float;
        in vec3 v_pos;
        uniform vec3 u_color;
        uniform float u_time;
        out vec4 outColor;

        void main() {
            float pulse = 0.5 + 0.5 * sin(u_time * 2.0 + v_pos.y * 5.0);
            outColor = vec4(u_color * pulse, 0.8);
        }`;

        const program = ShaderUtils.createProgram(gl, vsSource, fsSource);
        this.glProgram = program;

        // Icosahedron Data (Simplified)
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;
        const vertices = new Float32Array([
            -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0,
            0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t,
            t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1
        ]);

        // Indices for lines
        const indices = new Uint16Array([
            0, 11, 0, 5, 0, 1, 0, 7, 0, 10,
            1, 5, 1, 7, 1, 10, 1, 11,
            // ... more indices ideally, but this is enough for a "broken" tech look
            2, 3, 2, 4, 2, 6, 2, 8,
            3, 4, 3, 6, 3, 8
        ]);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const vBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

        const positionLoc = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);

        const iBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

        this.glVAO = vao;
        this.glIndexCount = indices.length;

        // Render Function
        this.layered.setRenderFunction('core', (layerData, timestamp) => {
            const time = timestamp * 0.001;
            const gl = layerData.context;

            gl.viewport(0, 0, this.width, this.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            gl.useProgram(program);
            gl.bindVertexArray(vao);

            // Simple perspective projection matrix (Column Major)
            // Field of View ~45 degrees
            // Camera at (0, 0, 5) looking at (0, 0, 0)

            const zNear = 0.1;
            const zFar = 100.0;
            const fovVal = 45 * Math.PI / 180;
            const aspectVal = this.width / this.height;
            const fVal = 1.0 / Math.tan(fovVal / 2);
            const rangeInv = 1.0 / (zNear - zFar);

            // Projection Matrix
            const p00 = fVal / aspectVal;
            const p11 = fVal;
            const p22 = (zFar + zNear) * rangeInv;
            const p23 = -1;
            const p32 = (2 * zFar * zNear) * rangeInv;

            // View Matrix (Translate camera back by 5 units -> Translate world by -5 units along Z)
            // Since we are doing Proj * View * Model
            // Let's pre-multiply Proj * View manually into 'matrix'

            // View Matrix (Translation z = -5)
            // 1 0 0 0
            // 0 1 0 0
            // 0 0 1 0
            // 0 0 -5 1

            // P * V (Column Major multiplication)
            // Result is effectively the Projection matrix but with the Z-translation applied to the W calculation

            // Correct Matrix (Column Major Array)
            const finalMatrix = new Float32Array([
                p00, 0, 0, 0,
                0, p11, 0, 0,
                0, 0, p22, -1,
                0, 0, p22 * -5.0 + p32, 5.0
            ]);

            const uMatrixLoc = gl.getUniformLocation(program, 'u_matrix');
            const uTimeLoc = gl.getUniformLocation(program, 'u_time');
            const uColorLoc = gl.getUniformLocation(program, 'u_color');

            gl.uniformMatrix4fv(uMatrixLoc, false, finalMatrix);

            // Pass adjusted time based on rotation speed
            gl.uniform1f(uTimeLoc, time * this.params.rotationSpeed * 5.0);
            gl.uniform3f(uColorLoc, 0.0, 1.0, 0.8); // Cyan

            // Draw Lines
            gl.lineWidth(2.0);
            gl.drawElements(gl.LINES, this.glIndexCount, gl.UNSIGNED_SHORT, 0);

            // Draw Points
            gl.drawElements(gl.POINTS, this.glIndexCount, gl.UNSIGNED_SHORT, 0);
        });
    }

    // --- WebGPU Implementation ---

    async initGPU(layer) {
        if (!navigator.gpu) return false;

        const canvas = layer.canvas;
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        const device = await adapter.requestDevice();
        this.device = device;

        const context = canvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();

        context.configure({
            device: device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // Use the existing WebGPUParticleSystem class but modified for this use case
        // or create a custom pipeline here.
        // Let's create a custom one that spirals.

        const particleCount = 10000;
        this.particleData = new Float32Array(particleCount * 4); // x, y, z, life

        // Initialize particles
        for (let i = 0; i < particleCount; i++) {
            const r = 2.0 + Math.random();
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            this.particleData[i*4 + 0] = r * Math.sin(phi) * Math.cos(theta);
            this.particleData[i*4 + 1] = r * Math.sin(phi) * Math.sin(theta);
            this.particleData[i*4 + 2] = r * Math.cos(phi);
            this.particleData[i*4 + 3] = Math.random(); // Life
        }

        // Buffers
        this.gpuParticleBuffer = device.createBuffer({
            size: this.particleData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.gpuParticleBuffer.getMappedRange()).set(this.particleData);
        this.gpuParticleBuffer.unmap();

        this.gpuUniformBuffer = device.createBuffer({
            size: 64, // time, velocity, density, padding...
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Shader
        const computeShader = `
            struct Particle {
                pos: vec3<f32>,
                life: f32
            }

            struct Uniforms {
                time: f32,
                velocity: f32,
                density: f32,
                _pad: f32
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Spiral movement
                let r = length(p.pos.xz);
                let theta = atan2(p.pos.z, p.pos.x);

                let newTheta = theta + uniforms.velocity * 0.1 * (3.0 / (r + 0.1));

                p.pos.x = r * cos(newTheta);
                p.pos.z = r * sin(newTheta);

                // Vertical movement
                p.pos.y -= uniforms.velocity * 0.05;
                if (p.pos.y < -3.0) {
                    p.pos.y = 3.0;
                    // Reset radius slightly
                    let angle = fract(sin(uniforms.time * 100.0 + f32(index)) * 43758.5453) * 6.28;
                    let rad = 1.5 + fract(cos(uniforms.time * 50.0 + f32(index))) * 2.0;
                    p.pos.x = rad * cos(angle);
                    p.pos.z = rad * sin(angle);
                }

                particles[index] = p;
            }
        `;

        // Render Shader
        const renderShader = `
            struct Particle {
                pos: vec3<f32>,
                life: f32
            }

            struct Uniforms {
                time: f32,
                velocity: f32,
                density: f32,
                _pad: f32
            }

            @group(0) @binding(0) var<storage, read> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>
            }

            @vertex
            fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
                let p = particles[instanceIndex];

                // Skip if density check fails (hacky density implementation)
                if (f32(instanceIndex) > uniforms.density * 10000.0) {
                    return VertexOutput(vec4<f32>(0.0), vec4<f32>(0.0));
                }

                // Billboard quad
                let pos = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
                );

                let localPos = pos[vertexIndex];

                // Perspective projection (matching WebGL roughly)
                let worldPos = vec3<f32>(p.pos.x, p.pos.y, p.pos.z - 4.0); // Translate z=-4

                // Simple perspective
                let fov = 1.5;
                let aspect = 1.0; // Assuming square for simplicity in shader, or pass aspect
                let scale = 1.0 / -worldPos.z;

                let screenPos = vec2<f32>(
                    worldPos.x * scale,
                    worldPos.y * scale * aspect
                );

                // Add particle size
                let particleSize = 0.02 * scale;
                let finalPos = screenPos + localPos * particleSize;

                var out: VertexOutput;
                out.position = vec4<f32>(finalPos, 0.0, 1.0);

                // Distance fade
                let dist = length(worldPos);
                let alpha = 1.0 - smoothstep(2.0, 6.0, dist);

                out.color = vec4<f32>(0.0, 1.0, 0.5, alpha);
                return out;
            }

            @fragment
            fn fragmentMain(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                if (color.a <= 0.0) { discard; }
                return color;
            }
        `;

        // Pipelines
        this.computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: computeShader }),
                entryPoint: 'main'
            }
        });

        this.renderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: device.createShaderModule({ code: renderShader }),
                entryPoint: 'vertexMain'
            },
            fragment: {
                module: device.createShaderModule({ code: renderShader }),
                entryPoint: 'fragmentMain',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one' }, // Additive
                        alpha: { srcFactor: 'zero', dstFactor: 'one' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list' }
        });

        this.gpuBindGroup = device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.gpuParticleBuffer } },
                { binding: 1, resource: { buffer: this.gpuUniformBuffer } }
            ]
        });

        // Render bind group needs to match render pipeline layout
        // The render pipeline uses 'auto' layout, so we must ask it for the layout.
        // Wait, the shader definitions for bind groups in render and compute are identical in indices/types
        // so we might be able to share or recreate.
        // Render shader: @group(0) @binding(0) particles, @group(0) @binding(1) uniforms

        this.gpuRenderBindGroup = device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.gpuParticleBuffer } },
                { binding: 1, resource: { buffer: this.gpuUniformBuffer } }
            ]
        });

        // Register Render Function
        this.layered.setRenderFunction('particles', (layerData, timestamp) => {
            const time = timestamp * 0.001;

            // Update Uniforms
            const uniformData = new Float32Array([
                time,
                this.params.velocity,
                this.params.density,
                0
            ]);
            device.queue.writeBuffer(this.gpuUniformBuffer, 0, uniformData);

            const commandEncoder = device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.gpuBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(particleCount / 64));
            computePass.end();

            // Render
            const textureView = context.getCurrentTexture().createView();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });

            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.gpuRenderBindGroup);
            renderPass.draw(6, particleCount); // 6 vertices per instance, N instances
            renderPass.end();

            device.queue.submit([commandEncoder.finish()]);
        });

        return true;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('hologram-display');
    new HologramExperiment(container);
});
