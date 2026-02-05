/**
 * Temporal Vortex Experiment
 * Combines WebGL2 (Wireframe Vortex Funnel) and WebGPU (Spiral Chroniton Particles).
 */

export class TemporalVortexExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.lastTime = Date.now();
        this.animationId = null;
        this.canvasSize = { width: 0, height: 0 };
        this.mouse = { x: 0, y: 0 };
        this.targetMouse = { x: 0, y: 0 };

        // Vortex State
        this.vortexSpeed = 1.0;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.numIndices = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 30000;

        this.handleResize = this.handleResize.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';
        this.container.style.cursor = 'crosshair';

        // 1. Initialize WebGL2 (The Vortex)
        this.initWebGL2();

        // 2. Initialize WebGPU (The Chronitons)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("TemporalVortex: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.resize();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);

        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);
        this.container.addEventListener('touchstart', this.handleMouseDown, {passive: false});
        window.addEventListener('touchend', this.handleMouseUp);

        this.animate();
    }

    handleResize() {
        this.resize();
    }

    handleMouseMove(e) {
        const x = (e.clientX / window.innerWidth) * 2 - 1;
        const y = -(e.clientY / window.innerHeight) * 2 + 1;
        this.targetMouse.x = x;
        this.targetMouse.y = y;
    }

    handleMouseDown(e) {
        if(e.type === 'touchstart') e.preventDefault();
        this.vortexSpeed = 5.0; // Hyper speed
    }

    handleMouseUp() {
        this.vortexSpeed = 1.0;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Wireframe Vortex)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Generate Vortex Funnel Geometry
        const vertices = [];
        const indices = [];

        const rings = 50;
        const segments = 32;
        const length = 50.0;
        const maxRadius = 15.0;
        const minRadius = 1.0;

        // Create rings along negative Z
        for (let i = 0; i <= rings; i++) {
            const t = i / rings; // 0 to 1
            const z = -t * length;
            // Funnel shape: curve radius
            // t=0 -> z=0 -> r=maxRadius
            // t=1 -> z=-length -> r=minRadius
            const r = maxRadius * (1.0 - t) + minRadius * t; // Linear taper for now, maybe pow for curve

            for (let j = 0; j < segments; j++) {
                const angle = (j / segments) * Math.PI * 2;
                const x = Math.cos(angle) * r;
                const y = Math.sin(angle) * r;
                vertices.push(x, y, z);
            }
        }

        // Generate lines
        // Ring connections
        for (let i = 0; i <= rings; i++) {
            for (let j = 0; j < segments; j++) {
                const current = i * segments + j;
                const next = i * segments + ((j + 1) % segments);
                indices.push(current, next);
            }
        }
        // Longitudinal connections
        for (let i = 0; i < rings; i++) {
            for (let j = 0; j < segments; j++) {
                const current = i * segments + j;
                const next = (i + 1) * segments + j;
                indices.push(current, next);
            }
        }

        this.numIndices = indices.length;

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);
        this.glVao = vao;

        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const posLoc = 0;
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        const vsSource = `#version 300 es
            in vec3 a_position;
            uniform float u_time;
            uniform vec2 u_mouse;
            uniform vec2 u_resolution;

            out float v_depth;

            void main() {
                vec3 p = a_position;

                // Rotate/Twist the vortex
                float angle = u_time * 0.5 + p.z * 0.1;
                float c = cos(angle);
                float s = sin(angle);
                float x = p.x * c - p.y * s;
                float y = p.x * s + p.y * c;
                p.x = x;
                p.y = y;

                // Mouse interaction: Bend the vortex
                // Bend more at the deep end (negative Z)
                float bend = p.z * 0.05;
                p.x += u_mouse.x * bend;
                p.y += u_mouse.y * bend;

                // Projection
                float fov = 1.0;
                float scale = 1.0 / tan(fov * 0.5);
                float aspect = u_resolution.x / u_resolution.y;

                float z = p.z;
                float px = p.x * scale / aspect;
                float py = p.y * scale;

                // Simple perspective projection
                // We shift Z so the camera is slightly outside the funnel start
                float camZ = 5.0;
                float zDist = camZ - z; // positive distance

                gl_Position = vec4(px, py, z * 0.01, zDist);
                v_depth = z;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in float v_depth;
            out vec4 outColor;

            void main() {
                // Fade into darkness deeper in the vortex
                float alpha = smoothstep(-50.0, -5.0, v_depth);

                // Color gradient
                vec3 purple = vec3(0.6, 0.0, 1.0);
                vec3 cyan = vec3(0.0, 1.0, 1.0);
                vec3 col = mix(purple, cyan, alpha);

                outColor = vec4(col, alpha * 0.5);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
    }

    createGLProgram(vs, fs) {
        const vShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vShader, vs);
        this.gl.compileShader(vShader);
        if (!this.gl.getShaderParameter(vShader, this.gl.COMPILE_STATUS)) return null;

        const fShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fShader, fs);
        this.gl.compileShader(fShader);
        if (!this.gl.getShaderParameter(fShader, this.gl.COMPILE_STATUS)) return null;

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vShader);
        this.gl.attachShader(prog, fShader);
        this.gl.linkProgram(prog);
        if (!this.gl.getProgramParameter(prog, this.gl.LINK_STATUS)) return null;
        return prog;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Spiral Particles)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2;
            pointer-events: none;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // Compute Shader
        const computeShader = `
            struct Particle {
                pos : vec4f, // x, y, z, angle
                vel : vec4f, // radius, speed, decay, size
            }

            struct Uniforms {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                speed : f32,
                pad1 : f32,
                pad2 : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            fn hash(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Initialization
                if (uniforms.time < 0.1 || p.vel.x <= 0.0) {
                    let seed = f32(index) * 0.001 + uniforms.time;
                    p.pos.z = -hash(seed) * 50.0; // Random depth
                    p.pos.w = hash(seed + 1.0) * 6.28; // Random angle

                    // Radius depends on depth to match funnel shape roughly
                    // maxRadius 15 at z=0, minRadius 1 at z=-50
                    let t = -p.pos.z / 50.0;
                    p.vel.x = 15.0 * (1.0 - t) + 1.0 * t; // Target radius

                    p.vel.y = 1.0 + hash(seed + 2.0); // Speed
                    p.vel.z = 0.99; // Decay?
                    p.vel.w = 0.5 + hash(seed + 3.0); // Size
                }

                // Spiral Motion
                // Move deeper
                p.pos.z -= p.vel.y * uniforms.dt * 10.0 * uniforms.speed;

                // Rotate
                p.pos.w += p.vel.y * uniforms.dt * 2.0;

                // Calculate X/Y based on current Z radius
                let t = clamp(-p.pos.z / 50.0, 0.0, 1.0);
                let currentRadius = 15.0 * (1.0 - t) + 1.0 * t;

                // Add some mouse influence (pull off center)
                let mx = uniforms.mouseX * 5.0;
                let my = uniforms.mouseY * 5.0;

                p.pos.x = cos(p.pos.w) * currentRadius + mx * t; // Influence affects deeper more?
                p.pos.y = sin(p.pos.w) * currentRadius + my * t;

                // Reset if too deep
                if (p.pos.z < -50.0) {
                    p.pos.z = 0.0;
                    p.vel.x = 15.0; // Reset radius
                }

                particles[index] = p;
            }
        `;

        // Render Shader
        const renderShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
                @location(1) alpha : f32,
            }

            struct Uniforms {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                aspect : f32,
                speed : f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(
                @builtin(vertex_index) vertexIndex : u32,
                @location(0) pos : vec4f, // x,y,z,angle
                @location(1) vel : vec4f  // rad,speed,decay,size
            ) -> VertexOutput {
                var output : VertexOutput;

                let p = pos.xyz;

                // Perspective
                let camZ = 5.0;
                let zDist = camZ - p.z;

                let fov = 1.0;
                let scale = 1.0 / tan(fov * 0.5);

                let px = p.x * scale / uniforms.aspect;
                let py = p.y * scale;

                output.position = vec4f(px, py, p.z * 0.01, zDist);

                // Size attenuation
                output.position.w = zDist;

                // Color based on speed/depth
                let depthAlpha = smoothstep(-60.0, 0.0, p.z);
                output.alpha = depthAlpha;

                // Hot colors
                output.color = vec4f(1.0, 0.5 + 0.5 * sin(uniforms.time + p.z * 0.1), 0.0, 1.0);

                // Point size hack for point-list? No, gl_PointSize is not in WGSL directly without specific config usually?
                // Actually WebGPU points have fixed size of 1.0 unless we use quads or verify implementation.
                // Assuming default 1px points for now, but usually we want billboards.
                // For simplicity in this template, we use point-list which renders 1px points.
                // To get larger points we usually need triangle-list billboards.
                // Let's stick to simple points or basic gl_PointSize equivalent if supported (it's not).
                // Wait, some implementations support it. But better to just accept small points or use instanced quads.
                // For this "Vortex" effect, small dense points look like dust.

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f, @location(1) alpha : f32) -> @location(0) vec4f {
                return vec4f(color.rgb, alpha);
            }
        `;

        // Setup buffers
        const particleData = new Float32Array(this.numParticles * 8);
        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 8 floats * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ],
        });

        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        const renderModule = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    }
                }]
            },
            primitive: { topology: 'point-list' },
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.style.cssText = `
            position: absolute; bottom: 10px; right: 10px;
            color: #ff5555; font-family: sans-serif; font-size: 12px;
            background: rgba(0,0,0,0.8); padding: 5px 10px; border-radius: 4px;
        `;
        msg.innerText = "WebGPU Not Available";
        this.container.appendChild(msg);
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width === 0 || height === 0) return;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

        const dw = Math.floor(width * dpr);
        const dh = Math.floor(height * dpr);

        if (this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            this.gl.viewport(0, 0, dw, dh);
        }

        if (this.gpuCanvas) {
            this.gpuCanvas.width = dw;
            this.gpuCanvas.height = dh;
        }
    }

    animate() {
        if (!this.isActive) return;

        const now = Date.now();
        const dt = Math.min((now - this.lastTime) * 0.001, 0.1);
        this.lastTime = now;
        const time = (now - this.startTime) * 0.001;

        // Smooth mouse
        this.mouse.x += (this.targetMouse.x - this.mouse.x) * 0.1;
        this.mouse.y += (this.targetMouse.y - this.mouse.y) * 0.1;

        // WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);
        }

        // WebGPU Render
        if (this.device && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const uniforms = new Float32Array([
                dt, time, this.mouse.x, this.mouse.y, aspect, this.vortexSpeed, 0, 0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, uniforms);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            const textureView = this.context.getCurrentTexture().createView();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);

        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);

        window.removeEventListener('mouseup', this.handleMouseUp);
        window.removeEventListener('touchend', this.handleMouseUp);

        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.TemporalVortexExperiment = TemporalVortexExperiment;
}
