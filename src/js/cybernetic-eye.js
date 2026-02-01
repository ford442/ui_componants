/**
 * Cybernetic Eye Experiment
 * Hybrid visualization combining:
 * - WebGL2: Renders a mechanical "Iris" that dilates and contracts.
 * - WebGPU: Simulates a "Scanner" particle system emitted from the pupil.
 */

export class CyberneticEyeExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.dilation = 0.5; // 0.0 (closed) to 1.0 (open)
        this.targetDilation = 0.5;
        this.isScanning = false;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndexCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 20000;

        // Handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000000';

        // 1. WebGL2 Layer (The Eye)
        this.initWebGL2();

        // 2. WebGPU Layer (The Scanner)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("CyberneticEye: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.resize();

        // Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp); // Window to catch release outside

        this.animate();
    }

    // ========================================================================
    // Events
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (this.glCanvas) {
            this.glCanvas.width = width * dpr;
            this.glCanvas.height = height * dpr;
            this.gl.viewport(0, 0, width * dpr, height * dpr);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = width * dpr;
            this.gpuCanvas.height = height * dpr;
        }
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        // Normalize mouse -1 to 1
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);

        // Dilation based on distance from center
        const dist = Math.sqrt(this.mouse.x * this.mouse.x + this.mouse.y * this.mouse.y);
        // Closer to center = more open (dilated)
        // Further away = more closed (contracted)
        this.targetDilation = Math.max(0.2, 1.0 - dist * 0.8);
    }

    onMouseDown() {
        this.isScanning = true;
        // On click, maybe dilate momentarily
        this.targetDilation = 1.0;
    }

    onMouseUp() {
        this.isScanning = false;
    }

    // ========================================================================
    // WebGL2 (The Mechanical Iris)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Geometry: A quad for raymarching the eye or simple geometry?
        // Let's use a quad and raymarch the iris for better detail.
        const positions = new Float32Array([
            -1, -1, 1, -1, -1, 1,
            -1, 1, 1, -1, 1, 1
        ]);

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);

        const buf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const posLoc = 0; // standard location
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.glVao = vao;
        this.glIndexCount = 6;

        // Shaders
        const vs = `#version 300 es
            layout(location=0) in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform float u_dilation;
            uniform vec2 u_mouse;
            out vec4 outColor;

            // Noise function
            float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
            float noise(vec2 p) {
                vec2 i = floor(p); vec2 f = fract(p);
                f = f*f*(3.0-2.0*f);
                return mix(mix(hash(i + vec2(0.0,0.0)), hash(i + vec2(1.0,0.0)), f.x),
                           mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), f.x), f.y);
            }
            float fbm(vec2 p) {
                float v = 0.0; float a = 0.5;
                for(int i=0; i<5; i++) { v += a*noise(p); p*=2.0; a*=0.5; }
                return v;
            }

            void main() {
                // Aspect correction
                vec2 uv = v_uv;
                uv.x *= u_resolution.x / u_resolution.y;

                // Eye Look At Mouse
                // Calculate eye rotation offset based on mouse
                vec2 lookOffset = u_mouse * 0.3; // Limit movement
                vec2 p = uv - lookOffset;

                float r = length(p);
                float a = atan(p.y, p.x);

                // Iris Radius (outer)
                float irisR = 0.8;
                // Pupil Radius (inner) - controlled by dilation
                float pupilR = mix(0.1, 0.5, u_dilation);

                vec3 col = vec3(0.0);

                if (r < irisR) {
                    // Inside Iris
                    if (r < pupilR) {
                        // Pupil (Black)
                        col = vec3(0.0);
                    } else {
                        // Iris Texture
                        // Radial coordinates for texture
                        float coord = (r - pupilR) / (irisR - pupilR); // 0 to 1 across iris width

                        // Fibers
                        float fibers = fbm(vec2(r * 10.0, a * 10.0 + u_time * 0.1));
                        fibers += fbm(vec2(r * 20.0, a * 20.0));

                        // Color Gradient
                        vec3 innerCol = vec3(0.2, 0.8, 1.0); // Cyan
                        vec3 outerCol = vec3(0.0, 0.1, 0.3); // Dark Blue

                        col = mix(innerCol, outerCol, coord);
                        col *= 0.5 + 0.5 * fibers;

                        // Highlights
                        col += vec3(0.1) * smoothstep(0.4, 0.6, fibers);
                    }

                    // Sclera (White part) edge softness
                    // Not rendering sclera here, assuming black background is "void" socket
                } else {
                    // Sclera / Metal Socket
                    float socket = smoothstep(irisR, irisR + 0.05, r);
                    col = vec3(0.1) * socket;

                    // Add some metallic ring details
                    if (r > irisR && r < irisR + 0.2) {
                        float ring = abs(sin(r * 50.0));
                        col += vec3(0.3, 0.3, 0.4) * ring * socket;
                    }
                }

                // Reflection (fake)
                vec2 refPos = p - vec2(0.2, 0.2);
                float reflection = 1.0 - smoothstep(0.05, 0.1, length(refPos));
                if (r < irisR) {
                    col += vec3(1.0) * reflection * 0.8;
                }

                outColor = vec4(col, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
            console.error('GL VS Error:', this.gl.getShaderInfoLog(vs));
            return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
            console.error('GL FS Error:', this.gl.getShaderInfoLog(fs));
            return null;
        }

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vs);
        this.gl.attachShader(prog, fs);
        this.gl.linkProgram(prog);
        return prog;
    }

    // ========================================================================
    // WebGPU (Scanner Particles)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // Compute Shader
        const computeShader = `
            struct Particle {
                pos : vec4f, // xyz, w=life
                vel : vec4f, // xyz, w=padding
            }
            struct Params {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                isScanning : f32,
                dilation : f32,
                pad1: f32,
                pad2: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : Params;

            fn rand(co: vec2f) -> f32 { return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453); }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let i = GlobalInvocationID.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Update
                if (p.pos.w > 0.0) {
                    p.pos.x += p.vel.x * params.dt;
                    p.pos.y += p.vel.y * params.dt;
                    p.pos.z += p.vel.z * params.dt;
                    p.pos.w -= params.dt * 0.5; // Decay

                    // Steer towards mouse
                    let target = vec3f(params.mouseX, params.mouseY, 0.0);
                    let dir = normalize(target - p.pos.xyz);
                    let dist = distance(target, p.pos.xyz);

                    // Attraction force
                    let strength = 2.0;
                    p.vel.x += dir.x * strength * params.dt;
                    p.vel.y += dir.y * strength * params.dt;

                    // Drag
                    p.vel.x *= 0.98;
                    p.vel.y *= 0.98;
                }

                // Respawn
                // If dead OR randomly if scanning is active
                let shouldRespawn = p.pos.w <= 0.0 || (params.isScanning > 0.5 && rand(vec2f(params.time, f32(i))) < 0.02);

                if (shouldRespawn) {
                    // Spawn at pupil edge
                    // Pupil radius calculation (must match WebGL roughly)
                    // Eye look offset
                    let lookOffset = vec2f(params.mouseX, params.mouseY) * 0.3;

                    let pupilR = mix(0.1, 0.5, params.dilation);
                    let angle = rand(vec2f(f32(i), params.time)) * 6.28;

                    // Spawn on pupil rim
                    p.pos.x = lookOffset.x + cos(angle) * pupilR;
                    p.pos.y = lookOffset.y + sin(angle) * pupilR;
                    p.pos.z = 0.0;
                    p.pos.w = 1.0; // Life

                    // Initial velocity outward
                    let speed = 0.5 + rand(vec2f(params.time, f32(i))) * 0.5;
                    p.vel.x = cos(angle) * speed;
                    p.vel.y = sin(angle) * speed;
                    p.vel.z = 0.0;
                }

                particles[i] = p;
            }
        `;

        // Render Shader
        const renderShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }
            struct Params {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                isScanning : f32,
                dilation : f32,
                aspect : f32, // Passed in last pad slot of compute params? No, we need separate or careful packing
                // Wait, I reused the struct definition above but Params in render shader needs to match binding or I create a new buffer?
                // I'll assume I write aspect to a separate uniform buffer or pack it.
                // Let's pack aspect into pad2 of the simulation buffer.
            }
            // Actually, let's redefine struct for Render to match the buffer layout
             struct Uniforms {
                dt : f32,
                time : f32,
                mouseX : f32,
                mouseY : f32,
                isScanning : f32,
                dilation : f32,
                pad1: f32,
                aspect: f32, // Using the last float for aspect
            }

            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(@location(0) particlePos : vec4f, @location(1) particleVel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Aspect correction
                // x needs to be divided by aspect to be square in view space if we treat y as 1.0
                // View space is -1 to 1.
                // particlePos is in "screen normalized" coordinates roughly (-1 to 1) but generated based on aspect?
                // Actually in compute I used raw coordinates.
                // Let's apply aspect correction here.

                output.position = vec4f(particlePos.x / uniforms.aspect, particlePos.y, 0.0, 1.0);
                output.position.w = 1.0;

                // Color
                let life = particlePos.w;
                // Green scanner color
                var col = vec3f(0.0, 1.0, 0.5);

                // If scanning (active), maybe Red?
                if (uniforms.isScanning > 0.5) {
                    col = vec3f(1.0, 0.2, 0.2);
                }

                output.color = vec4f(col, life);

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffer Init
        const pData = new Float32Array(this.numParticles * 8);
        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // 8 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
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
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.textContent = "WebGPU Not Supported";
        msg.style.cssText = "position:absolute;bottom:10px;right:10px;color:red;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Animation
    // ========================================================================

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const dt = 0.016;

        // Smooth Dilation
        this.dilation += (this.targetDilation - this.dilation) * 0.1;

        // WebGL2 Render
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Uniforms
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_dilation'), this.dilation);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
        }

        // WebGPU Render
        if (this.device && this.renderPipeline) {
            const aspect = this.gpuCanvas.width / this.gpuCanvas.height;
            const params = new Float32Array([
                dt, time, this.mouse.x, this.mouse.y,
                this.isScanning ? 1.0 : 0.0,
                this.dilation,
                0, aspect // packed aspect
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

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
                    storeOp: 'store'
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
        window.removeEventListener('mouseup', this.handleMouseUp);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('mousedown', this.handleMouseDown);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}
