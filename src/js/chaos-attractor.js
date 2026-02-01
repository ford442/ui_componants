/**
 * Chaos Attractor Experiment
 * Visualizing the Lorenz Attractor using WebGPU Compute Shaders.
 */

export class ChaosAttractor {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            particleCount: options.particleCount || 1000000,
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            speed: 0.5,
            ...options
        };

        this.isActive = true;

        // Camera State
        this.camera = {
            theta: 0.5, // Horizontal angle
            phi: 1.0,   // Vertical angle
            radius: 80.0,
            target: [0, 0, 25], // Look at center of attractor (approx z=25)
            matrix: new Float32Array(16),
            projection: new Float32Array(16),
            viewProjection: new Float32Array(16)
        };

        // Interaction State
        this.isDragging = false;
        this.isModifying = false; // Shift key state
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.init();
    }

    async init() {
        console.log("ChaosAttractor: Initializing...");

        // Setup Container
        this.setupContainer();

        // 1. Initialize WebGL2 (Background / Reference Grid)
        this.initWebGL2();

        // 2. Initialize WebGPU (Particles)
        let gpuSuccess = false;
        if (navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.error("WebGPU Init Error:", e);
            }
        }

        if (!gpuSuccess) {
            this.showError("WebGPU not supported. This experiment requires a WebGPU-enabled browser.");
            return;
        }

        // Start Loop
        this.startTime = performance.now();
        this.animate();

        // Events
        this.setupEvents();
    }

    setupContainer() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        // UI Layer
        this.uiLayer = document.createElement('div');
        this.uiLayer.style.cssText = `
            position: absolute;
            bottom: 20px;
            left: 20px;
            color: rgba(255, 255, 255, 0.8);
            font-family: 'Courier New', monospace;
            font-size: 12px;
            pointer-events: none;
            z-index: 10;
            text-align: left;
            text-shadow: 0 0 2px rgba(0,0,0,0.8);
            user-select: none;
        `;
        this.container.appendChild(this.uiLayer);
        this.updateUI();
    }

    updateUI(dt = 0.016) {
        if (!this.uiLayer) return;

        this.uiLayer.innerHTML = `
            <div style="margin-bottom: 8px; opacity: 0.7;">
                Particles: ${this.options.particleCount.toLocaleString()}<br>
                FPS: ${(1 / dt).toFixed(0)}
            </div>
            <div style="border-left: 2px solid #00ffcc; padding-left: 8px;">
                <div style="color: #00ffcc; font-weight: bold;">INTERACTION MODE</div>
                <div>Rotate: Left Click + Drag</div>
                <div>Zoom: Mouse Wheel</div>
                <div style="color: #ffcc00; margin-top: 4px;">➤ Mutate: Hold SHIFT + Drag</div>
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #aaa;">
                RHO (Chaos): <span style="color:#fff">${this.options.rho.toFixed(2)}</span><br>
                BETA (Shape): <span style="color:#fff">${this.options.beta.toFixed(2)}</span>
            </div>
        `;
    }

    showError(msg) {
        const div = document.createElement('div');
        div.className = 'webgpu-error';
        div.textContent = msg;
        this.container.appendChild(div);
    }

    // ========================================================================
    // WebGL2 (Grid & Box)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: 1; pointer-events: none;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Create Box Geometry
        const positions = new Float32Array([
            // Floor
            -50, -50, 0,  50, -50, 0,
            50, -50, 0,   50, 50, 0,
            50, 50, 0,    -50, 50, 0,
            -50, 50, 0,   -50, -50, 0,

            // Axes
            0, 0, 0, 10, 0, 0, // X
            0, 0, 0, 0, 10, 0, // Y
            0, 0, 0, 0, 0, 10, // Z
        ]);

        this.boxVertexCount = positions.length / 3;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const buf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        this.gl.enableVertexAttribArray(0);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);

        // Shader
        const vs = `#version 300 es
        layout(location=0) in vec3 a_pos;
        uniform mat4 u_viewProjection;
        void main() {
            gl_Position = u_viewProjection * vec4(a_pos, 1.0);
        }`;

        const fs = `#version 300 es
        precision mediump float;
        out vec4 color;
        void main() {
            color = vec4(0.2, 0.2, 0.2, 1.0);
        }`;

        this.glProgram = this.createGLProgram(vs, fs);
        this.glVPLoc = this.gl.getUniformLocation(this.glProgram, 'u_viewProjection');

        this.resizeGL();
    }

    createGLProgram(vsSrc, fsSrc) {
        const gl = this.gl;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSrc);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vs));

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSrc);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fs));

        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        return prog;
    }

    // ========================================================================
    // WebGPU (Particles)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });

        // 1. Particle Buffer
        // Struct: pos (vec4f) - x, y, z, unused
        const particleSize = 16; // 4 * 4 bytes
        const bufferSize = this.options.particleCount * particleSize;

        // Initial Data
        const initData = new Float32Array(this.options.particleCount * 4);
        for(let i=0; i<this.options.particleCount; i++) {
            // Start near a point on the attractor
            initData[i*4+0] = 0.1 + (Math.random() - 0.5) * 5.0;
            initData[i*4+1] = 0.0 + (Math.random() - 0.5) * 5.0;
            initData[i*4+2] = 20.0 + (Math.random() - 0.5) * 5.0;
            initData[i*4+3] = 1.0; // Life or unused
        }

        this.particleBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(initData);
        this.particleBuffer.unmap();

        // 2. Uniform Buffers

        // Sim Params
        // sigma, rho, beta, dt, time, aspectRatio, pad, pad
        this.simParamsBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // View Projection Matrix
        this.viewProjBuffer = this.device.createBuffer({
            size: 64, // mat4x4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // 3. Compute Pipeline
        const computeShader = `
            struct Particle {
                pos: vec4f,
            }

            struct Params {
                sigma: f32,
                rho: f32,
                beta: f32,
                dt: f32,
                time: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: Params;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.options.particleCount}) {
                    return;
                }

                var p = particles[index].pos;
                let x = p.x;
                let y = p.y;
                let z = p.z;

                let dx = params.sigma * (y - x);
                let dy = x * (params.rho - z) - y;
                let dz = x * y - params.beta * z;

                p.x += dx * params.dt;
                p.y += dy * params.dt;
                p.z += dz * params.dt;

                particles[index].pos = p;
            }
        `;

        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'main' }
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamsBuffer } }
            ]
        });

        // 4. Render Pipeline
        const renderShader = `
            struct Uniforms {
                viewProj: mat4x4f,
            }
            struct Params {
                sigma: f32,
                rho: f32,
                beta: f32,
                dt: f32,
                time: f32,
                aspectRatio: f32,
            }

            struct Particle {
                pos: vec4f,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> particles: array<Particle>;
            @group(0) @binding(2) var<uniform> params: Params;

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec4f,
                @location(1) uv: vec2f,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) v_index: u32, @builtin(instance_index) i_index: u32) -> VertexOutput {
                var out: VertexOutput;

                // Quad corners
                var corners = array<vec2f, 6>(
                    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
                );
                let corner = corners[v_index];
                out.uv = corner;

                let particle = particles[i_index];
                let pos = particle.pos.xyz;

                // Project center
                let p_clip = uniforms.viewProj * vec4f(pos, 1.0);

                // Billboard size (0.5 scale for radius vs diameter)
                let size = 0.5;
                var offset = corner * size;

                // Perspective scaling (approximate constant world size)
                out.position = p_clip + vec4f(offset.x / params.aspectRatio, offset.y, 0.0, 0.0);

                // Color based on Z
                let t = clamp(pos.z / 50.0, 0.0, 1.0);
                let col1 = vec3f(1.0, 0.2, 0.1); // Red/Orange
                let col2 = vec3f(0.1, 0.5, 1.0); // Blue
                let c = mix(col2, col1, t);

                out.color = vec4f(c, 1.0);

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f, @location(1) uv: vec2f) -> @location(0) vec4f {
                let dist = length(uv);
                if (dist > 1.0) {
                    discard;
                }
                // Soft glow
                let alpha = pow(1.0 - dist, 2.0);
                return vec4f(color.rgb, alpha * 0.8);
            }
        `;

        const renderModule = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.viewProjBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } },
                { binding: 2, resource: { buffer: this.simParamsBuffer } }
            ]
        });

        this.resizeGPU();
        return true;
    }

    // ========================================================================
    // Logic
    // ========================================================================

    updateMatrices() {
        const { theta, phi, radius, target } = this.camera;

        // Spherical to Cartesian
        const x = radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.cos(phi);
        const z = radius * Math.sin(phi) * Math.sin(theta);

        const eye = [target[0] + x, target[1] + y, target[2] + z];
        const up = [0, 1, 0];

        // Aspect
        const aspect = this.container.clientWidth / this.container.clientHeight;

        // Compute Projection
        mat4.perspective(this.camera.projection, Math.PI / 4, aspect, 0.1, 1000.0);

        // Compute View
        const view = new Float32Array(16);
        mat4.lookAt(view, eye, target, up);

        // Multiply
        mat4.multiply(this.camera.viewProjection, this.camera.projection, view);
    }

    animate() {
        if (!this.isActive) return;

        const now = performance.now();
        const dt = Math.min((now - this.lastTime) * 0.001, 0.1) || 0.016;
        this.lastTime = now;

        this.updateMatrices();

        // Update UI
        if (Math.random() < 0.1) { // Throttle UI updates
            this.updateUI(dt);
        }

        // WebGL2 Render
        if (this.gl) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(this.glVPLoc, false, this.camera.viewProjection);
            this.gl.bindVertexArray(this.glVao);
            this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT); // Clear only color, transparent background
            this.gl.drawArrays(this.gl.LINES, 0, this.boxVertexCount);
        }

        // WebGPU Render
        if (this.device) {
            // Update Uniforms
            const time = (now - this.startTime) * 0.001;
            const aspect = this.gpuCanvas.width / this.gpuCanvas.height;
            const simParams = new Float32Array([
                this.options.sigma,
                this.options.rho,
                this.options.beta,
                0.01 * this.options.speed, // Fixed small dt for integration stability * speed
                time, aspect, 0, 0 // Padding
            ]);
            this.device.queue.writeBuffer(this.simParamsBuffer, 0, simParams);
            this.device.queue.writeBuffer(this.viewProjBuffer, 0, this.camera.viewProjection);

            const encoder = this.device.createCommandEncoder();

            // Compute Pass
            const computePass = encoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.options.particleCount / 64));
            computePass.end();

            // Render Pass
            const textureView = this.context.getCurrentTexture().createView();
            const renderPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.renderBindGroup);
            renderPass.draw(6, this.options.particleCount);
            renderPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        requestAnimationFrame(() => this.animate());
    }

    resizeGL() {
        if (!this.glCanvas) return;
        const dpr = window.devicePixelRatio || 1;
        this.glCanvas.width = this.container.clientWidth * dpr;
        this.glCanvas.height = this.container.clientHeight * dpr;
    }

    resizeGPU() {
        if (!this.gpuCanvas) return;
        const dpr = window.devicePixelRatio || 1;
        this.gpuCanvas.width = this.container.clientWidth * dpr;
        this.gpuCanvas.height = this.container.clientHeight * dpr;
    }

    setupEvents() {
        this.resizeHandler = () => {
            this.resizeGL();
            this.resizeGPU();
        };
        window.addEventListener('resize', this.resizeHandler);

        // Key Modifiers
        this.keyDownHandler = (e) => {
            if (e.key === 'Shift') {
                this.isModifying = true;
                this.container.style.cursor = 'crosshair';
            }
        };
        window.addEventListener('keydown', this.keyDownHandler);

        this.keyUpHandler = (e) => {
            if (e.key === 'Shift') {
                this.isModifying = false;
                this.container.style.cursor = 'default';
            }
        };
        window.addEventListener('keyup', this.keyUpHandler);

        this.gpuCanvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        this.mouseMoveHandler = (e) => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;

            if (this.isModifying) {
                // Mutate Parameters
                // dx controls Rho (Chaos)
                this.options.rho = Math.max(1.0, Math.min(100.0, this.options.rho + dx * 0.1));

                // dy controls Beta (Shape/Aspect)
                this.options.beta = Math.max(0.1, Math.min(10.0, this.options.beta - dy * 0.01));
            } else {
                // Rotate Camera
                this.camera.theta -= dx * 0.01;
                this.camera.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.camera.phi - dy * 0.01));
            }
        };
        window.addEventListener('mousemove', this.mouseMoveHandler);

        this.mouseUpHandler = () => {
            this.isDragging = false;
        };
        window.addEventListener('mouseup', this.mouseUpHandler);

        this.gpuCanvas.addEventListener('wheel', (e) => {
            this.camera.radius = Math.max(10, Math.min(500, this.camera.radius + e.deltaY * 0.1));
            e.preventDefault();
        });
    }

    destroy() {
        this.isActive = false;
        window.removeEventListener('resize', this.resizeHandler);
        window.removeEventListener('keydown', this.keyDownHandler);
        window.removeEventListener('keyup', this.keyUpHandler);
        window.removeEventListener('mousemove', this.mouseMoveHandler);
        window.removeEventListener('mouseup', this.mouseUpHandler);

        if (this.uiLayer && this.uiLayer.parentNode) {
            this.uiLayer.parentNode.removeChild(this.uiLayer);
        }
    }
}

// Minimal Matrix Math (Column Major)
const mat4 = {
    perspective: (out, fovy, aspect, near, far) => {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        out.fill(0);
        out[0] = f / aspect;
        out[5] = f;
        out[10] = (far + near) * nf;
        out[11] = -1;
        out[14] = (2 * far * near) * nf;
    },
    lookAt: (out, eye, center, up) => {
        let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
        let eyex = eye[0], eyey = eye[1], eyez = eye[2];
        let upx = up[0], upy = up[1], upz = up[2];
        let centerx = center[0], centery = center[1], centerz = center[2];

        z0 = eyex - centerx; z1 = eyey - centery; z2 = eyez - centerz;
        len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        z0 *= len; z1 *= len; z2 *= len;

        x0 = upy * z2 - upz * z1;
        x1 = upz * z0 - upx * z2;
        x2 = upx * z1 - upy * z0;
        len = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
        if (!len) { x0 = 0; x1 = 0; x2 = 0; } else { len = 1 / len; x0 *= len; x1 *= len; x2 *= len; }

        y0 = z1 * x2 - z2 * x1;
        y1 = z2 * x0 - z0 * x2;
        y2 = z0 * x1 - z1 * x0;
        len = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
        if (!len) { y0 = 0; y1 = 0; y2 = 0; } else { len = 1 / len; y0 *= len; y1 *= len; y2 *= len; }

        out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
        out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
        out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
        out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
        out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
        out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
        out[15] = 1;
    },
    multiply: (out, a, b) => {
        let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        let a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        let a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        let a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
        let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
        out[0] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[1] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[2] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[3] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

        b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
        out[4] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[5] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[6] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[7] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

        b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
        out[8] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[9] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[10] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[11] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

        b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
        out[12] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[13] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[14] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[15] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
    }
};
