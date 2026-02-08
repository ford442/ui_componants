
export class QuantumEntanglementExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.canvasSize = { width: 0, height: 0 };

        // WebGL2
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.vertexCount = 0;

        // WebGPU
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 40000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050010';

        this.initWebGL2();

        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("QuantumEntanglement: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();
        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        window.addEventListener('mousemove', this.handleMouseMove);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    // ========================================================================
    // WebGL2 (Torus Knots)
    // ========================================================================
    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        const gl = this.gl;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.enable(gl.DEPTH_TEST);

        // Generate Torus Knot Geometry
        // Trefoil knot: x = sin(t) + 2sin(2t), y = cos(t) - 2cos(2t), z = -sin(3t)
        // Or (p,q) torus knot
        const vertices = [];
        const steps = 300;
        const p = 2, q = 3;
        const tubeRadius = 0.15;
        const radius = 0.8;

        // We'll generate a tube
        const tubeSteps = 12;

        for (let i = 0; i <= steps; i++) {
            const t = (i / steps) * Math.PI * 2;
            const tNext = ((i + 1) / steps) * Math.PI * 2;

            // Central path
            const getPos = (ang) => {
                const r = radius * (2 + Math.cos(q * ang));
                return [
                    r * Math.cos(p * ang),
                    r * Math.sin(p * ang),
                    radius * Math.sin(q * ang)
                ];
            };

            const p1 = getPos(t);
            const p2 = getPos(tNext);

            // Frenet frame approx
            const tangent = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
            const len = Math.sqrt(tangent[0]**2 + tangent[1]**2 + tangent[2]**2);
            const T = [tangent[0]/len, tangent[1]/len, tangent[2]/len];

            const up = [0, 0, 1];
            // N = T x up (rough)
            let N = [T[1]*up[2] - T[2]*up[1], T[2]*up[0] - T[0]*up[2], T[0]*up[1] - T[1]*up[0]];
            let nLen = Math.sqrt(N[0]**2 + N[1]**2 + N[2]**2);
            if (nLen < 0.001) N = [1, 0, 0]; // Degenerate case
            else N = [N[0]/nLen, N[1]/nLen, N[2]/nLen];

            const B = [T[1]*N[2] - T[2]*N[1], T[2]*N[0] - T[0]*N[2], T[0]*N[1] - T[1]*N[0]];

            // Generate circle at this segment
            for (let j = 0; j <= tubeSteps; j++) {
                const phi = (j / tubeSteps) * Math.PI * 2;
                const phiNext = ((j + 1) / tubeSteps) * Math.PI * 2;

                // We'll draw lines for wireframe look
                // Current ring vertex
                const cx = Math.cos(phi);
                const cy = Math.sin(phi);

                const vx = p1[0] + tubeRadius * (cx * N[0] + cy * B[0]);
                const vy = p1[1] + tubeRadius * (cx * N[1] + cy * B[1]);
                const vz = p1[2] + tubeRadius * (cx * N[2] + cy * B[2]);

                // Next ring vertex
                // Re-calculate frame for p2? For simplicity just use p1's frame for the segment connection
                // but that causes twisting artifacts.
                // For a wireframe experiment, let's just draw rings or longitudinal lines.
                // Let's just store points and draw GL_LINES_STRIP or similar.

                // Simpler: Just the center line curve
                // No, wireframe tube looks cooler.

                // Let's just output triangles for a tube and render as gl.LINES using barycentric coords in shader?
                // Or just standard wireframe mesh.

                vertices.push(vx, vy, vz);
            }
        }

        // Actually, for a clean wireframe, let's just draw the center curve with high thickness
        // NO, the prompt asked for "Structure".
        // Let's settle for just the center curve but drawn twice (left and right).

        const lineVertices = [];
        for(let i=0; i<=steps; i++) {
             const t = (i / steps) * Math.PI * 2 * 10; // multiple loops? No 0 to 2PI is full loop for knot
             // Torus knot closes at 2PI * something?
             // P=2, Q=3. Period is 2PI.
             const ang = (i / steps) * Math.PI * 2;
             const r = 0.6 * (2 + Math.cos(q * ang));
             const x = r * Math.cos(p * ang);
             const y = r * Math.sin(p * ang);
             const z = 0.6 * Math.sin(q * ang);
             lineVertices.push(x, y, z);
        }

        this.vertexCount = lineVertices.length / 3;

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lineVertices), gl.STATIC_DRAW);

        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_offset; // X offset for the knot
        uniform vec3 u_color;

        out vec3 v_color;

        void main() {
            vec3 pos = a_position;

            // Rotation
            float c = cos(u_time * 0.5);
            float s = sin(u_time * 0.5);

            // Rotate around Z locally
            float x = pos.x * c - pos.y * s;
            float y = pos.x * s + pos.y * c;
            pos.x = x;
            pos.y = y;

            // World position
            vec3 worldPos = pos + vec3(u_offset, 0.0, 0.0);

            // Camera
            vec3 camPos = vec3(0.0, 0.0, 4.0);
            vec3 target = vec3(0.0, 0.0, 0.0);

            // Simple perspective
            float aspect = u_resolution.x / u_resolution.y;
            float fov = 1.0;
            float f = 1.0 / tan(fov/2.0);

            // View transform (look at 0) - identity for now as cam is at Z=4 looking at Z=0
            vec3 p = worldPos - camPos;

            // Projection
            float z_near = 0.1;
            float z_far = 100.0;

            gl_Position = vec4(p.x * f / aspect, p.y * f, (p.z * (z_far+z_near)/(z_near-z_far)) + (2.0*z_far*z_near)/(z_near-z_far), -p.z);

            // Fade with depth
            float dist = length(worldPos - camPos);
            float alpha = 1.0 / (dist * 0.5);

            v_color = u_color * alpha;
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in vec3 v_color;
        out vec4 outColor;

        void main() {
            outColor = vec4(v_color, 0.6);
        }`;

        this.glProgram = this.createGLProgram(gl, vsSource, fsSource);
        if (this.glProgram) {
            const loc = gl.getAttribLocation(this.glProgram, 'a_position');
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
        }
        this.glVao = vao;
    }

    createGLProgram(gl, vsSource, fsSource) {
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSource);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(vs));
            return null;
        }
        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSource);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(fs));
            return null;
        }
        const p = gl.createProgram();
        gl.attachShader(p, vs);
        gl.attachShader(p, fs);
        gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Entangled Particles)
    // ========================================================================
    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2; pointer-events: none; background: transparent;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) {
            this.gpuCanvas.remove();
            return false;
        }
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        // Init Particles
        const particleData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            const idx = i * 8;

            // Even: Left (-1.5), Odd: Right (1.5)
            const isLeft = (i % 2 === 0);
            const centerX = isLeft ? -1.5 : 1.5;

            // Random start around torus knot? Or just sphere cloud?
            // Random sphere cloud around center
            const r = 0.5 * Math.random();
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            const lx = r * Math.sin(phi) * Math.cos(theta);
            const ly = r * Math.sin(phi) * Math.sin(theta);
            const lz = r * Math.cos(phi);

            particleData[idx] = centerX + lx; // x
            particleData[idx+1] = ly;         // y
            particleData[idx+2] = lz;         // z
            particleData[idx+3] = Math.random(); // life/phase

            particleData[idx+4] = 0; // vx
            particleData[idx+5] = 0; // vy
            particleData[idx+6] = 0; // vz
            particleData[idx+7] = 0; // pad
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);

        this.simParamBuffer = this.device.createBuffer({
            size: 32, // time, dt, mouseX, mouseY, aspect, padding...
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Compute Shader
        const computeCode = `
        struct Particle {
            pos : vec4f,
            vel : vec4f,
        }
        @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

        struct Params {
            time : f32,
            dt : f32,
            mouseX : f32,
            mouseY : f32,
            aspect : f32,
            pad1: f32,
            pad2: f32,
            pad3: f32,
        }
        @group(0) @binding(1) var<uniform> params : Params;

        fn random(st: vec2f) -> f32 {
            return fract(sin(dot(st, vec2f(12.9898, 78.233))) * 43758.5453123);
        }

        // Noise function for flow
        fn snoise(v : vec3f) -> f32 {
             return sin(v.x * 10.0 + params.time) * sin(v.y * 10.0) * sin(v.z * 10.0);
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id : vec3u) {
            let idx = id.x;
            if (idx >= arrayLength(&particles)) { return; }

            var p = particles[idx];
            let isLeft = (idx % 2 == 0);
            let centerX = select(1.5, -1.5, isLeft);

            // Base orbit/flow
            let localPos = p.pos.xyz - vec3f(centerX, 0.0, 0.0);

            // Attract to ring center
            let r = length(localPos);
            let attract = -normalize(localPos) * (r - 0.7) * 2.0; // Target radius 0.7

            // Rotate
            let tan = vec3f(-localPos.y, localPos.x, 0.0);

            var force = attract + tan * 2.0;

            // Mouse Interaction (Entanglement)
            // Real mouse position
            let mPos = vec3f(params.mouseX * params.aspect * 2.0, params.mouseY * 2.0, 0.0); // Rough projection mapping

            // Mirrored mouse position (The "Ghost" interaction)
            let ghostMPos = vec3f(-mPos.x, mPos.y, mPos.z);

            // Interaction logic:
            // Calculate force from Real Mouse
            let dirM = p.pos.xyz - mPos;
            let distM = length(dirM);
            var mouseForce = vec3f(0.0);
            if (distM < 1.0) {
                mouseForce += normalize(dirM) * 10.0 * (1.0 - distM); // Repel
            }

            // Calculate force from Ghost Mouse (Entanglement effect)
            let dirG = p.pos.xyz - ghostMPos;
            let distG = length(dirG);
            if (distG < 1.0) {
                // If the mouse is disturbing my partner's region (mirrored), I react too!
                mouseForce += normalize(dirG) * 10.0 * (1.0 - distG);
            }

            // Apply forces
            p.vel = vec4f(mix(p.vel.xyz, force + mouseForce, 0.1), 0.0);

            // Update pos
            p.pos = vec4f(p.pos.xyz + p.vel.xyz * params.dt, p.pos.w);

            // Life/Reset
            // p.pos.w stores life?
            // Keep constrained
            if (length(p.pos.xyz - vec3f(centerX, 0.0, 0.0)) > 2.5) {
                p.pos.x = centerX;
                p.pos.y = 0.0;
                p.pos.z = 0.0;
            }

            particles[idx] = p;
        }
        `;

        const computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'main' }
        });

        // Draw Shader
        const drawCode = `
        struct VertexOutput {
            @builtin(position) pos : vec4f,
            @location(0) color : vec4f,
            @location(1) uv : vec2f,
        }
        struct Params {
            time : f32,
            dt : f32,
            mouseX : f32,
            mouseY : f32,
            aspect : f32,
        }
        @group(0) @binding(1) var<uniform> params : Params;

        @vertex
        fn vs_main(
            @builtin(vertex_index) vIdx : u32,
            @location(0) pos : vec4f,
            @location(1) vel : vec4f,
            @builtin(instance_index) iIdx : u32
        ) -> VertexOutput {
            var out : VertexOutput;

            // Simple Perspective (matches WebGL)
            let camPos = vec3f(0.0, 0.0, 4.0);
            let p = pos.xyz - camPos;

            let fov = 1.0;
            let f = 1.0 / tan(fov/2.0);
            let z_near = 0.1;
            let z_far = 100.0;
            let aspect = params.aspect;

            let projPos = vec4f(
                p.x * f / aspect,
                p.y * f,
                (p.z * (z_far+z_near)/(z_near-z_far)) + (2.0*z_far*z_near)/(z_near-z_far),
                -p.z
            );

            // Billboard logic
            var corners = array<vec2f, 6>(
                vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
                vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
            );
            let corner = corners[vIdx];
            out.uv = corner * 0.5 + 0.5;

            let size = 0.02;
            let finalPos = projPos + vec4f(corner * size, 0.0, 0.0);
            out.pos = finalPos;

            // Color
            // Left (Even) = Cyan, Right (Odd) = Magenta
            let isLeft = (iIdx % 2 == 0);
            let baseColor = select(vec3f(1.0, 0.0, 1.0), vec3f(0.0, 1.0, 1.0), isLeft);

            // Velocity brightness
            let speed = length(vel.xyz);
            out.color = vec4f(baseColor + speed * 0.1, 1.0);

            return out;
        }

        @fragment
        fn fs_main(@location(0) color : vec4f, @location(1) uv : vec2f) -> @location(0) vec4f {
            let d = distance(uv, vec2f(0.5));
            let alpha = smoothstep(0.5, 0.2, d);
            if (alpha < 0.01) { discard; }
            return vec4f(color.rgb, color.a * alpha);
        }
        `;

        const drawModule = this.device.createShaderModule({ code: drawCode });

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.computePipeline.getBindGroupLayout(0)] }),
            vertex: {
                module: drawModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'instance',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: drawModule,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        if (this.container.querySelector('.webgpu-error')) return;
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.innerText = "WebGPU Not Available (WebGL2 Only)";
        msg.style.cssText = "position:absolute; bottom:20px; right:20px; color:white; background:rgba(100,0,0,0.8); padding:10px;";
        this.container.appendChild(msg);
    }

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.canvasSize.width = width;
        this.canvasSize.height = height;

        const dw = Math.floor(width * dpr);
        const dh = Math.floor(height * dpr);

        if(this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            if(this.gl) this.gl.viewport(0, 0, dw, dh);
        }
        if(this.gpuCanvas) {
            this.gpuCanvas.width = dw;
            this.gpuCanvas.height = dh;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;

        // 1. WebGL2 - Draw Two Knots
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.bindVertexArray(this.glVao);

            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);

            this.gl.clearColor(0,0,0,0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            // Left Knot (Cyan)
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_offset'), -1.5);
            this.gl.uniform3f(this.gl.getUniformLocation(this.glProgram, 'u_color'), 0.0, 1.0, 1.0);
            this.gl.drawArrays(this.gl.LINE_LOOP, 0, this.vertexCount);

            // Right Knot (Magenta)
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_offset'), 1.5);
            this.gl.uniform3f(this.gl.getUniformLocation(this.glProgram, 'u_color'), 1.0, 0.0, 1.0);
            this.gl.drawArrays(this.gl.LINE_LOOP, 0, this.vertexCount);
        }

        // 2. WebGPU
        if (this.device && this.context && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([
                time,
                0.016,
                this.mouse.x,
                this.mouse.y,
                aspect,
                0,0,0
            ]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r:0, g:0, b:0, a:0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup);
            renderPass.setVertexBuffer(0, this.particleBuffer);
            renderPass.draw(6, this.numParticles);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);
        if(this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        this.device?.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.QuantumEntanglementExperiment = QuantumEntanglementExperiment;
}
