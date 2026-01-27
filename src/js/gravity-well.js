
export class GravityWellExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.clickData = { time: -100, x: 0, y: 0 };
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
        this.numParticles = options.numParticles || 30000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000';

        this.initWebGL2();

        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("GravityWell: WebGPU init failed", e);
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
        this.container.addEventListener('mousedown', this.handleMouseDown);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    onMouseDown(e) {
        // Shockwave effect
        // Map screen mouse to approximate world coordinates on the XZ plane
        // Since camera rotates, this is an approximation, but sufficient for a "blast" effect
        // originating from where the user *thinks* they are clicking.
        // Ideally we'd raycast, but for this visual effect, using the mouse vector is okay.

        // Actually, let's just blast from the center if they click near it, or from a random point?
        // Let's try to map it roughly.
        // Camera is at radius 3.0.
        // We'll just set the click position to be "somewhere" based on screen coords.
        // But since the camera rotates, X on screen doesn't map to X in world constantly.
        // The shader knows the camera angle. Maybe we pass raw mouse coords and the shader calculates the world position?
        // Yes, let's pass the raw mouse coordinates at the time of click.

        this.clickData.time = (Date.now() - this.startTime) * 0.001;
        this.clickData.x = this.mouse.x;
        this.clickData.y = this.mouse.y;
    }

    // ========================================================================
    // WebGL2 (Wireframe Funnel)
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

        // Generate Grid
        const size = 30;
        const range = 2.0;
        const vertices = [];

        // Horizontal lines
        for (let i = 0; i <= size; i++) {
            const z = (i / size) * 2 * range - range;
            for (let j = 0; j < size; j++) {
                const x1 = (j / size) * 2 * range - range;
                const x2 = ((j + 1) / size) * 2 * range - range;
                vertices.push(x1, 0, z);
                vertices.push(x2, 0, z);
            }
        }
        // Vertical lines
        for (let i = 0; i <= size; i++) {
            const x = (i / size) * 2 * range - range;
            for (let j = 0; j < size; j++) {
                const z1 = (j / size) * 2 * range - range;
                const z2 = ((j + 1) / size) * 2 * range - range;
                vertices.push(x, 0, z1);
                vertices.push(x, 0, z2);
            }
        }

        this.vertexCount = vertices.length / 3;

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        const vsSource = `#version 300 es
        in vec3 a_position;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec2 u_mouse;
        out float v_depth;

        void main() {
            vec3 pos = a_position;

            // Funnel deformation
            float r = length(pos.xz);
            // Deep funnel at center
            pos.y = -0.5 / (r + 0.1);

            // Interactive wobble
            float wobble = sin(r * 5.0 - u_time * 2.0 + u_mouse.x * 2.0);
            pos.y += wobble * 0.05;

            // Camera Projection (Simple)
            // Rotate camera around center
            float camAngle = u_time * 0.1;
            float camDist = 3.0;
            vec3 camPos = vec3(cos(camAngle) * camDist, 1.5, sin(camAngle) * camDist);
            vec3 target = vec3(0.0, -0.5, 0.0);

            // LookAt Matrix construction
            vec3 f = normalize(target - camPos);
            vec3 u = vec3(0.0, 1.0, 0.0);
            vec3 s = normalize(cross(f, u));
            u = cross(s, f);

            mat4 view = mat4(
                s.x, u.x, -f.x, 0.0,
                s.y, u.y, -f.y, 0.0,
                s.z, u.z, -f.z, 0.0,
                -dot(s, camPos), -dot(u, camPos), dot(f, camPos), 1.0
            );

            // Perspective
            float aspect = u_resolution.x / u_resolution.y;
            float fov = 1.0;
            float f_n = 100.0;
            float n = 0.1;
            float tanHalfFov = tan(fov / 2.0);

            mat4 proj = mat4(
                1.0 / (aspect * tanHalfFov), 0.0, 0.0, 0.0,
                0.0, 1.0 / tanHalfFov, 0.0, 0.0,
                0.0, 0.0, -(f_n + n) / (f_n - n), -1.0,
                0.0, 0.0, -(2.0 * f_n * n) / (f_n - n), 0.0
            );

            gl_Position = proj * view * vec4(pos, 1.0);
            v_depth = gl_Position.z;
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        in float v_depth;
        out vec4 outColor;

        void main() {
            // Fade based on depth
            float alpha = 1.0 - smoothstep(0.0, 5.0, v_depth);
            outColor = vec4(0.2, 0.8, 0.4, 0.3 * alpha);
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
    // WebGPU (Particle System)
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

        // Particles: pos(4), vel(4) - align to 16 bytes
        const particleData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            const idx = i * 8;
            // Start at random radius > 1.0
            const angle = Math.random() * Math.PI * 2;
            const r = 1.5 + Math.random();
            particleData[idx] = Math.cos(angle) * r;   // x
            particleData[idx+1] = 0.0;                 // y (calc in shader)
            particleData[idx+2] = Math.sin(angle) * r; // z
            particleData[idx+3] = Math.random();       // life

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
            size: 32, // time, dt, mouseX, mouseY, aspect, clickTime, clickX, clickY
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Compute Shader
        const computeCode = `
        struct Particle {
            pos : vec4f, // xyz, life
            vel : vec4f,
        }
        @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

        struct Params {
            time : f32,
            dt : f32,
            mouseX : f32,
            mouseY : f32,
            aspect : f32,
            clickTime : f32,
            clickX : f32,
            clickY : f32,
        }
        @group(0) @binding(1) var<uniform> params : Params;

        fn random(st: vec2f) -> f32 {
            return fract(sin(dot(st, vec2f(12.9898, 78.233))) * 43758.5453123);
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id : vec3u) {
            let idx = id.x;
            if (idx >= arrayLength(&particles)) { return; }

            var p = particles[idx];

            // Physics
            // Move towards center (0,0) on XZ plane
            let posXZ = p.pos.xz;
            let r = length(posXZ);
            let dir = normalize(-posXZ); // Inward

            // Tangential (orbit)
            let tangent = vec2f(-dir.y, dir.x);

            // Forces
            // Stronger gravity closer to center
            let gravity = dir * (0.05 / (r + 0.1));
            // Orbit speed increases near center
            let orbit = tangent * (2.0 / (r + 0.5));

            // Shockwave interaction
            var shock : vec2f = vec2f(0.0);
            let shockTime = params.time - params.clickTime;
            if (shockTime > 0.0 && shockTime < 1.0) {
                // Determine shockwave origin
                // Since we don't unproject, let's just use the current camera angle
                // to estimate where the click was on the ring.
                // Or simpler: Shockwave radiates from center if click was near center,
                // or just adds a global radial push based on noise?

                // Let's implement a "Repulse from Center" blast for now, triggered by click.
                // It creates a ring of force that expands.
                let waveRadius = shockTime * 5.0; // Expands fast
                let distFromWave = abs(r - waveRadius);
                if (distFromWave < 0.5) {
                   shock = normalize(p.pos.xz) * 5.0 * (1.0 - distFromWave * 2.0);
                }
            }

            // Combine
            let targetVelXZ = gravity + orbit + shock;

            // Update velocity
            p.vel.x = mix(p.vel.x, targetVelXZ.x, 0.1);
            p.vel.z = mix(p.vel.z, targetVelXZ.y, 0.1);

            // Update Position
            p.pos.x += p.vel.x * params.dt;
            p.pos.z += p.vel.z * params.dt;

            // Calc Y based on funnel: y = -0.5 / (r + 0.1)
            let newR = length(p.pos.xz);
            p.pos.y = -0.5 / (newR + 0.1);

            // Reset logic
            // If too close to center (singularity) or too far
            if (newR < 0.1 || newR > 3.0) {
                // Respawn at edge
                let angle = random(vec2f(params.time, f32(idx))) * 6.28;
                let startR = 2.0 + random(vec2f(f32(idx), params.time)) * 0.5;
                p.pos.x = cos(angle) * startR;
                p.pos.z = sin(angle) * startR;
                p.pos.y = -0.5 / (startR + 0.1);
                p.vel = vec4f(0.0);
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
        // Share Projection logic with WebGL or approximate it
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
            clickTime : f32,
            clickX : f32,
            clickY : f32,
        }
        @group(0) @binding(1) var<uniform> params : Params;

        @vertex
        fn vs_main(
            @builtin(vertex_index) vIdx : u32,
            @location(0) pos : vec4f,
            @location(1) vel : vec4f
        ) -> VertexOutput {
            var out : VertexOutput;

            // Camera Logic
            let camAngle = params.time * 0.1;
            let camDist = 3.0;
            let camPos = vec3f(cos(camAngle) * camDist, 1.5, sin(camAngle) * camDist);
            let target = vec3f(0.0, -0.5, 0.0);

            let f = normalize(target - camPos);
            let u = vec3f(0.0, 1.0, 0.0);
            let s = normalize(cross(f, u));
            let u_new = cross(s, f);

            let view = mat4x4f(
                vec4f(s.x, u_new.x, -f.x, 0.0),
                vec4f(s.y, u_new.y, -f.y, 0.0),
                vec4f(s.z, u_new.z, -f.z, 0.0),
                vec4f(-dot(s, camPos), -dot(u_new, camPos), dot(f, camPos), 1.0)
            );

            let fov = 1.0;
            let f_n = 100.0;
            let n = 0.1;
            let tanHalfFov = tan(fov * 0.5);
            let aspect = params.aspect;

            let proj = mat4x4f(
                vec4f(1.0 / (aspect * tanHalfFov), 0.0, 0.0, 0.0),
                vec4f(0.0, 1.0 / tanHalfFov, 0.0, 0.0),
                vec4f(0.0, 0.0, -(f_n + n) / (f_n - n), -1.0),
                vec4f(0.0, 0.0, -(2.0 * f_n * n) / (f_n - n), 0.0)
            );

            // Billboard Quad Generation
            // 0, 1, 2
            // 2, 1, 3 (triangle strip order or indexed list)
            // Using triangle list: 0,1,2, 0,2,3 (or similar)
            // Let's use array of 6 vertices

            var corners = array<vec2f, 6>(
                vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
                vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
            );

            let corner = corners[vIdx];
            out.uv = corner * 0.5 + 0.5; // 0..1

            // Project center
            let p_center = view * vec4f(pos.xyz, 1.0);

            // Billboard size
            let size = 0.03 * (1.0 + pos.w * 0.5); // Use life for size var

            // Add offset in view space (billboarding)
            let p_view = p_center + vec4f(corner * size, 0.0, 0.0);

            out.pos = proj * p_view;

            // Color based on speed and depth
            let speed = length(vel.xyz);
            // Hotter colors near center/fast
            let colorBase = vec3f(1.0, 0.3, 0.1);
            let colorHot = vec3f(0.5, 0.8, 1.0);

            out.color = vec4f(mix(colorBase, colorHot, speed * 0.5), 1.0);

            // Fade out edges of quad (soft particle) done in fragment

            return out;
        }

        @fragment
        fn fs_main(@location(0) color : vec4f, @location(1) uv : vec2f) -> @location(0) vec4f {
            // Circular particle
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
                    stepMode: 'instance', // Changed to instance
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
            primitive: { topology: 'triangle-list' } // Changed to triangle-list
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

        // 1. WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'u_mouse'), this.mouse.x, this.mouse.y);

            this.gl.clearColor(0,0,0,1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.LINES, 0, this.vertexCount);
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
                this.clickData.time,
                this.clickData.x,
                this.clickData.y
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
            renderPass.draw(6, this.numParticles); // Draw 6 vertices per instance
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
        this.container.removeEventListener('mousedown', this.handleMouseDown);
        if(this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        this.device?.destroy();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.GravityWellExperiment = GravityWellExperiment;
}
