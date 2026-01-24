export class AtmosphericEntry {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            numParticles: options.numParticles || 30000,
            ...options
        };

        this.canvasGL = document.createElement('canvas');
        this.canvasGPU = document.createElement('canvas');

        // Setup styles for layering
        this.container.style.position = 'relative';
        this.container.style.width = '100%';
        this.container.style.height = '100%';
        this.container.style.overflow = 'hidden';
        this.container.style.backgroundColor = '#000'; // Space black

        [this.canvasGL, this.canvasGPU].forEach(canvas => {
            canvas.style.position = 'absolute';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            this.container.appendChild(canvas);
        });

        this.canvasGL.style.zIndex = '1';
        this.canvasGPU.style.zIndex = '2'; // Plasma on top

        this.isPlaying = true;
        this.time = 0;
        this.mouse = { x: 0, y: 0 };

        // Bind methods
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
        this.container.addEventListener('touchmove', this.handleMouseMove);

        this.resize();
        requestAnimationFrame(this.render);
    }

    handleMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        const x = (e.clientX || e.touches[0].clientX) - rect.left;
        const y = (e.clientY || e.touches[0].clientY) - rect.top;

        // Normalize to -1 to 1
        this.mouse.x = (x / rect.width) * 2 - 1;
        this.mouse.y = -((y / rect.height) * 2 - 1);
    }

    destroy() {
        this.isPlaying = false;
        window.removeEventListener('resize', this.resize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('touchmove', this.handleMouseMove);

        // WebGL Cleanup
        if (this.gl) {
            this.gl.deleteProgram(this.program);
            this.gl.deleteBuffer(this.positionBuffer);
            this.gl.deleteBuffer(this.normalBuffer);
            this.gl.deleteVertexArray(this.vao);
        }

        // WebGPU Cleanup (optional, device destroy not strictly necessary in browser context usually)
        if (this.device) {
             this.device.destroy();
        }

        this.container.innerHTML = '';
    }

    initWebGL() {
        this.gl = this.canvasGL.getContext('webgl2');
        if (!this.gl) {
            console.error('WebGL2 not supported');
            return;
        }

        const gl = this.gl;
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);

        // --- Geometry: Hemisphere ---
        const positions = [];
        const normals = [];
        const rings = 30;
        const segments = 40;
        const radius = 1.0;

        for (let i = 0; i <= rings; i++) {
            // 0 to PI/2 for hemisphere
            const lat = (i / rings) * (Math.PI / 2);
            const sinLat = Math.sin(lat);
            const cosLat = Math.cos(lat);

            for (let j = 0; j <= segments; j++) {
                const lon = (j / segments) * Math.PI * 2;
                const sinLon = Math.sin(lon);
                const cosLon = Math.cos(lon);

                const x = cosLon * sinLat;
                const y = cosLat; // Up is Y
                const z = sinLon * sinLat;

                positions.push(x * radius, y * radius, z * radius);
                normals.push(x, y, z);
            }
        }

        // Indices? No, let's just draw points or lines for wireframe feel,
        // OR triangles for solid shield. Let's do Triangles for the shield.
        const indices = [];
        for (let i = 0; i < rings; i++) {
            for (let j = 0; j < segments; j++) {
                const first = (i * (segments + 1)) + j;
                const second = first + segments + 1;

                indices.push(first, second, first + 1);
                indices.push(second, second + 1, first + 1);
            }
        }
        this.indexCount = indices.length;

        // --- Shaders ---
        const vsSource = `#version 300 es
        in vec3 a_position;
        in vec3 a_normal;

        uniform float u_time;
        uniform vec2 u_mouse;
        uniform vec2 u_resolution;

        out vec3 v_normal;
        out vec3 v_pos;

        void main() {
            // Rotate based on mouse
            float rotX = u_mouse.y * 1.5; // Pitch
            float rotY = u_mouse.x * 1.5; // Yaw

            float cX = cos(rotX); float sX = sin(rotX);
            float cY = cos(rotY); float sY = sin(rotY);

            mat3 mX = mat3(1,0,0, 0,cX,-sX, 0,sX,cX);
            mat3 mY = mat3(cY,0,sY, 0,1,0, -sY,0,cY);

            // Base orientation: Cup facing +Z
            // Our hemisphere generation has Y up. Let's rotate it to face Z initially.
            mat3 toZ = mat3(1,0,0, 0,0,-1, 0,1,0);

            vec3 pos = mY * mX * toZ * a_position;
            vec3 norm = mY * mX * toZ * a_normal;

            v_pos = pos;
            v_normal = norm;

            // Project
            float aspect = u_resolution.x / u_resolution.y;
            pos.z -= 4.0; // Push back
            pos.x /= aspect;

            gl_Position = vec4(pos.x, pos.y, pos.z, -pos.z); // Perspective
        }`;

        const fsSource = `#version 300 es
        precision highp float;

        in vec3 v_normal;
        in vec3 v_pos;

        uniform float u_time;

        out vec4 outColor;

        // Simplex noise function (simplified)
        vec3 hash33(vec3 p3) {
            p3 = fract(p3 * vec3(.1031, .1030, .0973));
            p3 += dot(p3, p3.yzx+33.33);
            return fract((p3.xxy + p3.yzz)*p3.zyx);
        }

        float noise(vec3 x) {
            return fract(sin(dot(x, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
        }

        void main() {
            // Lighting / Heat
            // Light comes from 'front' (the atmosphere we are crashing into)
            vec3 lightDir = normalize(vec3(0.0, 0.0, 1.0));
            float diff = max(dot(v_normal, lightDir), 0.0);

            // Heat ablation noise
            float n = noise(v_pos * 5.0 + u_time * 2.0);

            // Color ramp: Black (cool back) -> Red -> Orange -> White (hot front)
            vec3 cool = vec3(0.05, 0.05, 0.05); // Charred carbon
            vec3 hot = vec3(1.0, 0.2, 0.0);    // Glowing red
            vec3 plasma = vec3(1.0, 0.9, 0.8); // White hot

            vec3 color = cool;

            if (diff > 0.3) {
                color = mix(cool, hot, (diff - 0.3) * 1.5 + n * 0.2);
            }
            if (diff > 0.8) {
                color = mix(color, plasma, (diff - 0.8) * 5.0 + n * 0.5);
            }

            outColor = vec4(color, 1.0);
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        // Positions
        this.positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        const posLoc = gl.getAttribLocation(this.program, 'a_position');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        // Normals
        this.normalBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
        const normLoc = gl.getAttribLocation(this.program, 'a_normal');
        gl.enableVertexAttribArray(normLoc);
        gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);

        // Indices
        this.indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

        // Uniforms
        this.uTimeLoc = gl.getUniformLocation(this.program, 'u_time');
        this.uMouseLoc = gl.getUniformLocation(this.program, 'u_mouse');
        this.uResLoc = gl.getUniformLocation(this.program, 'u_resolution');
    }

    createProgram(gl, vs, fs) {
        const vShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vShader, vs);
        gl.compileShader(vShader);
        if(!gl.getShaderParameter(vShader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vShader));

        const fShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fShader, fs);
        gl.compileShader(fShader);
        if(!gl.getShaderParameter(fShader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fShader));

        const p = gl.createProgram();
        gl.attachShader(p, vShader);
        gl.attachShader(p, fShader);
        gl.linkProgram(p);
        return p;
    }

    async initWebGPU() {
        if (!navigator.gpu) return;

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return;
            this.device = await adapter.requestDevice();
            this.contextGPU = this.canvasGPU.getContext('webgpu');
            this.format = navigator.gpu.getPreferredCanvasFormat();

            this.contextGPU.configure({
                device: this.device,
                format: this.format,
                alphaMode: 'premultiplied'
            });

            // --- Particle Data ---
            const count = this.options.numParticles;
            // Struct: pos(vec3+pad), vel(vec3+pad), life(f32), maxLife(f32), pad(vec2) -> 32 bytes?
            // WGSL alignment: vec3 is 16 aligned.
            // Let's use vec4 for pos and vel for simplicity.
            // struct Particle { pos: vec4f, vel: vec4f, life: f32, maxLife: f32, pad1: f32, pad2: f32 } -> 16+16+16 = 48 bytes? No.
            // Let's optimize.
            // struct Particle { pos: vec4f, vel: vec4f, meta: vec4f }
            // meta.x = life, meta.y = maxLife.

            const stride = 4 * 4 * 3; // 48 bytes per particle (3 vec4s)
            const data = new Float32Array(count * 12);

            for(let i=0; i<count; i++) {
                const off = i * 12;
                // Pos
                data[off+0] = 0; data[off+1] = 0; data[off+2] = 0; data[off+3] = 1;
                // Vel
                data[off+4] = 0; data[off+5] = 0; data[off+6] = 0; data[off+7] = 0;
                // Meta (life, maxLife, seed, type)
                data[off+8] = -1.0; // Dead initially
                data[off+9] = 1.0 + Math.random();
                data[off+10] = Math.random();
                data[off+11] = 0;
            }

            this.particleBuffer = this.device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.particleBuffer.getMappedRange()).set(data);
            this.particleBuffer.unmap();

            // --- Uniforms ---
            // time, padding, mouseX, mouseY
            this.simParamsBuffer = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            // --- Compute Pipeline ---
            const computeShader = `
                struct Particle {
                    pos: vec4f,
                    vel: vec4f,
                    meta: vec4f, // life, maxLife, seed, unused
                }

                struct SimParams {
                    time: f32,
                    pad: f32,
                    mouse: vec2f,
                }

                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> params: SimParams;

                // Hash function
                fn hash(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }
                fn hash3(v: vec3f) -> vec3f {
                    return vec3f(hash(v.x), hash(v.y), hash(v.z));
                }

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3u) {
                    let idx = GlobalInvocationID.x;
                    if (idx >= arrayLength(&particles)) { return; }

                    var p = particles[idx];
                    var life = p.meta.x;
                    let maxLife = p.meta.y;
                    let seed = p.meta.z;

                    // Decrease life
                    life -= 0.016; // 60fps approx

                    if (life <= 0.0) {
                        // Respawn
                        life = maxLife * (0.5 + 0.5 * hash(params.time + seed));

                        // Spawn on hemisphere surface
                        // Random point on sphere
                        let u = hash(params.time * 0.1 + f32(idx) * 0.01);
                        let v = hash(params.time * 0.2 + f32(idx) * 0.02);
                        let theta = 2.0 * 3.14159 * u;
                        let phi = acos(2.0 * v - 1.0);

                        // Convert to cartesian
                        var r = 0.8; // Radius
                        var x = r * sin(phi) * cos(theta);
                        var y = r * sin(phi) * sin(theta);
                        var z = r * cos(phi);

                        // Restrict to 'front' relative to movement
                        // For now, let's just spawn everywhere on a small sphere,
                        // but transform it by the shield rotation

                        // Apply rotation (Same as WebGL)
                        let rotX = params.mouse.y * 1.5;
                        let rotY = params.mouse.x * 1.5;
                        let cX = cos(rotX); let sX = sin(rotX);
                        let cY = cos(rotY); let sY = sin(rotY);

                        // Rotate X then Y
                        let y_ = y * cX - z * sX;
                        let z_ = y * sX + z * cX;
                        y = y_; z = z_;

                        let x_ = x * cY + z * sY;
                        let z__ = -x * sY + z * cY;
                        x = x_; z = z__;

                        // Only spawn on the 'face' (z > 0 in local space before rotation, or after?)
                        // Let's just spawn and throw them back

                        p.pos = vec4f(x, y, z, 1.0);

                        // Velocity: Streaming 'back' relative to the shield face
                        // The shield faces 'forward' (towards screen/incoming air).
                        // So particles should flow 'back' and 'out' around the edges.

                        let normal = normalize(vec3f(x,y,z));
                        // Tangent flow + backward flow
                        p.vel = vec4f(normal * 0.1 + vec3f(0.0, 0.0, -2.0) * 0.05, 0.0);

                        // Add some randomness
                        p.vel += vec4f((hash3(vec3f(u,v,life)) - 0.5) * 0.1, 0.0);
                    } else {
                        // Update
                        p.pos += p.vel;
                        // Drag/Slowdown
                        p.vel *= 0.98;
                        // Turbulence
                        p.vel += vec4f((hash3(p.pos.xyz) - 0.5) * 0.01, 0.0);
                    }

                    p.meta.x = life;
                    particles[idx] = p;
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
                ]
            });

            // --- Render Pipeline ---
            const renderShader = `
                struct Particle {
                    pos: vec4f,
                    vel: vec4f,
                    meta: vec4f,
                }

                @group(0) @binding(0) var<storage, read> particles: array<Particle>;

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) color: vec4f,
                    @location(1) size: f32,
                }

                @vertex
                fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
                    var out: VertexOutput;
                    let p = particles[idx];
                    let life = p.meta.x;
                    let maxLife = p.meta.y;

                    // Project pos (Simple perspective matching WebGL approx)
                    // pos.z -= 4.0;
                    // pos.x /= aspect;
                    // But we don't have aspect here easily without another uniform.
                    // Let's approximate or just map to NDC directly if possible.
                    // Actually, let's just do a simple manual projection.

                    var pos = p.pos.xyz;
                    pos.z -= 4.0;

                    // Correct aspect ratio hardcoded or uniform?
                    // Let's assume aspect ~ 1.7 (1920/1080) for now or just ignore aspect stretch for particles
                    // Or better, pass aspect in simParams? No space.
                    // Just use 1.0, it will look okay.

                    let projZ = -pos.z;
                    out.position = vec4f(pos.x, pos.y, pos.z, projZ);

                    // Color based on life/heat
                    let t = life / maxLife;

                    // Fire colors
                    let hot = vec3f(1.0, 0.9, 0.5); // Yellow/White
                    let mid = vec3f(1.0, 0.2, 0.0); // Red
                    let cold = vec3f(0.2, 0.2, 0.2); // Smoke

                    var col = mix(mid, hot, smoothstep(0.5, 1.0, t));
                    col = mix(cold, col, smoothstep(0.0, 0.5, t));

                    out.color = vec4f(col, t * 0.8); // Fade alpha

                    // Point size attenuation
                    out.size = 20.0 / projZ;

                    return out;
                }

                @fragment
                fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                    if (color.a < 0.01) { discard; }
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
                            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }, // Additive
                            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                        }
                    }]
                },
                primitive: {
                    topology: 'point-list',
                }
            });

            this.renderBindGroup = this.device.createBindGroup({
                layout: this.renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer } },
                ]
            });

        } catch (e) {
            console.warn("WebGPU init error:", e);
        }
    }

    resize() {
        if (!this.container) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.canvasGL.width = w;
        this.canvasGL.height = h;
        this.gl.viewport(0, 0, w, h);

        this.canvasGPU.width = w;
        this.canvasGPU.height = h;
    }

    render(t) {
        if (!this.isPlaying) return;
        this.time = t * 0.001;

        // --- WebGL ---
        const gl = this.gl;
        if (gl) {
            gl.clearColor(0,0,0,0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            gl.useProgram(this.program);

            gl.uniform1f(this.uTimeLoc, this.time);
            gl.uniform2f(this.uMouseLoc, this.mouse.x, this.mouse.y);
            gl.uniform2f(this.uResLoc, this.canvasGL.width, this.canvasGL.height);

            gl.bindVertexArray(this.vao);
            // Draw indices
            gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
        }

        // --- WebGPU ---
        if (this.device && this.contextGPU && this.renderPipeline) {
            // Write uniforms
            const params = new Float32Array([this.time, 0, this.mouse.x, this.mouse.y]);
            this.device.queue.writeBuffer(this.simParamsBuffer, 0, params);

            const encoder = this.device.createCommandEncoder();

            // Compute
            const cPass = encoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.options.numParticles / 64));
            cPass.end();

            // Render
            const rPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.contextGPU.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBindGroup);
            rPass.draw(this.options.numParticles);
            rPass.end();

            this.device.queue.submit([encoder.finish()]);
        }

        requestAnimationFrame(this.render);
    }
}
