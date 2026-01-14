
export class ChronoDial {
    constructor(container) {
        console.log("ChronoDial: Initializing...");
        this.container = container;
        this.rotation = 0;
        this.targetRotation = 0;
        this.rotationSpeed = 0;
        this.width = container.clientWidth;
        this.height = container.clientHeight;

        // Interaction state
        this.isDragging = false;
        this.lastX = 0;

        // Initialize layers
        this.initLayers();

        // Initialize WebGL2 (The Dial)
        this.initWebGL();

        // Initialize WebGPU (The Time Dust)
        if (navigator.gpu) {
            this.initWebGPU();
        } else {
            console.warn("ChronoDial: WebGPU not supported - falling back to WebGL2 only");
        }

        this.attachEvents();

        // Bind animate loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    initLayers() {
        // WebGL2 Layer (Bottom)
        this.canvasGL = document.createElement('canvas');
        this.canvasGL.style.position = 'absolute';
        this.canvasGL.style.top = '0';
        this.canvasGL.style.left = '0';
        this.canvasGL.style.width = '100%';
        this.canvasGL.style.height = '100%';
        this.canvasGL.style.zIndex = '1';
        this.container.appendChild(this.canvasGL);

        // WebGPU Layer (Top)
        this.canvasGPU = document.createElement('canvas');
        this.canvasGPU.style.position = 'absolute';
        this.canvasGPU.style.top = '0';
        this.canvasGPU.style.left = '0';
        this.canvasGPU.style.width = '100%';
        this.canvasGPU.style.height = '100%';
        this.canvasGPU.style.zIndex = '2';
        this.canvasGPU.style.pointerEvents = 'none'; // Let events pass through to bottom or container
        this.container.appendChild(this.canvasGPU);
    }

    // ==========================================
    // WebGL2 Implementation
    // ==========================================

    initWebGL() {
        this.gl = this.canvasGL.getContext('webgl2', { alpha: false });
        if (!this.gl) return;

        const gl = this.gl;
        this.resizeWebGL();

        // Vertex Shader
        const vsSource = `#version 300 es
        in vec3 a_position;
        in vec3 a_normal;

        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        uniform float u_time;

        out vec3 v_normal;
        out vec3 v_pos;
        out float v_glow;

        void main() {
            vec4 pos = u_model * vec4(a_position, 1.0);
            v_pos = pos.xyz;
            v_normal = mat3(u_model) * a_normal;

            // Add some "breathing" to the ring
            float breath = sin(u_time * 2.0 + pos.y * 5.0) * 0.02;
            pos.xyz += a_normal * breath;

            gl_Position = u_projection * u_view * pos;

            // Calculate edge glow based on view angle
            vec3 viewDir = normalize(-pos.xyz); // Simplified view dir
            v_glow = pow(1.0 - abs(dot(normalize(v_normal), viewDir)), 3.0);
        }`;

        // Fragment Shader
        const fsSource = `#version 300 es
        precision highp float;

        in vec3 v_normal;
        in vec3 v_pos;
        in float v_glow;

        uniform float u_rotationSpeed;

        out vec4 outColor;

        void main() {
            vec3 baseColor = vec3(0.1, 0.1, 0.15);
            vec3 accentColor = vec3(0.0, 0.8, 1.0);

            // Simple lighting
            vec3 lightDir = normalize(vec3(1.0, 1.0, 2.0));
            float diff = max(dot(normalize(v_normal), lightDir), 0.0);

            // Grid pattern on the ring
            float grid = sin(v_pos.x * 20.0) * sin(v_pos.y * 20.0);
            float gridLine = smoothstep(0.9, 1.0, grid);

            // Mix colors
            vec3 color = baseColor + (diff * 0.1);
            color += accentColor * v_glow * (1.0 + abs(u_rotationSpeed) * 5.0); // Glow intensity increases with speed
            color += accentColor * gridLine * 0.5;

            outColor = vec4(color, 1.0);
        }`;

        this.programGL = this.createProgram(gl, vsSource, fsSource);

        // Create Torus Geometry
        const torus = this.createTorus(1.5, 0.3, 30, 50);

        this.vaoGL = gl.createVertexArray();
        gl.bindVertexArray(this.vaoGL);

        const posBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(torus.vertices), gl.STATIC_DRAW);
        const posLoc = gl.getAttribLocation(this.programGL, 'a_position');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        const normBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(torus.normals), gl.STATIC_DRAW);
        const normLoc = gl.getAttribLocation(this.programGL, 'a_normal');
        gl.enableVertexAttribArray(normLoc);
        gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);

        const idxBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(torus.indices), gl.STATIC_DRAW);

        this.indexCountGL = torus.indices.length;

        // Locations
        this.uModelLoc = gl.getUniformLocation(this.programGL, 'u_model');
        this.uViewLoc = gl.getUniformLocation(this.programGL, 'u_view');
        this.uProjLoc = gl.getUniformLocation(this.programGL, 'u_projection');
        this.uTimeLoc = gl.getUniformLocation(this.programGL, 'u_time');
        this.uRotSpeedLoc = gl.getUniformLocation(this.programGL, 'u_rotationSpeed');

        gl.enable(gl.DEPTH_TEST);
    }

    // ==========================================
    // WebGPU Implementation
    // ==========================================

    async initWebGPU() {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.warn("ChronoDial: WebGPU adapter not found.");
            return;
        }

        this.device = await adapter.requestDevice();
        this.context = this.canvasGPU.getContext('webgpu');
        this.resizeWebGPU();

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        this.numParticles = 20000;

        // Shader Module
        const shaderModule = this.device.createShaderModule({
            code: `
                struct Particle {
                    pos: vec2f,
                    vel: vec2f,
                    life: f32,
                    seed: f32, // pad
                }

                struct Uniforms {
                    time: f32,
                    rotationSpeed: f32,
                    rotation: f32,
                    screenRatio: f32,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                @group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

                // Random function
                fn rand(n: f32) -> f32 {
                    return fract(sin(n) * 43758.5453123);
                }

                @compute @workgroup_size(64)
                fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
                    let i = global_id.x;
                    if (i >= arrayLength(&particles)) { return; }

                    var p = particles[i];

                    // Rotation influence
                    let c = cos(uniforms.rotationSpeed * 0.1);
                    let s = sin(uniforms.rotationSpeed * 0.1);

                    // Orbit logic
                    let center = vec2f(0.0, 0.0);
                    let diff = p.pos - center;
                    let dist = length(diff);

                    // Tangential force from rotation
                    let tangent = vec2f(-diff.y, diff.x);

                    // Apply velocity
                    p.vel += tangent * uniforms.rotationSpeed * 0.01 / (dist + 0.1);
                    p.vel *= 0.98; // Friction

                    // Return to center gravity
                    p.vel -= diff * 0.001;

                    p.pos += p.vel;

                    // Reset if out of bounds or random life
                    p.life -= 0.005 + abs(uniforms.rotationSpeed) * 0.01;

                    if (p.life <= 0.0 || dist > 2.0) {
                        // Respawn in a ring
                        let angle = rand(uniforms.time + f32(i)) * 6.28;
                        let r = 1.3 + rand(uniforms.time * 2.0 + f32(i)) * 0.4;
                        p.pos = vec2f(cos(angle) * r, sin(angle) * r);
                        p.vel = vec2f(0.0, 0.0);
                        p.life = 1.0;
                        // Initial velocity along the tangent
                        let initialTan = vec2f(-sin(angle), cos(angle));
                        p.vel += initialTan * uniforms.rotationSpeed * 0.5;
                    }

                    particles[i] = p;
                }

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) color: vec4f,
                }

                @vertex
                fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
                    let p = particles[instanceIndex];

                    // Simple point expansion
                    let angle = f32(vertexIndex) / 6.0 * 6.28;
                    let radius = 0.01 * p.life; // Size fades with life

                    let posOffset = vec2f(cos(angle), sin(angle)) * radius;

                    // Adjust aspect ratio
                    let finalPos = p.pos + posOffset;
                    let correctedPos = vec2f(finalPos.x / uniforms.screenRatio, finalPos.y);

                    var out: VertexOutput;
                    out.position = vec4f(correctedPos, 0.0, 1.0);

                    // Color based on speed/rotation
                    let energy = length(p.vel) * 10.0 + abs(uniforms.rotationSpeed) * 5.0;
                    out.color = vec4f(0.2, 0.8, 1.0, p.life) * (1.0 + energy);
                    return out;
                }

                @fragment
                fn fragmentMain(@location(0) color: vec4f) -> @location(0) vec4f {
                    return color;
                }
            `
        });

        // Pipeline Setup
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'computeMain' }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list' }
        });

        // Buffers
        const particleData = new Float32Array(this.numParticles * 6); // pos(2), vel(2), life(1), pad(1)
        // Initialize randomly
        for(let i=0; i<this.numParticles; i++) {
            const r = 1.5;
            const theta = Math.random() * Math.PI * 2;
            particleData[i*6 + 0] = Math.cos(theta) * r;
            particleData[i*6 + 1] = Math.sin(theta) * r;
            particleData[i*6 + 4] = Math.random(); // life
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);

        // Uniform Buffer
        this.uniformBufferSize = 16; // 4 floats
        this.uniformBuffer = this.device.createBuffer({
            size: this.uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Bind Groups
        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } }
            ]
        });

        this.isWebGPUReady = true;
    }

    // ==========================================
    // Utils & Loop
    // ==========================================

    createTorus(radius, tube, radialSegments, tubularSegments) {
        const vertices = [];
        const normals = [];
        const indices = [];

        for (let j = 0; j <= radialSegments; j++) {
            for (let i = 0; i <= tubularSegments; i++) {
                const u = i / tubularSegments * Math.PI * 2;
                const v = j / radialSegments * Math.PI * 2;

                const cx = radius + tube * Math.cos(v);
                const cy = tube * Math.sin(v);

                const x = cx * Math.cos(u);
                const y = cx * Math.sin(u);
                const z = cy; // Laying flat on XY plane initially, but we might want Z up?
                // Let's keep it flat on XY for now as our camera is looking down Z.

                vertices.push(x, y, z);

                // Normal
                const nx = Math.cos(v) * Math.cos(u);
                const ny = Math.cos(v) * Math.sin(u);
                const nz = Math.sin(v);
                normals.push(nx, ny, nz);
            }
        }

        for (let j = 1; j <= radialSegments; j++) {
            for (let i = 1; i <= tubularSegments; i++) {
                const a = (tubularSegments + 1) * j + i;
                const b = (tubularSegments + 1) * (j - 1) + i;
                const c = (tubularSegments + 1) * (j - 1) + i - 1;
                const d = (tubularSegments + 1) * j + i - 1;

                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }

        return { vertices, normals, indices };
    }

    createProgram(gl, vs, fs) {
        const createShader = (type, src) => {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, src);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.error(gl.getShaderInfoLog(shader));
                return null;
            }
            return shader;
        };
        const p = gl.createProgram();
        gl.attachShader(p, createShader(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, createShader(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p);
        return p;
    }

    resizeWebGL() {
        this.canvasGL.width = this.width;
        this.canvasGL.height = this.height;
        this.gl.viewport(0, 0, this.width, this.height);
    }

    resizeWebGPU() {
        this.canvasGPU.width = this.width;
        this.canvasGPU.height = this.height;
    }

    attachEvents() {
        const onStart = (x) => {
            this.isDragging = true;
            this.lastX = x;
        };
        const onMove = (x) => {
            if (this.isDragging) {
                const dx = x - this.lastX;
                this.targetRotation += dx * 0.01;
                // Add momentum to speed
                this.rotationSpeed += dx * 0.005;
                this.lastX = x;
            }
        };
        const onEnd = () => {
            this.isDragging = false;
        };

        this.container.addEventListener('mousedown', e => onStart(e.clientX));
        window.addEventListener('mousemove', e => onMove(e.clientX));
        window.addEventListener('mouseup', onEnd);

        this.container.addEventListener('touchstart', e => onStart(e.touches[0].clientX));
        window.addEventListener('touchmove', e => onMove(e.touches[0].clientX));
        window.addEventListener('touchend', onEnd);

        window.addEventListener('resize', () => {
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;
            this.resizeWebGL();
            if (this.isWebGPUReady) this.resizeWebGPU();
        });
    }

    perspective(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ]);
    }

    rotateY(angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return new Float32Array([
            c, 0, -s, 0,
            0, 1, 0, 0,
            s, 0, c, 0,
            0, 0, 0, 1
        ]);
    }

    rotateX(angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return new Float32Array([
            1, 0, 0, 0,
            0, c, s, 0,
            0, -s, c, 0,
            0, 0, 0, 1
        ]);
    }

    multiply(a, b) {
        const out = new Float32Array(16);
        for(let r=0; r<4; ++r) {
            for(let c=0; c<4; ++c) {
                let sum = 0;
                for(let k=0; k<4; ++k) sum += a[r*4+k] * b[k*4+c]; // Row-major logic but GL expects col-major...
                // Wait, standard matrix mult: out[row][col] = sum(a[row][k] * b[k][col])
                // My arrays are flat.
                // If I want standard mult:
                // out[c*4 + r] (col-major index)
                // Let's just use a simple verified loop for Col-Major matrices
                // Actually, let's keep it simple: assume a and b are col-major.
            }
        }
        // Re-implement standard col-major mul
        for(let i=0; i<4; i++) { // col
             for(let j=0; j<4; j++) { // row
                 let s = 0;
                 for(let k=0; k<4; k++) s += a[k*4 + j] * b[i*4 + k]; // b[col i][row k], a[col k][row j]
                 out[i*4 + j] = s;
             }
        }
        return out;
    }

    animate(time) {
        const t = time * 0.001;

        // Physics
        if (!this.isDragging) {
            this.rotationSpeed *= 0.95; // Friction
            this.targetRotation += this.rotationSpeed;
        }
        // Smooth rotation
        this.rotation += (this.targetRotation - this.rotation) * 0.1;

        // Render WebGL
        if (this.gl) {
            this.gl.clearColor(0, 0, 0, 0); // Transparent background for GL layer too? No, bottom layer.
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.useProgram(this.programGL);

            // Projection
            const aspect = this.width / this.height;
            const proj = this.perspective(45 * Math.PI/180, aspect, 0.1, 100.0);

            // View (Camera)
            const view = new Float32Array([
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -5, 1
            ]);

            // Model (Rotation)
            // Initial tilt + user rotation
            const tilt = this.rotateX(0.5); // Tilt 30 deg
            const spin = this.rotateY(this.rotation); // Spin around Y (which is now tilted local Y?)
            // Actually, if I multiply Spin * Tilt, I spin around world Y.
            // If I multiply Tilt * Spin, I spin around local Y.
            // Let's do Tilt * Spin
            const model = this.multiply(spin, tilt); // Actually, let's just use rotationY for simplicity first.
            // My multiply function might be suspect. Let's just do manual simple matrix construction.

            // Manual Model Matrix: Rotate Y(rotation) * Rotate X(0.5)
            const cy = Math.cos(this.rotation);
            const sy = Math.sin(this.rotation);
            const cx = Math.cos(0.5);
            const sx = Math.sin(0.5);

            // R = Ry * Rx
            // Ry = [ cy 0 sx 0, 0 1 0 0, -sy 0 cy 0, 0 0 0 1 ] - Col Major
            // Rx = [ 1 0 0 0, 0 cx sx 0, 0 -sx cx 0, 0 0 0 1 ]

            // Mat4( col0, col1, col2, col3 )
            // col0 = (cy, 0, -sy, 0)
            // col1 = (sy*sx, cx, cy*sx, 0)
            // col2 = (sy*cx, -sx, cy*cx, 0)
            // col3 = (0, 0, 0, 1)

            const finalModel = new Float32Array([
                cy, 0, -sy, 0,
                sy*sx, cx, cy*sx, 0,
                sy*cx, -sx, cy*cx, 0,
                0, 0, 0, 1
            ]);

            this.gl.uniformMatrix4fv(this.uModelLoc, false, finalModel);
            this.gl.uniformMatrix4fv(this.uViewLoc, false, view);
            this.gl.uniformMatrix4fv(this.uProjLoc, false, proj);
            this.gl.uniform1f(this.uTimeLoc, t);
            this.gl.uniform1f(this.uRotSpeedLoc, this.rotationSpeed);

            this.gl.bindVertexArray(this.vaoGL);
            this.gl.drawElements(this.gl.TRIANGLES, this.indexCountGL, this.gl.UNSIGNED_SHORT, 0);
        }

        // Render WebGPU
        if (this.isWebGPUReady) {
            // Update Uniforms
            const uniforms = new Float32Array([
                t,
                this.rotationSpeed * 100.0, // Scale up effect
                this.rotation,
                this.width / this.height
            ]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute Pass
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            computePass.end();

            // Render Pass
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
            // We use the particle buffer as a storage buffer in vertex shader via BindGroup,
            // OR we can use it as a Vertex Buffer if we structured it that way.
            // In my shader I used `@group(0) @binding(1) var<storage> particles`.
            // So I don't need `setVertexBuffer`. I just draw instances.
            // But wait, to draw instances I need to call `draw`.
            // I'll use `draw(6, numParticles)` -> 6 verts per particle (2 triangles) or just points?
            // My vertex shader uses `@builtin(vertex_index)` to generate a quad/hex.
            renderPass.setBindGroup(0, this.computeBindGroup); // Same BG for both
            renderPass.draw(6, this.numParticles);

            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        requestAnimationFrame(this.animate);
    }
}
