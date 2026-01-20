export class DysonSwarmExperiment {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            particleCount: config.particleCount || 50000,
            sunColor: [1.0, 0.6, 0.1], // Orange-ish
            ...config
        };

        this.isPlaying = true;
        this.time = 0;
        this.mouse = { x: 0, y: 0 };
        this.animationFrameId = null;

        // Create Canvases
        this.canvasGL = document.createElement('canvas'); // For the Sun
        this.canvasGPU = document.createElement('canvas'); // For the Swarm

        // Setup container
        this.container.style.position = 'relative';
        this.container.style.width = '100%';
        this.container.style.height = '100%';
        this.container.style.backgroundColor = '#000000';
        this.container.style.overflow = 'hidden';

        // Setup canvases
        [this.canvasGL, this.canvasGPU].forEach(canvas => {
            canvas.style.position = 'absolute';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            canvas.style.pointerEvents = 'none'; // Let events pass through if needed
            this.container.appendChild(canvas);
        });

        // Layering
        this.canvasGL.style.zIndex = '1';
        this.canvasGPU.style.zIndex = '2';

        // Event Listeners
        this.handleResize = this.handleResize.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.render = this.render.bind(this);

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('touchmove', this.handleMouseMove);

        // Init
        this.init();
    }

    async init() {
        this.initWebGL();
        await this.initWebGPU();
        this.handleResize();
        this.render(0);
    }

    handleMouseMove(e) {
        // Support both mouse and touch
        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        const rect = this.container.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
            const x = (clientX - rect.left) / rect.width;
            const y = (clientY - rect.top) / rect.height;
            // Map to -1..1 range
            this.mouse.x = x * 2 - 1;
            this.mouse.y = -(y * 2 - 1);
        }
    }

    handleResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width === 0 || height === 0) return;

        this.canvasGL.width = width;
        this.canvasGL.height = height;
        if (this.gl) this.gl.viewport(0, 0, width, height);

        this.canvasGPU.width = width;
        this.canvasGPU.height = height;
        // WebGPU attachment is resized automatically by current texture request but good to keep canvas size synced
    }

    // --- WebGL2 (The Sun) ---
    initWebGL() {
        this.gl = this.canvasGL.getContext('webgl2');
        if (!this.gl) {
            console.warn("WebGL2 not supported");
            return;
        }
        const gl = this.gl;

        // Vertex Shader (Full screen quad)
        const vsSource = `#version 300 es
        in vec2 a_position;
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
        }`;

        // Fragment Shader (SDF Sun)
        const fsSource = `#version 300 es
        precision highp float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec3 u_color;
        out vec4 outColor;

        float random(in vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }

        float noise(in vec2 st) {
            vec2 i = floor(st);
            vec2 f = fract(st);
            float a = random(i);
            float b = random(i + vec2(1.0, 0.0));
            float c = random(i + vec2(0.0, 1.0));
            float d = random(i + vec2(1.0, 1.0));
            vec2 u = f * f * (3.0 - 2.0 * f);
            return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }

        float fbm(in vec2 st) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 0.0;
            for (int i = 0; i < 5; i++) {
                value += amplitude * noise(st);
                st *= 2.0;
                amplitude *= 0.5;
            }
            return value;
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / min(u_resolution.x, u_resolution.y);

            float dist = length(uv);
            float sunRadius = 0.3;

            // Corona/Glow
            float glow = 0.05 / abs(dist - sunRadius);
            glow = pow(glow, 1.2);

            // Sun Surface
            float surface = smoothstep(sunRadius + 0.01, sunRadius - 0.01, dist);

            // Dynamic plasma noise
            float n = fbm(uv * 10.0 + u_time * 0.5);
            vec3 sunColor = u_color + vec3(n * 0.3, n * 0.1, 0.0);

            vec3 finalColor = vec3(0.0);

            // Add surface
            finalColor += sunColor * surface;

            // Add glow (orange/red)
            finalColor += vec3(1.0, 0.4, 0.1) * glow * 0.8;

            // Black background (premultiplied for blending with WebGPU canvas if needed, but this is background layer)
            outColor = vec4(finalColor, 1.0);
        }`;

        this.program = this.createProgram(gl, vsSource, fsSource);
        this.posLoc = gl.getAttribLocation(this.program, 'a_position');
        this.timeLoc = gl.getUniformLocation(this.program, 'u_time');
        this.resLoc = gl.getUniformLocation(this.program, 'u_resolution');
        this.colorLoc = gl.getUniformLocation(this.program, 'u_color');

        // Quad data
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(this.posLoc);
        gl.vertexAttribPointer(this.posLoc, 2, gl.FLOAT, false, 0, 0);
        this.vao = vao;
    }

    createProgram(gl, vs, fs) {
        const vShader = this.compileShader(gl, gl.VERTEX_SHADER, vs);
        const fShader = this.compileShader(gl, gl.FRAGMENT_SHADER, fs);
        const prog = gl.createProgram();
        gl.attachShader(prog, vShader);
        gl.attachShader(prog, fShader);
        gl.linkProgram(prog);
        return prog;
    }

    compileShader(gl, type, source) {
        const s = gl.createShader(type);
        gl.shaderSource(s, source);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(s));
        }
        return s;
    }

    // --- WebGPU (The Swarm) ---
    async initWebGPU() {
        if (!navigator.gpu) return;

        try {
            this.adapter = await navigator.gpu.requestAdapter();
            if (!this.adapter) return;
            this.device = await this.adapter.requestDevice();
        } catch (e) {
            console.error("WebGPU init error:", e);
            return;
        }

        this.contextGPU = this.canvasGPU.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.contextGPU.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });

        const numParticles = this.config.particleCount;

        // Data: pos (vec4: x,y,z,life), vel (vec4: vx,vy,vz,pad)
        // 8 floats = 32 bytes per particle
        const data = new Float32Array(numParticles * 8);

        for (let i = 0; i < numParticles; i++) {
            const idx = i * 8;

            // Initial position: Disk distribution
            const theta = Math.random() * Math.PI * 2;
            const r = 0.4 + Math.random() * 0.5; // Radius 0.4 to 0.9
            const y = (Math.random() - 0.5) * 0.1; // Flat disk with slight thickness

            data[idx] = Math.cos(theta) * r;     // x
            data[idx + 1] = y;                   // y (vertical)
            data[idx + 2] = Math.sin(theta) * r; // z
            data[idx + 3] = Math.random();       // life/phase

            // Velocity: Tangential for orbit
            const speed = 0.5 / Math.sqrt(r); // Kepler-ish
            const vx = -Math.sin(theta) * speed;
            const vz = Math.cos(theta) * speed;

            data[idx + 4] = vx;
            data[idx + 5] = 0.0;
            data[idx + 6] = vz;
            data[idx + 7] = 0.0; // padding
        }

        this.particleBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(data);
        this.particleBuffer.unmap();

        // Uniforms: time(f32), pad(f32), mouse(vec2) => 16 bytes
        this.uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Shader
        const computeCode = `
            struct Particle {
                pos : vec4f, // x, y, z, life
                vel : vec4f, // vx, vy, vz, pad
            }

            struct Uniforms {
                time : f32,
                pad : f32,
                mouse : vec2f,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3u) {
                let i = global_id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];
                var pos = p.pos.xyz;
                var vel = p.vel.xyz;

                // 1. Gravity (Center Sun)
                let dist = length(pos);
                let dir = normalize(-pos);
                let force = dir * (0.2 / (dist * dist + 0.01)); // F ~ 1/r^2

                // 2. Mouse Influence (Perturbation)
                // Project mouse 2D to some effect in 3D
                // Mouse x rotates the plane, mouse y tilts
                let mouseEffect = uniforms.mouse;
                if (length(mouseEffect) > 0.01) {
                    // Add some noise/turbulence based on mouse
                    vel.y += mouseEffect.y * 0.001;

                    // Spin acceleration
                    let tan = vec3f(-dir.z, 0.0, dir.x);
                    vel += tan * mouseEffect.x * 0.001;
                }

                // Apply forces
                vel += force * 0.001;

                // Update position
                pos += vel;

                // Reset if too close (consumed) or too far (lost)
                if (dist < 0.1 || dist > 2.0) {
                     // Respawn at outer rim
                     let angle = uniforms.time * 0.1 + f32(i);
                     let r = 0.9;
                     pos = vec3f(cos(angle)*r, (fract(sin(f32(i))*43.0)-0.5)*0.1, sin(angle)*r);

                     // Tangential velocity
                     let speed = 0.5 / sqrt(r);
                     vel = vec3f(-sin(angle)*speed, 0.0, cos(angle)*speed);
                }

                p.pos = vec4f(pos, p.pos.w);
                p.vel = vec4f(vel, p.vel.w);
                particles[i] = p;
            }
        `;

        this.computeModule = this.device.createShaderModule({ code: computeCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.computeModule, entryPoint: 'main' }
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // Render Shader
        const renderCode = `
            struct Particle {
                pos : vec4f,
                vel : vec4f,
            }
            @group(0) @binding(0) var<storage, read> particles : array<Particle>;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
                let p = particles[vertex_index];

                // 3D Projection (Simple Perspective)
                // Camera at (0, 1.5, 2.0) looking at (0,0,0)
                let pos = p.pos.xyz;

                // Rotate view slightly over time
                let viewTime = 0.0; // Fixed view for now, or pass time if needed

                // Simple View Matrix (Camera raised and tilted down)
                // y' = y*cos - z*sin
                // z' = y*sin + z*cos
                let angle = 0.7; // ~40 degrees
                let y_rot = pos.y * cos(angle) - pos.z * sin(angle);
                let z_rot = pos.y * sin(angle) + pos.z * cos(angle);
                let x_rot = pos.x;

                // Translate Z
                let z_final = z_rot - 2.0;

                // Perspective division
                let projX = x_rot / -z_final;
                let projY = y_rot / -z_final;

                // Aspect correction (assuming square for simplicity or fix in CPU)
                // To fix aspect, we should pass aspect ratio.
                // For now, let's assume ~1.0 or close enough for "art"

                var output : VertexOutput;
                output.position = vec4f(projX, projY, 0.0, 1.0);

                // Color based on velocity/energy
                let speed = length(p.vel.xyz);
                let energy = smoothstep(0.4, 0.8, speed); // Normalized speed estimate

                // Blue/Cyan to White/Gold
                output.color = mix(vec4f(0.0, 0.5, 1.0, 0.6), vec4f(1.0, 0.9, 0.5, 0.8), energy);

                // Point size trick
                output.position.w = 1.0;
                // In WebGPU point size is fixed to 1.0 unless using specific features or quads
                // We'll just draw 1px points for the "swarm" look.

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        this.renderModule = this.device.createShaderModule({ code: renderCode });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.renderModule,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.presentationFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }, // Additive blending
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.particleBuffer } }]
        });
    }

    render(t) {
        if (!this.isPlaying) return;

        // Loop
        this.animationFrameId = requestAnimationFrame(this.render);

        const dt = (t - this.time) * 0.001;
        this.time = t;
        const timeSec = t * 0.001;

        // --- Render WebGL (Background Sun) ---
        if (this.gl) {
            const gl = this.gl;
            gl.viewport(0, 0, this.canvasGL.width, this.canvasGL.height);
            // Clear to black
            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(this.program);
            gl.uniform1f(this.timeLoc, timeSec);
            gl.uniform2f(this.resLoc, this.canvasGL.width, this.canvasGL.height);
            gl.uniform3fv(this.colorLoc, this.config.sunColor);

            gl.bindVertexArray(this.vao);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }

        // --- Render WebGPU (Foreground Swarm) ---
        if (this.device && this.contextGPU) {
            // Update Uniforms
            const uniData = new Float32Array([timeSec, 0.0, this.mouse.x, this.mouse.y]);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniData);

            const cmd = this.device.createCommandEncoder();

            // Compute
            const cPass = cmd.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.config.particleCount / 64));
            cPass.end();

            // Render
            const rPass = cmd.beginRenderPass({
                colorAttachments: [{
                    view: this.contextGPU.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 }, // Transparent!
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBindGroup);
            rPass.draw(this.config.particleCount);
            rPass.end();

            this.device.queue.submit([cmd.finish()]);
        }
    }

    destroy() {
        this.isPlaying = false;
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

        window.removeEventListener('resize', this.handleResize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('touchmove', this.handleMouseMove);

        if (this.gl) {
            this.gl.deleteProgram(this.program);
            this.gl.deleteBuffer(this.vao); // Simplified cleanup
        }

        if (this.device) {
            this.device.destroy();
        }

        this.container.innerHTML = '';
    }
}
