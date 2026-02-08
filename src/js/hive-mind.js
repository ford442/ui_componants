/**
 * Hive Mind Experiment
 * Hybrid WebGL2 + WebGPU
 * - WebGL2: Renders a hexagonal grid surface using Instanced Drawing.
 * - WebGPU: Simulates "thought particles" flowing over the grid.
 */

export class HiveMindExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // Interaction
        this.mouseX = 0;
        this.mouseY = 0;
        this.isHovering = false;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.instanceCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.renderBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.particleCount = options.particleCount || 50000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseEnter = () => { this.isHovering = true; };
        this.handleMouseLeave = () => { this.isHovering = false; };

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020205';

        // 1. WebGL2 Hex Grid
        this.initWebGL2();

        // 2. WebGPU Particles
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("HiveMind: WebGPU init failed:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mouseenter', this.handleMouseEnter);
        this.container.addEventListener('mouseleave', this.handleMouseLeave);

        this.resize();
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Hex Grid)
    // ========================================================================
    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2', { alpha: false });
        if (!this.gl) return;

        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE); // Additive blending for glow

        // Hexagon Geometry (Triangle Fan)
        // Center (0,0) + 6 vertices
        const size = 0.95; // Slightly less than 1.0 gap
        const hexVerts = [0, 0, 0]; // Center
        for (let i = 0; i <= 6; i++) {
            const angle = (i * 60) * Math.PI / 180;
            hexVerts.push(Math.cos(angle) * size, 0, Math.sin(angle) * size);
        }
        const vertices = new Float32Array(hexVerts);

        // Instance Positions (Axial Coordinates q, r)
        const instances = [];
        const radius = 15;
        for (let q = -radius; q <= radius; q++) {
            const r1 = Math.max(-radius, -q - radius);
            const r2 = Math.min(radius, -q + radius);
            for (let r = r1; r <= r2; r++) {
                instances.push(q, r);
            }
        }
        this.instanceCount = instances.length / 2;

        const vsSource = `#version 300 es
            in vec3 a_position;
            in vec2 a_gridPos; // q, r

            uniform mat4 u_viewProjection;
            uniform float u_time;
            uniform vec2 u_mouse;

            out vec3 v_pos;
            out float v_dist;
            out float v_activation;

            // Hex to World conversion
            vec3 hexToWorld(vec2 hex) {
                float size = 1.0;
                float x = size * 3.0/2.0 * hex.x;
                float z = size * sqrt(3.0) * (hex.y + hex.x/2.0);
                return vec3(x, 0.0, z);
            }

            void main() {
                vec3 instanceCenter = hexToWorld(a_gridPos);

                // Wave effect
                float distFromCenter = length(instanceCenter.xz);
                float wave = sin(u_time * 1.5 - distFromCenter * 0.3) * 1.0;
                instanceCenter.y = wave;

                // Mouse interaction (Activation)
                // Project mouse to approximate world plane
                // Simple approx: mouse ranges -1 to 1, map to world -20 to 20
                vec2 mouseWorld = u_mouse * 20.0;
                float dMouse = distance(mouseWorld, instanceCenter.xz);
                float activation = smoothstep(8.0, 0.0, dMouse); // 8.0 radius

                // Raise active hexes
                instanceCenter.y += activation * 2.0;

                vec3 pos = instanceCenter + a_position;

                gl_Position = u_viewProjection * vec4(pos, 1.0);
                v_pos = pos;
                v_dist = distFromCenter;
                v_activation = activation;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec3 v_pos;
            in float v_dist;
            in float v_activation;
            out vec4 outColor;

            void main() {
                // Base color: Dark Blue
                vec3 baseColor = vec3(0.0, 0.1, 0.2);

                // Active color: Bright Cyan/White
                vec3 activeColor = vec3(0.0, 0.8, 1.0);

                // Mix based on activation
                vec3 col = mix(baseColor, activeColor, v_activation * 0.8 + 0.1);

                // Edge glow (simple barycentric-like fake or distance from center)
                // Since we are using TRIANGLE_FAN, getting real edges is tricky without barycentrics.
                // We can use distance from instance center? v_pos - center?
                // But we don't have center passed to FS.
                // Let's just use a simple rim lighting or gradient.

                // Simple depth fade
                float alpha = 1.0 - smoothstep(15.0, 30.0, v_dist);

                outColor = vec4(col, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        // Vertices
        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        // Instances
        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(instances), this.gl.STATIC_DRAW);
        const gridLoc = this.gl.getAttribLocation(this.glProgram, 'a_gridPos');
        this.gl.enableVertexAttribArray(gridLoc);
        this.gl.vertexAttribPointer(gridLoc, 2, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(gridLoc, 1);
    }

    createGLProgram(vs, fs) {
        const createShader = (type, source) => {
            const s = this.gl.createShader(type);
            this.gl.shaderSource(s, source);
            this.gl.compileShader(s);
            if (!this.gl.getShaderParameter(s, this.gl.COMPILE_STATUS)) {
                console.error(this.gl.getShaderInfoLog(s));
                return null;
            }
            return s;
        };
        const p = this.gl.createProgram();
        const v = createShader(this.gl.VERTEX_SHADER, vs);
        const f = createShader(this.gl.FRAGMENT_SHADER, fs);
        if(!v || !f) return null;
        this.gl.attachShader(p, v);
        this.gl.attachShader(p, f);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Particles)
    // ========================================================================
    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2; pointer-events: none;
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            this.gpuCanvas.remove();
            return false;
        }
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device, format: format, alphaMode: 'premultiplied'
        });

        const computeShader = `
            struct Particle {
                pos: vec4f,
                vel: vec4f,
            }
            struct Params {
                dt: f32,
                time: f32,
                mouseX: f32,
                mouseY: f32,
            }
            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: Params;

            // Simple noise function
            fn hash(p: vec3f) -> f32 {
                let p3  = fract(p * .1031);
                let dotVal = dot(p3, vec3f(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
                return fract((p3.x + p3.y) * p3.z);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3u) {
                let i = GlobalInvocationID.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Flow Vector (Curl Noise approx or just sine waves)
                let scale = 0.2;
                let angle = sin(p.pos.x * scale + params.time) + cos(p.pos.z * scale + params.time);
                let flow = vec3f(cos(angle), 0.0, sin(angle));

                // Mouse Attraction
                let mouseWorld = vec3f(params.mouseX * 20.0, 0.0, params.mouseY * 20.0); // Match GL coords
                let toMouse = mouseWorld - vec3f(p.pos.x, 0.0, p.pos.z);
                let dist = length(toMouse);
                var attract = vec3f(0.0);
                if (dist < 15.0) {
                     attract = normalize(toMouse) * (15.0 - dist) * 0.1;
                }

                // Update Vel
                let targetVel = flow * 2.0 + attract;
                p.vel = mix(p.vel, vec4f(targetVel, 0.0), 0.05);

                // Update Pos
                p.pos += p.vel * params.dt;

                // Height sticking (Wave function matching GL)
                let distFromCenter = length(vec2f(p.pos.x, p.pos.z));
                let wave = sin(params.time * 1.5 - distFromCenter * 0.3) * 1.0;

                // Add mouse lift
                let mouseFactor = smoothstep(8.0, 0.0, distance(vec2f(p.pos.x, p.pos.z), vec2f(mouseWorld.x, mouseWorld.z)));

                p.pos.y = wave + mouseFactor * 2.0 + 0.5; // +0.5 to float above grid

                // Bounds wrap
                let bound = 25.0;
                if (p.pos.x > bound) { p.pos.x = -bound; }
                if (p.pos.x < -bound) { p.pos.x = bound; }
                if (p.pos.z > bound) { p.pos.z = -bound; }
                if (p.pos.z < -bound) { p.pos.z = bound; }

                particles[i] = p;
            }
        `;

        const renderShader = `
            struct Uniforms {
                mvp: mat4x4f,
            }
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;

            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs_main(@location(0) pPos: vec4f, @location(1) pVel: vec4f) -> VertexOut {
                var out: VertexOut;
                out.pos = uniforms.mvp * vec4f(pPos.xyz, 1.0);

                let speed = length(pVel.xyz);
                // Color based on speed
                out.color = mix(vec4f(0.0, 0.5, 1.0, 1.0), vec4f(1.0, 1.0, 1.0, 1.0), clamp(speed * 0.2, 0.0, 1.0));

                out.pos.w = 1.0; // Point size fix hack if needed? Not in WGSL clip space w, but actually gl_PointSize.
                // WebGPU doesn't have point size control in vertex shader unless topology is point-list and implementation supports it (often fixed to 1px).
                // We'll rely on quantity.

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const pData = new Float32Array(this.particleCount * 8);
        for(let i=0; i<this.particleCount; i++) {
            pData[i*8+0] = (Math.random() - 0.5) * 50;
            pData[i*8+1] = 0;
            pData[i*8+2] = (Math.random() - 0.5) * 50;
            pData[i*8+3] = 0;
            pData[i*8+4] = 0; // vel
            pData[i*8+5] = 0;
            pData[i*8+6] = 0;
            pData[i*8+7] = 0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, pData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 64, // Mat4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Pipeline
        const cModule = this.device.createShaderModule({ code: computeShader });
        const cLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [cLayout] }),
            compute: { module: cModule, entryPoint: 'main' }
        });
        this.computeBindGroup = this.device.createBindGroup({
            layout: cLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        // Render Pipeline
        const rModule = this.device.createShaderModule({ code: renderShader });
        const rLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }]
        });
        this.renderBindGroup = this.device.createBindGroup({
            layout: rLayout,
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [rLayout] }),
            vertex: {
                module: rModule, entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32, stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: rModule, entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.textContent = "WebGPU Not Supported - Running WebGL2 Mode Only";
        msg.style.cssText = "position:absolute; bottom:10px; right:10px; color:red; font-family:monospace;";
        this.container.appendChild(msg);
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        if (this.glCanvas) {
            this.glCanvas.width = w * dpr;
            this.glCanvas.height = h * dpr;
            this.gl.viewport(0, 0, w * dpr, h * dpr);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = w * dpr;
            this.gpuCanvas.height = h * dpr;
        }
    }

    animate() {
        if (!this.isActive) return;
        const time = (Date.now() - this.startTime) * 0.001;

        // Camera Logic
        const aspect = this.container.clientWidth / this.container.clientHeight;
        const fov = 60 * Math.PI / 180;
        const f = 1.0 / Math.tan(fov / 2);
        const far = 100.0, near = 0.1;

        const proj = new Float32Array([
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far+near)/(near-far), -1,
            0, 0, (2*far*near)/(near-far), 0
        ]);

        // Orbit Camera
        const radius = 20.0;
        const cx = Math.sin(time * 0.2) * radius;
        const cz = Math.cos(time * 0.2) * radius;
        const cy = 15.0;

        const zAxis = [cx, cy, cz];
        let len = Math.hypot(zAxis[0], zAxis[1], zAxis[2]);
        zAxis[0]/=len; zAxis[1]/=len; zAxis[2]/=len;

        const up = [0, 1, 0];
        const xAxis = [
            up[1]*zAxis[2] - up[2]*zAxis[1],
            up[2]*zAxis[0] - up[0]*zAxis[2],
            up[0]*zAxis[1] - up[1]*zAxis[0]
        ];
        len = Math.hypot(xAxis[0], xAxis[1], xAxis[2]);
        xAxis[0]/=len; xAxis[1]/=len; xAxis[2]/=len;

        const yAxis = [
            zAxis[1]*xAxis[2] - zAxis[2]*xAxis[1],
            zAxis[2]*xAxis[0] - zAxis[0]*xAxis[2],
            zAxis[0]*xAxis[1] - zAxis[1]*xAxis[0]
        ];

        const view = [
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -(xAxis[0]*cx + xAxis[1]*cy + xAxis[2]*cz),
            -(yAxis[0]*cx + yAxis[1]*cy + yAxis[2]*cz),
            -(zAxis[0]*cx + zAxis[1]*cy + zAxis[2]*cz),
            1
        ];

        const mvp = new Float32Array(16);
        for(let r=0; r<4; r++) {
            for(let c=0; c<4; c++) {
                let s=0;
                for(let k=0; k<4; k++) s += view[k*4+r] * proj[c*4+k];
                mvp[c*4+r] = s;
            }
        }

        // WebGL2 Draw
        if (this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(
                this.gl.getUniformLocation(this.glProgram, 'u_viewProjection'),
                false, mvp
            );
            this.gl.uniform1f(
                this.gl.getUniformLocation(this.glProgram, 'u_time'),
                time
            );
            this.gl.uniform2f(
                this.gl.getUniformLocation(this.glProgram, 'u_mouse'),
                this.mouseX, this.mouseY
            );
            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            // Draw Instanced Hexagons
            // 8 vertices in Fan: Center, 6 corners, 1 repeat
            this.gl.drawArraysInstanced(this.gl.TRIANGLE_FAN, 0, 8, this.instanceCount);
        }

        // WebGPU Draw
        if (this.device && this.renderPipeline) {
            // Update Sim Params
            const params = new Float32Array([0.016, time, this.mouseX, this.mouseY]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            // Update Uniforms
            this.device.queue.writeBuffer(this.uniformBuffer, 0, mvp);

            const enc = this.device.createCommandEncoder();

            // Compute
            const cPass = enc.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
            cPass.end();

            // Render
            const tView = this.context.getCurrentTexture().createView();
            const rPass = enc.beginRenderPass({
                colorAttachments: [{
                    view: tView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear', storeOp: 'store'
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.renderBindGroup);
            rPass.setVertexBuffer(0, this.particleBuffer);
            rPass.draw(this.particleCount);
            rPass.end();

            this.device.queue.submit([enc.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('mouseenter', this.handleMouseEnter);
        this.container.removeEventListener('mouseleave', this.handleMouseLeave);
        if(this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if(this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}
