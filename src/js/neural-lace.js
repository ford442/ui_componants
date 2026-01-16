/**
 * Neural Lace Experiment
 * Hybrid WebGL2 + WebGPU
 * - WebGL2: Renders a static 3D lattice structure (Instanced Cubes).
 * - WebGPU: Simulates "neural impulses" (particles) traveling along/through the lattice.
 */

export class NeuralLaceExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;

        // Interaction
        this.mouseX = 0;
        this.mouseY = 0;
        this.pulseStrength = 0.0;

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
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 50000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleClick = this.onClick.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#020205'; // Deep dark background

        // 1. WebGL2 for the Structural Lattice
        this.initWebGL2();

        // 2. WebGPU for the Particles
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("NeuralLace: WebGPU init failed:", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('click', this.handleClick);

        this.resize();
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    }

    onClick() {
        this.pulseStrength = 2.0;
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Lattice Structure)
    // ========================================================================
    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2', { alpha: false });
        if (!this.gl) return;

        // Enable depth test
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        // Define a simple Cube Geometry (Lines)
        // 8 corners, 12 edges (24 vertices for lines)
        const vertices = new Float32Array([
            // Bottom square
            -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,  0.5, -0.5,  0.5,
             0.5, -0.5,  0.5, -0.5, -0.5,  0.5,
            -0.5, -0.5,  0.5, -0.5, -0.5, -0.5,
            // Top square
            -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,
             0.5,  0.5, -0.5,  0.5,  0.5,  0.5,
             0.5,  0.5,  0.5, -0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5, -0.5,  0.5, -0.5,
            // Pillars
            -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,
             0.5, -0.5, -0.5,  0.5,  0.5, -0.5,
             0.5, -0.5,  0.5,  0.5,  0.5,  0.5,
            -0.5, -0.5,  0.5, -0.5,  0.5,  0.5,
        ]);

        // Instance Positions (A 3D grid)
        const instances = [];
        const dim = 4; // 4x4x4 grid
        const spacing = 1.5;
        for(let x = -dim; x <= dim; x++) {
            for(let y = -dim; y <= dim; y++) {
                for(let z = -dim; z <= dim; z++) {
                    if (Math.abs(x) + Math.abs(y) + Math.abs(z) < 2) continue; // Hollow out center slightly
                    instances.push(x * spacing, y * spacing, z * spacing);
                }
            }
        }
        this.instanceCount = instances.length / 3;

        const vsSource = `#version 300 es
            in vec3 a_position;
            in vec3 a_instancePos;

            uniform mat4 u_viewProjection;
            uniform float u_time;

            out vec3 v_pos;
            out float v_dist;

            void main() {
                vec3 pos = a_position + a_instancePos;

                // Subtle breathing animation
                float dist = length(a_instancePos);
                float wave = sin(u_time * 0.5 + dist * 0.5) * 0.05;
                pos += normalize(a_instancePos + vec3(0.001)) * wave;

                gl_Position = u_viewProjection * vec4(pos, 1.0);
                v_pos = pos;
                v_dist = dist;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;
            in vec3 v_pos;
            in float v_dist;
            out vec4 outColor;

            void main() {
                // Fade out distant nodes
                float alpha = 1.0 - smoothstep(5.0, 15.0, v_dist);
                vec3 col = vec3(0.1, 0.4, 0.5); // Tealish
                outColor = vec4(col, alpha * 0.3);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        // Vertex Buffer
        const vBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        // Instance Buffer
        const iBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, iBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(instances), this.gl.STATIC_DRAW);
        const instLoc = this.gl.getAttribLocation(this.glProgram, 'a_instancePos');
        this.gl.enableVertexAttribArray(instLoc);
        this.gl.vertexAttribPointer(instLoc, 3, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(instLoc, 1);
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

        // WGSL Shaders
        const computeShader = `
            struct Particle {
                pos: vec4f, // xyz, w=phase
                vel: vec4f, // xyz, w=padding
            }
            struct Params {
                dt: f32,
                time: f32,
                pulse: f32,
                pad: f32,
            }
            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> params: Params;

            // Pseudo-random
            fn hash(u: u32) -> f32 {
                var p = u;
                p = (p << 13u) ^ p;
                return (1.0 - f32((p * (p * p * 15731u + 789221u) + 1376312589u) & 0x7fffffffu) / 1073741824.0);
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3u) {
                let i = GlobalInvocationID.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];

                // Move
                p.pos.x += p.vel.x * params.dt;
                p.pos.y += p.vel.y * params.dt;
                p.pos.z += p.vel.z * params.dt;

                // Neural-like movement: abruptly change direction to align with axes
                // Based on noise/time
                let noiseVal = hash(i + u32(params.time * 50.0));
                if (noiseVal > 0.98) {
                    // Pick a random axis
                    let axis = hash(i + u32(params.time * 100.0));
                    p.vel = vec4f(0.0);
                    let speed = 2.0 + params.pulse * 5.0; // Burst speed on pulse
                    if (axis < 0.33) { p.vel.x = speed * sign(hash(i)-0.5); }
                    else if (axis < 0.66) { p.vel.y = speed * sign(hash(i+1u)-0.5); }
                    else { p.vel.z = speed * sign(hash(i+2u)-0.5); }
                }

                // Constrain to box
                let bound = 8.0;
                if (abs(p.pos.x) > bound) { p.pos.x = -p.pos.x; }
                if (abs(p.pos.y) > bound) { p.pos.y = -p.pos.y; }
                if (abs(p.pos.z) > bound) { p.pos.z = -p.pos.z; }

                // Attract to center slightly
                p.pos.x -= p.pos.x * 0.1 * params.dt;
                p.pos.y -= p.pos.y * 0.1 * params.dt;
                p.pos.z -= p.pos.z * 0.1 * params.dt;

                particles[i] = p;
            }
        `;

        const drawShader = `
            struct Uniforms {
                mvp: mat4x4f,
                color1: vec4f,
                color2: vec4f,
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
                let mixVal = clamp(speed * 0.5, 0.0, 1.0);

                // Mix between dim purple and bright cyan based on speed
                out.color = mix(uniforms.color1, uniforms.color2, mixVal);

                // Point size trick not fully supported in pure WGSL without extension,
                // but points render as 1px squares usually.
                // We rely on sheer number of particles (50k).

                return out;
            }

            @fragment
            fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Create Buffers
        const pSize = 32; // 8 floats
        const pData = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            pData[i*8+0] = (Math.random()-0.5) * 10;
            pData[i*8+1] = (Math.random()-0.5) * 10;
            pData[i*8+2] = (Math.random()-0.5) * 10;
            pData[i*8+3] = Math.random(); // phase
        }
        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, pData);

        this.simParamBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 64 + 16 + 16, // Mat4 + 2 Vec4s
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Compute Pipeline
        const cModule = this.device.createShaderModule({ code: computeShader });
        const cLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [cLayout] }),
            compute: { module: cModule, entryPoint: 'main' },
        });
        this.computeBindGroup = this.device.createBindGroup({
            layout: cLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } },
            ]
        });

        // Render Pipeline
        const rModule = this.device.createShaderModule({ code: drawShader });
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
                    arrayStride: pSize, stepMode: 'vertex',
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
        if(this.glCanvas) {
            this.glCanvas.width = w * dpr;
            this.glCanvas.height = h * dpr;
            this.gl.viewport(0, 0, w * dpr, h * dpr);
        }
        if(this.gpuCanvas) {
            this.gpuCanvas.width = w * dpr;
            this.gpuCanvas.height = h * dpr;
        }
    }

    animate() {
        if(!this.isActive) return;
        const time = (Date.now() - this.startTime) * 0.001;

        // Decay pulse
        this.pulseStrength *= 0.95;

        // Camera Logic
        const aspect = this.container.clientWidth / this.container.clientHeight;
        const fov = 60 * Math.PI / 180;
        const f = 1.0 / Math.tan(fov/2);
        const far = 100.0, near = 0.1;

        // Projection
        const proj = new Float32Array([
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far+near)/(near-far), -1,
            0, 0, (2*far*near)/(near-far), 0
        ]);

        // View (Orbit)
        const radius = 12.0;
        const cx = Math.sin(time*0.1 + this.mouseX) * radius;
        const cz = Math.cos(time*0.1 + this.mouseX) * radius;
        const cy = this.mouseY * 5.0;

        // Simple LookAt 0,0,0
        const zAxis = [cx, cy, cz]; // eye - target(0)
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

        // MVP = View * Proj (Column Major multiply)
        const mvp = new Float32Array(16);
        for(let r=0; r<4; r++) {
            for(let c=0; c<4; c++) {
                let s=0;
                for(let k=0; k<4; k++) s += view[k*4+r] * proj[c*4+k];
                mvp[c*4+r] = s;
            }
        }

        // Render WebGL
        if(this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(
                this.gl.getUniformLocation(this.glProgram, 'u_viewProjection'),
                false, mvp
            );
            this.gl.uniform1f(
                this.gl.getUniformLocation(this.glProgram, 'u_time'),
                time
            );
            this.gl.clearColor(0,0,0,0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            // Draw Instanced Lines (cube is 24 verts)
            this.gl.drawArraysInstanced(this.gl.LINES, 0, 24, this.instanceCount);
        }

        // Render WebGPU
        if(this.device && this.renderPipeline) {
            // Update Sim Params
            const params = new Float32Array([0.016, time, this.pulseStrength, 0]);
            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            // Update Uniforms
            // mvp (64) + color1 (16) + color2 (16)
            const uniformsData = new Float32Array(16 + 4 + 4);
            uniformsData.set(mvp, 0);
            uniformsData.set([0.2, 0.0, 0.5, 0.5], 16); // Purple low energy
            uniformsData.set([0.0, 1.0, 1.0, 1.0], 20); // Cyan high energy
            this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformsData);

            const enc = this.device.createCommandEncoder();

            // Compute
            const cPass = enc.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
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
            // Even though we don't strictly use a vertex buffer to pull attrs (we use storage),
            // sometimes explicit vertex buffer calls or dummy draw calls are needed.
            // But we can pull from storage using gl_VertexID (vertex_index) if mapped correctly.
            // Wait, my shader uses @location(0) pPos, which implies vertex buffer inputs.
            // But I bound storage buffer in Compute, and I need to bind it as Vertex Buffer here?
            // Yes, I defined buffers: [{ arrayStride... }] in pipeline.
            rPass.setVertexBuffer(0, this.particleBuffer);
            rPass.draw(this.numParticles);
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
        this.container.removeEventListener('click', this.handleClick);
        if(this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if(this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}
