/**
 * Genetic Splicing Experiment
 * Combines WebGL2 (DNA Helix Wireframe) and WebGPU (Enzyme Swarm).
 */

export class GeneticSplicingExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.isMouseDown = false;
        this.canvasSize = { width: 0, height: 0 };
        this.numParticles = options.numParticles || 50000;

        // Geometry Data
        this.helixVertices = null; // Float32Array
        this.helixIndices = null; // Uint16Array

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndexCount = 0;

        // WebGPU State (Placeholder for now)
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;

        // Bind handlers
        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseDown = this.onMouseDown.bind(this);
        this.handleMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000505';

        // Generate Geometry (DNA Helix)
        this.generateGeometry();

        // 1. Initialize WebGL2
        this.initWebGL2();

        // 2. Initialize WebGPU
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("GeneticSplicing: WebGPU initialization error:", e);
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
        window.addEventListener('mousedown', this.handleMouseDown);
        window.addEventListener('mouseup', this.handleMouseUp);
        window.addEventListener('touchstart', this.handleMouseDown);
        window.addEventListener('touchend', this.handleMouseUp);
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = (e.clientX - rect.left) / rect.width * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height * 2 - 1);
    }

    onMouseDown() { this.isMouseDown = true; }
    onMouseUp() { this.isMouseDown = false; }

    generateGeometry() {
        const vertices = [];
        const indices = [];

        const numSegments = 100;
        const radius = 1.0;
        const height = 8.0;
        const turns = 4;

        for (let i = 0; i <= numSegments; i++) {
            const t = i / numSegments;
            const angle = t * Math.PI * 2 * turns;
            const y = (t - 0.5) * height; // Center vertically

            const x1 = Math.cos(angle) * radius;
            const z1 = Math.sin(angle) * radius;

            const x2 = Math.cos(angle + Math.PI) * radius; // Opposite strand
            const z2 = Math.sin(angle + Math.PI) * radius;

            // Helix 1 Vertex
            vertices.push(x1, y, z1);
            // Helix 2 Vertex
            vertices.push(x2, y, z2);

            // Indices
            const idx = i * 2;
            if (i > 0) {
                // Strand 1 line
                indices.push(idx - 2, idx);
                // Strand 2 line
                indices.push(idx - 1, idx + 1);
            }

            // Rung (Base pair) connecting strands
            if (i % 2 === 0) { // Add rung every other segment
                 indices.push(idx, idx + 1);
            }
        }

        this.helixVertices = new Float32Array(vertices);
        this.helixIndices = new Uint16Array(indices);
        this.glIndexCount = indices.length;
    }

    // ========================================================================
    // WebGL2 (DNA Helix)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;`;
        this.container.appendChild(this.glCanvas);
        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        const vs = `#version 300 es
            in vec3 a_position;
            uniform mat4 u_mvp;
            out float v_y;
            void main() {
                v_y = a_position.y;
                gl_Position = u_mvp * vec4(a_position, 1.0);
            }
        `;
        const fs = `#version 300 es
            precision highp float;
            in float v_y;
            uniform float u_time;
            out vec4 outColor;
            void main() {
                // Gradient color based on height and time
                float t = (v_y + 4.0) / 8.0; // Normalize y to 0..1

                vec3 col1 = vec3(0.0, 1.0, 1.0); // Cyan
                vec3 col2 = vec3(1.0, 0.0, 1.0); // Magenta

                vec3 finalColor = mix(col1, col2, t + sin(u_time + t * 5.0) * 0.2);

                float alpha = 0.6 + 0.2 * sin(u_time * 3.0 + v_y);
                outColor = vec4(finalColor, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const vb = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vb);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.helixVertices, this.gl.STATIC_DRAW);
        const loc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 3, this.gl.FLOAT, false, 0, 0);

        const ib = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, ib);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, this.helixIndices, this.gl.STATIC_DRAW);
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);

        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) {
             console.error('VS Error:', this.gl.getShaderInfoLog(vs));
             return null;
        }

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);

        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) {
             console.error('FS Error:', this.gl.getShaderInfoLog(fs));
             return null;
        }

        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);

        if (!this.gl.getProgramParameter(p, this.gl.LINK_STATUS)) {
             console.error('Link Error:', this.gl.getProgramInfoLog(p));
             return null;
        }

        return p;
    }

    // ========================================================================
    // WebGPU (Enzyme Swarm)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none;`;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' });

        const wgslCommon = `
            struct Particle {
                pos : vec4f,
                vel : vec4f,
            }
            struct Params {
                mvp : mat4x4f,
                mouse : vec2f,
                time : f32,
                isMouseDown : f32,
            }
        `;

        const computeShader = `
            ${wgslCommon}
            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : Params;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                var p = particles[idx];

                // Forces
                var force = vec3f(0.0);

                // 1. Attraction to Helix Radius (r=1.2)
                let r = 1.2;
                let centerDist = length(p.pos.xz);
                let centerDir = normalize(p.pos.xz);
                force.x += (centerDir.x * r - p.pos.x) * 1.0; // Spring to radius
                force.z += (centerDir.y * r - p.pos.z) * 1.0; // Note: centerDir is 2D (x, z->y)

                // 2. Spiral Motion
                // Tangent vector: (-z, 0, x)
                let tangent = vec3f(-centerDir.y, 0.0, centerDir.x);
                force += tangent * 2.0;

                // 3. Vertical Motion (Oscillate)
                // Keep within height -4 to 4
                if (p.pos.y > 4.0) { force.y -= 1.0; }
                if (p.pos.y < -4.0) { force.y += 1.0; }

                // 4. Mouse Splicing Beam
                if (params.isMouseDown > 0.5) {
                    // Map mouse (NDC) to roughly world space
                    // This is approximate. Assuming view is looking at origin.
                    // Mouse X affects rotation or position?
                    // Let's make a "Beam" through the center that pushes particles out
                    let beamDist = length(p.pos.xz - vec2f(params.mouse.x * 5.0, 0.0));
                    if (beamDist < 2.0) {
                        force += normalize(p.pos.xyz) * 20.0;
                        p.vel += vec4f(normalize(p.pos.xyz) * 0.5, 0.0);
                    }
                }

                // Integration
                p.vel = p.vel * 0.95 + vec4f(force, 0.0) * 0.016;
                p.pos = p.pos + p.vel * 0.016;

                // Reset if lost
                if (length(p.pos.xyz) > 20.0) {
                     p.pos = vec4f(0.0, 0.0, 0.0, 1.0);
                     p.vel = vec4f(0.0);
                }

                particles[idx] = p;
            }
        `;

        const renderShader = `
            ${wgslCommon}
            @group(0) @binding(1) var<uniform> params : Params;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f, @builtin(vertex_index) vIdx : u32) -> VertexOutput {
                var out : VertexOutput;
                out.position = params.mvp * vec4f(pos.xyz, 1.0);

                let speed = length(vel.xyz);
                let t = smoothstep(0.0, 2.0, speed);

                // Enzyme colors: Green/Yellow
                let cSlow = vec3f(0.0, 1.0, 0.5);
                let cFast = vec3f(1.0, 1.0, 0.2);

                out.color = vec4f(mix(cSlow, cFast, t), 1.0);

                // Size adjustment? (Simulated by brightness/alpha blending)
                return out;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const initP = new Float32Array(this.numParticles * 8);
        for(let i=0; i<this.numParticles; i++) {
            initP[i*8+0] = (Math.random()-0.5) * 4.0;
            initP[i*8+1] = (Math.random()-0.5) * 8.0;
            initP[i*8+2] = (Math.random()-0.5) * 4.0;
            initP[i*8+3] = 1.0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: initP.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, initP);

        this.simParamBuffer = this.device.createBuffer({
            size: 80, // mat4(64) + vec2(8) + time(4) + mouse(4)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Pipeline
        const moduleCompute = this.device.createShaderModule({ code: computeShader });
        const moduleRender = this.device.createShaderModule({ code: renderShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.simParamBuffer } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: moduleCompute, entryPoint: 'main' }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: moduleRender,
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
                module: moduleRender,
                entryPoint: 'fs_main',
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' },
            depthStencil: undefined
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const el = document.createElement('div');
        el.className = 'webgpu-error';
        el.style.cssText = 'position:absolute; top:10px; left:10px; color:red; background:rgba(0,0,0,0.8); padding:5px;';
        el.textContent = "WebGPU Not Available - Swarm Disabled";
        this.container.appendChild(el);
    }

    // ========================================================================
    // Loop
    // ========================================================================

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.canvasSize.width = w;
        this.canvasSize.height = h;

        const dw = Math.floor(w * dpr);
        const dh = Math.floor(h * dpr);

        if(this.glCanvas) {
            this.glCanvas.width = dw;
            this.glCanvas.height = dh;
            this.gl.viewport(0, 0, dw, dh);
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;
        const aspect = this.canvasSize.width / this.canvasSize.height;

        // Camera
        const fov = 45 * Math.PI / 180;
        const zNear = 0.1;
        const zFar = 100.0;
        const f = 1.0 / Math.tan(fov / 2);
        const rangeInv = 1 / (zNear - zFar);

        const p = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (zNear + zFar) * rangeInv, -1,
            0, 0, zNear * zFar * rangeInv * 2, 0
        ];

        // View Matrix (Rotate around)
        const radius = 12.0;
        const cx = Math.sin(time * 0.3) * radius;
        const cz = Math.cos(time * 0.3) * radius;
        const cy = Math.sin(time * 0.1) * 2.0;

        const eye = [cx, cy, cz];
        const center = [0, 0, 0];
        const up = [0, 1, 0];

        const z0 = eye[0]-center[0], z1 = eye[1]-center[1], z2 = eye[2]-center[2];
        const len = 1/Math.sqrt(z0*z0 + z1*z1 + z2*z2);
        const zx=z0*len, zy=z1*len, zz=z2*len;
        const x0=up[1]*zz - up[2]*zy, x1=up[2]*zx - up[0]*zz, x2=up[0]*zy - up[1]*zx;
        const lenX = 1/Math.sqrt(x0*x0+x1*x1+x2*x2);
        const xx=x0*lenX, xy=x1*lenX, xz=x2*lenX;
        const y0=zy*xz - zz*xy, y1=zz*xx - zx*xz, y2=zx*xy - zy*xx;
        const lenY = 1/Math.sqrt(y0*y0+y1*y1+y2*y2);
        const yx=y0*lenY, yy=y1*lenY, yz=y2*lenY;

        const v = [
            xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx*eye[0]+xy*eye[1]+xz*eye[2]),
            -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
            -(zx*eye[0]+zy*eye[1]+zz*eye[2]),
            1
        ];

        const mvp = new Float32Array(16);
        for(let row=0; row<4; row++) {
            for(let col=0; col<4; col++) {
                let sum = 0;
                for(let k=0; k<4; k++) {
                    sum += p[k*4 + row] * v[col*4 + k];
                }
                mvp[col*4 + row] = sum;
            }
        }

        // WebGL Draw
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Enable blending for transparency
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

            const loc = this.gl.getUniformLocation(this.glProgram, 'u_mvp');
            this.gl.uniformMatrix4fv(loc, false, mvp);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_time'), time);

            this.gl.clearColor(0,0,0,0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.LINES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. WebGPU Draw
        if (this.device && this.renderPipeline) {
            // Write Uniforms
            const params = new Float32Array(20);
            params.set(mvp, 0);
            params[16] = this.mouse.x;
            params[17] = this.mouse.y;
            params[18] = time;
            params[19] = this.isMouseDown ? 1.0 : 0.0;

            this.device.queue.writeBuffer(this.simParamBuffer, 0, params);

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const cPass = commandEncoder.beginComputePass();
            cPass.setPipeline(this.computePipeline);
            cPass.setBindGroup(0, this.computeBindGroup);
            cPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            cPass.end();

            // Render
            const rPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });
            rPass.setPipeline(this.renderPipeline);
            rPass.setBindGroup(0, this.computeBindGroup);
            rPass.draw(this.numParticles);
            rPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('mousemove', this.handleMouseMove);
        window.removeEventListener('mousedown', this.handleMouseDown);
        window.removeEventListener('mouseup', this.handleMouseUp);
        window.removeEventListener('touchstart', this.handleMouseDown);
        window.removeEventListener('touchend', this.handleMouseUp);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.GeneticSplicingExperiment = GeneticSplicingExperiment;
}
