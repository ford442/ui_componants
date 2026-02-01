/**
 * Encryption Lock Experiment
 * Combines WebGL2 (Holographic Rotating Rings) and WebGPU (Particle Key Injection).
 */

export class EncryptionLock {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.canvasSize = { width: 0, height: 0 };
        this.isDecrypting = false;
        this.decryptionProgress = 0.0; // 0.0 to 1.0

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;

        // We will store geometry for 3 rings
        this.rings = [
            { r: 1.5, tube: 0.05, speed: 0.5, axis: [1, 0, 0], vao: null, count: 0 },
            { r: 2.2, tube: 0.08, speed: -0.3, axis: [0, 1, 0], vao: null, count: 0 },
            { r: 3.0, tube: 0.12, speed: 0.2, axis: [0, 0, 1], vao: null, count: 0 }
        ];

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.simParamBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 40000;

        // Bindings
        this.handleResize = this.resize.bind(this);
        this.handleMouseDown = () => { this.isDecrypting = true; };
        this.handleMouseUp = () => { this.isDecrypting = false; };
        this.handleMouseLeave = () => { this.isDecrypting = false; };

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#000510'; // Dark Blue-Black

        // 1. WebGL2
        this.initWebGL2();

        // 2. WebGPU
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("EncryptionLock: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.resize();
        this.isActive = true;

        // Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousedown', this.handleMouseDown);
        this.container.addEventListener('mouseup', this.handleMouseUp);
        this.container.addEventListener('mouseleave', this.handleMouseLeave);
        this.container.addEventListener('touchstart', this.handleMouseDown);
        this.container.addEventListener('touchend', this.handleMouseUp);

        this.animate();
    }

    // ========================================================================
    // WebGL2
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Shader
        const vs = `#version 300 es
            in vec3 a_position;
            in vec3 a_normal;

            uniform float u_time;
            uniform mat4 u_view;
            uniform mat4 u_projection;
            uniform float u_rotationSpeed;
            uniform vec3 u_rotationAxis;
            uniform float u_progress; // 0 to 1

            out vec3 v_normal;
            out vec3 v_viewDir;
            out float v_progress;

            mat4 rotationMatrix(vec3 axis, float angle) {
                axis = normalize(axis);
                float s = sin(angle);
                float c = cos(angle);
                float oc = 1.0 - c;
                return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                            oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                            oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                            0.0,                                0.0,                                0.0,                                1.0);
            }

            void main() {
                // Rotate based on time, but if progress is high, slow down/stop at alignment
                // Alignment: effectively angle = 0 when progress = 1

                float angle = u_time * u_rotationSpeed;

                // When progress -> 1, we want angle -> 0 (or specific target)
                // Let's mix the continuous rotation with a fixed "Locked" rotation (0)
                angle = mix(angle, 0.0, u_progress * u_progress); // Ease in alignment

                mat4 model = rotationMatrix(u_rotationAxis, angle);

                vec4 worldPos = model * vec4(a_position, 1.0);
                gl_Position = u_projection * u_view * worldPos;

                v_normal = mat3(model) * a_normal;
                v_viewDir = normalize(-worldPos.xyz); // Camera at origin relative to world transform if view is simple
                v_progress = u_progress;
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            in vec3 v_normal;
            in vec3 v_viewDir;
            in float v_progress;

            out vec4 outColor;

            void main() {
                vec3 normal = normalize(v_normal);
                vec3 view = normalize(v_viewDir);

                // Rim lighting (Fresnel)
                float fresnel = pow(1.0 - max(dot(view, normal), 0.0), 3.0);

                // Colors
                vec3 colorLocked = vec3(0.0, 0.8, 1.0); // Cyan
                vec3 colorUnlocked = vec3(0.2, 1.0, 0.2); // Green

                vec3 baseColor = mix(colorLocked, colorUnlocked, v_progress);

                float alpha = fresnel * 0.8 + 0.1;

                // Scanline effect
                float scan = sin(gl_FragCoord.y * 0.1) * 0.1;

                outColor = vec4(baseColor + scan, alpha);
            }
        `;

        this.glProgram = this.createGLProgram(vs, fs);
        if (!this.glProgram) return;

        // Create Geometries
        this.rings.forEach(ring => {
            const { positions, normals, indices } = this.createTorus(ring.r, ring.tube, 64, 16);

            const vao = this.gl.createVertexArray();
            this.gl.bindVertexArray(vao);

            const posBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

            const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
            this.gl.enableVertexAttribArray(posLoc);
            this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

            const normBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, normBuffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(normals), this.gl.STATIC_DRAW);

            const normLoc = this.gl.getAttribLocation(this.glProgram, 'a_normal');
            this.gl.enableVertexAttribArray(normLoc);
            this.gl.vertexAttribPointer(normLoc, 3, this.gl.FLOAT, false, 0, 0);

            const idxBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
            this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

            ring.vao = vao;
            ring.count = indices.length;
        });

        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE); // Additive
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.depthMask(false); // Transparent objects don't write depth usually
    }

    createTorus(R, r, radialSegments, tubularSegments) {
        const positions = [];
        const normals = [];
        const indices = [];

        for (let j = 0; j <= radialSegments; j++) {
            for (let i = 0; i <= tubularSegments; i++) {
                const u = j / radialSegments * Math.PI * 2;
                const v = i / tubularSegments * Math.PI * 2;

                const centerX = R * Math.cos(u);
                const centerY = R * Math.sin(u);

                const x = (R + r * Math.cos(v)) * Math.cos(u);
                const y = (R + r * Math.cos(v)) * Math.sin(u);
                const z = r * Math.sin(v);

                positions.push(x, y, z);

                // Normal
                const nx = Math.cos(v) * Math.cos(u);
                const ny = Math.cos(v) * Math.sin(u);
                const nz = Math.sin(v);
                normals.push(nx, ny, nz);
            }
        }

        for (let j = 1; j <= radialSegments; j++) {
            for (let i = 1; i <= tubularSegments; i++) {
                const a = (tubularSegments + 1) * j + i - 1;
                const b = (tubularSegments + 1) * (j - 1) + i - 1;
                const c = (tubularSegments + 1) * (j - 1) + i;
                const d = (tubularSegments + 1) * j + i;

                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }
        return { positions, normals, indices };
    }

    createGLProgram(vsSource, fsSource) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        if (!this.gl.getShaderParameter(vs, this.gl.COMPILE_STATUS)) return null;

        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        if (!this.gl.getShaderParameter(fs, this.gl.COMPILE_STATUS)) return null;

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vs);
        this.gl.attachShader(prog, fs);
        this.gl.linkProgram(prog);
        return prog;
    }

    // ========================================================================
    // WebGPU
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
                vel : vec4f,
            }
            struct Uniforms {
                time : f32,
                dt : f32,
                progress : f32, // decryption progress
                padding : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            fn hash(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Behavior:
                // Progress low: Orbit chaotic at radius ~4.0
                // Progress high: Suck into center (Radius 0)

                let targetRadius = mix(4.0, 0.5, uniforms.progress);
                let speed = mix(1.0, 5.0, uniforms.progress);

                // Attraction to target radius
                let dist = length(p.pos.xyz);
                let dir = normalize(p.pos.xyz);

                // Tangent force (Orbit)
                let tangent = vec3f(-dir.y, dir.x, 0.0); // Simple Z-axis orbit
                if (abs(dir.z) > 0.9) { tangent = vec3f(1.0, 0.0, 0.0); } // Handle pole

                // Forces
                var force = vec3f(0.0);

                // Radial Spring
                force += -dir * (dist - targetRadius) * 2.0;

                // Tangent Vortex
                force += tangent * 2.0;

                // Random noise
                let seed = uniforms.time + f32(index) * 0.01;
                force += vec3f(hash(seed)-0.5, hash(seed+1.0)-0.5, hash(seed+2.0)-0.5) * 5.0;

                // Integration
                p.vel = p.vel * 0.95 + vec4f(force * uniforms.dt * speed, 0.0);
                p.pos = p.pos + p.vel * uniforms.dt;

                particles[index] = p;
            }
        `;

        // Render Shader
        const renderShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }
            struct Uniforms {
                time : f32,
                dt : f32,
                progress : f32,
                aspect : f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(@location(0) pos : vec4f, @location(1) vel : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Camera
                let viewPos = pos.xyz;
                let camZ = 10.0;
                let z = camZ - viewPos.z;

                output.position = vec4f(viewPos.x / z / uniforms.aspect * 2.5, viewPos.y / z * 2.5, 0.0, 1.0);

                let speed = length(vel.xyz);

                // Color: Blue -> Green -> White
                var col = mix(vec3f(0.0, 0.5, 1.0), vec3f(0.0, 1.0, 0.2), uniforms.progress);
                col += vec3f(speed * 0.1); // Add brightness with speed

                output.color = vec4f(col, 1.0);
                output.position.w = 1.0; // Point size
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Buffers
        const pData = new Float32Array(this.numParticles * 8); // pos(4), vel(4)
        for(let i=0; i<this.numParticles; i++) {
            const r = 4.0 + (Math.random()-0.5)*2.0;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            pData[i*8+0] = r * Math.sin(phi) * Math.cos(theta);
            pData[i*8+1] = r * Math.sin(phi) * Math.sin(theta);
            pData[i*8+2] = r * Math.cos(phi);
            pData[i*8+3] = 1.0; // Life
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
                targets: [{ format, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
                } }]
            },
            primitive: { topology: 'point-list' }
        });

        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute; bottom: 20px; right: 20px;
            background: rgba(200, 50, 50, 0.9); color: white;
            padding: 8px 16px; border-radius: 8px; font-family: monospace;
            pointer-events: none;
        `;
        msg.textContent = "WebGPU Not Supported - Core Only";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Logic
    // ========================================================================

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        if (this.glCanvas) {
            this.glCanvas.width = w * dpr;
            this.glCanvas.height = h * dpr;
            this.gl.viewport(0, 0, w * dpr, h * dpr);
        }
        if (this.gpuCanvas) {
            this.gpuCanvas.width = w * dpr;
            this.gpuCanvas.height = h * dpr;
        }
        this.canvasSize = { width: w, height: h };
    }

    animate() {
        if (!this.isActive) return;

        const dt = 0.016;
        const time = (Date.now() - this.startTime) * 0.001;

        // Update Logic
        if (this.isDecrypting) {
            this.decryptionProgress = Math.min(this.decryptionProgress + dt * 0.5, 1.0);
        } else {
            this.decryptionProgress = Math.max(this.decryptionProgress - dt * 1.0, 0.0);
        }

        // WebGL
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);
            this.gl.clearColor(0.0, 0.0, 0.0, 0.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            const uTime = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const uView = this.gl.getUniformLocation(this.glProgram, 'u_view');
            const uProj = this.gl.getUniformLocation(this.glProgram, 'u_projection');
            const uProgress = this.gl.getUniformLocation(this.glProgram, 'u_progress');

            this.gl.uniform1f(uTime, time);
            this.gl.uniform1f(uProgress, this.decryptionProgress);

            // Camera (Static)
            const view = new Float32Array([
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -10, 1
            ]);
            this.gl.uniformMatrix4fv(uView, false, view);

            // Projection
            const aspect = this.glCanvas.width / this.glCanvas.height;
            const fov = 45 * Math.PI / 180;
            const f = 1.0 / Math.tan(fov / 2);
            const zNear = 0.1; const zFar = 100.0;
            const proj = new Float32Array([
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (zFar + zNear) / (zNear - zFar), -1,
                0, 0, (2 * zFar * zNear) / (zNear - zFar), 0
            ]);
            this.gl.uniformMatrix4fv(uProj, false, proj);

            // Draw Rings
            this.rings.forEach(ring => {
                const uRotSpeed = this.gl.getUniformLocation(this.glProgram, 'u_rotationSpeed');
                const uRotAxis = this.gl.getUniformLocation(this.glProgram, 'u_rotationAxis');

                this.gl.uniform1f(uRotSpeed, ring.speed);
                this.gl.uniform3fv(uRotAxis, ring.axis);

                this.gl.bindVertexArray(ring.vao);
                this.gl.drawElements(this.gl.TRIANGLES, ring.count, this.gl.UNSIGNED_SHORT, 0);
            });
        }

        // WebGPU
        if (this.device && this.renderPipeline) {
            const aspect = this.canvasSize.width / this.canvasSize.height;
            const params = new Float32Array([time, dt, this.decryptionProgress, aspect]);
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

        // Remove listeners
        this.container.removeEventListener('mousedown', this.handleMouseDown);
        this.container.removeEventListener('mouseup', this.handleMouseUp);
        this.container.removeEventListener('mouseleave', this.handleMouseLeave);
        this.container.removeEventListener('touchstart', this.handleMouseDown);
        this.container.removeEventListener('touchend', this.handleMouseUp);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();
        this.container.innerHTML = '';
    }
}
