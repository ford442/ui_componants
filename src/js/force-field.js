/**
 * Force Field Experiment
 * Combines WebGL2 (Central Artifact) and WebGPU (Compute-driven Force Field).
 */

class ForceFieldExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;

        // State
        this.isActive = false;
        this.startTime = Date.now();
        this.animationId = null;
        this.mouseX = 0;
        this.mouseY = 0;
        this.isHovering = false;

        // WebGL2 State
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVao = null;
        this.glIndexCount = 0;

        // WebGPU State
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.computePipeline = null;
        this.computeBindGroup = null;
        this.uniformBuffer = null;
        this.particleBuffer = null;
        this.numParticles = options.numParticles || 20000;

        this.handleResize = this.resize.bind(this);
        this.handleMouseMove = this.onMouseMove.bind(this);
        this.handleMouseEnter = () => { this.isHovering = true; };
        this.handleMouseLeave = () => { this.isHovering = false; };

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050508';

        // 1. Initialize WebGL2 (The Artifact)
        this.initWebGL2();

        // 2. Initialize WebGPU (The Shield)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("ForceField: WebGPU init failed", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;

        // Listeners
        window.addEventListener('resize', this.handleResize);
        this.container.addEventListener('mousemove', this.handleMouseMove);
        this.container.addEventListener('mouseenter', this.handleMouseEnter);
        this.container.addEventListener('mouseleave', this.handleMouseLeave);

        this.animate();
    }

    // ========================================================================
    // WebGL2 IMPLEMENTATION (Central Artifact)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        `;
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Create an Icosahedron
        const t = (1.0 + Math.sqrt(5.0)) / 2.0;
        const v = [
            -1,  t,  0,
             1,  t,  0,
            -1, -t,  0,
             1, -t,  0,
             0, -1,  t,
             0,  1,  t,
             0, -1, -t,
             0,  1, -t,
             t,  0, -1,
             t,  0,  1,
            -t,  0, -1,
            -t,  0,  1
        ];

        // Normalize vertices to project onto sphere
        const vertices = [];
        for (let i = 0; i < v.length; i += 3) {
            const x = v[i], y = v[i+1], z = v[i+2];
            const len = Math.sqrt(x*x + y*y + z*z);
            vertices.push(x/len, y/len, z/len); // Position
            vertices.push(x/len, y/len, z/len); // Normal (same as position for sphere)
        }

        const indices = [
            0, 11, 5,
            0, 5, 1,
            0, 1, 7,
            0, 7, 10,
            0, 10, 11,
            1, 5, 9,
            5, 11, 4,
            11, 10, 2,
            10, 7, 6,
            7, 1, 8,
            3, 9, 4,
            3, 4, 2,
            3, 2, 6,
            3, 6, 8,
            3, 8, 9,
            4, 9, 5,
            2, 4, 11,
            6, 2, 10,
            8, 6, 7,
            9, 8, 1
        ];

        this.glIndexCount = indices.length;

        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW);

        const indexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW);

        // Shaders
        const vsSource = `#version 300 es
            in vec3 a_position;
            in vec3 a_normal;

            uniform float u_time;
            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_projection;

            out vec3 v_normal;
            out vec3 v_pos;

            void main() {
                v_normal = mat3(u_model) * a_normal;
                vec4 worldPos = u_model * vec4(a_position, 1.0);
                v_pos = worldPos.xyz;
                gl_Position = u_projection * u_view * worldPos;
            }
        `;

        const fsSource = `#version 300 es
            precision highp float;

            in vec3 v_normal;
            in vec3 v_pos;
            uniform float u_time;

            out vec4 outColor;

            void main() {
                vec3 normal = normalize(v_normal);
                vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));

                // Ambient
                float ambient = 0.2;

                // Diffuse
                float diff = max(dot(normal, lightDir), 0.0);

                // Specular
                vec3 cameraPos = vec3(0.0, 0.0, 4.0);
                vec3 viewDir = normalize(cameraPos - v_pos);
                vec3 reflectDir = reflect(-lightDir, normal);
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

                // Pulsating emission
                float emission = 0.5 + 0.5 * sin(u_time * 2.0);

                vec3 baseColor = vec3(0.2, 0.8, 1.0); // Cyan
                vec3 color = baseColor * (ambient + diff) + vec3(1.0) * spec * 0.5;
                color += baseColor * emission * 0.2;

                outColor = vec4(color, 1.0);
            }
        `;

        this.glProgram = this.createGLProgram(vsSource, fsSource);
        if (!this.glProgram) return;

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);

        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_position');
        const normLoc = this.gl.getAttribLocation(this.glProgram, 'a_normal');

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 24, 0); // 6 floats * 4 bytes = 24 stride
        this.gl.enableVertexAttribArray(normLoc);
        this.gl.vertexAttribPointer(normLoc, 3, this.gl.FLOAT, false, 24, 12);

        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.CULL_FACE);

        this.resize();
    }

    createGLProgram(vs, fs) {
        const vShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vShader, vs);
        this.gl.compileShader(vShader);
        if (!this.gl.getShaderParameter(vShader, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(vShader));
            return null;
        }

        const fShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fShader, fs);
        this.gl.compileShader(fShader);
        if (!this.gl.getShaderParameter(fShader, this.gl.COMPILE_STATUS)) {
            console.error(this.gl.getShaderInfoLog(fShader));
            return null;
        }

        const prog = this.gl.createProgram();
        this.gl.attachShader(prog, vShader);
        this.gl.attachShader(prog, fShader);
        this.gl.linkProgram(prog);
        return prog;
    }

    // ========================================================================
    // WebGPU IMPLEMENTATION (Force Field Particles)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2;
            pointer-events: none; /* Let clicks pass through if needed, but we capture on container */
        `;
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // Compute Shader
        const computeShader = `
            struct Particle {
                pos : vec4f, // xyz, w=padding
                origPos : vec4f,
                vel : vec4f,
            }

            struct Uniforms {
                time : f32,
                dt : f32,
                mouseX : f32,
                mouseY : f32,
                isHover : f32,
                padding0 : f32,
                padding1 : f32,
                padding2 : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
                let index = GlobalInvocationID.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // 1. Rotate original position based on time to create orbital feel
                let theta = uniforms.time * 0.2;
                let cosT = cos(theta);
                let sinT = sin(theta);

                // Rotate around Y axis
                let rotX = p.origPos.x * cosT - p.origPos.z * sinT;
                let rotZ = p.origPos.x * sinT + p.origPos.z * cosT;
                let targetPos = vec3f(rotX, p.origPos.y, rotZ);

                // 2. Interaction (Mouse Repel)
                // Map mouse (-1 to 1) to approximate world space coordinates
                // Assuming view is approx +/- 2.0 in X and Y at z=0
                let mouseWorld = vec3f(uniforms.mouseX * 2.0, -uniforms.mouseY * 2.0, 1.0); // Mouse is 'front'

                var force = vec3f(0.0);
                if (uniforms.isHover > 0.5) {
                    let dir = p.pos.xyz - mouseWorld;
                    let dist = length(dir);
                    if (dist < 1.0) {
                        force = normalize(dir) * (1.0 - dist) * 2.0;
                    }
                }

                // 3. Spring back to target
                let spring = (targetPos - p.pos.xyz) * 2.0;

                // 4. Update velocity
                let acc = spring + force;
                p.vel = p.vel * 0.9 + vec4f(acc * uniforms.dt, 0.0);

                // 5. Update position
                p.pos = p.pos + p.vel;

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
                mouseX : f32,
                mouseY : f32,
                isHover : f32,
            }
            @group(0) @binding(1) var<uniform> uniforms : Uniforms;

            @vertex
            fn vs_main(@location(0) pos : vec4f) -> VertexOutput {
                var output : VertexOutput;

                // Simple perspective projection
                // Camera at (0, 0, 4) looking at (0, 0, 0)
                let p = pos.xyz;
                let camZ = 4.0;
                let z = camZ - p.z;

                // Perspective division
                output.position = vec4f(p.x / z * 2.5, p.y / z * 2.5, 0.0, 1.0); // 2.5 = zoom factor

                // Color based on depth and interaction
                let depth = (p.z + 1.5) / 3.0;

                var col = vec3f(0.2, 0.5, 1.0); // Blue
                if (uniforms.isHover > 0.5) {
                    col = mix(col, vec3f(1.0, 0.2, 0.5), 0.5); // Add red on hover
                }

                output.color = vec4f(col * (0.5 + 0.5 * depth), 1.0);

                // Point size trick? WebGPU point list defaults to 1px usually.
                // We'll rely on dense points for the field effect.
                output.position.w = 1.0;
                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Initialize Particles on a Sphere
        const particleData = new Float32Array(this.numParticles * 12); // pos(4), orig(4), vel(4)
        for (let i = 0; i < this.numParticles; i++) {
            // Fibonacci Sphere distribution
            const offset = 2 / this.numParticles;
            const increment = Math.PI * (3 - Math.sqrt(5));

            const y = ((i * offset) - 1) + (offset / 2);
            const r = Math.sqrt(1 - Math.pow(y, 2));
            const phi = ((i + 1) % this.numParticles) * increment;

            const x = Math.cos(phi) * r;
            const z = Math.sin(phi) * r;

            const radius = 1.6; // Slightly larger than the icosahedron (roughly size 1.0)

            const idx = i * 12;

            // Pos
            particleData[idx] = x * radius;
            particleData[idx+1] = y * radius;
            particleData[idx+2] = z * radius;
            particleData[idx+3] = 1.0;

            // Orig Pos
            particleData[idx+4] = x * radius;
            particleData[idx+5] = y * radius;
            particleData[idx+6] = z * radius;
            particleData[idx+7] = 1.0;

            // Vel
            particleData[idx+8] = 0;
            particleData[idx+9] = 0;
            particleData[idx+10] = 0;
            particleData[idx+11] = 0;
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);

        // Uniform Buffer
        this.uniformBuffer = this.device.createBuffer({
            size: 32, // 8 floats * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind Group Layout
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ],
        });

        // Compute Pipeline
        const computeModule = this.device.createShaderModule({ code: computeShader });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        // Render Pipeline
        const renderModule = this.device.createShaderModule({ code: renderShader });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 48, // 12 floats * 4 bytes
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' }, // pos
                    ],
                }],
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{ format: format, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
                } }], // Additive blending for glow
            },
            primitive: { topology: 'point-list' },
        });

        this.resize();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.className = 'webgpu-error';
        msg.style.cssText = `
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(200, 50, 50, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-family: monospace;
            z-index: 10;
            pointer-events: none;
        `;
        msg.textContent = "WebGPU Not Supported - Showing Core Only";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // LOGIC
    // ========================================================================

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.resizeGL(width * dpr, height * dpr);
        this.resizeGPU(width * dpr, height * dpr);
    }

    resizeGL(w, h) {
        if (!this.glCanvas) return;
        if (this.glCanvas.width !== w || this.glCanvas.height !== h) {
            this.glCanvas.width = w;
            this.glCanvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
    }

    resizeGPU(w, h) {
        if (!this.gpuCanvas) return;
        if (this.gpuCanvas.width !== w || this.gpuCanvas.height !== h) {
            this.gpuCanvas.width = w;
            this.gpuCanvas.height = h;
        }
    }

    onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        // Normalize mouse to -1 to 1
        this.mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseY = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    }

    animate() {
        if (!this.isActive) return;

        const time = (Date.now() - this.startTime) * 0.001;

        // 1. Render WebGL2
        if (this.gl && this.glProgram) {
            this.gl.useProgram(this.glProgram);

            // Model Matrix (Rotate object)
            const angle = time * 0.5;
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            // Simple rotation around Y
            const model = [
                c, 0, s, 0,
                0, 1, 0, 0,
                -s, 0, c, 0,
                0, 0, 0, 1
            ];

            // View Matrix (Camera at 0,0,4)
            const view = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -4, 1
            ];

            // Projection
            const aspect = this.glCanvas.width / this.glCanvas.height;
            const fov = 45 * Math.PI / 180;
            const f = 1.0 / Math.tan(fov / 2);
            const nf = 1 / (0.1 - 100);
            const proj = [
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (100 + 0.1) * nf, -1,
                0, 0, 2 * 100 * 0.1 * nf, 0
            ];

            const locTime = this.gl.getUniformLocation(this.glProgram, 'u_time');
            const locModel = this.gl.getUniformLocation(this.glProgram, 'u_model');
            const locView = this.gl.getUniformLocation(this.glProgram, 'u_view');
            const locProj = this.gl.getUniformLocation(this.glProgram, 'u_projection');

            this.gl.uniform1f(locTime, time);
            this.gl.uniformMatrix4fv(locModel, false, model);
            this.gl.uniformMatrix4fv(locView, false, view);
            this.gl.uniformMatrix4fv(locProj, false, proj);

            this.gl.clearColor(0.02, 0.02, 0.05, 1.0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.bindVertexArray(this.glVao);
            this.gl.drawElements(this.gl.TRIANGLES, this.glIndexCount, this.gl.UNSIGNED_SHORT, 0);
        }

        // 2. Render WebGPU
        if (this.device && this.renderPipeline) {
            // Update Uniforms
            const uniforms = new Float32Array([
                time,
                0.016,
                this.mouseX,
                this.mouseY,
                this.isHovering ? 1.0 : 0.0,
                0, 0, 0 // Padding
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
                    storeOp: 'store',
                }]
            });
            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.computeBindGroup); // Access uniforms in vertex shader
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
        this.container.removeEventListener('mousemove', this.handleMouseMove);
        this.container.removeEventListener('mouseenter', this.handleMouseEnter);
        this.container.removeEventListener('mouseleave', this.handleMouseLeave);

        if (this.gl) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
        if (this.device) this.device.destroy();

        this.container.innerHTML = '';
    }
}

if (typeof window !== 'undefined') {
    window.ForceFieldExperiment = ForceFieldExperiment;
}
