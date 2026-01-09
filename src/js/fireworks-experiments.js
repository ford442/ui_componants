/**
 * Fireworks Experiment
 * WebGPU Particle System simulating fireworks explosions
 */

export class FireworksExperiment {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        this.options = {
            particleCount: options.particleCount || 10000,
            gravity: options.gravity || 0.5,
            ...options
        };

        this.canvas = document.createElement('canvas');
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.display = 'block';
        this.container.appendChild(this.canvas);

        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.computePipeline = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.initialized = false;
        this.isRunning = false;

        this.mouseX = 0;
        this.mouseY = 0;
        this.isMouseDown = 0;

        this.resize = this.resize.bind(this);
        this.render = this.render.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);

        this.init();
    }

    async init() {
        if (!navigator.gpu) {
            this.showError('WebGPU not supported on this device.');
            return;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                this.showError('No WebGPU adapter found.');
                return;
            }

            this.device = await adapter.requestDevice();

            this.context = this.canvas.getContext('webgpu');
            this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

            this.resize();

            const observer = new ResizeObserver(() => this.resize());
            observer.observe(this.container);

            // Event Listeners
            this.canvas.addEventListener('mousemove', this.handleMouseMove);
            this.canvas.addEventListener('mousedown', this.handleMouseDown);
            this.canvas.addEventListener('mouseup', this.handleMouseUp);
            // Touch support
            this.canvas.addEventListener('touchmove', (e) => {
                e.preventDefault();
                const touch = e.touches[0];
                this.handleMouseMove(touch);
            }, { passive: false });
             this.canvas.addEventListener('touchstart', (e) => {
                e.preventDefault();
                this.handleMouseDown();
                const touch = e.touches[0];
                this.handleMouseMove(touch);
            }, { passive: false });
            this.canvas.addEventListener('touchend', this.handleMouseUp);


            await this.createAssets();

            this.initialized = true;
            this.isRunning = true;
            requestAnimationFrame(this.render);

        } catch (e) {
            console.error('Fireworks initialization failed:', e);
            this.showError('Initialization failed: ' + e.message);
        }
    }

    showError(msg) {
        const error = document.createElement('div');
        error.style.color = '#ff4444';
        error.style.padding = '20px';
        error.style.textAlign = 'center';
        error.textContent = msg;
        this.container.innerHTML = '';
        this.container.appendChild(error);
    }

    async createAssets() {
        // Particle data structure:
        // pos (vec2), vel (vec2), color (vec4), life (f32), type (f32), padding (vec2)
        // Total: 4 * (2 + 2 + 4 + 1 + 1 + 2) = 48 bytes per particle
        // Ensuring 16-byte alignment

        const numParticles = this.options.particleCount;
        const particleData = new Float32Array(numParticles * 12);

        // Initialize particles
        // We will manage emission in the compute shader or JS, let's do JS reset for simplicity or compute respawn
        // For this demo, we'll let compute shader handle respawning fireworks

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        // Initialize with zeros/defaults
        new Float32Array(this.particleBuffer.getMappedRange()).fill(0);
        this.particleBuffer.unmap();

        // Uniform buffer
        // time, deltaTime, resolution(2), gravity, seed, mouseX, mouseY, mouseDown
        // Size: 8 floats = 32 bytes.
        // But let's align to 16 bytes chunks.
        // Uniforms struct:
        // time (f32), deltaTime (f32), resX (f32), resY (f32)
        // gravity (f32), seed (f32), mouseX (f32), mouseY (f32)
        // mouseDown (f32), pad1, pad2, pad3
        // Total: 12 floats -> 48 bytes. Round up to 64 bytes for alignment safety if needed or keep packed.
        // Let's use 48 bytes (3 * vec4 size).

        this.uniformBuffer = this.device.createBuffer({
            size: 64, // Sufficient for our needs
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Compute Shader
        const computeShader = `
            struct Particle {
                pos: vec2<f32>,
                vel: vec2<f32>,
                color: vec4<f32>,
                life: f32,
                pType: f32, // 0: trail, 1: explosion
                pad1: f32,
                pad2: f32
            }

            struct Uniforms {
                time: f32,
                deltaTime: f32,
                resX: f32,
                resY: f32,
                gravity: f32,
                seed: f32,
                mouseX: f32,
                mouseY: f32,
                mouseDown: f32
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            // Pseudo-random function
            fn hash(p: u32) -> f32 {
                var p_mut = p;
                p_mut = p_mut ^ 61u;
                p_mut = p_mut ^ (p_mut >> 16u);
                p_mut = p_mut * 9u;
                p_mut = p_mut ^ (p_mut >> 4u);
                p_mut = p_mut * 668265261u;
                p_mut = p_mut ^ (p_mut >> 15u);
                return f32(p_mut) / 4294967296.0;
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&particles)) { return; }

                var p = particles[index];

                // Update Life
                p.life -= uniforms.deltaTime * 0.5;

                // Mouse interaction: Attraction when mouse down
                if (uniforms.mouseDown > 0.5) {
                    let mousePos = vec2<f32>(uniforms.mouseX, uniforms.mouseY);
                    // Convert mouse from [0, res] to [-1, 1] (approximately, preserving aspect ratio)
                    let aspect = uniforms.resX / uniforms.resY;
                    let mX = (mousePos.x / uniforms.resX) * 2.0 - 1.0;
                    let mY = (1.0 - mousePos.y / uniforms.resY) * 2.0 - 1.0; // Flip Y
                    // Adjust X for aspect if we did that for particles.
                    // Our vertex shader doesn't correct aspect, it just uses raw pos.
                    // But assume particles are in standard clip space [-1, 1].

                    let target = vec2<f32>(mX, mY);
                    let delta = target - p.pos;
                    let dist = length(delta);
                    if (dist < 0.5) {
                        let force = normalize(delta) * 5.0 * uniforms.deltaTime;
                        p.vel += force;
                    }
                }

                // Respawn
                if (p.life <= 0.0) {
                    let seed = u32(uniforms.time * 1000.0) + index * 123u;
                    let rand = hash(seed);

                    if (uniforms.mouseDown > 0.5 && (index % 10u) == 0u) {
                        // Spawn at mouse if clicked
                         let mousePos = vec2<f32>(uniforms.mouseX, uniforms.mouseY);
                         let mX = (mousePos.x / uniforms.resX) * 2.0 - 1.0;
                         let mY = (1.0 - mousePos.y / uniforms.resY) * 2.0 - 1.0;

                         p.pos = vec2<f32>(mX, mY);
                         let angle = hash(seed + 1u) * 6.28;
                         let speed = hash(seed + 2u) * 2.0;
                         p.vel = vec2<f32>(cos(angle) * speed, sin(angle) * speed);
                    } else {
                        // Reset to bottom center-ish
                        p.pos = vec2<f32>((hash(seed + 1u) - 0.5) * 1.5, -1.2);
                        // Launch velocity
                        p.vel = vec2<f32>((hash(seed + 2u) - 0.5) * 0.5, 1.5 + hash(seed + 3u) * 1.0);
                    }

                    // Random Color
                    let hue = hash(seed + 4u);
                    let c = vec3<f32>(
                        0.5 + 0.5 * cos(6.28318 * (hue + 0.0)),
                        0.5 + 0.5 * cos(6.28318 * (hue + 0.33)),
                        0.5 + 0.5 * cos(6.28318 * (hue + 0.67))
                    );
                    p.color = vec4<f32>(c, 1.0);

                    p.life = 1.0 + hash(seed + 5u);
                    p.pType = 1.0; // Explosion type
                } else {
                    // Physics
                    p.vel.y -= uniforms.gravity * uniforms.deltaTime;

                    // Air resistance
                    p.vel *= 0.99;

                    p.pos += p.vel * uniforms.deltaTime;

                    // Fade out
                    if (p.life < 0.5) {
                        p.color.a = p.life * 2.0;
                    }

                    // "Explosion" logic simulation (simple expansion for now)
                    if (p.pType > 0.5 && p.vel.y < 0.0 && p.life > 0.8) {
                        // If falling and young, spread out like an explosion
                        let seed = u32(index) + u32(uniforms.time * 60.0);
                        p.vel.x += (hash(seed) - 0.5) * 0.1;
                    }
                }

                particles[index] = p;
            }
        `;

        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: computeShader }),
                entryPoint: 'main'
            }
        });

        // Render Shader
        const renderShader = `
            struct Particle {
                pos: vec2<f32>,
                vel: vec2<f32>,
                color: vec4<f32>,
                life: f32,
                pType: f32,
                pad1: f32,
                pad2: f32
            }

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>
            }

            @group(0) @binding(0) var<storage, read> particles: array<Particle>;

            @vertex
            fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
                let p = particles[instanceIndex];

                // Quad vertices
                let corners = array<vec2<f32>, 4>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(1.0, -1.0),
                    vec2<f32>(-1.0, 1.0),
                    vec2<f32>(1.0, 1.0)
                );

                let corner = corners[vertexIndex % 4u];
                let size = 0.01 * (p.life + 0.2); // Shrink as dying

                var pos = p.pos + corner * size;

                var out: VertexOutput;
                out.position = vec4<f32>(pos, 0.0, 1.0);
                out.color = p.color;
                return out;
            }

            @fragment
            fn fragmentMain(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                // Circular particle
                // We'd need UVs for proper circle, but simple quad is fine for thousands of particles
                return color;
            }
        `;

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'vertexMain'
            },
            fragment: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'fragmentMain',
                targets: [{
                    format: this.presentationFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one', // Additive blending for glow look
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'zero',
                            dstFactor: 'one',
                            operation: 'add'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-strip'
            }
        });

        // Bind Groups
        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } }
            ]
        });
    }

    resize() {
        const rect = this.container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        // Ensure non-zero dimensions
        const width = Math.max(1, rect.width * dpr);
        const height = Math.max(1, rect.height * dpr);

        this.canvas.width = width;
        this.canvas.height = height;

        if (this.device && this.context) {
             this.context.configure({
                device: this.device,
                format: this.presentationFormat,
                alphaMode: 'premultiplied'
            });
        }
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const clientX = (e.touches && e.touches.length > 0) ? e.touches[0].clientX : e.clientX;
        const clientY = (e.touches && e.touches.length > 0) ? e.touches[0].clientY : e.clientY;

        // Fallback if event is just a Touch object passed directly
        const x = clientX !== undefined ? clientX : e.clientX;
        const y = clientY !== undefined ? clientY : e.clientY;

        this.mouseX = x - rect.left;
        this.mouseY = y - rect.top;
    }

    handleMouseDown(e) {
        this.isMouseDown = 1.0;
        if(e) {
            const rect = this.canvas.getBoundingClientRect();
             // Check if it's a TouchEvent or MouseEvent or direct Touch object
            let clientX, clientY;
            if (e.touches && e.touches.length > 0) {
                 clientX = e.touches[0].clientX;
                 clientY = e.touches[0].clientY;
            } else {
                 clientX = e.clientX;
                 clientY = e.clientY;
            }

            this.mouseX = clientX - rect.left;
            this.mouseY = clientY - rect.top;
        }
    }

    handleMouseUp(e) {
        this.isMouseDown = 0.0;
    }

    render(time) {
        if (!this.isRunning) return;
        requestAnimationFrame(this.render);

        const t = time / 1000;
        const dt = 0.016; // Fixed step for stability

        // Update Uniforms
        // Struct: time, deltaTime, resX, resY, gravity, seed, mouseX, mouseY, mouseDown
        const uniforms = new Float32Array([
            t,
            dt,
            this.canvas.width,
            this.canvas.height,
            this.options.gravity,
            Math.random(), // seed
            this.mouseX,
            this.mouseY,
            this.isMouseDown,
            0, 0, 0 // padding
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

        const commandEncoder = this.device.createCommandEncoder();

        // Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.options.particleCount / 64));
        computePass.end();

        // Render Pass
        const textureView = this.context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(4, this.options.particleCount); // 4 vertices per instance
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    destroy() {
        this.isRunning = false;
        if (this.particleBuffer) this.particleBuffer.destroy();
        if (this.uniformBuffer) this.uniformBuffer.destroy();
        this.container.innerHTML = '';
        this.canvas.removeEventListener('mousemove', this.handleMouseMove);
        this.canvas.removeEventListener('mousedown', this.handleMouseDown);
        this.canvas.removeEventListener('mouseup', this.handleMouseUp);
        // ... remove touch listeners if stored
    }
}
