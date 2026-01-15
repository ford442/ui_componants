/**
 * webgpu-particle-system.js
 * 
 * A reusable class for creating a WebGPU-based particle system.
 * This is extracted from the buttons.js file to be managed by a WebGPUManager.
 */

class WebGPUParticleSystem {
    constructor(manager, options = {}) {
        this.manager = manager;
        this.device = manager.device;
        this.element = options.element;

        this.particleCount = options.particleCount || 10000;
        this.particleSize = options.particleSize || 2;
        this.color = options.color || [0, 1, 0.5, 0.8];
        this.attractorStrength = options.attractorStrength || 0.3;
        this.damping = options.damping || 0.98;

        this.computePipeline = null;
        this.renderPipeline = null;
        this.particleBuffers = [];
        this.uniformBuffer = null;
        this.computeBindGroups = [];
        
        this.frame = 0;
    }

    async init() {
        if (!this.device) return false;

        const computeShader = `
            struct Particle {
                pos: vec2<f32>,
                vel: vec2<f32>,
            };

            struct Uniforms {
                deltaTime: f32,
                attractor: vec2<f32>,
            };

            @group(0) @binding(0) var<uniform> u: Uniforms;
            @group(0) @binding(1) var<storage, read> particlesA: array<Particle>;
            @group(0) @binding(2) var<storage, write> particlesB: array<Particle>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
                let index = GlobalInvocationID.x;
                if (index >= ${this.particleCount}) {
                    return;
                }
                var pA: Particle = particlesA[index];

                var vel = pA.vel;
                var pos = pA.pos;

                let delta = u.deltaTime;
                let attractor = u.attractor;

                let to_attractor = attractor - pos;
                let dist = length(to_attractor);
                
                if (dist > 0.01) {
                    vel = vel + (to_attractor / dist) * ${this.attractorStrength} * delta;
                }

                vel = vel * ${this.damping};
                
                pos = pos + vel * delta;

                if (pos.x > 1.0) { pos.x = -1.0; }
                if (pos.x < -1.0) { pos.x = 1.0; }
                if (pos.y > 1.0) { pos.y = -1.0; }
                if (pos.y < -1.0) { pos.y = 1.0; }
                
                particlesB[index].pos = pos;
                particlesB[index].vel = vel;
            }
        `;

        const renderShader = `
            @vertex
            fn vs_main(@location(0) in_pos: vec2<f32>) -> @builtin(position) vec4<f32> {
                return vec4<f32>(in_pos, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(${this.color[0]}, ${this.color[1]}, ${this.color[2]}, ${this.color[3]});
            }
        `;

        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: computeShader }),
                entryPoint: 'main',
            },
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 16,
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
                }],
            },
            fragment: {
                module: this.device.createShaderModule({ code: renderShader }),
                entryPoint: 'fs_main',
                targets: [{ format: this.manager.format }],
            },
            primitive: {
                topology: 'point-list',
            },
        });
        
        const initialParticles = new Float32Array(this.particleCount * 4);
        for (let i = 0; i < this.particleCount; i++) {
            initialParticles[i * 4 + 0] = (Math.random() - 0.5) * 2; // pos.x
            initialParticles[i * 4 + 1] = (Math.random() - 0.5) * 2; // pos.y
            initialParticles[i * 4 + 2] = (Math.random() - 0.5) * 0.1; // vel.x
            initialParticles[i * 4 + 3] = (Math.random() - 0.5) * 0.1; // vel.y
        }
        
        for (let i = 0; i < 2; i++) {
            this.particleBuffers[i] = this.device.createBuffer({
                size: initialParticles.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Float32Array(this.particleBuffers[i].getMappedRange()).set(initialParticles);
            this.particleBuffers[i].unmap();
        }
        
        this.uniformBuffer = this.device.createBuffer({
            size: 16, // deltaTime (4) + attractor (8) + padding (4)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        for (let i = 0; i < 2; i++) {
            this.computeBindGroups[i] = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: { buffer: this.particleBuffers[i] } },
                    { binding: 2, resource: { buffer: this.particleBuffers[(i + 1) % 2] } },
                ],
            });
        }
        
        return true;
    }

    updateUniforms(deltaTime, mouseX, mouseY) {
        this.device.queue.writeBuffer(
            this.uniformBuffer, 0,
            new Float32Array([deltaTime, mouseX, mouseY])
        );
    }
    
    render(passEncoder) {
        passEncoder.setPipeline(this.renderPipeline);
        passEncoder.setVertexBuffer(0, this.particleBuffers[this.frame % 2]);
        passEncoder.draw(this.particleCount);
    }

    compute(deltaTime) {
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.computePipeline);
        passEncoder.setBindGroup(0, this.computeBindGroups[this.frame % 2]);
        passEncoder.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        this.frame++;
    }
}

window.WebGPUParticleSystem = WebGPUParticleSystem;