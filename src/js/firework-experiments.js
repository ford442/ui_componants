/**
 * Firework Experiments
 * Contains two hybrid WebGL2/WebGPU classes:
 * 1. CityCelebration: City skyline + Fireworks
 * 2. CyberBurst: Synthwave sun + Volumetric explosions
 */

export class CityCelebration {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;
        this.isActive = false;
        this.particleCount = options.particleCount || 10000;

        this.startTime = Date.now();
        this.animationId = null;

        // WebGL2 (Skyline)
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;

        // WebGPU (Fireworks)
        this.gpuCanvas = null;
        this.device = null;

        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.background = '#020205';

        this.initWebGL2();
        if (navigator.gpu) await this.initWebGPU();

        this.resizeObserver.observe(this.container);
        this.resize();
        this.isActive = true;
        this.animate();
    }

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 1; pointer-events: none;';
        this.container.appendChild(this.glCanvas);
        this.gl = this.glCanvas.getContext('webgl2');

        const vs = `#version 300 es
        in vec2 position;
        out vec2 vUv;
        void main() {
            vUv = position * 0.5 + 0.5;
            gl_Position = vec4(position, 0.0, 1.0);
        }`;

        const fs = `#version 300 es
        precision highp float;
        in vec2 vUv;
        uniform float time;
        uniform vec2 resolution;
        out vec4 fragColor;

        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }

        float building(vec2 uv, float width, float height, float offset) {
            float x = step(offset, uv.x) - step(offset + width, uv.x);
            float y = step(0.0, uv.y) - step(height, uv.y);
            return x * y;
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;

            // Generate Skyline (simplified)
            float city = 0.0;
            for(int i=0; i<10; i++) {
                float f = float(i);
                float h = 0.1 + 0.3 * random(vec2(f, 1.0));
                float w = 0.05 + 0.05 * random(vec2(f, 2.0));
                float o = f * 0.1;
                city += building(uv, w, h, o);
            }
            // Add windows
            float windows = 0.0;
            if (city > 0.0) {
                 float wx = step(0.5, fract(uv.x * 50.0 + time * 0.1)); // flickering lights
                 float wy = step(0.5, fract(uv.y * 50.0));
                 if (random(vec2(floor(uv.x*50.0), floor(uv.y*50.0))) > 0.8) {
                    windows = wx * wy * sin(time + uv.x * 10.0);
                 }
            }

            vec3 skyColor = vec3(0.05, 0.05, 0.1) * (1.0 - uv.y);
            vec3 cityColor = vec3(0.01);
            vec3 windowColor = vec3(1.0, 0.9, 0.5);

            vec3 finalColor = mix(skyColor, cityColor, step(0.01, city));
            finalColor += windows * windowColor * step(0.01, city);

            fragColor = vec4(finalColor, 1.0);
        }`;

        this.glProgram = this.createProgram(this.gl, vs, fs);

        const positions = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
        const vbo = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vbo);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const loc = this.gl.getAttribLocation(this.glProgram, 'position');
        this.gl.enableVertexAttribArray(loc);
        this.gl.vertexAttribPointer(loc, 2, this.gl.FLOAT, false, 0, 0);
    }

    createProgram(gl, vs, fs) {
        const p = gl.createProgram();
        const v = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(v, vs); gl.compileShader(v);
        const f = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(f, fs); gl.compileShader(f);
        gl.attachShader(p, v); gl.attachShader(p, f);
        gl.linkProgram(p);
        return p;
    }

    async initWebGPU() {
        const adapter = await navigator.gpu.requestAdapter();
        if(!adapter) return;
        this.device = await adapter.requestDevice();
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 2; pointer-events: none;';
        this.container.appendChild(this.gpuCanvas);
        this.context = this.gpuCanvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' });

        const computeShader = `
            struct Particle {
                pos: vec4f, // xy, vx, vy
                data: vec4f, // life, type, r, g
            }
            struct Uniforms {
                time: f32,
                dt: f32,
            }
            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            fn rand(co: vec2f) -> f32 { return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453); }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }

                var p = particles[i];
                let spark_type = p.data.y;

                // Physics
                if (spark_type == 0.0) { // Rocket
                    p.pos.y += p.pos.w * uniforms.dt; // vy
                    p.pos.w -= 0.2 * uniforms.dt; // gravity drag?
                    p.data.x -= uniforms.dt * 0.5; // life

                    if (p.data.x <= 0.0) { // Explode
                        p.data.y = 1.0; // Become spark
                        p.data.x = 1.0 + rand(vec2f(f32(i), uniforms.time)); // New life
                        let angle = rand(vec2f(f32(i), p.pos.x)) * 6.28;
                        let speed = 0.5 + rand(vec2f(p.pos.y, f32(i))) * 0.5;
                        p.pos.z = cos(angle) * speed;
                        p.pos.w = sin(angle) * speed;
                    }
                } else { // Spark
                    p.pos.x += p.pos.z * uniforms.dt;
                    p.pos.y += p.pos.w * uniforms.dt;
                    p.pos.w -= 0.5 * uniforms.dt; // Gravity
                    p.data.x -= uniforms.dt; // Decay

                    if (p.data.x <= 0.0) { // Reset to Rocket
                        p.data.y = 0.0;
                        p.data.x = 2.0; // Life
                        p.pos.x = (rand(vec2f(uniforms.time, f32(i))) * 2.0 - 1.0) * 0.8;
                        p.pos.y = -1.0;
                        p.pos.z = 0.0;
                        p.pos.w = 1.0 + rand(vec2f(f32(i), uniforms.time)) * 0.5; // Launch speed

                        // New Color
                        p.data.z = rand(vec2f(f32(i), 1.0));
                        p.data.w = rand(vec2f(f32(i), 2.0));
                    }
                }

                particles[i] = p;
            }
        `;

        const renderShader = `
             struct Particle {
                pos: vec4f,
                data: vec4f,
            }
            @group(0) @binding(0) var<storage, read> particles: array<Particle>;

            struct VO {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f,
            }

            @vertex
            fn vs(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) iIdx: u32) -> VO {
                let p = particles[iIdx];
                var output: VO;
                output.pos = vec4f(p.pos.x, p.pos.y, 0.0, 1.0);

                let life = p.data.x;
                let spark_type = p.data.y;

                var col = vec3f(1.0, 0.8, 0.2); // Default
                if (spark_type == 1.0) {
                     col = vec3f(p.data.z, p.data.w, 1.0 - p.data.z);
                }

                output.color = vec4f(col, life);

                // Point size hack
                var size = 0.005;
                if (spark_type == 1.0) { size = 0.003; }

                // Expand point to quad in vertex shader (simplified just returning center point, needs topology point-list)
                return output;
            }

            @fragment
            fn fs(@location(0) color: vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Setup Buffers
        const pData = new Float32Array(this.particleCount * 8);
        for(let i=0; i<this.particleCount; i++) {
             // Init as sparks mostly so they look good immediately, or staggered rockets?
             // Let's init as dead sparks to reset.
             pData[i*8 + 4] = -1.0; // Life < 0 -> trigger reset immediately
             pData[i*8 + 5] = 1.0; // Spark type
        }

        this.particleBuffer = this.device.createBuffer({
            size: pData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(pData);
        this.particleBuffer.unmap();

        this.uniformBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        const layout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // Using 'read-only' trick? No, Compute needs RW.
                // Let's stick to the 2-layout pattern for safety.
            ]
        });

        // To save chars, I'll assume I can fix the layout issue or use separate.
        // Let's use separate layouts again to be robust.
        const cLayout = this.device.createBindGroupLayout({ entries: [{binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:'storage'}}, {binding:1, visibility:GPUShaderStage.COMPUTE, buffer:{type:'uniform'}}] });
        const rLayout = this.device.createBindGroupLayout({ entries: [{binding:0, visibility:GPUShaderStage.VERTEX, buffer:{type:'read-only-storage'}}] });

        this.cBG = this.device.createBindGroup({ layout: cLayout, entries: [{binding:0, resource:{buffer:this.particleBuffer}}, {binding:1, resource:{buffer:this.uniformBuffer}}] });
        this.rBG = this.device.createBindGroup({ layout: rLayout, entries: [{binding:0, resource:{buffer:this.particleBuffer}}] });

        const cMod = this.device.createShaderModule({ code: computeShader });
        this.cPipe = this.device.createComputePipeline({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [cLayout] }), compute: { module: cMod, entryPoint: 'main' } });

        const rMod = this.device.createShaderModule({ code: renderShader });
        this.rPipe = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [rLayout] }),
            vertex: { module: rMod, entryPoint: 'vs' },
            fragment: { module: rMod, entryPoint: 'fs', targets: [{ format: navigator.gpu.getPreferredCanvasFormat(), blend: {color:{srcFactor:'src-alpha',dstFactor:'one',operation:'add'},alpha:{srcFactor:'src-alpha',dstFactor:'one',operation:'add'}} }] },
            primitive: { topology: 'point-list' }
        });
    }

    resize() {
        if(!this.container) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if(this.glCanvas) { this.glCanvas.width = w; this.glCanvas.height = h; this.gl.viewport(0,0,w,h); }
        if(this.gpuCanvas) { this.gpuCanvas.width = w; this.gpuCanvas.height = h; }
    }

    animate() {
        if (!this.isActive) return;
        const time = (Date.now() - this.startTime) / 1000;

        if (this.gl) {
            this.gl.useProgram(this.glProgram);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'time'), time);
            this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'resolution'), this.glCanvas.width, this.glCanvas.height);
            this.gl.clearColor(0,0,0,0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.bindVertexArray(this.glVao);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        if (this.device && this.context) {
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([time, 0.016]));
            const enc = this.device.createCommandEncoder();
            const cp = enc.beginComputePass();
            cp.setPipeline(this.cPipe); cp.setBindGroup(0, this.cBG); cp.dispatchWorkgroups(Math.ceil(this.particleCount/64)); cp.end();
            const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: {r:0,g:0,b:0,a:0}, loadOp: 'clear', storeOp: 'store' }] });
            rp.setPipeline(this.rPipe); rp.setBindGroup(0, this.rBG); rp.draw(this.particleCount); rp.end();
            this.device.queue.submit([enc.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

export class CyberBurst {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;
        this.isActive = false;
        this.particleCount = options.particleCount || 10000;
        this.startTime = Date.now();
        this.animationId = null;
        this.glCanvas = null; this.gl = null; this.glProgram = null;
        this.gpuCanvas = null; this.device = null;
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.background = '#1a051a';
        this.initWebGL2();
        if(navigator.gpu) await this.initWebGPU();
        this.resizeObserver.observe(this.container);
        this.resize();
        this.isActive = true;
        this.animate();
    }

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 1; pointer-events: none;';
        this.container.appendChild(this.glCanvas);
        this.gl = this.glCanvas.getContext('webgl2');

        const vs = `#version 300 es
        in vec2 position;
        out vec2 vUv;
        void main() { vUv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }`;
        const fs = `#version 300 es
        precision highp float;
        in vec2 vUv;
        uniform float time;
        uniform vec2 resolution;
        out vec4 fragColor;
        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / resolution.y;
            // Sun
            float d = length(uv - vec2(0.0, 0.2));
            float sun = step(d, 0.3);
            // Scanlines on sun
            float lines = step(0.05, sin(uv.y * 100.0 + time));
            if (uv.y < 0.2) sun *= lines; // Only bottom half

            // Grid
            float gridY = 1.0 / (abs(uv.y + 0.5) + 0.01);

            vec3 col = vec3(1.0, 0.0, 0.5) * sun;
            col += vec3(0.0, 0.2, 0.4) * gridY * step(uv.y, -0.1);

            fragColor = vec4(col, 1.0);
        }`;

        this.glProgram = this.createProgram(this.gl, vs, fs);
        const p = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
        const vbo = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vbo);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, p, this.gl.STATIC_DRAW);
        this.glVao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVao);
        const l = this.gl.getAttribLocation(this.glProgram, 'position');
        this.gl.enableVertexAttribArray(l);
        this.gl.vertexAttribPointer(l, 2, this.gl.FLOAT, false, 0, 0);
    }

    createProgram(gl, vs, fs) {
        const p = gl.createProgram();
        const v = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(v, vs); gl.compileShader(v);
        const f = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(f, fs); gl.compileShader(f);
        gl.attachShader(p, v); gl.attachShader(p, f);
        gl.linkProgram(p);
        return p;
    }

    async initWebGPU() {
         const adapter = await navigator.gpu.requestAdapter();
        if(!adapter) return;
        this.device = await adapter.requestDevice();
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position: absolute; inset: 0; z-index: 2; pointer-events: none;';
        this.container.appendChild(this.gpuCanvas);
        this.context = this.gpuCanvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' });

        // Similar particle system but explosive burst
         const computeShader = `
            struct Particle { pos: vec4f, vel: vec4f }
            struct Uniforms { time: f32, dt: f32 }
            @group(0) @binding(0) var<storage, read_write> p: array<Particle>;
            @group(0) @binding(1) var<uniform> u: Uniforms;
            fn r(c: vec2f) -> f32 { return fract(sin(dot(c, vec2f(12.9, 78.2))) * 43758.5); }
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x; if(i>=arrayLength(&p)) {return;}
                var pt = p[i];
                pt.pos.x += pt.vel.x * u.dt;
                pt.pos.y += pt.vel.y * u.dt;
                pt.pos.z += pt.vel.z * u.dt; // z unused usually but maybe 3d?
                pt.pos.w -= u.dt; // life
                if (pt.pos.w <= 0.0) {
                     pt.pos = vec4f(0.0, 0.0, 0.0, 1.0 + r(vec2f(f32(i), u.time))); // Reset center
                     let a = r(vec2f(f32(i), 1.0))*6.28;
                     let s = 1.0 + r(vec2f(f32(i), 2.0));
                     pt.vel = vec4f(cos(a)*s, sin(a)*s, 0.0, 0.0);
                }
                p[i] = pt;
            }
        `;
        const renderShader = `
            struct Particle { pos: vec4f, vel: vec4f }
            @group(0) @binding(0) var<storage, read> p: array<Particle>;
            @vertex fn vs(@builtin(instance_index) i: u32) -> @builtin(position) vec4f {
                return vec4f(p[i].pos.x, p[i].pos.y, 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 1.0, 1.0); }
        `;
        // Setup ... (abbreviated for memory, using same pattern as CityCelebration)
        // Actually, to ensure quality, I'll copy the robust pattern from above but simplified logic.

        const pData = new Float32Array(this.particleCount * 8);
        this.particleBuffer = this.device.createBuffer({size:pData.byteLength, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
        this.uniformBuffer = this.device.createBuffer({size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});

        const cL = this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}}]});
        const rL = this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}}]});

        this.cBG = this.device.createBindGroup({layout:cL, entries:[{binding:0,resource:{buffer:this.particleBuffer}},{binding:1,resource:{buffer:this.uniformBuffer}}]});
        this.rBG = this.device.createBindGroup({layout:rL, entries:[{binding:0,resource:{buffer:this.particleBuffer}}]});

        this.cPipe = this.device.createComputePipeline({layout:this.device.createPipelineLayout({bindGroupLayouts:[cL]}), compute:{module:this.device.createShaderModule({code:computeShader}), entryPoint:'main'}});
        this.rPipe = this.device.createRenderPipeline({
            layout:this.device.createPipelineLayout({bindGroupLayouts:[rL]}),
            vertex:{module:this.device.createShaderModule({code:renderShader}), entryPoint:'vs'},
            fragment:{module:this.device.createShaderModule({code:renderShader}), entryPoint:'fs', targets:[{format:navigator.gpu.getPreferredCanvasFormat(), blend:{color:{srcFactor:'src-alpha',dstFactor:'one',operation:'add'},alpha:{srcFactor:'src-alpha',dstFactor:'one',operation:'add'}}}]},
            primitive:{topology:'point-list'}
        });
    }

    resize() {
         if(!this.container) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if(this.glCanvas) { this.glCanvas.width = w; this.glCanvas.height = h; this.gl.viewport(0,0,w,h); }
        if(this.gpuCanvas) { this.gpuCanvas.width = w; this.gpuCanvas.height = h; }
    }

    animate() {
        if(!this.isActive) return;
        const time = (Date.now() - this.startTime) / 1000;

        if(this.gl) {
             this.gl.useProgram(this.glProgram);
             this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'time'), time);
             this.gl.uniform2f(this.gl.getUniformLocation(this.glProgram, 'resolution'), this.glCanvas.width, this.glCanvas.height);
             this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        }

        if(this.device && this.context) {
             this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([time, 0.016]));
             const enc = this.device.createCommandEncoder();
             const cp = enc.beginComputePass(); cp.setPipeline(this.cPipe); cp.setBindGroup(0, this.cBG); cp.dispatchWorkgroups(Math.ceil(this.particleCount/64)); cp.end();
             const rp = enc.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(), loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:0}}]});
             rp.setPipeline(this.rPipe); rp.setBindGroup(0, this.rBG); rp.draw(this.particleCount); rp.end();
             this.device.queue.submit([enc.finish()]);
        }
        requestAnimationFrame(()=>this.animate());
    }
}
