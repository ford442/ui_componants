/**
 * Neural Network Visualization
 * Hybrid Rendering:
 * - WebGL2: Renders the static 3D network structure (Nodes & Connections)
 * - WebGPU: Simulates and renders data pulses traveling along the connections
 */

class NeuralNetworkExperiment {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            nodeCount: options.nodeCount || 100,
            connectionDensity: options.connectionDensity || 3,
            pulseCount: options.pulseCount || 1000,
            ...options
        };

        this.isActive = false;
        this.animationId = null;

        // Graph Data
        this.nodes = [];
        this.connections = []; // Pairs of node indices

        // WebGL2
        this.glCanvas = null;
        this.gl = null;
        this.glProgram = null;
        this.glVaoNodes = null;
        this.glVaoLines = null;
        this.uMatrixLoc = null;

        // WebGPU
        this.gpuCanvas = null;
        this.device = null;
        this.context = null;
        this.computePipeline = null;
        this.renderPipeline = null;
        this.computeBindGroup = null;
        this.pulseBuffer = null;
        this.pathBuffer = null; // Stores connection paths for GPU
        this.nodeBuffer = null; // Stores node positions for GPU

        this.init();
    }

    async init() {
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';
        this.container.style.background = '#050510'; // Deep dark blue/black

        this.generateGraph();

        // 1. WebGL2 (Structure)
        this.initWebGL2();

        // 2. WebGPU (Pulses)
        let gpuSuccess = false;
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                gpuSuccess = await this.initWebGPU();
            } catch (e) {
                console.warn("NeuralNet: WebGPU error", e);
            }
        }

        if (!gpuSuccess) {
            this.addWebGPUNotSupportedMessage();
        }

        this.isActive = true;
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    generateGraph() {
        // Generate random nodes in a sphere
        for (let i = 0; i < this.options.nodeCount; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 0.8 * Math.cbrt(Math.random()); // Even distribution inside sphere

            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);

            this.nodes.push(x, y, z);
        }

        // Generate connections based on distance
        const positions = [];
        for (let i = 0; i < this.options.nodeCount; i++) {
            const p1 = { x: this.nodes[i*3], y: this.nodes[i*3+1], z: this.nodes[i*3+2] };

            // Find k nearest neighbors
            const neighbors = [];
            for (let j = 0; j < this.options.nodeCount; j++) {
                if (i === j) continue;
                const p2 = { x: this.nodes[j*3], y: this.nodes[j*3+1], z: this.nodes[j*3+2] };
                const dist = Math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2);
                neighbors.push({ idx: j, dist });
            }

            neighbors.sort((a,b) => a.dist - b.dist);

            // Connect to closest few
            for (let k = 0; k < Math.min(this.options.connectionDensity, neighbors.length); k++) {
                const neighborIdx = neighbors[k].idx;
                // Avoid duplicates (only connect if i < neighborIdx)
                // But for directed pulses, maybe we want bidirectional?
                // Let's just add all directed edges for the pulse simulation logic.
                this.connections.push(i, neighborIdx);
            }
        }
    }

    // ========================================================================
    // WebGL2 (Static Network Structure)
    // ========================================================================

    initWebGL2() {
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = 'position:absolute; top:0; left:0; width:100%; height:100%; z-index:1;';
        this.container.appendChild(this.glCanvas);

        this.gl = this.glCanvas.getContext('webgl2');
        if (!this.gl) return;

        // Enable Depth Test and Blending
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        // --- Shaders ---
        const vs = `#version 300 es
            in vec3 a_pos;
            uniform mat4 u_matrix;
            uniform float u_pointSize;
            void main() {
                gl_Position = u_matrix * vec4(a_pos, 1.0);
                gl_PointSize = u_pointSize / gl_Position.w;
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            uniform vec4 u_color;
            out vec4 outColor;
            void main() {
                // Circular points
                vec2 coord = gl_PointCoord * 2.0 - 1.0;
                float dist = length(coord);
                if (dist > 1.0) discard;
                float alpha = 1.0 - smoothstep(0.8, 1.0, dist);
                outColor = vec4(u_color.rgb, u_color.a * alpha);
            }
        `;

        this.glProgram = this.createProgram(vs, fs);
        this.uMatrixLoc = this.gl.getUniformLocation(this.glProgram, 'u_matrix');

        // Buffer Nodes
        const nodeData = new Float32Array(this.nodes);
        this.glVaoNodes = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVaoNodes);
        const nodeBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, nodeBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, nodeData, this.gl.STATIC_DRAW);
        const posLoc = this.gl.getAttribLocation(this.glProgram, 'a_pos');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        // Buffer Lines
        const lineData = [];
        for (let i = 0; i < this.connections.length; i+=2) {
            const idx1 = this.connections[i];
            const idx2 = this.connections[i+1];

            lineData.push(
                this.nodes[idx1*3], this.nodes[idx1*3+1], this.nodes[idx1*3+2],
                this.nodes[idx2*3], this.nodes[idx2*3+1], this.nodes[idx2*3+2]
            );
        }

        this.glVaoLines = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.glVaoLines);
        const lineBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, lineBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(lineData), this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 3, this.gl.FLOAT, false, 0, 0);

        this.resizeGL();
    }

    createProgram(vsSrc, fsSrc) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vs, vsSrc);
        this.gl.compileShader(vs);
        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fs, fsSrc);
        this.gl.compileShader(fs);
        const p = this.gl.createProgram();
        this.gl.attachShader(p, vs);
        this.gl.attachShader(p, fs);
        this.gl.linkProgram(p);
        return p;
    }

    // ========================================================================
    // WebGPU (Dynamic Pulses)
    // ========================================================================

    async initWebGPU() {
        this.gpuCanvas = document.createElement('canvas');
        this.gpuCanvas.style.cssText = 'position:absolute; top:0; left:0; width:100%; height:100%; z-index:2; pointer-events:none;';
        this.container.appendChild(this.gpuCanvas);

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        this.context = this.gpuCanvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: format, alphaMode: 'premultiplied' });

        // --- Data Prep for GPU ---

        // 1. Path Buffer: Array of [startX, startY, startZ, endX, endY, endZ] (padded to vec4)
        // Actually simpler: Just store indices? No, storing positions avoids indirection in shader.
        // Each connection is a potential path.
        const pathDataArray = [];
        for (let i = 0; i < this.connections.length; i+=2) {
            const idx1 = this.connections[i];
            const idx2 = this.connections[i+1];
            // vec4 start, vec4 end
            pathDataArray.push(
                this.nodes[idx1*3], this.nodes[idx1*3+1], this.nodes[idx1*3+2], 0, // start
                this.nodes[idx2*3], this.nodes[idx2*3+1], this.nodes[idx2*3+2], 0  // end
            );
        }
        const numPaths = this.connections.length / 2;
        const pathBuffer = this.device.createBuffer({
            size: pathDataArray.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(pathBuffer, 0, new Float32Array(pathDataArray));
        this.pathBuffer = pathBuffer;

        // 2. Pulse Buffer: [pathIndex (f32), progress (f32), speed (f32), padding]
        const pulseData = new Float32Array(this.options.pulseCount * 4);
        for(let i=0; i<this.options.pulseCount; i++) {
            pulseData[i*4+0] = Math.floor(Math.random() * numPaths); // random path
            pulseData[i*4+1] = Math.random(); // random progress
            pulseData[i*4+2] = 0.5 + Math.random(); // speed
            pulseData[i*4+3] = 0; // padding
        }

        this.pulseBuffer = this.device.createBuffer({
            size: pulseData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.pulseBuffer, 0, pulseData);

        // 3. Uniforms
        this.uniformBuffer = this.device.createBuffer({
            size: 80, // 16 floats (mat4) + padding/time
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // --- Compute Pipeline ---
        const computeShader = `
            struct Pulse {
                pathIdx : f32,
                progress : f32,
                speed : f32,
                pad : f32,
            }

            struct Path {
                start : vec4f,
                end : vec4f,
            }

            @group(0) @binding(0) var<storage, read_write> pulses : array<Pulse>;
            @group(0) @binding(1) var<storage, read> paths : array<Path>;
            @group(0) @binding(2) var<uniform> uniforms : vec4f; // x = dt, y = numPaths

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id : vec3u) {
                let i = id.x;
                if (i >= arrayLength(&pulses)) { return; }

                var p = pulses[i];

                // Move pulse
                p.progress += p.speed * uniforms.x;

                // If finished, pick new random path (simple fake random)
                if (p.progress >= 1.0) {
                    p.progress = 0.0;
                    // Pseudo random based on id and time
                    let r = fract(sin(dot(vec2f(f32(i), uniforms.z), vec2f(12.9898, 78.233))) * 43758.5453);
                    p.pathIdx = floor(r * uniforms.y);
                }

                pulses[i] = p;
            }
        `;

        const computeModule = this.device.createShaderModule({ code: computeShader });
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' }
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.pulseBuffer } },
                { binding: 1, resource: { buffer: this.pathBuffer } },
                { binding: 2, resource: { buffer: this.uniformBuffer } } // actually using same uniform buffer but only part of it?
                // Let's create a separate small uniform buffer for compute params to be safe/clean
            ]
        });

        // Fix: Separate uniform buffer for compute params
        this.computeParamsBuffer = this.device.createBuffer({
            size: 16, // vec4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        // Re-create bind group with correct buffer
        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.pulseBuffer } },
                { binding: 1, resource: { buffer: this.pathBuffer } },
                { binding: 2, resource: { buffer: this.computeParamsBuffer } }
            ]
        });


        // --- Render Pipeline ---
        const drawShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct Pulse {
                pathIdx : f32,
                progress : f32,
                speed : f32,
                pad : f32,
            }

            struct Path {
                start : vec4f,
                end : vec4f,
            }

            struct Uniforms {
                modelViewProjection : mat4x4f,
            }

            @group(0) @binding(0) var<uniform> uniforms : Uniforms;
            @group(0) @binding(1) var<storage, read> paths : array<Path>;

            @vertex
            fn vs_main(
                @location(0) pathIdx : f32,
                @location(1) progress : f32
            ) -> VertexOutput {
                var output : VertexOutput;

                let idx = u32(pathIdx);
                let path = paths[idx];

                let pos = mix(path.start.xyz, path.end.xyz, progress);

                output.position = uniforms.modelViewProjection * vec4f(pos, 1.0);
                output.color = vec4f(0.0, 1.0, 1.0, 1.0 - abs(progress - 0.5)*1.5); // Fade at ends

                // Point size hack not available in WGSL directly without point-list topology adjustment or quad expansion
                // For simplicity, we assume point-list with gl_PointSize equivalent if supported,
                // but WebGPU defaults don't scale points.
                // We'll rely on 1px points or similar.
                // Actually, let's make them slightly brighter/additive.

                return output;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        const renderModule = this.device.createShaderModule({ code: drawShader });

        // Render Bind Group
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } }, // MVP matrix
                { binding: 1, resource: { buffer: this.pathBuffer } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 16, // Pulse struct size
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32' }, // pathIdx
                        { shaderLocation: 1, offset: 4, format: 'float32' }, // progress
                    ]
                }]
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }, // Additive blending
                        alpha: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }
                    }
                }]
            },
            primitive: { topology: 'point-list' },
            depthStencil: undefined // No depth write for additive particles usually
        });

        this.resizeGPU();
        return true;
    }

    addWebGPUNotSupportedMessage() {
        const msg = document.createElement('div');
        msg.innerHTML = "⚠️ WebGPU Not Available (Neural Network runs in reduced mode)";
        msg.style.cssText = "position:absolute; bottom:10px; right:10px; color:white; background:rgba(255,0,0,0.5); padding:5px; border-radius:4px;";
        this.container.appendChild(msg);
    }

    // ========================================================================
    // Loop
    // ========================================================================

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        this.resizeGL(w*dpr, h*dpr);
        this.resizeGPU(w*dpr, h*dpr);
    }

    resizeGL(w, h) {
        if(this.glCanvas) {
            this.glCanvas.width = w;
            this.glCanvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
    }

    resizeGPU(w, h) {
        if(this.gpuCanvas) {
            this.gpuCanvas.width = w;
            this.gpuCanvas.height = h;
        }
    }

    animate() {
        if (!this.isActive) return;

        const time = performance.now() * 0.001;

        // Camera Orbit
        const radius = 2.5;
        const camX = Math.sin(time * 0.2) * radius;
        const camZ = Math.cos(time * 0.2) * radius;
        const camY = Math.sin(time * 0.1) * 0.5;

        // Matrix (Simple Projection * View)
        const aspect = this.container.clientWidth / this.container.clientHeight;
        const fov = 60 * Math.PI / 180;
        const f = 1.0 / Math.tan(fov / 2);
        const zNear = 0.1;
        const zFar = 100.0;

        // Projection
        const proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (zFar + zNear) / (zNear - zFar), -1,
            0, 0, (2 * zFar * zNear) / (zNear - zFar), 0
        ];

        // View (LookAt 0,0,0)
        // Simplified LookAt from [camX, camY, camZ] to [0,0,0]
        const zAxis = this.normalize([camX, camY, camZ]);
        const xAxis = this.normalize(this.cross([0,1,0], zAxis));
        const yAxis = this.cross(zAxis, xAxis);

        const view = [
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -this.dot(xAxis, [camX,camY,camZ]), -this.dot(yAxis, [camX,camY,camZ]), -this.dot(zAxis, [camX,camY,camZ]), 1
        ];

        const vpMatrix = this.multiplyMatrices(proj, view);

        // --- Render WebGL2 ---
        if(this.gl) {
            this.gl.clearColor(0.02, 0.02, 0.05, 1);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            this.gl.useProgram(this.glProgram);
            this.gl.uniformMatrix4fv(this.uMatrixLoc, false, vpMatrix);

            // Draw Nodes
            this.gl.bindVertexArray(this.glVaoNodes);
            this.gl.uniform4f(this.gl.getUniformLocation(this.glProgram, 'u_color'), 0.2, 0.5, 1.0, 1.0);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_pointSize'), 200.0);
            this.gl.drawArrays(this.gl.POINTS, 0, this.nodes.length / 3);

            // Draw Connections
            this.gl.bindVertexArray(this.glVaoLines);
            this.gl.uniform4f(this.gl.getUniformLocation(this.glProgram, 'u_color'), 0.2, 0.5, 1.0, 0.2);
            this.gl.uniform1f(this.gl.getUniformLocation(this.glProgram, 'u_pointSize'), 1.0);
            this.gl.drawArrays(this.gl.LINES, 0, this.connections.length);
        }

        // --- Render WebGPU ---
        if(this.device && this.context) {
            // Update Uniforms
            this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array(vpMatrix));
            this.device.queue.writeBuffer(this.computeParamsBuffer, 0, new Float32Array([0.016, this.connections.length/2, time, 0]));

            const commandEncoder = this.device.createCommandEncoder();

            // Compute
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.options.pulseCount / 64));
            computePass.end();

            // Render
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
            renderPass.setBindGroup(0, this.renderBindGroup);
            renderPass.setVertexBuffer(0, this.pulseBuffer);
            renderPass.draw(this.options.pulseCount);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    // Math Helpers
    normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        return [v[0]/len, v[1]/len, v[2]/len];
    }

    cross(a, b) {
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ];
    }

    dot(a, b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    multiplyMatrices(a, b) {
        const out = [];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += b[i * 4 + k] * a[k * 4 + j];
                }
                out.push(sum);
            }
        }
        return out;
    }

    destroy() {
        this.isActive = false;
        if(this.animationId) cancelAnimationFrame(this.animationId);
        // Clean up GL/GPU resources if strictly needed,
        // though page navigation handles most in this MPA setup.
    }
}

if (typeof window !== 'undefined') {
    window.NeuralNetworkExperiment = NeuralNetworkExperiment;
}
