
// --- Global State ---
let appState = {
    vortexSpeed: 1.0,
    vortexColor: 0.5, // Hue
    gateRotation: 0.0,
    gateOpen: 0.0, // 0 to 1
    time: 0
};

// --- UI Initialization ---
function initHardwareUI() {
    const UI = window.UIComponents;
    if (!UI) {
        console.error("UIComponents not loaded");
        return;
    }

    const vortexContainer = document.getElementById('vortex-controls');
    const gateContainer = document.getElementById('gate-controls');
    const displayEl = document.getElementById('main-display');

    // Vortex Speed
    if (vortexContainer) {
        new UI.RotaryKnob(vortexContainer, {
            size: 70,
            min: 0,
            max: 5,
            value: 1,
            color: '#00ffff',
            label: 'SPEED',
            onChange: (val) => {
                appState.vortexSpeed = val;
                if(displayEl) displayEl.innerHTML = `VORTEX SPEED<br><span style="color:#fff">${val.toFixed(2)}</span>`;
            }
        });

        // Vortex Color (Hue)
        new UI.RotaryKnob(vortexContainer, {
            size: 70,
            min: 0,
            max: 1,
            value: 0.5,
            color: '#ff00ff',
            label: 'COLOR',
            onChange: (val) => {
                appState.vortexColor = val;
            }
        });
    }

    // Gate Controls
    if (gateContainer) {
        new UI.RotaryKnob(gateContainer, {
            size: 70,
            min: 0,
            max: 1,
            value: 0,
            color: '#ffaa00',
            label: 'OPEN',
            onChange: (val) => {
                appState.gateOpen = val;
                if(displayEl) displayEl.innerHTML = `GATE STATUS<br><span style="color:#fff">${(val*100).toFixed(0)}%</span>`;
            }
        });
    }
}

// --- Backing Layer: Starfield ---
function initBackingLayer() {
    const canvas = document.getElementById('backing-canvas');
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');

    let stars = [];
    const numStars = 200;

    // Resize
    const observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            canvas.width = entry.contentRect.width;
            canvas.height = entry.contentRect.height;
            initStars();
        }
    });
    observer.observe(canvas);

    function initStars() {
        stars = [];
        for(let i=0; i<numStars; i++) {
            stars.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 2,
                speed: Math.random() * 0.5 + 0.1
            });
        }
    }
    initStars();

    return {
        render: (t) => {
            ctx.fillStyle = '#050505';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#ffffff';
            stars.forEach(star => {
                ctx.globalAlpha = Math.random() * 0.5 + 0.5;
                ctx.beginPath();
                ctx.arc(star.x, star.y, star.size, 0, Math.PI*2);
                ctx.fill();

                // Parallax/Movement
                let dx = (canvas.width/2 - star.x) * 0.001 * appState.vortexSpeed;
                let dy = (canvas.height/2 - star.y) * 0.001 * appState.vortexSpeed;
                star.x += dx;
                star.y += dy;

                // Respawn if too close to center
                if(Math.abs(star.x - canvas.width/2) < 10 && Math.abs(star.y - canvas.height/2) < 10) {
                     star.x = Math.random() * canvas.width;
                     star.y = Math.random() * canvas.height;
                }
            });
            ctx.globalAlpha = 1.0;
        }
    };
}

// --- WebGPU: Particle Vortex ---
async function initWebGPU() {
    const canvas = document.getElementById('webgpu-canvas');
    const statusEl = document.getElementById('gpu-status');

    if (!navigator.gpu) {
        if(statusEl) statusEl.innerText = "N/A";
        return null;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if(!adapter) throw new Error("No adapter");
        const device = await adapter.requestDevice();
        const context = canvas.getContext('webgpu');
        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: 'premultiplied' });

        const numParticles = 100000;

        // Shader Modules
        const computeShader = `
            struct Particle {
                pos : vec2f,
                vel : vec2f,
                life : f32,
                pad : f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> params : vec4f; // x: time, y: speed, z: center_x, w: center_y

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3u) {
                let idx = global_id.x;
                if (idx >= ${numParticles}) { return; }

                var p = particles[idx];
                let center = params.zw;
                let dt = 0.016 * params.y;

                // Physics: Spiral towards center
                let diff = center - p.pos;
                let dist = length(diff);
                let dir = normalize(diff);

                // Tangent force
                let tangent = vec2f(-dir.y, dir.x);

                p.vel += (dir * 100.0 + tangent * 500.0) * dt / (dist + 10.0);
                p.vel *= 0.96; // Damping

                p.pos += p.vel * dt;

                // Respawn
                if (dist < 5.0 || dist > 1500.0 || p.life <= 0.0) {
                    // Random pos on outer ring
                    let angle = f32(idx) * 0.123 + params.x;
                    let r = 800.0 + fract(sin(f32(idx))*43758.54)*200.0;
                    p.pos = center + vec2f(cos(angle), sin(angle)) * r;
                    p.vel = vec2f(0.0);
                    p.life = 1.0;
                } else {
                    p.life -= 0.001 * params.y;
                }

                particles[idx] = p;
            }
        `;

        const drawShader = `
            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) color : vec4f,
            }

            struct Particle {
                pos : vec2f,
                vel : vec2f,
                life : f32,
                pad : f32,
            }

            @group(0) @binding(0) var<storage, read> particles : array<Particle>;
            @group(0) @binding(1) var<uniform> viewport : vec2f;
            @group(0) @binding(2) var<uniform> colorParam : f32; // Hue

            // Helper for HSV to RGB
            fn hsv2rgb(c : vec3f) -> vec3f {
                let K = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vIdx : u32) -> VertexOutput {
                let p = particles[vIdx];
                var out : VertexOutput;

                // Map to clip space
                let clipX = (p.pos.x / viewport.x) * 2.0 - 1.0;
                let clipY = (1.0 - (p.pos.y / viewport.y)) * 2.0 - 1.0; // Flip Y

                out.position = vec4f(clipX, clipY, 0.0, 1.0);

                // Color based on velocity/life
                let speed = length(p.vel);
                let hue = colorParam + speed * 0.001;
                let rgb = hsv2rgb(vec3f(hue, 0.8, 1.0));

                out.color = vec4f(rgb, p.life);
                return out;
            }

            @fragment
            fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
                return color;
            }
        `;

        // Compute Pipeline
        const computeModule = device.createShaderModule({ code: computeShader });
        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'main' }
        });

        // Render Pipeline
        const drawModule = device.createShaderModule({ code: drawShader });
        const drawPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: drawModule, entryPoint: 'vs_main' },
            fragment: { module: drawModule, entryPoint: 'fs_main', targets: [{ format, blend: {
                color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' }
            } }] },
            primitive: { topology: 'point-list' }
        });

        // Buffers
        const particleBufferSize = numParticles * 32; // 4 floats * 4 bytes * 2 vec2 + 2 floats? Wait struct is vec2, vec2, f32, f32 -> 8 floats = 32 bytes
        const particleBuffer = device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
            mappedAtCreation: true
        });

        // Init particles
        const pData = new Float32Array(particleBuffer.getMappedRange());
        for (let i = 0; i < numParticles; i++) {
            const off = i * 8;
            pData[off] = Math.random() * 800; // x
            pData[off+1] = Math.random() * 600; // y
            pData[off+2] = 0; // vx
            pData[off+3] = 0; // vy
            pData[off+4] = Math.random(); // life
        }
        particleBuffer.unmap();

        const uniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const viewportBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const colorBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        // Bind Groups
        const computeBindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: uniformBuffer } }
            ]
        });

        const drawBindGroup = device.createBindGroup({
            layout: drawPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: viewportBuffer } },
                { binding: 2, resource: { buffer: colorBuffer } }
            ]
        });

        if(statusEl) statusEl.innerText = "ACTIVE - " + numParticles + " pts";

        // Resize
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                canvas.width = Math.max(1, entry.contentRect.width);
                canvas.height = Math.max(1, entry.contentRect.height);
            }
        });
        observer.observe(canvas);

        return {
            render: (t) => {
                // Update Uniforms
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([t, appState.vortexSpeed, centerX, centerY]));
                device.queue.writeBuffer(viewportBuffer, 0, new Float32Array([canvas.width, canvas.height]));
                device.queue.writeBuffer(colorBuffer, 0, new Float32Array([appState.vortexColor]));

                const commandEncoder = device.createCommandEncoder();

                // Compute Pass
                const computePass = commandEncoder.beginComputePass();
                computePass.setPipeline(computePipeline);
                computePass.setBindGroup(0, computeBindGroup);
                computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));
                computePass.end();

                // Render Pass
                const textureView = context.getCurrentTexture().createView();
                const renderPass = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: textureView,
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: 'clear',
                        storeOp: 'store'
                    }]
                });
                renderPass.setPipeline(drawPipeline);
                renderPass.setBindGroup(0, drawBindGroup);
                renderPass.draw(numParticles);
                renderPass.end();

                device.queue.submit([commandEncoder.finish()]);
            }
        };

    } catch(e) {
        console.error("WebGPU Init Failed:", e);
        if(statusEl) statusEl.innerText = "ERROR";
        return null;
    }
}

// --- WebGL2: 3D Gate Geometry ---
function initWebGL2() {
    const canvas = document.getElementById('webgl-canvas');
    const statusEl = document.getElementById('gl-status');
    const gl = canvas.getContext('webgl2', { alpha: true });

    if (!gl) {
        if(statusEl) statusEl.innerText = "N/A";
        return null;
    }
    if(statusEl) statusEl.innerText = "ACTIVE";

    // Resize
    const observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            canvas.width = Math.max(1, entry.contentRect.width);
            canvas.height = Math.max(1, entry.contentRect.height);
            gl.viewport(0, 0, canvas.width, canvas.height);
        }
    });
    observer.observe(canvas);

    // Simple Cube/Torus Geometry
    // We'll generate a ring of cubes
    const positions = [];
    const normals = [];
    const count = 12; // number of segments

    function createCube(mx, my, mz, scale) {
        // ... simplified cube generation ...
        // Vertices for a cube face
        const v = [
             -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1, // Back
             -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1, // Front
             -1,-1,-1, -1, 1,-1, -1, 1, 1, -1,-1, 1, // Left
              1,-1,-1,  1, 1,-1,  1, 1, 1,  1,-1, 1, // Right
             -1,-1,-1,  1,-1,-1,  1,-1, 1, -1,-1, 1, // Bottom
             -1, 1,-1,  1, 1,-1,  1, 1, 1, -1, 1, 1  // Top
        ];
        const n = [
             0, 0,-1,  0, 0,-1,  0, 0,-1,  0, 0,-1,
             0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1,
            -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
             1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0,
             0,-1, 0,  0,-1, 0,  0,-1, 0,  0,-1, 0,
             0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0
        ];
        // Indices would be needed for indexed draw, but let's just do triangles
        const idx = [
            0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11,
            12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
        ];

        for(let i=0; i<idx.length; i++) {
            let ii = idx[i] * 3;
            positions.push(v[ii]*scale+mx, v[ii+1]*scale+my, v[ii+2]*scale+mz);
            normals.push(n[ii], n[ii+1], n[ii+2]);
        }
    }

    // Initialize Geometry in loop in shader instead? No, let's just push static geo.
    // Actually, let's use instancing to draw the ring segments.
    // Base geometry: One Cube
    createCube(0,0,0, 0.5); // unitish cube

    const vs = `#version 300 es
    in vec3 a_pos;
    in vec3 a_norm;

    uniform mat4 u_proj;
    uniform mat4 u_view;
    uniform float u_time;
    uniform float u_open; // Expansion of the gate

    out vec3 v_norm;
    out float v_instance;

    void main() {
        int i = gl_InstanceID;
        float fi = float(i);
        float count = 12.0;

        float angle = (fi / count) * 3.14159 * 2.0 + u_time * 0.5;
        float radius = 4.0 + u_open * 2.0; // Expand

        // Transform
        float c = cos(angle);
        float s = sin(angle);

        // Rotate around Z (circle in XY plane)
        mat4 rot = mat4(
            c, s, 0, 0,
           -s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );

        // Translate out
        vec3 offset = vec3(c * radius, s * radius, 0.0);

        // Local rotation of the piece
        float localRot = u_time * 2.0 + fi;
        float lc = cos(localRot);
        float ls = sin(localRot);
        mat4 lRot = mat4(
            lc, 0, ls, 0,
            0, 1, 0, 0,
           -ls, 0, lc, 0,
            0, 0, 0, 1
        );

        vec4 pos = lRot * vec4(a_pos, 1.0);
        pos = rot * pos; // Rotate to ring position orientation?
        // Actually, just translate
        pos.xyz += offset;

        // Tilt the whole ring
        mat4 tilt = mat4(
            1, 0, 0, 0,
            0, 0.8, -0.6, 0,
            0, 0.6, 0.8, 0,
            0, 0, 0, 1
        );

        gl_Position = u_proj * u_view * tilt * pos;
        v_norm = (tilt * rot * lRot * vec4(a_norm, 0.0)).xyz;
        v_instance = fi;
    }`;

    const fs = `#version 300 es
    precision mediump float;

    in vec3 v_norm;
    in float v_instance;
    uniform float u_time;

    out vec4 outColor;

    void main() {
        vec3 normal = normalize(v_norm);
        vec3 light = normalize(vec3(0.5, 0.5, 1.0));

        float diff = max(0.0, dot(normal, light));

        vec3 baseCol = vec3(1.0, 0.5, 0.0);
        if (mod(v_instance, 2.0) == 0.0) baseCol = vec3(0.0, 0.8, 1.0);

        vec3 col = baseCol * (0.2 + 0.8 * diff);

        // Emission pulse
        float pulse = sin(u_time * 5.0 + v_instance) * 0.5 + 0.5;
        col += baseCol * pulse * 0.5;

        outColor = vec4(col, 1.0);
    }`;

    // Compile Shaders
    const p = gl.createProgram();
    const createS = (t, s) => { const x=gl.createShader(t); gl.shaderSource(x,s); gl.compileShader(x); return x; };
    const vShader = createS(gl.VERTEX_SHADER, vs);
    const fShader = createS(gl.FRAGMENT_SHADER, fs);
    gl.attachShader(p, vShader);
    gl.attachShader(p, fShader);
    gl.linkProgram(p);

    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        console.error("GL Link Error", gl.getProgramInfoLog(p));
        // Check shader logs
        console.error("VS Log:", gl.getShaderInfoLog(vShader));
        console.error("FS Log:", gl.getShaderInfoLog(fShader));
        return null;
    }

    const locs = {
        pos: gl.getAttribLocation(p, 'a_pos'),
        norm: gl.getAttribLocation(p, 'a_norm'),
        proj: gl.getUniformLocation(p, 'u_proj'),
        view: gl.getUniformLocation(p, 'u_view'),
        time: gl.getUniformLocation(p, 'u_time'),
        open: gl.getUniformLocation(p, 'u_open'),
    };

    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const normBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(locs.pos);
    gl.vertexAttribPointer(locs.pos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
    gl.enableVertexAttribArray(locs.norm);
    gl.vertexAttribPointer(locs.norm, 3, gl.FLOAT, false, 0, 0);

    return {
        render: (t) => {
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.clearColor(0,0,0,0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            gl.useProgram(p);

            // Matrices
            const aspect = canvas.width / canvas.height;
            const fov = 45 * Math.PI / 180;
            const zNear = 0.1;
            const zFar = 100.0;
            const f = 1.0 / Math.tan(fov / 2);
            const proj = new Float32Array([
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (zFar + zNear) / (zNear - zFar), -1,
                0, 0, (2 * zFar * zNear) / (zNear - zFar), 0
            ]);

            // View (Camera at 0,0,20)
            const view = new Float32Array([
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -20, 1
            ]);

            gl.uniformMatrix4fv(locs.proj, false, proj);
            gl.uniformMatrix4fv(locs.view, false, view);
            gl.uniform1f(locs.time, t);
            gl.uniform1f(locs.open, appState.gateOpen);

            // Draw 12 instances
            gl.drawArraysInstanced(gl.TRIANGLES, 0, positions.length/3, 12);
        }
    };
}


async function main() {
    initHardwareUI();
    const backing = initBackingLayer();
    const gpu = await initWebGPU();
    const gl = initWebGL2();

    function frame(now) {
        const t = now * 0.001;
        appState.time = t;

        if (backing) backing.render(t);
        if (gpu) gpu.render(t);
        if (gl) gl.render(t);

        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
} else {
    main();
}
