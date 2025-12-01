
    // --- UI Logic ---
    const errorLog = document.getElementById('error-log');
    const logError = (msg) => { if(errorLog) errorLog.textContent = msg; console.error(msg); };

    const labContainer = document.getElementById('lab-container');
    const backingCanvas = document.getElementById('backing-canvas');
    const webglCanvas = document.getElementById('webgl-canvas');
    const webgpuCanvas = document.getElementById('webgpu-canvas');
    const blendSelect = document.getElementById('blend-mode-select');
    const glBlendSelect = document.getElementById('gl-blend-select');
    const layerOrderSelect = document.getElementById('layer-order-select');
    const backingStyleSelect = document.getElementById('backing-style-select');
    const toggleBtn = document.getElementById('toggle-btn');

    let glContextState = {
        blendMode: 'additive',
        backingStyle: 'none'
    };

    // Layer order control
    if (layerOrderSelect) {
        layerOrderSelect.addEventListener('change', (e) => {
            // Remove all order classes
            labContainer.classList.remove('order-gpu-gl', 'order-gl-gpu', 'order-three-layer');
            // Add selected order class
            labContainer.classList.add(e.target.value);
            
            // Show backing canvas only in three-layer mode or when backing style is set
            updateBackingVisibility();
        });
        // Initialize
        labContainer.classList.add(layerOrderSelect.value);
    }

    // Backing style control
    if (backingStyleSelect) {
        backingStyleSelect.addEventListener('change', (e) => {
            glContextState.backingStyle = e.target.value;
            updateBackingVisibility();
        });
    }

    function updateBackingVisibility() {
        const isThreeLayer = layerOrderSelect && layerOrderSelect.value === 'order-three-layer';
        const hasBackingStyle = glContextState.backingStyle !== 'none';
        
        if (backingCanvas) {
            backingCanvas.style.display = (isThreeLayer || hasBackingStyle) ? 'block' : 'none';
        }
    }

    // Auto-enable backing for overlay/soft-light modes
    function autoAdjustForBlendMode(blendMode) {
        // These blend modes work better with a mid-gray backing
        const needsBacking = ['overlay', 'soft-light', 'color-dodge', 'color-burn'].includes(blendMode);
        
        if (needsBacking && glContextState.backingStyle === 'none') {
            // Automatically set mid-gray backing and trigger UI update
            glContextState.backingStyle = 'mid-gray';
            if (backingStyleSelect) {
                backingStyleSelect.value = 'mid-gray';
            }
        }
        updateBackingVisibility();
    }

    if (blendSelect) {
        blendSelect.addEventListener('change', (e) => {
            webglCanvas.style.mixBlendMode = e.target.value;
            autoAdjustForBlendMode(e.target.value);
        });
        // Initialize
        if (webglCanvas) webglCanvas.style.mixBlendMode = blendSelect.value;
    }

    if (glBlendSelect) {
        glBlendSelect.addEventListener('change', (e) => {
            glContextState.blendMode = e.target.value;
        });
    }

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            webglCanvas.style.opacity = webglCanvas.style.opacity === '0' ? '1' : '0';
        });
    }

    // --- Backing Layer: 2D Canvas for mid-gray/gradient backgrounds ---
    function initBackingLayer() {
        const canvas = backingCanvas;
        const statusEl = document.getElementById('backing-status');
        
        if (!canvas) {
            if (statusEl) statusEl.innerText = "N/A";
            return null;
        }
        
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            if (statusEl) statusEl.innerText = "N/A";
            return null;
        }
        
        if (statusEl) statusEl.innerText = "Active";
        
        // Resize observer
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                const width = entry.contentBoxSize
                    ? entry.contentBoxSize[0].inlineSize
                    : entry.contentRect.width;
                const height = entry.contentBoxSize
                    ? entry.contentBoxSize[0].blockSize
                    : entry.contentRect.height;
                
                canvas.width = Math.max(1, width);
                canvas.height = Math.max(1, height);
            }
        });
        observer.observe(canvas);
        
        // Initially hide backing canvas
        canvas.style.display = 'none';
        
        return {
            render: (t) => {
                const style = glContextState.backingStyle;
                const w = canvas.width;
                const h = canvas.height;
                
                if (style === 'none') {
                    ctx.clearRect(0, 0, w, h);
                    return;
                }
                
                if (style === 'mid-gray') {
                    // Solid mid-gray for overlay blend mode to work
                    ctx.fillStyle = '#808080';
                    ctx.fillRect(0, 0, w, h);
                } else if (style === 'gradient') {
                    // Radial gradient from center
                    const gradient = ctx.createRadialGradient(w/2, h/2, 0, w/2, h/2, Math.max(w, h) * 0.6);
                    gradient.addColorStop(0, '#606060');
                    gradient.addColorStop(0.5, '#404040');
                    gradient.addColorStop(1, '#202020');
                    ctx.fillStyle = gradient;
                    ctx.fillRect(0, 0, w, h);
                } else if (style === 'noise') {
                    // Animated noise pattern
                    // Noise parameters: base gray level and variation range
                    const NOISE_BASE_GRAY = 98;  // Base gray level (out of 255)
                    const NOISE_VARIATION = 60;   // Amount of variation around base
                    const NOISE_SPEED = 0.01;     // Animation speed factor
                    
                    const imageData = ctx.createImageData(w, h);
                    const data = imageData.data;
                    const seed = t * 1000;
                    
                    for (let i = 0; i < data.length; i += 4) {
                        const noise = (Math.sin(seed + i * NOISE_SPEED) * 0.5 + 0.5) * NOISE_VARIATION + NOISE_BASE_GRAY;
                        data[i] = noise;     // R
                        data[i+1] = noise;   // G
                        data[i+2] = noise;   // B
                        data[i+3] = 255;     // A
                    }
                    ctx.putImageData(imageData, 0, 0);
                }
            }
        };
    }

    // --- WebGPU: The Circuit (Background) ---
    async function initWebGPU() {
        const canvas = document.getElementById('webgpu-canvas');
        const statusEl = document.getElementById('gpu-status');

        if (!navigator.gpu) {
            if(statusEl) statusEl.innerText = "N/A";
            return null;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                if(statusEl) statusEl.innerText = "No Adapter";
                return null;
            }
            const device = await adapter.requestDevice();
            const context = canvas.getContext('webgpu');
            const format = navigator.gpu.getPreferredCanvasFormat();
            context.configure({ device, format, alphaMode: 'opaque' });

            const module = device.createShaderModule({
                code: `
                    struct VertexOutput {
                        @builtin(position) position : vec4f,
                        @location(0) uv : vec2f,
                    }

                    @vertex
                    fn vs_main(@builtin(vertex_index) vIdx : u32) -> VertexOutput {
                        var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
                        var out : VertexOutput;
                        out.position = vec4f(pos[vIdx], 0.0, 1.0);
                        out.uv = pos[vIdx] * 0.5 + 0.5;
                        return out;
                    }

                    @group(0) @binding(0) var<uniform> time : f32;
                    @group(0) @binding(1) var<uniform> res : vec2f;

                    @fragment
                    fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
                        var aspect = res.x / res.y;
                        var cUV = uv - 0.5;
                        cUV.x *= aspect;

                        var col = vec3f(0.02, 0.02, 0.03);
                        let grid = step(0.98, fract(uv * 50.0));
                        col += vec3f(0.05) * grid.x;

                        let lineY = abs(cUV.y);
                        let rail = smoothstep(0.04, 0.035, abs(lineY - 0.05));
                        col += vec3f(0.2) * rail;

                        let rowWidth = 1.8;
                        let blipPos = (fract(time * 0.5) - 0.5) * rowWidth;

                        let dx = cUV.x - blipPos;
                        let dist = length(vec2f(dx, cUV.y));

                        let energy = 0.02 / (dist + 0.001);
                        let trail = exp(-10.0 * max(0.0, -dx));
                        let beam = smoothstep(0.05, 0.0, abs(cUV.y)) * trail * step(dx, 0.0);

                        let glowColor = vec3f(0.0, 0.8, 1.0);
                        col += glowColor * (pow(energy, 1.5) * 0.5);
                        col += glowColor * beam * 2.0;

                        return vec4f(col, 1.0);
                    }
                `
            });

            const U_SIZE = 16 + 16;
            const uBuf = device.createBuffer({ size: U_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

            const pipeline = device.createRenderPipeline({
                layout: 'auto',
                vertex: { module, entryPoint: 'vs_main' },
                fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
                primitive: { topology: 'triangle-list' },
            });

            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uBuf, offset: 0, size: 4 } },
                    { binding: 1, resource: { buffer: uBuf, offset: 16, size: 8 } }
                ],
            });

            if(statusEl) statusEl.innerText = "Active";

            const observer = new ResizeObserver(entries => {
                for (const entry of entries) {
                    const width = entry.contentBoxSize
                        ? entry.contentBoxSize[0].inlineSize
                        : entry.contentRect.width;
                    const height = entry.contentBoxSize
                        ? entry.contentBoxSize[0].blockSize
                        : entry.contentRect.height;

                    canvas.width = Math.max(1, width);
                    canvas.height = Math.max(1, height);
                }
            });
            observer.observe(canvas);

            return {
                render: (t) => {
                    device.queue.writeBuffer(uBuf, 0, new Float32Array([t]));
                    device.queue.writeBuffer(uBuf, 16, new Float32Array([canvas.width, canvas.height]));
                    const enc = device.createCommandEncoder();
                    const pass = enc.beginRenderPass({
                        colorAttachments: [{
                            view: context.getCurrentTexture().createView(),
                            clearValue: { r: 0, g: 0, b: 0, a: 1 },
                            loadOp: 'clear', storeOp: 'store'
                        }]
                    });
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(0, bindGroup);
                    pass.draw(3);
                    pass.end();
                    device.queue.submit([enc.finish()]);
                }
            };
        } catch(e) { logError(e); }
    }

    // --- WebGL2: The Lens / LED (Foreground) ---
    function initWebGL2() {
        const canvas = document.getElementById('webgl-canvas');
        const statusEl = document.getElementById('gl-status');
        const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });

        if (!gl) {
            if(statusEl) statusEl.innerText = "N/A";
            return null;
        }
        if(statusEl) statusEl.innerText = "Active";

        // Modified resize logic to use ResizeObserver
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                const width = entry.contentBoxSize
                    ? entry.contentBoxSize[0].inlineSize
                    : entry.contentRect.width;
                const height = entry.contentBoxSize
                    ? entry.contentBoxSize[0].blockSize
                    : entry.contentRect.height;

                canvas.width = Math.max(1, width);
                canvas.height = Math.max(1, height);
                gl.viewport(0, 0, canvas.width, canvas.height);
            }
        });
        observer.observe(canvas);

        const vs = `#version 300 es
        in vec2 a_pos;
        uniform float u_time;
        uniform vec2 u_res;
        uniform float u_scale;
        out vec4 v_col;
        out vec2 v_uv;

        void main() {
            float aspect = u_res.y / u_res.x;

            float rowWidth = 1.8;
            float numLeds = 16.0;
            float idx = float(gl_InstanceID);
            float xOffset = (idx / (numLeds - 1.0) - 0.5) * rowWidth;

            float blipPos = (fract(u_time * 0.5) - 0.5) * rowWidth;
            float dist = abs(xOffset - blipPos);

            float activeVal = 1.0 - smoothstep(0.0, 0.08, dist);

            vec3 colOff = vec3(0.2, 0.2, 0.2);
            vec3 colOn = vec3(0.8, 1.0, 1.0);
            vec3 finalRgb = mix(colOff, colOn, activeVal);

            float alpha = mix(0.5, 1.0, activeVal);

            v_col = vec4(finalRgb, alpha);
            v_uv = a_pos * 2.0;

            vec2 pos = a_pos * u_scale + vec2(xOffset, 0.0);
            pos.x *= aspect;
            gl_Position = vec4(pos, 0.0, 1.0);
        }`;

        const fs = `#version 300 es
        precision mediump float;
        in vec4 v_col;
        in vec2 v_uv;
        out vec4 outColor;

        void main() {
            float r = length(v_uv);

            float lensR = 0.65;
            float bezelR = 0.85;

            if (r > bezelR) discard;

            vec3 rgb;
            float alpha;

            if (r > lensR) {
                // --- PLASTIC BEZEL UPDATED ---
                // Using lighter Silver/Chrome colors (0.5 - 0.7) to survive Hard Light blend
                vec3 bezelColor = vec3(0.6, 0.6, 0.65);

                float bezelPos = (r - lensR) / (bezelR - lensR);

                // Rim Lighting
                vec2 dir = normalize(v_uv);
                float light = dot(dir, vec2(-0.707, 0.707));

                float highlight = smoothstep(0.0, 1.0, light) * 0.4; // Stronger highlight
                float shadow = smoothstep(0.0, -1.0, light) * 0.3;

                rgb = bezelColor + highlight - shadow;
                alpha = 1.0;

                alpha *= smoothstep(bezelR, bezelR - 0.02, r);
            } else {
                // --- GLASS LENS ---
                float lensNormR = r / lensR;
                float z = sqrt(1.0 - lensNormR*lensNormR);
                vec3 normal = vec3(v_uv / lensR, z);

                vec3 lightDir = normalize(vec3(-0.5, 0.5, 1.0));
                float diffuse = max(0.0, dot(normal, lightDir));
                float specular = pow(max(0.0, dot(reflect(-lightDir, normal), vec3(0,0,1))), 20.0);
                float fresnel = pow(1.0 - z, 3.0);

                rgb = v_col.rgb;
                rgb *= (0.5 + 0.5 * diffuse);

                // LENS FLARE STREAK
                float brightness = v_col.a;
                if (brightness > 0.8) {
                    float hDist = abs(v_uv.y);
                    float hFade = smoothstep(0.9, 0.0, abs(v_uv.x));
                    float streak = smoothstep(0.1, 0.0, hDist);

                    vec3 flareCol = vec3(0.5, 0.8, 1.0);
                    rgb += flareCol * streak * hFade * (brightness - 0.7);
                }

                rgb += vec3(1.0) * specular * 0.8 * v_col.a;
                rgb += vec3(1.0, 0.5, 0.0) * fresnel * 0.5;

                alpha = v_col.a;
            }

            // Output premultiplied
            outColor = vec4(rgb * alpha, alpha);
        }`;

        const p = gl.createProgram();
        const createS = (t, s) => { const x=gl.createShader(t); gl.shaderSource(x,s); gl.compileShader(x); return x; };
        gl.attachShader(p, createS(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, createS(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p);

        const locs = {
            pos: gl.getAttribLocation(p, 'a_pos'),
            time: gl.getUniformLocation(p, 'u_time'),
            res: gl.getUniformLocation(p, 'u_res'),
            scale: gl.getUniformLocation(p, 'u_scale')
        };

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        gl.enableVertexAttribArray(locs.pos);
        gl.vertexAttribPointer(locs.pos, 2, gl.FLOAT, false, 0, 0);

        gl.enable(gl.BLEND);

        return {
            render: (t) => {
                gl.clearColor(0,0,0,0);
                gl.clear(gl.COLOR_BUFFER_BIT);

                if (glContextState.blendMode === 'additive') {
                    gl.blendFunc(gl.ONE, gl.ONE);
                } else {
                    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
                }

                gl.useProgram(p);
                gl.uniform2f(locs.res, canvas.width, canvas.height);
                gl.uniform1f(locs.time, t);
                gl.uniform1f(locs.scale, 0.18);

                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 16);
            }
        };
    }

    async function main() {
        const backing = initBackingLayer();
        const gpu = await initWebGPU();
        const gl = initWebGL2();

        function frame(now) {
            const t = now * 0.001;
            if (backing) backing.render(t);
            if (gpu) gpu.render(t);
            if (gl) gl.render(t);
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    main();
