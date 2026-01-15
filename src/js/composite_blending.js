
    // --- Global State & Configuration ---
    const BLEND_MODES = [
        'normal', 'screen', 'overlay', 'hard-light', 'soft-light',
        'difference', 'exclusion', 'plus-lighter', 'multiply',
        'color-dodge', 'color-burn', 'luminosity', 'saturation', 'hue'
    ];

    const PRESETS = {
        'default': {
            blendMode: 'hard-light',
            layerOrder: 'order-gpu-gl', // GPU behind, GL in front
            backing: 'none',
            ledColors: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.6, 0.0], [0.2, 1.0, 0.2]]
        },
        'hologram': {
            blendMode: 'screen',
            layerOrder: 'order-gl-gpu', // GL behind GPU (affects blending if transparency used)
            backing: 'none',
            ledColors: [[0.0, 0.5, 1.0], [0.0, 0.8, 1.0], [0.0, 0.5, 1.0], [0.0, 0.8, 1.0]]
        },
        'xray': {
            blendMode: 'difference',
            layerOrder: 'order-gpu-gl',
            backing: 'mid-gray', // Difference needs contrast
            ledColors: [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        },
        'interference': {
            blendMode: 'exclusion',
            layerOrder: 'order-gpu-gl',
            backing: 'noise',
            ledColors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        }
    };

    let appState = {
        blendMode: 'hard-light',
        layerOrder: 'order-gpu-gl',
        backingStyle: 'none',
        glVisible: true,
        ledColors: PRESETS.default.ledColors,
        time: 0
    };

    // --- DOM Elements ---
    const canvasViewport = document.getElementById('canvas-viewport');
    const webglCanvas = document.getElementById('webgl-canvas');
    const webgpuCanvas = document.getElementById('webgpu-canvas');
    const backingCanvas = document.getElementById('backing-canvas');
    const displayEl = document.getElementById('main-display');

    // --- UI Initialization ---
    function initHardwareUI() {
        const UI = window.UIComponents;
        if (!UI) {
            console.error("UIComponents not loaded");
            return;
        }

        // 1. Blend Mode Knob
        const blendKnobContainer = document.getElementById('blend-knob-container');
        if (blendKnobContainer) {
            new UI.RotaryKnob(blendKnobContainer, {
                size: 80,
                min: 0,
                max: BLEND_MODES.length - 1,
                value: BLEND_MODES.indexOf(appState.blendMode),
                color: '#00ff88',
                label: 'BLEND',
                onChange: (val) => {
                    const idx = Math.round(val);
                    const mode = BLEND_MODES[idx];
                    updateBlendMode(mode);
                }
            });
        }

        // 2. Layer Order Switch
        const layerSwitch = document.getElementById('layer-order-switch');
        if (layerSwitch) {
            layerSwitch.addEventListener('click', () => {
                layerSwitch.classList.toggle('active');
                const isSwapped = layerSwitch.classList.contains('active');
                appState.layerOrder = isSwapped ? 'order-gl-gpu' : 'order-gpu-gl';
                updateLayerOrder();
                updateDisplay("LAYER ORDER", isSwapped ? "GL -> GPU" : "GPU -> GL");
            });
        }

        // 3. GL Toggle Switch
        const glSwitch = document.getElementById('gl-toggle-switch');
        if (glSwitch) {
            glSwitch.addEventListener('click', () => {
                glSwitch.classList.toggle('active');
                appState.glVisible = glSwitch.classList.contains('active');
                webglCanvas.style.opacity = appState.glVisible ? '1' : '0';
                updateDisplay("GL LAYER", appState.glVisible ? "VISIBLE" : "HIDDEN");
            });
        }

        // 4. Preset Buttons
        const presetBtns = document.querySelectorAll('.preset-btn');
        presetBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // UI Update
                presetBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Logic Update
                const presetName = btn.dataset.preset;
                applyPreset(presetName);
            });
        });
    }

    // --- State Update Helpers ---

    function updateBlendMode(mode) {
        if (appState.blendMode === mode) return;
        appState.blendMode = mode;
        
        // Update CSS
        if (webglCanvas) webglCanvas.style.mixBlendMode = mode;

        // Update Display
        updateDisplay("BLEND MODE", mode.toUpperCase());

        // Auto-adjust backing for certain modes if in default preset workflow?
        // Kept simple: Presets override everything, Manual knob just changes blend mode.
    }

    function updateLayerOrder() {
        if (canvasViewport) {
            canvasViewport.classList.remove('order-gl-gpu', 'order-gpu-gl');
            canvasViewport.classList.add(appState.layerOrder);
        }
    }

    function updateDisplay(line1, line2) {
        if (displayEl) {
            displayEl.innerHTML = `${line1}<br><span style="color:#fff">${line2}</span>`;
        }
    }

    function applyPreset(name) {
        const preset = PRESETS[name];
        if (!preset) return;

        // Apply state
        appState.blendMode = preset.blendMode;
        appState.layerOrder = preset.layerOrder;
        appState.backingStyle = preset.backing;
        appState.ledColors = preset.ledColors;

        // Sync UI - Canvas
        if (webglCanvas) webglCanvas.style.mixBlendMode = appState.blendMode;
        updateLayerOrder();
        
        // Sync UI - Controls (This is tricky with the knob, we might need to update the knob instance if we had reference)
        // For now, the visual knob position might lag behind presets, which is a common "soft takeover" issue in hardware.
        // We will just update the text display.
        updateDisplay("PRESET LOADED", name.toUpperCase());

        // Backing Visibility
        if (backingCanvas) {
            backingCanvas.style.display = (appState.backingStyle !== 'none') ? 'block' : 'none';
        }
        
        // Layer Order Switch Visual
        const layerSwitch = document.getElementById('layer-order-switch');
        if (layerSwitch) {
            if (appState.layerOrder === 'order-gl-gpu') layerSwitch.classList.add('active');
            else layerSwitch.classList.remove('active');
        }
    }


    // --- Backing Layer: 2D Canvas ---
    function initBackingLayer() {
        const canvas = backingCanvas;
        if (!canvas) return null;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        
        // Resize observer
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                const width = entry.contentRect.width;
                const height = entry.contentRect.height;
                canvas.width = Math.max(1, width);
                canvas.height = Math.max(1, height);
            }
        });
        observer.observe(canvas);
        
        return {
            render: (t) => {
                const style = appState.backingStyle;
                const w = canvas.width;
                const h = canvas.height;
                
                if (style === 'none') {
                    ctx.clearRect(0, 0, w, h);
                    return;
                }
                
                if (style === 'mid-gray') {
                    ctx.fillStyle = '#808080';
                    ctx.fillRect(0, 0, w, h);
                } else if (style === 'noise') {
                    // Simple noise
                    const imageData = ctx.createImageData(w, h);
                    const data = imageData.data;
                    for (let i = 0; i < data.length; i += 4) {
                        const val = Math.random() * 100 + 50;
                        data[i] = val;
                        data[i+1] = val;
                        data[i+2] = val;
                        data[i+3] = 255;
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

  //  add WebGPU extensions
        const requiredFeatures = [];
        if (adapter.features.has('float32-filterable')) {
            requiredFeatures.push('float32-filterable');
        } else {
            console.log("Device does not support 'float32-filterable'");
        }
        if (adapter.features.has('float32-blendable')) {
            requiredFeatures.push('float32-blendable');
        } else {
            console.log("Device does not support 'float32-blendable'.");
        }
        if (adapter.features.has('clip-distances')) {
            requiredFeatures.push('clip-distances');
        } else {
            console.log("Device does not support 'clip-distances'.");
        }
        if (adapter.features.has('depth32float-stencil8')) {
            requiredFeatures.push('depth32float-stencil8');
        } else {
            console.log("Device does not support 'depth32float-stencil8'.");
        }
        if (adapter.features.has('dual-source-blending')) {
            requiredFeatures.push('dual-source-blending');
        } else {
            console.log("Device does not support 'dual-source-blending'.");
        }
                if (adapter.features.has('subgroups')) {
            requiredFeatures.push('subgroups');
        } else {
            console.log("Device does not support 'subgroups'.");
        }
        if (adapter.features.has('texture-component-swizzle')) {
            requiredFeatures.push('texture-component-swizzle');
        } else {
            console.log("Device does not support 'texture-component-swizzle'.");
        }
        
        const device = await adapter.requestDevice({
            requiredFeatures,
        });
            const context = canvas.getContext('webgpu');
            const format = navigator.gpu.getPreferredCanvasFormat();
            context.configure({ device, format, alphaMode: 'opaque' });

            const module = device.createShaderModule({
                code: `
                    struct Uniforms {
                        time: f32,
                        res: vec2f,
                    };

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

                    @group(0) @binding(0) var<uniform> u : Uniforms;

                    @fragment
                    fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
                        var aspect = u.res.x / u.res.y;
                        var cUV = uv - 0.5;
                        cUV.x *= aspect;

                        var col = vec3f(0.02, 0.02, 0.03);
                        // Grid
                        let grid = step(vec2f(0.98), fract(uv * 50.0));
                        col += vec3f(0.05) * grid.x;

                        // Circuit lines
                        let lineY = abs(cUV.y);
                        let rail = smoothstep(0.04, 0.035, abs(lineY - 0.05));
                        col += vec3f(0.2) * rail;

                        let rowWidth = 1.8;
                        let blipPos = (fract(u.time * 0.5) - 0.5) * rowWidth;

                        let dx = cUV.x - blipPos;
                        let dist = length(vec2f(dx, cUV.y));

                        let energy = 0.02 / (dist + 0.001);
                        let trail = exp(-10.0 * max(0.0, -dx));

                        let glowColor = vec3f(0.0, 0.8, 1.0);
                        col += glowColor * (pow(energy, 1.5) * 0.5);

                        return vec4f(col, 1.0);
                    }
                `
            });

            // WGSL uniform struct layout:
            // time: f32 -> offset 0, size 4, align 4
            // (padding of 4 bytes)
            // res: vec2f -> offset 8, size 8, align 8
            // Total size: 16 bytes.
            const uBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            const uniformData = new Float32Array(4); // 4 * 4 bytes = 16 bytes

            const pipeline = device.createRenderPipeline({
                layout: 'auto',
                vertex: { module, entryPoint: 'vs_main' },
                fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
                primitive: { topology: 'triangle-list' },
            });

            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uBuf } },
                ],
            });

            if(statusEl) statusEl.innerText = "ACTIVE";

            // Resize Observer
            const observer = new ResizeObserver(entries => {
                for (const entry of entries) {
                    const width = entry.contentRect.width;
                    const height = entry.contentRect.height;
                    canvas.width = Math.max(1, width);
                    canvas.height = Math.max(1, height);
                }
            });
            observer.observe(canvas);

            return {
                render: (t) => {
                    // Update uniform data
                    uniformData[0] = t; // time
                    uniformData[2] = canvas.width; // res.x
                    uniformData[3] = canvas.height; // res.y
                    device.queue.writeBuffer(uBuf, 0, uniformData);
                    
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
        } catch(e) { console.error(e); }
    }

    // --- WebGL2: Multi-Row LED Display ---
    function initWebGL2() {
        const canvas = document.getElementById('webgl-canvas');
        const statusEl = document.getElementById('gl-status');
        const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });

        if (!gl) {
            if(statusEl) statusEl.innerText = "N/A";
            return null;
        }
        if(statusEl) statusEl.innerText = "ACTIVE";

        // Resize
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                const width = entry.contentRect.width;
                const height = entry.contentRect.height;
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

        // Colors for 4 rows
        uniform vec3 u_rowColors[4];

        void main() {
            float aspect = u_res.y / u_res.x;

            // Grid Configuration
            float rows = 4.0;
            float cols = 16.0;

            float instance = float(gl_InstanceID);
            float row = floor(instance / cols);
            float col = mod(instance, cols);

            // Horizontal layout
            float rowWidth = 1.8; // Normalized width coverage
            float xOffset = (col / (cols - 1.0) - 0.5) * rowWidth;

            // Vertical layout
            float ySpacing = 0.3;
            float yOffset = (row - (rows-1.0)*0.5) * ySpacing * aspect; // Adjust for aspect to keep spacing square-ish

            // Animation Logic per row
            float speed = 0.5 + row * 0.2;
            float direction = (mod(row, 2.0) == 0.0) ? 1.0 : -1.0;
            float blipPos = (fract(u_time * speed * direction) - 0.5) * rowWidth;

            float dist = abs(xOffset - blipPos);

            // Activate LED based on proximity to "blip"
            float activeVal = 1.0 - smoothstep(0.0, 0.15, dist);

            // Row Color
            int rIdx = int(row);
            vec3 baseColor = u_rowColors[rIdx];

            vec3 colOff = vec3(0.1, 0.1, 0.1);
            vec3 colOn = baseColor; // Use uniform color
            vec3 finalRgb = mix(colOff, colOn, activeVal);

            float alpha = mix(0.4, 1.0, activeVal);

            v_col = vec4(finalRgb, alpha);
            v_uv = a_pos * 2.0; // Pass Quad UVs (-1 to 1 range approx)

            // Final Position
            vec2 pos = a_pos * u_scale + vec2(xOffset, yOffset);
            pos.x *= aspect; // Correct aspect ratio distortion

            gl_Position = vec4(pos, 0.0, 1.0);
        }`;

        const fs = `#version 300 es
        precision mediump float;
        in vec4 v_col;
        in vec2 v_uv;
        out vec4 outColor;

        void main() {
            float r = length(v_uv);
            float lensR = 0.7;
            float bezelR = 0.9;

            if (r > bezelR) discard;

            vec3 rgb;
            float alpha;

            if (r > lensR) {
                // Bezel
                rgb = vec3(0.3); // Dark metal bezel
                alpha = 1.0;
            } else {
                // Lens/LED
                float lensNormR = r / lensR;
                float z = sqrt(1.0 - lensNormR*lensNormR);
                vec3 normal = vec3(v_uv / lensR, z);

                // Lighting
                vec3 lightDir = normalize(vec3(-0.5, 0.5, 1.0));
                float diffuse = max(0.0, dot(normal, lightDir));
                float specular = pow(max(0.0, dot(reflect(-lightDir, normal), vec3(0,0,1))), 10.0);

                rgb = v_col.rgb * (0.5 + 0.8 * diffuse);
                rgb += vec3(1.0) * specular * 0.5 * v_col.a;

                alpha = v_col.a;
            }

            outColor = vec4(rgb * alpha, alpha);
        }`;

        const p = gl.createProgram();
        const createS = (t, s) => { const x=gl.createShader(t); gl.shaderSource(x,s); gl.compileShader(x); return x; };
        gl.attachShader(p, createS(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, createS(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p);

        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            console.error(gl.getProgramInfoLog(p));
            return null;
        }

        const locs = {
            pos: gl.getAttribLocation(p, 'a_pos'),
            time: gl.getUniformLocation(p, 'u_time'),
            res: gl.getUniformLocation(p, 'u_res'),
            scale: gl.getUniformLocation(p, 'u_scale'),
            rowColors: gl.getUniformLocation(p, 'u_rowColors')
        };

        // Quad Geometry
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        gl.enableVertexAttribArray(locs.pos);
        gl.vertexAttribPointer(locs.pos, 2, gl.FLOAT, false, 0, 0);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); // Standard Pre-multiplied blend

        return {
            render: (t) => {
                gl.clearColor(0,0,0,0);
                gl.clear(gl.COLOR_BUFFER_BIT);

                gl.useProgram(p);
                gl.uniform2f(locs.res, canvas.width, canvas.height);
                gl.uniform1f(locs.time, t);
                gl.uniform1f(locs.scale, 0.08); // Smaller scale for more LEDs

                // Upload array of 4 colors
                const colors = appState.ledColors.flat();
                gl.uniform3fv(locs.rowColors, new Float32Array(colors));

                // Draw 4 rows * 16 cols = 64 instances
                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 64);
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
            if (gl && appState.glVisible) gl.render(t);

            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', main);
    } else {
        main();
    }
