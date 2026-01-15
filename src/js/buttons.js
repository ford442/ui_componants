/**
 * Buttons Page JavaScript
 * Creates and manages LED button components with layered canvas effects
 */

document.addEventListener('DOMContentLoaded', () => {
    const webgl2Canvas = document.getElementById('webgl2-manager-canvas');
    let webgl2Manager;

    checkWebGL2Support().then(supported => {
        if (!supported) {
            document.getElementById('webgl2-warning')?.setAttribute('style', 'display: block;');
            document.body.classList.add('no-webgl2');
        } else {
            if (webgl2Canvas) {
                webgl2Manager = new window.WebGL2Manager(webgl2Canvas);
                webgl2Canvas.width = webgl2Canvas.clientWidth;
                webgl2Canvas.height = webgl2Canvas.clientHeight;
                window.addEventListener('resize', () => {
                    webgl2Canvas.width = webgl2Canvas.clientWidth;
                    webgl2Canvas.height = webgl2Canvas.clientHeight;
                });
            }
        }

        // Initialize all components regardless of support,
        // but pass the manager to those that need it.
        initBasicButtons(); // This one uses WebGL1, will need its own refactor later
        initRGBButtons(webgl2Manager);
        initMomentaryButtons();
        initPulsingButtons(webgl2Manager);
        initButtonMatrix(webgl2Manager);
        initLayeredDemo(webgl2Manager);
        initArcadeButtons();
        initIndustrialButtons();
        initHolographicButtons();
        initOrganicButtons();

        // New Experiments
        initLiquidMetalButtons(webgl2Manager);
        initKineticTypographyButtons();
        initEMPButtons(webgl2Manager);
        initMagneticFieldButtons(webgl2Manager);
    });

    checkWebGPUSupport().then(supported => {
        if (supported) {
            const webgpuCanvas = document.getElementById('webgpu-manager-canvas');
            const webgpuManager = new window.WebGPUManager(webgpuCanvas);
            webgpuManager.init().then(success => {
                if (success) {
                    document.getElementById('enable-webgpu-btn')?.addEventListener('click', () => {
                        initWebGPUExperiments(webgl2Manager, webgpuManager);
                        document.getElementById('webgpu-experiments-container').style.display = 'block';
                        document.getElementById('webgpu-enable-section').style.display = 'none';
                    });
                } else {
                    document.getElementById('webgpu-warning')?.setAttribute('style', 'display: block;');
                    document.body.classList.add('no-webgpu');
                }
            });
        } else {
            document.getElementById('webgpu-warning')?.setAttribute('style', 'display: block;');
            document.body.classList.add('no-webgpu');
        }
    });
});

function initWebGPUExperiments(webgl2Manager, webgpuManager) {
    initParticleSwarmButtons(webgl2Manager, webgpuManager);
    initQuantumFluxButtons(webgpuManager);
    initNeuralNetworkButtons();
    initCompositingShowcase(webgl2Manager, webgpuManager);
}

/**
 * Initialize basic LED buttons
 */
function initBasicButtons() {
    const container = document.getElementById('basic-buttons');
    if (!container) return;

    const colors = [
        { color: [0, 1, 0.5], label: 'Power' },
        { color: [1, 0.3, 0], label: 'Alert' },
        { color: [0, 0.6, 1], label: 'Info' },
        { color: [1, 0, 0.5], label: 'Mute' }
    ];

    colors.forEach(({ color, label }) => {
        new UIComponents.LEDButton(container, {
            width: 80,
            height: 50,
            color: color,
            label: label
        });
    });
}

/**
 * Initialize RGB cycling buttons
 */
function initRGBButtons(manager) {
    const container = document.getElementById('rgb-buttons');
    if (!container || !manager) return;

    const fragmentShader = `#version 300 es
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_on;
        out vec4 fragColor;
        
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            float hue = fract(u_time * 0.2);
            vec3 color = hsv2rgb(vec3(hue, 1.0, 1.0));
            
            float core = 1.0 - smoothstep(0.0, 0.15, dist);
            float glow = 1.0 - smoothstep(0.0, 0.4, dist);
            glow = pow(glow, 2.0);
            
            float intensity = mix(0.1, 1.0, u_on);
            vec3 finalColor = color * (core + glow * 0.5) * intensity;
            float alpha = (core + glow * 0.3) * intensity;
            
            fragColor = vec4(finalColor, alpha);
        }
    `;

    const vertexShader = `#version 300 es
        in vec4 a_position;
        void main() {
            gl_Position = a_position;
        }
    `;

    const program = manager.createProgram(vertexShader, fragmentShader);
    if (!program) return;

    const createRGBButton = (label) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'rgb-button-wrapper';
        wrapper.style.cssText = `
            width: 100px;
            height: 60px;
            position: relative;
        `;

        const button = document.createElement('button');
        button.textContent = label;
        button.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
            background: linear-gradient(145deg, #2a2a3a, #1a1a2a);
            border-radius: 10px;
            cursor: pointer;
            position: relative;
            z-index: 1;
            color: #888;
            font-size: 0.8rem;
            font-weight: bold;
            transition: all 0.1s ease;
            box-shadow: 
                inset 0 2px 4px rgba(255, 255, 255, 0.1),
                0 4px 8px rgba(0, 0, 0, 0.5);
        `;

        wrapper.appendChild(button);
        container.appendChild(wrapper);

        let isOn = false;
        button.addEventListener('click', () => {
            isOn = !isOn;
        });

        const uniforms = {
            time: manager.gl.getUniformLocation(program, 'u_time'),
            resolution: manager.gl.getUniformLocation(program, 'u_resolution'),
            on: manager.gl.getUniformLocation(program, 'u_on')
        };
        
        manager.addRenderable({
            element: wrapper,
            program: program,
            uniformsCallback: (gl, time) => {
                const rect = wrapper.getBoundingClientRect();
                gl.uniform1f(uniforms.time, time);
                gl.uniform2f(uniforms.resolution, rect.width, rect.height);
                gl.uniform1f(uniforms.on, isOn ? 1.0 : 0.0);
            }
        });
    };

    createRGBButton('RGB 1');
    createRGBButton('RGB 2');
    createRGBButton('RGB 3');
}

/**
 * Initialize momentary buttons
 */
function initMomentaryButtons() {
    const container = document.getElementById('momentary-buttons');
    if (!container) return;

    const labels = ['A', 'B', 'C', 'D'];
    const colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0.5, 1],
        [1, 1, 0]
    ];

    labels.forEach((label, i) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'momentary-btn-wrapper';
        wrapper.style.cssText = `
            width: 60px;
            height: 60px;
            position: relative;
        `;

        // SVG filter layer
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.cssText = 'position: absolute; top: 0; left: 0; pointer-events: none; z-index: 2;';

        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', `glow-${i}`);
        filter.innerHTML = `
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        `;
        defs.appendChild(filter);
        svg.appendChild(defs);

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', '30');
        circle.setAttribute('cy', '30');
        circle.setAttribute('r', '25');
        circle.setAttribute('fill', 'transparent');
        circle.setAttribute('stroke', `rgb(${colors[i][0] * 255}, ${colors[i][1] * 255}, ${colors[i][2] * 255})`);
        circle.setAttribute('stroke-width', '2');
        circle.setAttribute('stroke-opacity', '0');
        circle.setAttribute('filter', `url(#glow-${i})`);
        svg.appendChild(circle);

        // Button element
        const button = document.createElement('button');
        button.textContent = label;
        button.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
            background: radial-gradient(circle at 40% 40%, #3a3a4a, #1a1a2a);
            border-radius: 50%;
            cursor: pointer;
            position: relative;
            z-index: 1;
            color: #666;
            font-size: 1rem;
            font-weight: bold;
            transition: all 0.05s ease;
            box-shadow: 
                inset 0 2px 4px rgba(255, 255, 255, 0.1),
                0 4px 8px rgba(0, 0, 0, 0.5);
        `;

        wrapper.appendChild(svg);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        // Add momentary behavior
        let isPressed = false;

        button.addEventListener('mousedown', () => {
            isPressed = true;
            button.style.transform = 'scale(0.95)';
            button.style.boxShadow = `
                inset 0 2px 4px rgba(255, 255, 255, 0.05),
                0 2px 4px rgba(0, 0, 0, 0.5)
            `;
            circle.setAttribute('stroke-opacity', '1');
        });

        const release = () => {
            if (isPressed) {
                isPressed = false;
                button.style.transform = 'scale(1)';
                button.style.boxShadow = `
                    inset 0 2px 4px rgba(255, 255, 255, 0.1),
                    0 4px 8px rgba(0, 0, 0, 0.5)
                `;
                circle.setAttribute('stroke-opacity', '0');
            }
        };

        button.addEventListener('mouseup', release);
        button.addEventListener('mouseleave', release);
    });
}

/**
 * Initialize pulsing buttons
 */
function initPulsingButtons(manager) {
    const container = document.getElementById('pulsing-buttons');
    if (!container || !manager) return;

    const configs = [
        { label: 'Slow', speed: 1, color: [0, 1, 0.5] },
        { label: 'Medium', speed: 2, color: [1, 0.5, 0] },
        { label: 'Fast', speed: 4, color: [1, 0, 0.3] },
        { label: 'Strobe', speed: 10, color: [0, 0.7, 1] }
    ];

    const vertexShader = `#version 300 es
        in vec4 a_position;
        void main() {
            gl_Position = a_position;
        }
    `;

    const fragmentShader = `#version 300 es
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec3 u_color;
        uniform float u_speed;
        out vec4 fragColor;
        
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            float pulse = 0.5 + 0.5 * sin(u_time * u_speed);
            
            float core = 1.0 - smoothstep(0.0, 0.15, dist);
            float glow = 1.0 - smoothstep(0.0, 0.4, dist);
            glow = pow(glow, 2.0);
            
            vec3 finalColor = u_color * (core + glow * 0.5) * pulse;
            float alpha = (core + glow * 0.3) * pulse;
            
            fragColor = vec4(finalColor, alpha);
        }
    `;
    
    const program = manager.createProgram(vertexShader, fragmentShader);
    if (!program) return;

    configs.forEach(({ label, speed, color }) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            width: 80px;
            height: 50px;
            position: relative;
        `;

        const button = document.createElement('div');
        button.textContent = label;
        button.style.cssText = `
            width: 100%;
            height: 100%;
            background: linear-gradient(145deg, #2a2a3a, #1a1a2a);
            border-radius: 10px;
            position: relative;
            z-index: 1;
            color: #888;
            font-size: 0.8rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 
                inset 0 2px 4px rgba(255, 255, 255, 0.1),
                0 4px 8px rgba(0, 0, 0, 0.5);
        `;
        
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        const uniforms = {
            time: manager.gl.getUniformLocation(program, 'u_time'),
            resolution: manager.gl.getUniformLocation(program, 'u_resolution'),
            color: manager.gl.getUniformLocation(program, 'u_color'),
            speed: manager.gl.getUniformLocation(program, 'u_speed')
        };

        manager.addRenderable({
            element: wrapper,
            program: program,
            uniformsCallback: (gl, time) => {
                const rect = wrapper.getBoundingClientRect();
                gl.uniform1f(uniforms.time, time);
                gl.uniform2f(uniforms.resolution, rect.width, rect.height);
                gl.uniform3fv(uniforms.color, color);
                gl.uniform1f(uniforms.speed, speed);
            }
        });
    });
}

/**
 * Initialize button matrix
 */
function initButtonMatrix(manager) {
    const container = document.getElementById('button-matrix-container');
    if (!container || !manager) return;

    const buttons = [];
    const colors = [
        [1, 0, 0.3], [0, 1, 0.5], [0, 0.6, 1], [1, 0.5, 0],
        [1, 0, 0.8], [0.5, 1, 0], [0, 1, 1], [1, 0.3, 0],
        [0.7, 0, 1], [0, 1, 0.3], [0.3, 0.7, 1], [1, 1, 0],
        [1, 0, 0.5], [0, 0.8, 0.5], [0, 0.4, 1], [1, 0.6, 0]
    ];
    
    const matrixSize = 4;
    const buttonSize = 60;
    const gap = 10;
    const canvasSize = matrixSize * buttonSize + (matrixSize - 1) * gap;

    // Create a container for the buttons that can receive clicks
    const clickContainer = document.createElement('div');
    clickContainer.style.width = `${canvasSize}px`;
    clickContainer.style.height = `${canvasSize}px`;
    clickContainer.style.position = 'relative';
    container.appendChild(clickContainer);

    for (let i = 0; i < 16; i++) {
        const row = Math.floor(i / matrixSize);
        const col = i % matrixSize;
        const btn = {
            x: col * (buttonSize + gap),
            y: row * (buttonSize + gap),
            width: buttonSize,
            height: buttonSize,
            color: colors[i],
            isOn: false
        };
        buttons.push(btn);

        // Create divs for clicking
        const btnDiv = document.createElement('div');
        btnDiv.style.position = 'absolute';
        btnDiv.style.left = `${btn.x}px`;
        btnDiv.style.top = `${btn.y}px`;
        btnDiv.style.width = `${btn.width}px`;
        btnDiv.style.height = `${btn.height}px`;
        btnDiv.style.cursor = 'pointer';
        btnDiv.addEventListener('click', () => {
            btn.isOn = !btn.isOn;
        });
        clickContainer.appendChild(btnDiv);
    }

    const fragmentShader = `#version 300 es
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec3 u_color;
        uniform float u_on;
        out vec4 fragColor;
        
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            float core = 1.0 - smoothstep(0.0, 0.15, dist);
            float glow = 1.0 - smoothstep(0.0, 0.4, dist);
            glow = pow(glow, 2.0);
            
            float pulse = 0.9 + 0.1 * sin(u_time * 3.0);
            
            float intensity = mix(0.1, 1.0, u_on);
            vec3 color = u_color * (core + glow * 0.5) * intensity * pulse;
            float alpha = (core + glow * 0.3) * intensity;
            
            fragColor = vec4(color, alpha);
        }
    `;

    const vertexShader = `#version 300 es
        in vec4 a_position;
        void main() {
            gl_Position = a_position;
        }
    `;

    const program = manager.createProgram(vertexShader, fragmentShader);
    if (!program) return;

    const uniforms = {
        time: manager.gl.getUniformLocation(program, 'u_time'),
        resolution: manager.gl.getUniformLocation(program, 'u_resolution'),
        color: manager.gl.getUniformLocation(program, 'u_color'),
        on: manager.gl.getUniformLocation(program, 'u_on')
    };

    manager.addRenderable({
        element: clickContainer,
        program: program,
        customDraw: (gl, time) => {
            gl.uniform1f(uniforms.time, time);
            
            const matrixRect = clickContainer.getBoundingClientRect();
            const canvasRect = manager.canvas.getBoundingClientRect();

            buttons.forEach(btn => {
                const btnRect = {
                    x: matrixRect.left - canvasRect.left + btn.x,
                    y: canvasRect.height - (matrixRect.top - canvasRect.top + btn.y + btn.height),
                    width: btn.width,
                    height: btn.height
                };

                gl.viewport(btnRect.x, btnRect.y, btnRect.width, btnRect.height);
                
                gl.uniform2f(uniforms.resolution, btn.width, btn.height);
                gl.uniform3fv(uniforms.color, btn.color);
                gl.uniform1f(uniforms.on, btn.isOn ? 1.0 : 0.0);

                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            });
        }
    });

    // Matrix control buttons
    document.getElementById('matrix-random')?.addEventListener('click', () => {
        buttons.forEach(btn => {
            btn.isOn = Math.random() > 0.5;
        });
    });

    document.getElementById('matrix-wave')?.addEventListener('click', () => {
        buttons.forEach(btn => btn.isOn = false);

        let index = 0;
        const interval = setInterval(() => {
            if (index >= 16) {
                clearInterval(interval);
                return;
            }
            buttons[index].isOn = true;
            if (index > 0) buttons[index - 1].isOn = false;
            index++;
        }, 150);

        setTimeout(() => {
            if (buttons[15]) buttons[15].isOn = false;
        }, 150 * 17);
    });

    document.getElementById('matrix-chase')?.addEventListener('click', () => {
        const sequence = [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 5, 6, 10, 9];
        buttons.forEach(btn => btn.isOn = false);

        let index = 0;
        const interval = setInterval(() => {
            if (index >= sequence.length) {
                clearInterval(interval);
                return;
            }
            buttons[sequence[index]].isOn = true;
            if (index > 0) buttons[sequence[index - 1]].isOn = false;
            index++;
        }, 100);

        setTimeout(() => {
            buttons[sequence[sequence.length - 1]].isOn = false;
        }, 100 * (sequence.length + 1));
    });

    document.getElementById('matrix-clear')?.addEventListener('click', () => {
        buttons.forEach(btn => btn.isOn = false);
    });
}

/**
 * Initialize layered demo
 */
function initLayeredDemo(manager) {
    const container = document.getElementById('layered-demo');
    if (!container || !manager) return;

    // The manager's canvas will be used for WebGL content.
    // We just need to add the SVG layer to the container.
    container.style.position = 'relative';
    container.style.width = '800px';
    container.style.height = '400px';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 800 400');
    svg.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 2;
        pointer-events: none;
    `;
    container.appendChild(svg);
    
    // --- Renderable 1: Base Grid Layer ---
    const baseVs = `#version 300 es
        in vec4 a_position; void main() { gl_Position = a_position; }
    `;
    const baseFs = `#version 300 es
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        out vec4 fragColor;
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 grid = fract(uv * 20.0);
            float gridLine = step(0.95, grid.x) + step(0.95, grid.y);
            vec3 bg = mix(vec3(0.02, 0.02, 0.05), vec3(0.05, 0.08, 0.1), uv.y);
            vec3 color = bg + vec3(0.0, 0.1, 0.15) * gridLine * 0.3;
            fragColor = vec4(color, 1.0);
        }
    `;
    const baseProgram = manager.createProgram(baseVs, baseFs);
    const baseUniforms = {
        time: manager.gl.getUniformLocation(baseProgram, 'u_time'),
        resolution: manager.gl.getUniformLocation(baseProgram, 'u_resolution'),
    };
    const baseRenderable = {
        id: 'layered-demo-base',
        element: container,
        program: baseProgram,
        uniformsCallback: (gl, time) => {
            const rect = container.getBoundingClientRect();
            gl.uniform1f(baseUniforms.time, time);
            gl.uniform2f(baseUniforms.resolution, rect.width, rect.height);
        }
    };

    // --- Renderable 2: Effects Layer ---
    const effectsVs = `#version 300 es
        in vec4 a_position; void main() { gl_Position = a_position; }
    `;
    const effectsFs = `#version 300 es
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        out vec4 fragColor;
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            float glow = 0.0;
            for (int i = 0; i < 5; i++) {
                float fi = float(i);
                vec2 center = vec2(
                    0.2 + 0.15 * fi + 0.1 * sin(u_time + fi),
                    0.5 + 0.2 * sin(u_time * 0.7 + fi * 1.5)
                );
                float dist = distance(uv, center);
                glow += 0.03 / (dist * dist + 0.01);
            }
            vec3 color = vec3(0.0, glow * 0.5, glow);
            float alpha = min(glow * 0.5, 1.0);
            fragColor = vec4(color, alpha);
        }
    `;
    const effectsProgram = manager.createProgram(effectsVs, effectsFs);
    const effectsUniforms = {
        time: manager.gl.getUniformLocation(effectsProgram, 'u_time'),
        resolution: manager.gl.getUniformLocation(effectsProgram, 'u_resolution'),
    };
    const effectsRenderable = {
        id: 'layered-demo-effects',
        element: container,
        program: effectsProgram,
        uniformsCallback: (gl, time) => {
            const rect = container.getBoundingClientRect();
            gl.uniform1f(effectsUniforms.time, time);
            gl.uniform2f(effectsUniforms.resolution, rect.width, rect.height);
        }
    };
    
    // --- SVG Setup ---
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `<filter id="svg-glow"><feGaussianBlur stdDeviation="3" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
    svg.appendChild(defs);
    for (let i = 0; i < 3; i++) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', 200 + i * 200);
        circle.setAttribute('cy', 200);
        circle.setAttribute('r', 40 + i * 10);
        circle.setAttribute('fill', 'none');
        circle.setAttribute('stroke', `rgba(0, 255, 200, ${0.3 - i * 0.08})`);
        circle.setAttribute('stroke-width', '2');
        circle.setAttribute('filter', 'url(#svg-glow)');
        const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
        animate.setAttribute('attributeName', 'r');
        animate.setAttribute('values', `${40 + i * 10};${50 + i * 10};${40 + i * 10}`);
        animate.setAttribute('dur', `${2 + i * 0.5}s`);
        animate.setAttribute('repeatCount', 'indefinite');
        circle.appendChild(animate);
        svg.appendChild(circle);
    }

    // --- Controls ---
    const layers = {
        'layer-webgl': baseRenderable,
        'layer-webgl2': effectsRenderable,
    };

    // Initial state
    manager.addRenderable(baseRenderable);
    manager.addRenderable(effectsRenderable);

    Object.keys(layers).forEach(id => {
        document.getElementById(id)?.addEventListener('change', (e) => {
            const renderable = layers[id];
            if (e.target.checked) {
                manager.addRenderable(renderable);
            } else {
                manager.renderables = manager.renderables.filter(r => r.id !== renderable.id);
            }
        });
    });

    document.getElementById('layer-svg')?.addEventListener('change', (e) => {
        svg.style.display = e.target.checked ? 'block' : 'none';
    });

    document.getElementById('layer-css')?.addEventListener('change', (e) => {
        container.style.filter = e.target.checked ? 'contrast(1.1) saturate(1.2)' : 'none';
    });
}

/**
 * Initialize arcade buttons
 */
function initArcadeButtons() {
    const container = document.getElementById('arcade-buttons');
    if (!container) return;

    const colors = ['', 'green', 'blue', 'yellow'];

    colors.forEach(color => {
        const button = document.createElement('button');
        button.className = `arcade-btn ${color}`;

        button.addEventListener('click', () => {
            button.style.animation = 'none';
            button.offsetHeight; // Trigger reflow
            button.style.animation = 'arcadePress 0.2s ease';
        });

        container.appendChild(button);
    });

    // Add animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes arcadePress {
            0% { filter: brightness(1); }
            50% { filter: brightness(1.5); }
            100% { filter: brightness(1); }
        }
    `;
    document.head.appendChild(style);
}

/**
 * Initialize industrial buttons
 */
function initIndustrialButtons() {
    const container = document.getElementById('industrial-buttons');
    if (!container) return;

    const states = ['active', 'warning', 'success', ''];

    states.forEach(state => {
        const button = document.createElement('button');
        button.className = `industrial-btn ${state}`;

        button.addEventListener('click', () => {
            // Toggle active state
            if (button.classList.contains('active')) {
                button.classList.remove('active');
            } else {
                button.classList.add('active');
            }
        });

        container.appendChild(button);
    });
}

/**
 * Initialize holographic buttons
 */
function initHolographicButtons() {
    const container = document.getElementById('holographic-buttons');
    if (!container) return;

    const labels = ['ENGAGE', 'SYSTEM', 'PORTAL'];

    labels.forEach(label => {
        const button = document.createElement('button');
        button.className = 'holo-btn';
        button.textContent = label;
        container.appendChild(button);

        button.addEventListener('click', () => {
            button.classList.add('holo-active');
            setTimeout(() => button.classList.remove('holo-active'), 300);
        });
    });
}

/**
 * Initialize organic buttons
 */
function initOrganicButtons() {
    const container = document.getElementById('organic-buttons');
    if (!container) return;

    const colors = ['', 'purple', 'orange'];

    colors.forEach((color, i) => {
        const button = document.createElement('button');
        button.className = `organic-btn ${color}`;
        button.style.animationDelay = `${i * 0.5}s`;
        container.appendChild(button);

        button.addEventListener('click', () => {
            button.classList.add('organic-active');
            setTimeout(() => button.classList.remove('organic-active'), 400);
        });
    });
}

// Helper function for WebGL2
function createProgram(gl, vertexSource, fragmentSource) {
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);

    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
        return null;
    }

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);

    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
        return null;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        return null;
    }

    return program;
}

function setupQuad(gl, program) {
    const positions = new Float32Array([
        -1, -1,
        1, -1,
        -1, 1,
        1, 1,
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
}

function setupQuadWebGL1(gl, program) {
    const positions = new Float32Array([
        -1, -1, 0, 0,
        1, -1, 1, 0,
        -1, 1, 0, 1,
        1, 1, 1, 1,
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const texCoordLoc = gl.getAttribLocation(program, 'a_texCoord');

    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 16, 0);

    if (texCoordLoc !== -1) {
        gl.enableVertexAttribArray(texCoordLoc);
        gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 16, 8);
    }
}

/**
 * Check WebGPU Support
 */
async function checkWebGPUSupport() {
    if (!navigator.gpu) {
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch (e) {
        return false;
    }
}

/**
 * Check WebGL2 Support
 */
async function checkWebGL2Support() {
    try {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('webgl2');
        return !!context;
    } catch (e) {
        return false;
    }
}

/**
 * Initialize Particle Swarm Buttons
 * WebGPU compute shader with 10K particles that react to mouse interaction
 */
function initParticleSwarmButtons(webgl2Manager, webgpuManager) {
    const container = document.getElementById('particle-swarm-buttons');
    if (!container || !webgl2Manager || !webgpuManager) return;

    const config = { label: 'ATTRACT', color: [0, 1, 0.5, 0.8], physics: 'attract' };

    const wrapper = document.createElement('div');
    wrapper.className = 'particle-button-wrapper';
    wrapper.style.cssText = `
        width: 140px;
        height: 80px;
        position: relative;
        margin: 0.5rem;
    `;

    const button = document.createElement('button');
    button.textContent = config.label;
    button.style.cssText = `
        width: 100%;
        height: 100%;
        background: rgba(20, 20, 30, 0.7);
        border: 2px solid rgba(${config.color[0] * 255}, ${config.color[1] * 255}, ${config.color[2] * 255}, 0.5);
        border-radius: 8px;
        color: rgb(${config.color[0] * 255}, ${config.color[1] * 255}, ${config.color[2] * 255});
        font-size: 0.9rem;
        font-weight: bold;
        cursor: pointer;
        position: relative;
        z-index: 2;
        backdrop-filter: blur(2px);
        transition: all 0.2s;
    `;

    wrapper.appendChild(button);
    container.appendChild(wrapper);

    const particleSystem = new window.WebGPUParticleSystem(webgpuManager, {
        element: wrapper,
        particleCount: 10000,
        particleSize: 2,
        color: config.color,
        attractorStrength: 0.3,
        damping: 0.98
    });

    (async () => {
        const initialized = await particleSystem.init();
        if (!initialized) return;

        let frame = 0;
        let mouseX = 0;
        let mouseY = 0;

        webgpuManager.addRenderable({
            element: wrapper,
            render: (passEncoder, time, deltaTime) => {
                particleSystem.updateUniforms(deltaTime, mouseX, mouseY);
                particleSystem.compute(deltaTime);
                particleSystem.render(passEncoder);
                frame++;
            }
        });
        
        const glowShader = `#version 300 es
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec3 u_color;
            uniform float u_active;
            out vec4 fragColor;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(uv, center);
                
                float glow = 1.0 - smoothstep(0.0, 0.6, dist);
                glow = pow(glow, 3.0);
                glow *= u_active * 0.5;
                glow *= 0.9 + 0.1 * sin(u_time * 3.0);
                
                fragColor = vec4(u_color * glow, glow * 0.3);
            }
        `;

        const vertexShader = `#version 300 es
            in vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `;

        const program = webgl2Manager.createProgram(vertexShader, glowShader);
        if (!program) return;

        const uniforms = {
            time: webgl2Manager.gl.getUniformLocation(program, 'u_time'),
            resolution: webgl2Manager.gl.getUniformLocation(program, 'u_resolution'),
            color: webgl2Manager.gl.getUniformLocation(program, 'u_color'),
            active: webgl2Manager.gl.getUniformLocation(program, 'u_active')
        };

        let isActive = false;
        
        button.addEventListener('mouseenter', () => { isActive = true; });
        button.addEventListener('mouseleave', () => { isActive = false; });
        button.addEventListener('click', () => {
            button.style.transform = 'scale(0.95)';
            setTimeout(() => { button.style.transform = 'scale(1)'; }, 100);
        });

        wrapper.addEventListener('mousemove', (e) => {
            const rect = wrapper.getBoundingClientRect();
            mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        });

        webgl2Manager.addRenderable({
            element: wrapper,
            program: program,
            uniformsCallback: (gl, time) => {
                const rect = wrapper.getBoundingClientRect();
                gl.uniform1f(uniforms.time, time);
                gl.uniform2f(uniforms.resolution, rect.width, rect.height);
                gl.uniform3f(uniforms.color, config.color[0], config.color[1], config.color[2]);
                gl.uniform1f(uniforms.active, isActive ? 1.0 : 0.0);
            }
        });
    })();
}

/**
 * Initialize Quantum Flux Buttons
 * WebGPU rendering with probabilistic shader effects
 */
function initQuantumFluxButtons(webgpuManager) {
    const container = document.getElementById('quantum-flux-buttons');
    if (!container || !webgpuManager) return;

    const label = 'QUANTUM';
    const index = 0; // Only one button

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 140px;
        height: 80px;
        position: relative;
        margin: 0.5rem;
    `;

    const button = document.createElement('button');
    button.textContent = label;
    button.style.cssText = `
        width: 100%;
        height: 100%;
        background: rgba(10, 10, 20, 0.5);
        border: 2px solid rgba(100, 150, 255, 0.6);
        border-radius: 8px;
        color: #6af;
        font-size: 0.8rem;
        font-weight: bold;
        cursor: pointer;
        position: relative;
        z-index: 2;
        backdrop-filter: blur(3px);
    `;

    wrapper.appendChild(button);
    container.appendChild(wrapper);

    // Initialize WebGPU volumetric renderer for quantum effects
    const volumetric = new UIComponents.WebGPUVolumetricRenderer(webgpuManager, {
        element: wrapper,
        raySteps: 32,
        density: 0.3 + index * 0.1
    });

    (async () => {
        const initialized = await volumetric.init();
        if (!initialized) return;

        let isActive = false;
        button.addEventListener('mouseenter', () => { isActive = true; });
        button.addEventListener('mouseleave', () => { isActive = false; });

        webgpuManager.addRenderable({
            element: wrapper,
            render: (passEncoder, time, deltaTime) => {
                volumetric.render(passEncoder, time, deltaTime);

                if (isActive) {
                    button.style.borderColor = `rgba(${100 + Math.sin(time * 5) * 50}, 150, 255, 0.9)`;
                }
            }
        });
    })();
}

/**
 * Initialize Magnetic Field Buttons
 * Compute shader field line simulation
 */
function initMagneticFieldButtons(manager) {
    const container = document.getElementById('magnetic-field-buttons');
    if (!container || !manager) return;

    const configs = [
        { label: 'N', polarity: 1, color: '#f44' },
        { label: 'S', polarity: -1, color: '#44f' }
    ];
    
    const vertexShader = `#version 300 es
        in vec4 a_position;
        void main() {
            gl_Position = a_position;
        }
    `;

    const fragmentShader = `#version 300 es
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_polarity;
        out vec4 fragColor;
        
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 center = vec2(0.5, 0.5);
            vec2 toCenter = uv - center;
            float dist = length(toCenter);
            float angle = atan(toCenter.y, toCenter.x);
            
            // Field lines
            float fieldLine = sin(angle * 6.0 + u_time) * 0.5 + 0.5;
            fieldLine *= smoothstep(0.5, 0.2, dist) * smoothstep(0.05, 0.2, dist);
            
            vec3 color = u_polarity > 0.0 ? vec3(1.0, 0.3, 0.3) : vec3(0.3, 0.3, 1.0);
            fragColor = vec4(color * fieldLine, fieldLine * 0.5);
        }
    `;

    const program = manager.createProgram(vertexShader, fragmentShader);
    if (!program) return;

    configs.forEach((config) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            width: 100px;
            height: 100px;
            position: relative;
            margin: 0.5rem;
        `;

        const button = document.createElement('button');
        button.textContent = config.label;
        button.style.cssText = `
            width: 60%;
            height: 60%;
            position: absolute;
            top: 20%;
            left: 20%;
            background: radial-gradient(circle, ${config.color}40, ${config.color}20);
            border: 3px solid ${config.color};
            border-radius: 50%;
            color: ${config.color};
            font-size: 1.5rem;
            font-weight: bold;
            cursor: pointer;
            z-index: 2;
            box-shadow: 0 0 20px ${config.color}80;
        `;

        wrapper.appendChild(button);
        container.appendChild(wrapper);

        const uniforms = {
            time: manager.gl.getUniformLocation(program, 'u_time'),
            resolution: manager.gl.getUniformLocation(program, 'u_resolution'),
            polarity: manager.gl.getUniformLocation(program, 'u_polarity')
        };

        manager.addRenderable({
            element: wrapper,
            program: program,
            uniformsCallback: (gl, time) => {
                const rect = wrapper.getBoundingClientRect();
                gl.uniform1f(uniforms.time, time);
                gl.uniform2f(uniforms.resolution, rect.width, rect.height);
                gl.uniform1f(uniforms.polarity, config.polarity);
            }
        });
    });
}

/**
 * Initialize Neural Network Buttons
 * Animated node connections with impulse propagation
 */
function initNeuralNetworkButtons() {
    const container = document.getElementById('neural-network-buttons');
    if (!container) return;

    const labels = ['INPUT', 'PROCESS', 'OUTPUT'];

    labels.forEach((label) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            width: 140px;
            height: 100px;
            position: relative;
            margin: 0.5rem;
        `;

        // SVG for neural connections
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            z-index: 0;
        `;

        // Create neural network structure
        const nodes = 8;
        for (let i = 0; i < nodes; i++) {
            const x = 20 + (i % 4) * 30;
            const y = 20 + Math.floor(i / 4) * 60;

            // Node circle
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', '5');
            circle.setAttribute('fill', '#0af');
            circle.setAttribute('opacity', '0.7');
            svg.appendChild(circle);

            // Connections
            if (i < 4) {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', x);
                line.setAttribute('y1', y);
                line.setAttribute('x2', 20 + (i % 4) * 30);
                line.setAttribute('y2', 80);
                line.setAttribute('stroke', '#0af');
                line.setAttribute('stroke-width', '1');
                line.setAttribute('opacity', '0.3');

                const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
                animate.setAttribute('attributeName', 'opacity');
                animate.setAttribute('values', '0.1;0.6;0.1');
                animate.setAttribute('dur', '2s');
                animate.setAttribute('begin', `${i * 0.5}s`);
                animate.setAttribute('repeatCount', 'indefinite');
                line.appendChild(animate);

                svg.appendChild(line);
            }
        }

        const button = document.createElement('button');
        button.textContent = label;
        button.style.cssText = `
            width: 100%;
            height: 100%;
            background: rgba(10, 10, 30, 0.8);
            border: 2px solid #0af;
            border-radius: 10px;
            color: #0af;
            font-size: 0.9rem;
            font-weight: bold;
            cursor: pointer;
            position: relative;
            z-index: 2;
            backdrop-filter: blur(3px);
        `;

        wrapper.appendChild(svg);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        button.addEventListener('click', () => {
            button.style.animation = 'none';
            button.offsetHeight;
            button.style.animation = 'neuralPulse 0.5s ease';
        });
    });

    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes neuralPulse {
            0%, 100% { box-shadow: 0 0 10px #0af; }
            50% { box-shadow: 0 0 30px #0af, 0 0 50px #0af; }
        }
    `;
    document.head.appendChild(style);
}

/**
 * Initialize Multi-Layer Compositing Showcase
 * Real-time layer management with blend modes
 */
function initCompositingShowcase(webgl2Manager, webgpuManager) {
    const container = document.getElementById('compositing-display');
    if (!container || !webgl2Manager || !webgpuManager) return;

    container.style.cssText = `
        width: 100%;
        height: 400px;
        position: relative;
        background: #000;
        border-radius: 12px;
        overflow: hidden;
    `;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        z-index: 3;
        pointer-events: none;
    `;
    svg.id = 'comp-svg-layer';

    for (let i = 0; i < 3; i++) {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', 50 + i * 250);
        rect.setAttribute('y', 150);
        rect.setAttribute('width', 100);
        rect.setAttribute('height', 100);
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke', '#0f8');
        rect.setAttribute('stroke-width', '2');
        rect.setAttribute('rx', '10');
        svg.appendChild(rect);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', 100 + i * 250);
        text.setAttribute('y', 290);
        text.setAttribute('fill', '#0f8');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '14');
        text.textContent = `ZONE ${i + 1}`;
        svg.appendChild(text);
    }

    container.appendChild(svg);

    (async () => {
        const particleSystem = new window.WebGPUParticleSystem(webgpuManager, {
            element: container,
            particleCount: 5000,
            color: [0, 1, 0.5, 0.6],
            attractorStrength: 0.2,
            damping: 0.95
        });
        const initialized = await particleSystem.init();
        if (!initialized) return;

        let frame = 0;
        webgpuManager.addRenderable({
            id: 'compositing-particles',
            element: container,
            render: (passEncoder, time, deltaTime) => {
                particleSystem.updateUniforms(deltaTime, 0, 0);
                particleSystem.compute(deltaTime);
                particleSystem.render(passEncoder);
                frame++;
            }
        });

        const shaderCode = `#version 300 es
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            out vec4 fragColor;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                
                vec3 color = vec3(0.0);
                for (int i = 0; i < 3; i++) {
                    float fi = float(i);
                    vec2 center = vec2(0.2 + fi * 0.3, 0.5);
                    center.x += 0.05 * sin(u_time + fi);
                    center.y += 0.05 * cos(u_time * 0.7 + fi);
                    
                    float dist = distance(uv, center);
                    float glow = 0.05 / (dist * dist + 0.01);
                    color += vec3(0.0, glow * 0.5, glow) * 0.3;
                }
                
                fragColor = vec4(color, min(length(color), 1.0));
            }
        `;
        const vertexShader = `#version 300 es
            in vec4 a_position; void main() { gl_Position = a_position; }`;
        const program = webgl2Manager.createProgram(vertexShader, shaderCode);
        if (!program) return;

        const uniforms = {
            time: webgl2Manager.gl.getUniformLocation(program, 'u_time'),
            resolution: webgl2Manager.gl.getUniformLocation(program, 'u_resolution')
        };
        
        const glowRenderable = {
            id: 'compositing-glow',
            element: container,
            program: program,
            uniformsCallback: (gl, time) => {
                const rect = container.getBoundingClientRect();
                gl.uniform1f(uniforms.time, time);
                gl.uniform2f(uniforms.resolution, rect.width, rect.height);
            }
        };
        webgl2Manager.addRenderable(glowRenderable);

        // Setup layer controls
        document.getElementById('comp-webgpu')?.addEventListener('change', (e) => {
            const r = webgpuManager.renderables.find(r => r.id === 'compositing-particles');
            if (r) r.element.style.display = e.target.checked ? 'block' : 'none';
        });

        document.getElementById('comp-webgl2')?.addEventListener('change', (e) => {
            const r = webgl2Manager.renderables.find(r => r.id === 'compositing-glow');
            if (r) r.element.style.display = e.target.checked ? 'block' : 'none';
        });

        document.getElementById('comp-svg')?.addEventListener('change', (e) => {
            svg.style.display = e.target.checked ? 'block' : 'none';
        });
        
        // Opacity and blend mode are handled by CSS, no change needed there.
    })();
}

// ============================================================================
// NEW EXPERIMENTS
// ============================================================================

/**
 * Liquid Metal / Mercury Buttons
 * Morphing metallic surfaces with reflective ripples
 */
function initLiquidMetalButtons(manager) {
    const container = document.getElementById('liquid-metal-buttons');
    if (!container || !manager) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
        border-radius: 12px;
        cursor: pointer;
    `;

    container.appendChild(wrapper);

    const vs = `#version 300 es
        in vec4 a_position;
        out vec2 v_uv;
        void main() {
            v_uv = a_position.xy * 0.5 + 0.5;
            gl_Position = a_position;
        }
    `;

    const fs = `#version 300 es
        precision highp float;
        in vec2 v_uv;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec2 u_mouse;
        uniform float u_click;
        out vec4 fragColor;

        vec3 getEnv(vec3 dir) {
            float y = dir.y * 0.5 + 0.5;
            vec3 sky = mix(vec3(0.15, 0.2, 0.3), vec3(0.8, 0.9, 1.0), pow(y, 0.4));
            vec3 ground = mix(vec3(0.1, 0.08, 0.05), vec3(0.3, 0.25, 0.2), 1.0 - y);
            return mix(ground, sky, smoothstep(-0.1, 0.1, dir.y));
        }

        float metaball(vec2 p, vec2 center, float r) {
            float d = length(p - center);
            return r * r / (d * d + 0.001);
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
            vec2 mouse = (u_mouse - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
            
            float blob = 0.0;
            float t = u_time;
            
            blob += metaball(uv, vec2(0.0, 0.0), 0.15);
            
            for (float i = 0.0; i < 5.0; i++) {
                float angle = t * 0.5 + i * 1.256;
                float r = 0.2 + 0.1 * sin(t + i);
                vec2 pos = vec2(cos(angle), sin(angle)) * r;
                blob += metaball(uv, pos, 0.08 + 0.02 * sin(t * 2.0 + i));
            }
            
            blob += metaball(uv, mouse, 0.12) * (1.0 + u_click * 0.5);
            
            if (u_click > 0.0) {
                float ripple = sin(length(uv - mouse) * 30.0 - u_time * 10.0);
                blob += ripple * 0.1 * u_click;
            }
            
            float surface = smoothstep(0.8, 0.9, blob);
            
            if (surface < 0.1) {
                fragColor = vec4(0.05, 0.05, 0.08, 1.0);
                return;
            }
            
            vec2 e = vec2(0.01, 0.0);
            float bx = blob;
            float bx1 = 0.0, by1 = 0.0;
            
            vec2 uvx = uv + e.xy;
            vec2 uvy = uv + e.yx;
            
            bx1 += metaball(uvx, vec2(0.0, 0.0), 0.15);
            by1 += metaball(uvy, vec2(0.0, 0.0), 0.15);
            for (float i = 0.0; i < 5.0; i++) {
                float angle = t * 0.5 + i * 1.256;
                float r = 0.2 + 0.1 * sin(t + i);
                vec2 pos = vec2(cos(angle), sin(angle)) * r;
                float rad = 0.08 + 0.02 * sin(t * 2.0 + i);
                bx1 += metaball(uvx, pos, rad);
                by1 += metaball(uvy, pos, rad);
            }
            bx1 += metaball(uvx, mouse, 0.12);
            by1 += metaball(uvy, mouse, 0.12);
            
            vec3 normal = normalize(vec3(bx - bx1, bx - by1, 0.1));
            
            vec3 viewDir = normalize(vec3(uv, -1.0));
            vec3 reflectDir = reflect(viewDir, normal);
            vec3 reflection = getEnv(reflectDir);
            
            float fresnel = pow(1.0 - max(dot(-viewDir, normal), 0.0), 3.0);
            
            vec3 chrome = vec3(0.8, 0.85, 0.9);
            vec3 color = mix(chrome * 0.3, reflection, 0.7 + fresnel * 0.3);
            
            vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
            float spec = pow(max(dot(reflectDir, lightDir), 0.0), 60.0);
            color += vec3(1.0) * spec;
            
            color += vec3(0.3, 0.5, 0.7) * fresnel * 0.5;
            
            fragColor = vec4(color * surface, 1.0);
        }
    `;

    const program = manager.createProgram(vs, fs);
    if (!program) return;

    const uTime = manager.gl.getUniformLocation(program, 'u_time');
    const uRes = manager.gl.getUniformLocation(program, 'u_resolution');
    const uMouse = manager.gl.getUniformLocation(program, 'u_mouse');
    const uClick = manager.gl.getUniformLocation(program, 'u_click');

    let mouseX = 0, mouseY = 0;
    let clickTime = -1000;

    wrapper.addEventListener('mousemove', e => {
        const rect = wrapper.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = rect.height - (e.clientY - rect.top);
    });

    wrapper.addEventListener('click', () => {
        clickTime = performance.now();
    });

    manager.addRenderable({
        element: wrapper,
        program: program,
        uniformsCallback: (gl, time) => {
            const rect = wrapper.getBoundingClientRect();
            const click = Math.max(0, 1 - (performance.now() - clickTime) / 500);
            gl.uniform1f(uTime, time);
            gl.uniform2f(uRes, rect.width, rect.height);
            gl.uniform2f(uMouse, mouseX, mouseY);
            gl.uniform1f(uClick, click);
        }
    });
}

/**
 * Kinetic Typography Buttons
 * Text explodes into particles on hover and reforms with spring physics
 */
function initKineticTypographyButtons() {
    const container = document.getElementById('kinetic-typography-buttons');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 150px;
        position: relative;
        margin: 1rem auto;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 300;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 12px;
        cursor: pointer;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    `;

    wrapper.appendChild(canvas);
    container.appendChild(wrapper);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const text = "CLICK ME";
    const fontSize = 48;
    const particles = [];
    let isExploded = false;
    let mouseX = canvas.width / 2;
    let mouseY = canvas.height / 2;

    // Create text particles
    function createTextParticles() {
        particles.length = 0;

        ctx.font = `bold ${fontSize}px 'Segoe UI', sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#fff';
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        const step = 3; // Sample every 3 pixels
        for (let y = 0; y < canvas.height; y += step) {
            for (let x = 0; x < canvas.width; x += step) {
                const i = (y * canvas.width + x) * 4;
                if (data[i + 3] > 128) {
                    particles.push({
                        homeX: x,
                        homeY: y,
                        x: x,
                        y: y,
                        vx: 0,
                        vy: 0,
                        size: step * 0.8,
                        hue: Math.random() * 60 + 160, // Cyan to blue
                    });
                }
            }
        }
    }

    function explode() {
        particles.forEach(p => {
            const angle = Math.atan2(p.y - canvas.height / 2, p.x - canvas.width / 2);
            const force = 5 + Math.random() * 10;
            p.vx = Math.cos(angle) * force + (Math.random() - 0.5) * 5;
            p.vy = Math.sin(angle) * force + (Math.random() - 0.5) * 5;
        });
        isExploded = true;
    }

    function reform() {
        isExploded = false;
    }

    createTextParticles();

    canvas.addEventListener('mouseenter', explode);
    canvas.addEventListener('mouseleave', reform);
    canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        mouseX = (e.clientX - rect.left) * 2;
        mouseY = (e.clientY - rect.top) * 2;
    });

    function animate() {
        ctx.fillStyle = 'rgba(26, 26, 46, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
            if (isExploded) {
                // Repel from mouse
                const dx = p.x - mouseX;
                const dy = p.y - mouseY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 100) {
                    const force = (100 - dist) / 100 * 2;
                    p.vx += (dx / dist) * force;
                    p.vy += (dy / dist) * force;
                }

                // Add some chaos
                p.vx += (Math.random() - 0.5) * 0.5;
                p.vy += (Math.random() - 0.5) * 0.5;
            } else {
                // Spring back to home position
                const dx = p.homeX - p.x;
                const dy = p.homeY - p.y;
                p.vx += dx * 0.08;
                p.vy += dy * 0.08;
            }

            // Apply velocity
            p.vx *= 0.92;
            p.vy *= 0.92;
            p.x += p.vx;
            p.y += p.vy;

            // Draw particle
            ctx.fillStyle = `hsla(${p.hue}, 80%, 60%, 0.9)`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();

            // Glow effect
            ctx.fillStyle = `hsla(${p.hue}, 80%, 70%, 0.3)`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * 2, 0, Math.PI * 2);
            ctx.fill();
        });

        requestAnimationFrame(animate);
    }

    animate();
}

/**
 * EMP Buttons
 * Shockwave rings with glitch interference effects
 */
function initEMPButtons(manager) {
    const container = document.getElementById('emp-buttons');
    if (!container || !manager) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
        border-radius: 12px;
        cursor: pointer;
        overflow: hidden;
    `;

    // Create SVG scanline overlay
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    `;
    svg.innerHTML = `
        <defs>
            <pattern id="scanlines" patternUnits="userSpaceOnUse" width="4" height="4">
                <line x1="0" y1="0" x2="4" y2="0" stroke="rgba(0,255,255,0.03)" stroke-width="1"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#scanlines)"/>
    `;

    wrapper.appendChild(svg);
    container.appendChild(wrapper);
    
    const vs = `#version 300 es
    in vec4 a_position;
    out vec2 v_uv;
    void main() {
        v_uv = a_position.xy * 0.5 + 0.5;
        gl_Position = a_position;
    }
    `;

    const fs = `#version 300 es
    precision highp float;
    in vec2 v_uv;
    uniform float u_time;
    uniform vec2 u_resolution;
    uniform vec3 u_emps[5]; // x, y, startTime
    out vec4 fragColor;

    // Noise function
    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
        
        vec3 color = vec3(0.02, 0.03, 0.05);
        
        // Central button glow
        float buttonDist = length(uv);
        float button = smoothstep(0.2, 0.15, buttonDist);
        color += vec3(0.0, 0.3, 0.5) * button;
        
        // Button edge
        float edge = smoothstep(0.18, 0.15, buttonDist) - smoothstep(0.15, 0.12, buttonDist);
        color += vec3(0.0, 0.8, 1.0) * edge * 2.0;
        
        // EMP shockwaves
        for (int i = 0; i < 5; i++) {
            vec3 emp = u_emps[i];
            float t = u_time - emp.z;
            
            if (t > 0.0 && t < 2.0) {
                vec2 empPos = (emp.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
                float d = length(uv - empPos);
                
                // Expanding ring
                float ringRadius = t * 0.5;
                float ringWidth = 0.05;
                float ring = smoothstep(ringWidth, 0.0, abs(d - ringRadius));
                ring *= exp(-t * 2.0); // Fade over time
                
                // Multiple rings
                float ring2 = smoothstep(ringWidth * 0.5, 0.0, abs(d - ringRadius * 0.7));
                ring2 *= exp(-t * 2.5);
                
                float ring3 = smoothstep(ringWidth * 0.3, 0.0, abs(d - ringRadius * 1.3));
                ring3 *= exp(-t * 1.5);
                
                // Add color (cyan to magenta gradient)
                color += vec3(0.0, 0.8, 1.0) * ring;
                color += vec3(0.5, 0.0, 1.0) * ring2;
                color += vec3(1.0, 0.0, 0.5) * ring3 * 0.5;
                
                // Glitch distortion
                if (t < 0.5) {
                    float glitch = hash(uv * 100.0 + t * 1000.0);
                    if (glitch > 0.98) {
                        color = vec3(1.0, 1.0, 1.0);
                    }
                }
            }
        }
        
        // Scanline effect
        float scanline = sin(gl_FragCoord.y * 1.0) * 0.02 + 0.98;
        color *= scanline;
        
        // Vignette
        float vignette = 1.0 - length(uv) * 0.5;
        color *= vignette;
        
        fragColor = vec4(color, 1.0);
    }
    `;

    const program = manager.createProgram(vs, fs);
    if (!program) return;

    const uTime = manager.gl.getUniformLocation(program, 'u_time');
    const uRes = manager.gl.getUniformLocation(program, 'u_resolution');
    const uEmps = manager.gl.getUniformLocation(program, 'u_emps');

    const emps = new Float32Array(15).fill(-100);
    let empIndex = 0;

    wrapper.addEventListener('click', e => {
        const rect = wrapper.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = rect.height - (e.clientY - rect.top);

        emps[empIndex * 3] = x;
        emps[empIndex * 3 + 1] = y;
        emps[empIndex * 3 + 2] = performance.now() * 0.001;

        empIndex = (empIndex + 1) % 5;

        // Glitch the wrapper briefly
        wrapper.style.transform = `translate(${(Math.random() - 0.5) * 5}px, ${(Math.random() - 0.5) * 5}px)`;
        setTimeout(() => wrapper.style.transform = 'none', 100);
    });

    manager.addRenderable({
        element: wrapper,
        program: program,
        uniformsCallback: (gl, time) => {
            const rect = wrapper.getBoundingClientRect();
            gl.uniform1f(uTime, time);
            gl.uniform2f(uRes, rect.width, rect.height);
            gl.uniform3fv(uEmps, emps);
        },
        // This shader should clear the background
        customDraw: (gl, time) => {
            const rect = wrapper.getBoundingClientRect();
            const canvasRect = manager.canvas.getBoundingClientRect();
            gl.viewport(rect.left - canvasRect.left, canvasRect.height - rect.bottom, rect.width, rect.height);
            gl.clearColor(0.02, 0.03, 0.05, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            
            gl.uniform1f(uTime, time);
            gl.uniform2f(uRes, rect.width, rect.height);
            gl.uniform3fv(uEmps, emps);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
    });
}
