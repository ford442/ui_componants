/**
 * Buttons Page JavaScript
 * Creates and manages LED button components with layered canvas effects
 */

document.addEventListener('DOMContentLoaded', () => {
    checkWebGL2Support().then(supported => {
        if (!supported) {
            document.getElementById('webgl2-warning')?.setAttribute('style', 'display: block;');
            document.body.classList.add('no-webgl2');
        }
    });

    checkWebGPUSupport().then(supported => {
        if (supported) {
            initParticleSwarmButtons();
            initQuantumFluxButtons();
            initMagneticFieldButtons();
            initNeuralNetworkButtons();
            initCompositingShowcase();
        } else {
            document.getElementById('webgpu-warning')?.setAttribute('style', 'display: block;');
            document.body.classList.add('no-webgpu');
        }
    });

    initBasicButtons();
    initRGBButtons();
    initMomentaryButtons();
    initPulsingButtons();
    initButtonMatrix();
    initLayeredDemo();
    initArcadeButtons();
    initIndustrialButtons();
    initHolographicButtons();
    initOrganicButtons();

    // New Experiments
    initLiquidMetalButtons();
    initKineticTypographyButtons();
    initEMPButtons();
});

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
function initRGBButtons() {
    const container = document.getElementById('rgb-buttons');
    if (!container) return;

    // Create custom RGB button with color cycling
    const createRGBButton = (label) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'rgb-button-wrapper';
        wrapper.style.cssText = `
            width: 100px;
            height: 60px;
            position: relative;
        `;

        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 120;
        canvas.style.cssText = `
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
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

        wrapper.appendChild(canvas);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        // Initialize WebGL2 for RGB effect
        const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
        if (!gl) return;

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
            in vec2 a_texCoord;
            out vec2 v_texCoord;
            
            void main() {
                gl_Position = a_position;
                v_texCoord = a_texCoord;
            }
        `;

        const program = createProgram(gl, vertexShader, fragmentShader);
        if (!program) return;

        setupQuad(gl, program);

        const uniforms = {
            time: gl.getUniformLocation(program, 'u_time'),
            resolution: gl.getUniformLocation(program, 'u_resolution'),
            on: gl.getUniformLocation(program, 'u_on')
        };

        let isOn = false;

        button.addEventListener('click', () => {
            isOn = !isOn;
        });

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const animate = (timestamp) => {
            const time = timestamp * 0.001;

            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(program);

            gl.uniform1f(uniforms.time, time);
            gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
            gl.uniform1f(uniforms.on, isOn ? 1.0 : 0.0);

            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
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
function initPulsingButtons() {
    const container = document.getElementById('pulsing-buttons');
    if (!container) return;

    const configs = [
        { label: 'Slow', speed: 1, color: [0, 1, 0.5] },
        { label: 'Medium', speed: 2, color: [1, 0.5, 0] },
        { label: 'Fast', speed: 4, color: [1, 0, 0.3] },
        { label: 'Strobe', speed: 10, color: [0, 0.7, 1] }
    ];

    configs.forEach(({ label, speed, color }) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            width: 80px;
            height: 50px;
            position: relative;
        `;

        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 100;
        canvas.style.cssText = `
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
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

        wrapper.appendChild(canvas);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        // Initialize WebGL for pulse effect
        const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
        if (!gl) return;

        const fragmentShader = `
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec3 u_color;
            uniform float u_speed;
            
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
                
                gl_FragColor = vec4(finalColor, alpha);
            }
        `;

        const program = UIComponents.ShaderUtils.createProgram(
            gl,
            UIComponents.ShaderUtils.vertexShader2D,
            fragmentShader
        );

        if (!program) return;

        setupQuadWebGL1(gl, program);

        const uniforms = {
            time: gl.getUniformLocation(program, 'u_time'),
            resolution: gl.getUniformLocation(program, 'u_resolution'),
            color: gl.getUniformLocation(program, 'u_color'),
            speed: gl.getUniformLocation(program, 'u_speed')
        };

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const animate = (timestamp) => {
            const time = timestamp * 0.001;

            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(program);

            gl.uniform1f(uniforms.time, time);
            gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
            gl.uniform3fv(uniforms.color, color);
            gl.uniform1f(uniforms.speed, speed);

            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    });
}

/**
 * Initialize button matrix
 */
function initButtonMatrix() {
    const container = document.getElementById('button-matrix');
    if (!container) return;

    const buttons = [];
    const colors = [
        [1, 0, 0.3], [0, 1, 0.5], [0, 0.6, 1], [1, 0.5, 0],
        [1, 0, 0.8], [0.5, 1, 0], [0, 1, 1], [1, 0.3, 0],
        [0.7, 0, 1], [0, 1, 0.3], [0.3, 0.7, 1], [1, 1, 0],
        [1, 0, 0.5], [0, 0.8, 0.5], [0, 0.4, 1], [1, 0.6, 0]
    ];

    for (let i = 0; i < 16; i++) {
        const wrapper = document.createElement('div');
        wrapper.className = 'matrix-btn';

        const btn = new UIComponents.LEDButton(wrapper, {
            width: 60,
            height: 60,
            color: colors[i],
            onToggle: (isOn) => {
                if (isOn) {
                    wrapper.classList.add('animate-on');
                    wrapper.addEventListener('animationend', () => {
                        wrapper.classList.remove('animate-on');
                    }, { once: true });
                }
            }
        });

        buttons.push(btn);
        container.appendChild(wrapper);
    }

    // Matrix control buttons
    document.getElementById('matrix-random')?.addEventListener('click', () => {
        buttons.forEach(btn => {
            btn.setOn(Math.random() > 0.5);
        });
    });

    document.getElementById('matrix-wave')?.addEventListener('click', () => {
        buttons.forEach(btn => btn.setOn(false));

        let index = 0;
        const interval = setInterval(() => {
            if (index >= 16) {
                clearInterval(interval);
                return;
            }
            buttons[index].setOn(true);
            if (index > 0) buttons[index - 1].setOn(false);
            index++;
        }, 150);

        setTimeout(() => {
            if (buttons[15]) buttons[15].setOn(false);
        }, 150 * 17);
    });

    document.getElementById('matrix-chase')?.addEventListener('click', () => {
        const sequence = [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 5, 6, 10, 9];
        buttons.forEach(btn => btn.setOn(false));

        let index = 0;
        const interval = setInterval(() => {
            if (index >= sequence.length) {
                clearInterval(interval);
                return;
            }
            buttons[sequence[index]].setOn(true);
            if (index > 0) buttons[sequence[index - 1]].setOn(false);
            index++;
        }, 100);

        setTimeout(() => {
            buttons[sequence[sequence.length - 1]]?.setOn(false);
        }, 100 * (sequence.length + 1));
    });

    document.getElementById('matrix-clear')?.addEventListener('click', () => {
        buttons.forEach(btn => btn.setOn(false));
    });
}

/**
 * Initialize layered demo
 */
function initLayeredDemo() {
    const container = document.getElementById('layered-demo');
    if (!container) return;

    const layeredCanvas = new UIComponents.LayeredCanvas(container, {
        width: 800,
        height: 400
    });

    // Add WebGL base layer
    const webglLayer = layeredCanvas.addLayer('webgl-base', 'webgl', 0);

    // Add WebGL2 effects layer
    const webgl2Layer = layeredCanvas.addLayer('webgl2-effects', 'webgl2', 1);

    // Add SVG overlay layer
    const svgLayer = layeredCanvas.addSVGLayer('svg-overlay', 2);

    // Setup WebGL base layer shader
    if (webglLayer.context) {
        const gl = webglLayer.context;

        const fragmentShader = `
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                
                // Grid pattern
                vec2 grid = fract(uv * 20.0);
                float gridLine = step(0.95, grid.x) + step(0.95, grid.y);
                
                // Background gradient
                vec3 bg = mix(vec3(0.02, 0.02, 0.05), vec3(0.05, 0.08, 0.1), uv.y);
                
                vec3 color = bg + vec3(0.0, 0.1, 0.15) * gridLine * 0.3;
                
                gl_FragColor = vec4(color, 1.0);
            }
        `;

        const program = UIComponents.ShaderUtils.createProgram(
            gl,
            UIComponents.ShaderUtils.vertexShader2D,
            fragmentShader
        );

        if (program) {
            setupQuadWebGL1(gl, program);

            const uniforms = {
                time: gl.getUniformLocation(program, 'u_time'),
                resolution: gl.getUniformLocation(program, 'u_resolution')
            };

            layeredCanvas.setRenderFunction('webgl-base', (layer, timestamp) => {
                const time = timestamp * 0.001;

                gl.viewport(0, 0, layer.canvas.width, layer.canvas.height);
                gl.clearColor(0, 0, 0, 1);
                gl.clear(gl.COLOR_BUFFER_BIT);

                gl.useProgram(program);
                gl.uniform1f(uniforms.time, time);
                gl.uniform2f(uniforms.resolution, layer.canvas.width, layer.canvas.height);

                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            });
        }
    }

    // Setup WebGL2 effects layer
    if (webgl2Layer.context) {
        const gl = webgl2Layer.context;

        const fragmentShader = `#version 300 es
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            out vec4 fragColor;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                
                // Multiple glowing orbs
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

        const vertexShader = `#version 300 es
            in vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `;

        const program = createProgram(gl, vertexShader, fragmentShader);

        if (program) {
            setupQuad(gl, program);

            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            const uniforms = {
                time: gl.getUniformLocation(program, 'u_time'),
                resolution: gl.getUniformLocation(program, 'u_resolution')
            };

            layeredCanvas.setRenderFunction('webgl2-effects', (layer, timestamp) => {
                const time = timestamp * 0.001;

                gl.viewport(0, 0, layer.canvas.width, layer.canvas.height);
                gl.clearColor(0, 0, 0, 0);
                gl.clear(gl.COLOR_BUFFER_BIT);

                gl.useProgram(program);
                gl.uniform1f(uniforms.time, time);
                gl.uniform2f(uniforms.resolution, layer.canvas.width, layer.canvas.height);

                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            });
        }
    }

    // Setup SVG overlay
    if (svgLayer.element) {
        const svg = svgLayer.element;
        svg.setAttribute('viewBox', '0 0 800 400');

        // Add decorative elements
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <filter id="svg-glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        `;
        svg.appendChild(defs);

        // Add animated rings
        for (let i = 0; i < 3; i++) {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', 200 + i * 200);
            circle.setAttribute('cy', 200);
            circle.setAttribute('r', 40 + i * 10);
            circle.setAttribute('fill', 'none');
            circle.setAttribute('stroke', `rgba(0, 255, 200, ${0.3 - i * 0.08})`);
            circle.setAttribute('stroke-width', '2');
            circle.setAttribute('filter', 'url(#svg-glow)');

            // Add animation
            const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
            animate.setAttribute('attributeName', 'r');
            animate.setAttribute('values', `${40 + i * 10};${50 + i * 10};${40 + i * 10}`);
            animate.setAttribute('dur', `${2 + i * 0.5}s`);
            animate.setAttribute('repeatCount', 'indefinite');
            circle.appendChild(animate);

            svg.appendChild(circle);
        }
    }

    // Start animation
    layeredCanvas.startAnimation();

    // Layer toggle controls
    const toggleLayer = (checkbox, layerName) => {
        checkbox?.addEventListener('change', (e) => {
            const layer = layeredCanvas.getLayer(layerName);
            if (layer) {
                if (layer.canvas) {
                    layer.canvas.style.display = e.target.checked ? 'block' : 'none';
                }
                if (layer.element) {
                    layer.element.style.display = e.target.checked ? 'block' : 'none';
                }
            }
        });
    };

    toggleLayer(document.getElementById('layer-webgl'), 'webgl-base');
    toggleLayer(document.getElementById('layer-webgl2'), 'webgl2-effects');
    toggleLayer(document.getElementById('layer-svg'), 'svg-overlay');

    // CSS filter toggle
    document.getElementById('layer-css')?.addEventListener('change', (e) => {
        container.style.filter = e.target.checked
            ? 'contrast(1.1) saturate(1.2)'
            : 'none';
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
        console.error('Vertex shader error:', gl.getShaderInfoLog(vertexShader));
        return null;
    }

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);

    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
        console.error('Fragment shader error:', gl.getShaderInfoLog(fragmentShader));
        return null;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
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
function initParticleSwarmButtons() {
    const container = document.getElementById('particle-swarm-buttons');
    if (!container) return;

    const configs = [
        { label: 'ATTRACT', color: [0, 1, 0.5, 0.8], physics: 'attract' },
        { label: 'REPEL', color: [1, 0.3, 0, 0.8], physics: 'repel' },
        { label: 'ORBIT', color: [0.3, 0.6, 1, 0.8], physics: 'orbit' }
    ];

    configs.forEach(async (config) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'particle-button-wrapper';
        wrapper.style.cssText = `
            width: 140px;
            height: 80px;
            position: relative;
            margin: 0.5rem;
        `;

        // WebGPU particle canvas (back layer)
        const particleCanvas = document.createElement('canvas');
        particleCanvas.width = 280;
        particleCanvas.height = 160;
        particleCanvas.style.cssText = `
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 0;
        `;

        // WebGL2 glow canvas (middle layer)
        const glowCanvas = document.createElement('canvas');
        glowCanvas.width = 280;
        glowCanvas.height = 160;
        glowCanvas.style.cssText = `
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
            pointer-events: none;
        `;

        // CSS button (top layer)
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

        wrapper.appendChild(particleCanvas);
        wrapper.appendChild(glowCanvas);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        // Initialize WebGPU particle system
        const particleSystem = new UIComponents.WebGPUParticleSystem(particleCanvas, {
            particleCount: 10000,
            particleSize: 2,
            color: config.color,
            attractorStrength: 0.3,
            damping: 0.98
        });

        const initialized = await particleSystem.init();
        if (!initialized) return;

        // Initialize WebGL2 glow effect
        const gl2 = glowCanvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
        if (gl2) {
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

            const program = createProgram(gl2, vertexShader, glowShader);
            if (program) {
                setupQuad(gl2, program);

                gl2.enable(gl2.BLEND);
                gl2.blendFunc(gl2.SRC_ALPHA, gl2.ONE_MINUS_SRC_ALPHA);

                const uniforms = {
                    time: gl2.getUniformLocation(program, 'u_time'),
                    resolution: gl2.getUniformLocation(program, 'u_resolution'),
                    color: gl2.getUniformLocation(program, 'u_color'),
                    active: gl2.getUniformLocation(program, 'u_active')
                };

                let isActive = false;
                let mouseX = 0;
                let mouseY = 0;

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

                let lastTime = 0;
                const animate = (timestamp) => {
                    const time = timestamp * 0.001;
                    const deltaTime = timestamp - lastTime > 0 ? (timestamp - lastTime) * 0.001 : 0.016;
                    lastTime = timestamp;

                    // Update particle system
                    particleSystem.updateUniforms(time, deltaTime, mouseX, mouseY);
                    particleSystem.render(time, deltaTime);

                    // Render glow
                    gl2.viewport(0, 0, glowCanvas.width, glowCanvas.height);
                    gl2.clearColor(0, 0, 0, 0);
                    gl2.clear(gl2.COLOR_BUFFER_BIT);

                    gl2.useProgram(program);
                    gl2.uniform1f(uniforms.time, time);
                    gl2.uniform2f(uniforms.resolution, glowCanvas.width, glowCanvas.height);
                    gl2.uniform3f(uniforms.color, config.color[0], config.color[1], config.color[2]);
                    gl2.uniform1f(uniforms.active, isActive ? 1.0 : 0.0);

                    gl2.drawArrays(gl2.TRIANGLE_STRIP, 0, 4);

                    requestAnimationFrame(animate);
                };

                requestAnimationFrame(animate);
            }
        }
    });
}

/**
 * Initialize Quantum Flux Buttons
 * WebGPU rendering with probabilistic shader effects
 */
function initQuantumFluxButtons() {
    const container = document.getElementById('quantum-flux-buttons');
    if (!container) return;

    const labels = ['QUANTUM A', 'QUANTUM B', 'QUANTUM C'];

    labels.forEach(async (label, index) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            width: 140px;
            height: 80px;
            position: relative;
            margin: 0.5rem;
        `;

        const canvas = document.createElement('canvas');
        canvas.width = 280;
        canvas.height = 160;
        canvas.style.cssText = `
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
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

        wrapper.appendChild(canvas);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        // Initialize WebGPU volumetric renderer for quantum effects
        const volumetric = new UIComponents.WebGPUVolumetricRenderer(canvas, {
            raySteps: 32,
            density: 0.3 + index * 0.1
        });

        const initialized = await volumetric.init();
        if (!initialized) return;

        let isActive = false;
        button.addEventListener('mouseenter', () => { isActive = true; });
        button.addEventListener('mouseleave', () => { isActive = false; });

        const animate = (timestamp) => {
            const time = timestamp * 0.001 * (1 + index * 0.3);
            volumetric.render(time);

            if (isActive) {
                button.style.borderColor = `rgba(${100 + Math.sin(time * 5) * 50}, 150, 255, 0.9)`;
            }

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    });
}

/**
 * Initialize Magnetic Field Buttons
 * Compute shader field line simulation
 */
function initMagneticFieldButtons() {
    const container = document.getElementById('magnetic-field-buttons');
    if (!container) return;

    const configs = [
        { label: 'N', polarity: 1, color: '#f44' },
        { label: 'S', polarity: -1, color: '#44f' }
    ];

    configs.forEach((config) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            width: 100px;
            height: 100px;
            position: relative;
            margin: 0.5rem;
        `;

        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 200;
        canvas.style.cssText = `
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
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

        wrapper.appendChild(canvas);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        // Draw magnetic field lines with WebGL
        const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
        if (gl) {
            const fragmentShader = `
                precision mediump float;
                uniform float u_time;
                uniform vec2 u_resolution;
                uniform float u_polarity;
                
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
                    gl_FragColor = vec4(color * fieldLine, fieldLine * 0.5);
                }
            `;

            const program = UIComponents.ShaderUtils.createProgram(
                gl,
                UIComponents.ShaderUtils.vertexShader2D,
                fragmentShader
            );

            if (program) {
                setupQuadWebGL1(gl, program);

                gl.enable(gl.BLEND);
                gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

                const uniforms = {
                    time: gl.getUniformLocation(program, 'u_time'),
                    resolution: gl.getUniformLocation(program, 'u_resolution'),
                    polarity: gl.getUniformLocation(program, 'u_polarity')
                };

                const animate = (timestamp) => {
                    const time = timestamp * 0.001;

                    gl.viewport(0, 0, canvas.width, canvas.height);
                    gl.clearColor(0, 0, 0, 0);
                    gl.clear(gl.COLOR_BUFFER_BIT);

                    gl.useProgram(program);
                    gl.uniform1f(uniforms.time, time);
                    gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
                    gl.uniform1f(uniforms.polarity, config.polarity);

                    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

                    requestAnimationFrame(animate);
                };

                requestAnimationFrame(animate);
            }
        }
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
function initCompositingShowcase() {
    const container = document.getElementById('compositing-display');
    if (!container) return;

    container.style.cssText = `
        width: 100%;
        height: 400px;
        position: relative;
        background: #000;
        border-radius: 12px;
        overflow: hidden;
    `;

    // WebGPU particle layer
    const gpuCanvas = document.createElement('canvas');
    gpuCanvas.width = 800;
    gpuCanvas.height = 400;
    gpuCanvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 1;
    `;
    gpuCanvas.id = 'comp-gpu-layer';

    // WebGL2 glow layer
    const gl2Canvas = document.createElement('canvas');
    gl2Canvas.width = 800;
    gl2Canvas.height = 400;
    gl2Canvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 2;
        mix-blend-mode: screen;
    `;
    gl2Canvas.id = 'comp-gl2-layer';

    // SVG UI layer
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

    // Add UI elements to SVG
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

    container.appendChild(gpuCanvas);
    container.appendChild(gl2Canvas);
    container.appendChild(svg);

    // Initialize particle system
    (async () => {
        const particleSystem = new UIComponents.WebGPUParticleSystem(gpuCanvas, {
            particleCount: 5000,
            color: [0, 1, 0.5, 0.6],
            attractorStrength: 0.2,
            damping: 0.95
        });

        const initialized = await particleSystem.init();
        if (initialized) {
            let lastTime = 0;
            const animate = (timestamp) => {
                const time = timestamp * 0.001;
                const deltaTime = timestamp - lastTime > 0 ? (timestamp - lastTime) * 0.001 : 0.016;
                lastTime = timestamp;

                particleSystem.render(time, deltaTime);
                requestAnimationFrame(animate);
            };
            requestAnimationFrame(animate);
        }
    })();

    // Initialize WebGL2 glow
    const gl2 = gl2Canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
    if (gl2) {
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
            in vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `;

        const program = createProgram(gl2, vertexShader, shaderCode);
        if (program) {
            setupQuad(gl2, program);

            gl2.enable(gl2.BLEND);
            gl2.blendFunc(gl2.SRC_ALPHA, gl2.ONE_MINUS_SRC_ALPHA);

            const uniforms = {
                time: gl2.getUniformLocation(program, 'u_time'),
                resolution: gl2.getUniformLocation(program, 'u_resolution')
            };

            const animate = (timestamp) => {
                const time = timestamp * 0.001;

                gl2.viewport(0, 0, gl2Canvas.width, gl2Canvas.height);
                gl2.clearColor(0, 0, 0, 0);
                gl2.clear(gl2.COLOR_BUFFER_BIT);

                gl2.useProgram(program);
                gl2.uniform1f(uniforms.time, time);
                gl2.uniform2f(uniforms.resolution, gl2Canvas.width, gl2Canvas.height);

                gl2.drawArrays(gl2.TRIANGLE_STRIP, 0, 4);

                requestAnimationFrame(animate);
            };

            requestAnimationFrame(animate);
        }
    }

    // Setup layer controls
    document.getElementById('comp-webgpu')?.addEventListener('change', (e) => {
        gpuCanvas.style.display = e.target.checked ? 'block' : 'none';
    });

    document.getElementById('comp-webgl2')?.addEventListener('change', (e) => {
        gl2Canvas.style.display = e.target.checked ? 'block' : 'none';
    });

    document.getElementById('comp-svg')?.addEventListener('change', (e) => {
        svg.style.display = e.target.checked ? 'block' : 'none';
    });

    document.getElementById('comp-css')?.addEventListener('change', (e) => {
        container.style.filter = e.target.checked ? 'contrast(1.1) saturate(1.2)' : 'none';
    });

    document.getElementById('comp-webgpu-opacity')?.addEventListener('input', (e) => {
        gpuCanvas.style.opacity = e.target.value / 100;
    });

    document.getElementById('comp-webgl2-opacity')?.addEventListener('input', (e) => {
        gl2Canvas.style.opacity = e.target.value / 100;
    });

    document.getElementById('comp-svg-opacity')?.addEventListener('input', (e) => {
        svg.style.opacity = e.target.value / 100;
    });

    document.getElementById('comp-blend-mode')?.addEventListener('change', (e) => {
        gl2Canvas.style.mixBlendMode = e.target.value;
    });

    // Layer order controls
    document.querySelectorAll('.order-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.getAttribute('data-action');
            if (action === 'swap-gpu-gl2') {
                const z1 = gpuCanvas.style.zIndex;
                gpuCanvas.style.zIndex = gl2Canvas.style.zIndex;
                gl2Canvas.style.zIndex = z1;
            } else if (action === 'svg-to-back') {
                svg.style.zIndex = '0';
            } else if (action === 'reset-order') {
                gpuCanvas.style.zIndex = '1';
                gl2Canvas.style.zIndex = '2';
                svg.style.zIndex = '3';
            }
        });
    });
}

// ============================================================================
// NEW EXPERIMENTS
// ============================================================================

/**
 * Liquid Metal / Mercury Buttons
 * Morphing metallic surfaces with reflective ripples
 */
function initLiquidMetalButtons() {
    const container = document.getElementById('liquid-metal-buttons');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 400;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 12px;
        cursor: pointer;
    `;

    wrapper.appendChild(canvas);
    container.appendChild(wrapper);

    const gl = canvas.getContext('webgl2');
    if (!gl) {
        container.innerHTML = '<p style="color: #ff6666;">WebGL2 required</p>';
        return;
    }

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

    // Environment reflection
    vec3 getEnv(vec3 dir) {
        float y = dir.y * 0.5 + 0.5;
        vec3 sky = mix(vec3(0.15, 0.2, 0.3), vec3(0.8, 0.9, 1.0), pow(y, 0.4));
        vec3 ground = mix(vec3(0.1, 0.08, 0.05), vec3(0.3, 0.25, 0.2), 1.0 - y);
        return mix(ground, sky, smoothstep(-0.1, 0.1, dir.y));
    }

    // Metaball function
    float metaball(vec2 p, vec2 center, float r) {
        float d = length(p - center);
        return r * r / (d * d + 0.001);
    }

    void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
        vec2 mouse = (u_mouse - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
        
        // Multiple metaballs
        float blob = 0.0;
        float t = u_time;
        
        // Central blob
        blob += metaball(uv, vec2(0.0, 0.0), 0.15);
        
        // Orbiting blobs
        for (float i = 0.0; i < 5.0; i++) {
            float angle = t * 0.5 + i * 1.256;
            float r = 0.2 + 0.1 * sin(t + i);
            vec2 pos = vec2(cos(angle), sin(angle)) * r;
            blob += metaball(uv, pos, 0.08 + 0.02 * sin(t * 2.0 + i));
        }
        
        // Mouse interaction
        blob += metaball(uv, mouse, 0.12) * (1.0 + u_click * 0.5);
        
        // Click ripple
        if (u_click > 0.0) {
            float ripple = sin(length(uv - mouse) * 30.0 - u_time * 10.0);
            blob += ripple * 0.1 * u_click;
        }
        
        // Threshold for surface
        float surface = smoothstep(0.8, 0.9, blob);
        
        if (surface < 0.1) {
            fragColor = vec4(0.05, 0.05, 0.08, 1.0);
            return;
        }
        
        // Calculate normal
        vec2 e = vec2(0.01, 0.0);
        float bx = blob;
        float bx1 = 0.0, by1 = 0.0;
        
        vec2 uvx = uv + e.xy;
        vec2 uvy = uv + e.yx;
        
        // Recalculate for gradient
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
        
        // Reflection
        vec3 viewDir = normalize(vec3(uv, -1.0));
        vec3 reflectDir = reflect(viewDir, normal);
        vec3 reflection = getEnv(reflectDir);
        
        // Fresnel
        float fresnel = pow(1.0 - max(dot(-viewDir, normal), 0.0), 3.0);
        
        // Chrome color
        vec3 chrome = vec3(0.8, 0.85, 0.9);
        vec3 color = mix(chrome * 0.3, reflection, 0.7 + fresnel * 0.3);
        
        // Specular highlights
        vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
        float spec = pow(max(dot(reflectDir, lightDir), 0.0), 60.0);
        color += vec3(1.0) * spec;
        
        // Edge glow
        color += vec3(0.3, 0.5, 0.7) * fresnel * 0.5;
        
        fragColor = vec4(color * surface, 1.0);
    }
    `;

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uRes = gl.getUniformLocation(program, 'u_resolution');
    const uMouse = gl.getUniformLocation(program, 'u_mouse');
    const uClick = gl.getUniformLocation(program, 'u_click');

    let mouseX = canvas.width / 2, mouseY = canvas.height / 2;
    let clickTime = 0;

    canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        mouseX = (e.clientX - rect.left) * 2;
        mouseY = (rect.height - (e.clientY - rect.top)) * 2;
    });

    canvas.addEventListener('click', () => {
        clickTime = performance.now();
    });

    function render(time) {
        const click = Math.max(0, 1 - (time - clickTime) / 500);

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.05, 0.05, 0.08, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform2f(uMouse, mouseX, mouseY);
        gl.uniform1f(uClick, click);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
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
function initEMPButtons() {
    const container = document.getElementById('emp-buttons');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 400;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 12px;
        cursor: pointer;
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
        border-radius: 12px;
        overflow: hidden;
    `;
    svg.innerHTML = `
        <defs>
            <pattern id="scanlines" patternUnits="userSpaceOnUse" width="4" height="4">
                <line x1="0" y1="0" x2="4" y2="0" stroke="rgba(0,255,255,0.03)" stroke-width="1"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#scanlines)"/>
    `;

    wrapper.appendChild(canvas);
    wrapper.appendChild(svg);
    container.appendChild(wrapper);

    const gl = canvas.getContext('webgl2');
    if (!gl) return;

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

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uRes = gl.getUniformLocation(program, 'u_resolution');
    const uEmps = gl.getUniformLocation(program, 'u_emps');

    const emps = new Float32Array(15).fill(-100);
    let empIndex = 0;

    canvas.addEventListener('click', e => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * 2;
        const y = (rect.height - (e.clientY - rect.top)) * 2;

        emps[empIndex * 3] = x;
        emps[empIndex * 3 + 1] = y;
        emps[empIndex * 3 + 2] = performance.now() * 0.001;

        empIndex = (empIndex + 1) % 5;

        // Glitch the wrapper briefly
        wrapper.style.transform = `translate(${(Math.random() - 0.5) * 5}px, ${(Math.random() - 0.5) * 5}px)`;
        setTimeout(() => wrapper.style.transform = 'none', 100);
    });

    function render(time) {
        const t = time * 0.001;

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.02, 0.03, 0.05, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, t);
        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform3fv(uEmps, emps);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}
