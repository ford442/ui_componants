/**
 * Buttons Page JavaScript
 * Creates and manages LED button components with layered canvas effects
 */

document.addEventListener('DOMContentLoaded', () => {
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
                    setTimeout(() => wrapper.classList.remove('animate-on'), 300);
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
        -1,  1,
         1,  1,
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
        -1,  1, 0, 1,
         1,  1, 1, 1,
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
