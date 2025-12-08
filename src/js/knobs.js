/**
 * Knobs Page JavaScript
 * Creates and manages rotary knob components with layered canvas effects
 */

document.addEventListener('DOMContentLoaded', () => {
    initBasicKnobs();
    initLEDRingKnobs();
    initVintageKnobs();
    initDigitalEncoders();
    initLargeKnobDemo();
    initMixerConsole();
    initMeterKnobs();
    initDualKnobs();
    initIlluminatedKnobs();
    initStepKnobs();

    // New Experiments
    initFerrofluidKnob();
    initTimeWarpKnob();
    initWaveformRingKnob();
});

/**
 * Initialize basic rotary knobs
 */
function initBasicKnobs() {
    const container = document.getElementById('basic-knobs');
    if (!container) return;

    const configs = [
        { label: 'Volume', color: '#00ff88', value: 75 },
        { label: 'Pan', color: '#00aaff', value: 50 },
        { label: 'Gain', color: '#ffaa00', value: 25 },
        { label: 'Filter', color: '#ff44aa', value: 60 }
    ];

    configs.forEach(config => {
        new UIComponents.RotaryKnob(container, {
            size: 70,
            min: 0,
            max: 100,
            value: config.value,
            color: config.color,
            label: config.label
        });
    });
}

/**
 * Initialize LED ring knobs
 */
function initLEDRingKnobs() {
    const container = document.getElementById('led-ring-knobs');
    if (!container) return;

    const createLEDRingKnob = (label, color) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'knob-wrapper';
        wrapper.style.width = '90px';

        // Label
        const labelEl = document.createElement('div');
        labelEl.className = 'knob-label';
        labelEl.textContent = label;

        // Knob container
        const knobContainer = document.createElement('div');
        knobContainer.className = 'led-ring-knob';

        // Create LED ring
        const ledRing = document.createElement('div');
        ledRing.className = 'led-ring';

        const numLeds = 12;
        const leds = [];

        for (let i = 0; i < numLeds; i++) {
            const led = document.createElement('div');
            led.className = 'led-dot';
            const angle = (i / numLeds) * 270 - 135; // -135 to 135 degrees
            const radius = 35;
            const x = 40 + radius * Math.cos((angle - 90) * Math.PI / 180);
            const y = 40 + radius * Math.sin((angle - 90) * Math.PI / 180);
            led.style.left = `${x - 3}px`;
            led.style.top = `${y - 3}px`;
            led.style.setProperty('--led-color', color);
            leds.push(led);
            ledRing.appendChild(led);
        }

        // Create center knob
        const centerKnob = document.createElement('div');
        centerKnob.style.cssText = `
            position: absolute;
            top: 15px;
            left: 15px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(145deg, #3a3a4a, #1a1a2a);
            cursor: grab;
            box-shadow: 
                0 3px 10px rgba(0, 0, 0, 0.5),
                inset 0 2px 5px rgba(255, 255, 255, 0.05);
        `;

        const indicator = document.createElement('div');
        indicator.style.cssText = `
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            width: 3px;
            height: 12px;
            background: ${color};
            border-radius: 2px;
            box-shadow: 0 0 10px ${color};
        `;
        centerKnob.appendChild(indicator);

        // Value display
        const valueEl = document.createElement('div');
        valueEl.className = 'knob-value';
        valueEl.textContent = '0';

        knobContainer.appendChild(ledRing);
        knobContainer.appendChild(centerKnob);
        wrapper.appendChild(labelEl);
        wrapper.appendChild(knobContainer);
        wrapper.appendChild(valueEl);
        container.appendChild(wrapper);

        // Add interaction
        let value = 0;
        let rotation = -135;
        let isDragging = false;
        let startY = 0;
        let startRotation = 0;

        const updateDisplay = () => {
            centerKnob.style.transform = `rotate(${rotation}deg)`;
            valueEl.textContent = Math.round(value);

            // Update LEDs
            const activeLeds = Math.floor((value / 100) * numLeds);
            leds.forEach((led, i) => {
                if (i < activeLeds) {
                    led.classList.add('active');
                    led.style.background = color;
                    led.style.boxShadow = `0 0 10px ${color}`;
                } else {
                    led.classList.remove('active');
                    led.style.background = '#333';
                    led.style.boxShadow = 'inset 0 1px 2px rgba(0, 0, 0, 0.5)';
                }
            });
        };

        centerKnob.addEventListener('mousedown', (e) => {
            isDragging = true;
            startY = e.clientY;
            startRotation = rotation;
            centerKnob.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaY = startY - e.clientY;
            rotation = Math.max(-135, Math.min(135, startRotation + deltaY));
            value = ((rotation + 135) / 270) * 100;
            updateDisplay();
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            centerKnob.style.cursor = 'grab';
        });

        updateDisplay();
    };

    createLEDRingKnob('Level', '#00ff88');
    createLEDRingKnob('Tone', '#00aaff');
    createLEDRingKnob('Mix', '#ff8800');
}

/**
 * Initialize vintage style knobs
 */
function initVintageKnobs() {
    const container = document.getElementById('vintage-knobs');
    if (!container) return;

    const labels = ['Treble', 'Mid', 'Bass', 'Master'];

    labels.forEach(label => {
        const wrapper = document.createElement('div');
        wrapper.className = 'knob-wrapper';
        wrapper.style.width = '80px';

        const labelEl = document.createElement('div');
        labelEl.className = 'knob-label';
        labelEl.textContent = label;

        const knob = document.createElement('div');
        knob.className = 'vintage-knob';

        const valueEl = document.createElement('div');
        valueEl.className = 'knob-value';
        valueEl.textContent = '5';

        wrapper.appendChild(labelEl);
        wrapper.appendChild(knob);
        wrapper.appendChild(valueEl);
        container.appendChild(wrapper);

        // Add interaction
        let value = 5;
        let rotation = 0;
        let isDragging = false;
        let startY = 0;
        let startRotation = 0;

        knob.addEventListener('mousedown', (e) => {
            isDragging = true;
            startY = e.clientY;
            startRotation = rotation;
            knob.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaY = startY - e.clientY;
            rotation = Math.max(-135, Math.min(135, startRotation + deltaY));
            value = Math.round(((rotation + 135) / 270) * 10);
            knob.style.transform = `rotate(${rotation}deg)`;
            valueEl.textContent = value;
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            knob.style.cursor = 'grab';
        });
    });
}

/**
 * Initialize digital encoders
 */
function initDigitalEncoders() {
    const container = document.getElementById('digital-encoders');
    if (!container) return;

    const configs = [
        { label: 'Preset', min: 1, max: 99, step: 1 },
        { label: 'BPM', min: 60, max: 200, step: 1 },
        { label: 'Channel', min: 1, max: 16, step: 1 }
    ];

    configs.forEach(config => {
        const wrapper = document.createElement('div');
        wrapper.className = 'digital-encoder';

        const display = document.createElement('div');
        display.className = 'encoder-display';
        display.textContent = config.min;

        const bodyContainer = document.createElement('div');
        bodyContainer.style.position = 'relative';

        const body = document.createElement('div');
        body.className = 'encoder-body';

        const notch = document.createElement('div');
        notch.className = 'encoder-notch';
        body.appendChild(notch);

        const labelEl = document.createElement('div');
        labelEl.className = 'knob-label';
        labelEl.textContent = config.label;

        bodyContainer.appendChild(body);
        wrapper.appendChild(display);
        wrapper.appendChild(bodyContainer);
        wrapper.appendChild(labelEl);
        container.appendChild(wrapper);

        // Add interaction
        let value = config.min;
        let rotation = 0;
        let isDragging = false;
        let startY = 0;
        let startValue = config.min;

        body.addEventListener('mousedown', (e) => {
            isDragging = true;
            startY = e.clientY;
            startValue = value;
            body.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaY = Math.floor((startY - e.clientY) / 5);
            value = Math.max(config.min, Math.min(config.max, startValue + deltaY * config.step));
            rotation += (e.clientY - startY) < 0 ? 10 : -10;
            notch.style.transform = `rotate(${rotation}deg)`;
            display.textContent = value;
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            body.style.cursor = 'grab';
        });
    });
}

/**
 * Initialize large knob demo with layered canvases
 */
function initLargeKnobDemo() {
    const container = document.getElementById('large-knob-demo');
    if (!container) return;

    const size = 180;
    let value = 50;
    let rotation = 0;

    // Create layered canvas structure
    const layers = {};

    // WebGL base layer - metallic knob texture
    const webglCanvas = document.createElement('canvas');
    webglCanvas.width = size * 2;
    webglCanvas.height = size * 2;
    webglCanvas.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        z-index: 1;
    `;
    webglCanvas.id = 'large-knob-webgl';

    // WebGL2 glow layer
    const webgl2Canvas = document.createElement('canvas');
    webgl2Canvas.width = size * 2;
    webgl2Canvas.height = size * 2;
    webgl2Canvas.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        z-index: 0;
    `;
    webgl2Canvas.id = 'large-knob-webgl2';

    // SVG detail layer
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.style.cssText = `
        position: absolute;
        z-index: 2;
        pointer-events: none;
    `;
    svg.id = 'large-knob-svg';

    // Add SVG elements
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
        <filter id="knob-shadow">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.5"/>
        </filter>
        <linearGradient id="knob-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#5a5a6a"/>
            <stop offset="50%" stop-color="#3a3a4a"/>
            <stop offset="100%" stop-color="#2a2a3a"/>
        </linearGradient>
        <filter id="glow-filter">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
    `;
    svg.appendChild(defs);

    const center = size / 2;

    // Outer ring with tick marks
    for (let i = 0; i < 11; i++) {
        const angle = -135 + (i / 10) * 270;
        const radians = (angle - 90) * Math.PI / 180;
        const x1 = center + 85 * Math.cos(radians);
        const y1 = center + 85 * Math.sin(radians);
        const x2 = center + 75 * Math.cos(radians);
        const y2 = center + 75 * Math.sin(radians);

        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        tick.setAttribute('x1', x1);
        tick.setAttribute('y1', y1);
        tick.setAttribute('x2', x2);
        tick.setAttribute('y2', y2);
        tick.setAttribute('stroke', '#555');
        tick.setAttribute('stroke-width', i % 5 === 0 ? '3' : '1');
        svg.appendChild(tick);
    }

    // Knob body
    const knobCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    knobCircle.setAttribute('cx', center);
    knobCircle.setAttribute('cy', center);
    knobCircle.setAttribute('r', 65);
    knobCircle.setAttribute('fill', 'url(#knob-gradient)');
    knobCircle.setAttribute('filter', 'url(#knob-shadow)');
    svg.appendChild(knobCircle);

    // Indicator group
    const indicatorGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    indicatorGroup.id = 'knob-indicator';
    indicatorGroup.style.transformOrigin = `${center}px ${center}px`;

    const indicator = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    indicator.setAttribute('x1', center);
    indicator.setAttribute('y1', center - 55);
    indicator.setAttribute('x2', center);
    indicator.setAttribute('y2', center - 35);
    indicator.setAttribute('stroke', '#00ff88');
    indicator.setAttribute('stroke-width', '4');
    indicator.setAttribute('stroke-linecap', 'round');
    indicator.setAttribute('filter', 'url(#glow-filter)');
    indicatorGroup.appendChild(indicator);

    svg.appendChild(indicatorGroup);

    // CSS effects wrapper
    const cssWrapper = document.createElement('div');
    cssWrapper.id = 'large-knob-css';
    cssWrapper.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        z-index: 3;
        pointer-events: none;
        border-radius: 50%;
        transition: filter 0.3s ease;
    `;

    // Interactive layer
    const interactiveLayer = document.createElement('div');
    interactiveLayer.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        z-index: 4;
        cursor: grab;
        border-radius: 50%;
    `;

    container.appendChild(webgl2Canvas);
    container.appendChild(webglCanvas);
    container.appendChild(svg);
    container.appendChild(cssWrapper);
    container.appendChild(interactiveLayer);

    layers.webgl = webglCanvas;
    layers.webgl2 = webgl2Canvas;
    layers.svg = svg;
    layers.css = cssWrapper;

    // Initialize WebGL base layer
    const gl = webglCanvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
    if (gl) {
        const fragmentShader = `
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform float u_rotation;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 center = vec2(0.5, 0.5);
                vec2 pos = uv - center;
                float dist = length(pos);
                float angle = atan(pos.y, pos.x);
                
                // Metallic reflection
                float reflection = 0.5 + 0.5 * sin(angle * 8.0 + u_rotation * 0.02);
                
                // Knob shape
                float knob = 1.0 - smoothstep(0.35, 0.36, dist);
                
                // Concentric rings
                float rings = 0.5 + 0.5 * sin(dist * 40.0);
                
                vec3 color = vec3(0.2 + reflection * 0.1, 0.2 + reflection * 0.1, 0.25 + reflection * 0.1);
                color *= rings * 0.3 + 0.7;
                
                gl_FragColor = vec4(color * knob, knob);
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
                rotation: gl.getUniformLocation(program, 'u_rotation')
            };

            layers.webglProgram = program;
            layers.webglUniforms = uniforms;
            layers.gl = gl;
        }
    }

    // Initialize WebGL2 glow layer
    const gl2 = webgl2Canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
    if (gl2) {
        const fragmentShader = `#version 300 es
            precision mediump float;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform float u_value;
            out vec4 fragColor;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(uv, center);
                
                // Outer glow ring
                float ring = abs(dist - 0.45);
                float glow = 0.02 / (ring * ring + 0.001);
                glow *= u_value;
                glow *= 0.9 + 0.1 * sin(u_time * 2.0);
                
                vec3 color = vec3(0.0, glow, glow * 0.5);
                float alpha = min(glow * 0.3, 1.0);
                
                fragColor = vec4(color, alpha);
            }
        `;

        const vertexShader = `#version 300 es
            in vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `;

        const program = createProgram(gl2, vertexShader, fragmentShader);

        if (program) {
            setupQuad(gl2, program);

            gl2.enable(gl2.BLEND);
            gl2.blendFunc(gl2.SRC_ALPHA, gl2.ONE_MINUS_SRC_ALPHA);

            const uniforms = {
                time: gl2.getUniformLocation(program, 'u_time'),
                resolution: gl2.getUniformLocation(program, 'u_resolution'),
                value: gl2.getUniformLocation(program, 'u_value')
            };

            layers.webgl2Program = program;
            layers.webgl2Uniforms = uniforms;
            layers.gl2 = gl2;
        }
    }

    // Animation loop
    const animate = (timestamp) => {
        const time = timestamp * 0.001;

        // Render WebGL base
        if (layers.gl && layers.webglProgram) {
            const gl = layers.gl;
            gl.viewport(0, 0, webglCanvas.width, webglCanvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(layers.webglProgram);
            gl.uniform1f(layers.webglUniforms.time, time);
            gl.uniform2f(layers.webglUniforms.resolution, webglCanvas.width, webglCanvas.height);
            gl.uniform1f(layers.webglUniforms.rotation, rotation);

            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }

        // Render WebGL2 glow
        if (layers.gl2 && layers.webgl2Program) {
            const gl2 = layers.gl2;
            gl2.viewport(0, 0, webgl2Canvas.width, webgl2Canvas.height);
            gl2.clearColor(0, 0, 0, 0);
            gl2.clear(gl2.COLOR_BUFFER_BIT);

            gl2.useProgram(layers.webgl2Program);
            gl2.uniform1f(layers.webgl2Uniforms.time, time);
            gl2.uniform2f(layers.webgl2Uniforms.resolution, webgl2Canvas.width, webgl2Canvas.height);
            gl2.uniform1f(layers.webgl2Uniforms.value, value / 100);

            gl2.drawArrays(gl2.TRIANGLE_STRIP, 0, 4);
        }

        requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);

    // Interaction
    let isDragging = false;
    let startY = 0;
    let startRotation = 0;

    const valueDisplay = document.getElementById('large-knob-value');
    const rotationDisplay = document.getElementById('large-knob-rotation');

    const updateDisplay = () => {
        indicatorGroup.style.transform = `rotate(${rotation}deg)`;
        if (valueDisplay) valueDisplay.textContent = Math.round(value);
        if (rotationDisplay) rotationDisplay.textContent = `${Math.round(rotation)}°`;
    };

    interactiveLayer.addEventListener('mousedown', (e) => {
        isDragging = true;
        startY = e.clientY;
        startRotation = rotation;
        interactiveLayer.style.cursor = 'grabbing';
        cssWrapper.style.filter = 'brightness(1.1)';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const deltaY = startY - e.clientY;
        rotation = Math.max(-135, Math.min(135, startRotation + deltaY));
        value = ((rotation + 135) / 270) * 100;
        updateDisplay();
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            interactiveLayer.style.cursor = 'grab';
            cssWrapper.style.filter = 'none';
        }
    });

    updateDisplay();

    // Layer toggles
    document.getElementById('knob-layer-webgl')?.addEventListener('change', (e) => {
        webglCanvas.style.display = e.target.checked ? 'block' : 'none';
    });

    document.getElementById('knob-layer-webgl2')?.addEventListener('change', (e) => {
        webgl2Canvas.style.display = e.target.checked ? 'block' : 'none';
    });

    document.getElementById('knob-layer-svg')?.addEventListener('change', (e) => {
        svg.style.display = e.target.checked ? 'block' : 'none';
    });

    document.getElementById('knob-layer-css')?.addEventListener('change', (e) => {
        cssWrapper.style.display = e.target.checked ? 'block' : 'none';
    });
}

/**
 * Initialize mixer console
 */
function initMixerConsole() {
    const container = document.getElementById('mixer-console');
    if (!container) return;

    const channels = ['Kick', 'Snare', 'HiHat', 'Bass', 'Synth', 'Pad', 'Lead', 'FX'];

    channels.forEach((label, i) => {
        const channel = document.createElement('div');
        channel.className = 'mixer-channel';

        const channelLabel = document.createElement('div');
        channelLabel.className = 'channel-label';
        channelLabel.textContent = label;

        // Create mini knob
        const knobContainer = document.createElement('div');
        knobContainer.style.cssText = 'width: 50px; height: 50px;';

        const valueEl = document.createElement('div');
        valueEl.className = 'channel-value';
        valueEl.textContent = '50';

        const meter = document.createElement('div');
        meter.className = 'channel-meter';

        const meterFill = document.createElement('div');
        meterFill.className = 'meter-fill';
        meterFill.style.height = '50%';
        meter.appendChild(meterFill);

        new UIComponents.RotaryKnob(knobContainer, {
            size: 50,
            min: 0,
            max: 100,
            value: 50 + Math.random() * 30 - 15,
            color: ['#00ff88', '#00aaff', '#ffaa00', '#ff44aa'][i % 4],
            label: '',
            onChange: (val) => {
                valueEl.textContent = Math.round(val);
                meterFill.style.height = `${val}%`;
            }
        });

        channel.appendChild(channelLabel);
        channel.appendChild(knobContainer);
        channel.appendChild(valueEl);
        channel.appendChild(meter);
        container.appendChild(channel);
    });
}

/**
 * Initialize meter knobs
 */
function initMeterKnobs() {
    const container = document.getElementById('meter-knobs');
    if (!container) return;

    const createMeterKnob = (label, color) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'meter-knob';

        // Create SVG meter arc
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100');
        svg.setAttribute('height', '100');
        svg.style.position = 'absolute';

        const arc = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        arc.setAttribute('fill', 'none');
        arc.setAttribute('stroke', '#333');
        arc.setAttribute('stroke-width', '8');
        arc.setAttribute('stroke-linecap', 'round');

        // Arc path for background
        const radius = 45;
        const startAngle = -135 * Math.PI / 180;
        const endAngle = 135 * Math.PI / 180;
        const x1 = 50 + radius * Math.cos(startAngle);
        const y1 = 50 + radius * Math.sin(startAngle);
        const x2 = 50 + radius * Math.cos(endAngle);
        const y2 = 50 + radius * Math.sin(endAngle);
        arc.setAttribute('d', `M ${x1} ${y1} A ${radius} ${radius} 0 1 1 ${x2} ${y2}`);
        svg.appendChild(arc);

        // Value arc
        const valueArc = arc.cloneNode();
        valueArc.setAttribute('stroke', color);
        valueArc.setAttribute('stroke-dasharray', '0 1000');
        svg.appendChild(valueArc);

        // Center knob
        const knobBody = document.createElement('div');
        knobBody.className = 'meter-knob-body';

        const indicator = document.createElement('div');
        indicator.style.cssText = `
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            width: 3px;
            height: 12px;
            background: ${color};
            border-radius: 2px;
        `;
        knobBody.appendChild(indicator);

        const labelEl = document.createElement('div');
        labelEl.className = 'knob-label';
        labelEl.textContent = label;
        labelEl.style.cssText = 'position: absolute; bottom: -25px; width: 100%; text-align: center;';

        wrapper.appendChild(svg);
        wrapper.appendChild(knobBody);
        wrapper.appendChild(labelEl);
        container.appendChild(wrapper);

        // Interaction
        let value = 50;
        let rotation = 0;
        let isDragging = false;
        let startY = 0;
        let startRotation = 0;

        const arcLength = radius * (270 * Math.PI / 180);

        const updateDisplay = () => {
            knobBody.style.transform = `rotate(${rotation}deg)`;
            const dashLength = (value / 100) * arcLength;
            valueArc.setAttribute('stroke-dasharray', `${dashLength} 1000`);
        };

        knobBody.addEventListener('mousedown', (e) => {
            isDragging = true;
            startY = e.clientY;
            startRotation = rotation;
            knobBody.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaY = startY - e.clientY;
            rotation = Math.max(-135, Math.min(135, startRotation + deltaY));
            value = ((rotation + 135) / 270) * 100;
            updateDisplay();
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            knobBody.style.cursor = 'grab';
        });

        updateDisplay();
    };

    createMeterKnob('Input', '#00ff88');
    createMeterKnob('Output', '#ff8800');
}

/**
 * Initialize dual concentric knobs
 */
function initDualKnobs() {
    const container = document.getElementById('dual-knobs');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'dual-knob';

    const outerRing = document.createElement('div');
    outerRing.className = 'outer-ring';

    const innerKnob = document.createElement('div');
    innerKnob.className = 'inner-knob';

    const labelEl = document.createElement('div');
    labelEl.className = 'knob-label';
    labelEl.textContent = 'Freq / Res';
    labelEl.style.marginTop = '90px';

    wrapper.appendChild(outerRing);
    wrapper.appendChild(innerKnob);
    container.appendChild(wrapper);
    container.appendChild(labelEl);

    // Outer ring interaction
    let outerRotation = 0;
    let outerDragging = false;
    let outerStartY = 0;
    let outerStartRotation = 0;

    outerRing.addEventListener('mousedown', (e) => {
        if (e.target === innerKnob) return;
        outerDragging = true;
        outerStartY = e.clientY;
        outerStartRotation = outerRotation;
        outerRing.style.cursor = 'grabbing';
        e.stopPropagation();
    });

    document.addEventListener('mousemove', (e) => {
        if (outerDragging) {
            const deltaY = outerStartY - e.clientY;
            outerRotation = Math.max(-135, Math.min(135, outerStartRotation + deltaY));
            outerRing.style.transform = `rotate(${outerRotation}deg)`;
        }
    });

    document.addEventListener('mouseup', () => {
        outerDragging = false;
        outerRing.style.cursor = 'grab';
    });

    // Inner knob interaction
    let innerRotation = 0;
    let innerDragging = false;
    let innerStartY = 0;
    let innerStartRotation = 0;

    innerKnob.addEventListener('mousedown', (e) => {
        innerDragging = true;
        innerStartY = e.clientY;
        innerStartRotation = innerRotation;
        innerKnob.style.cursor = 'grabbing';
        e.stopPropagation();
    });

    document.addEventListener('mousemove', (e) => {
        if (innerDragging) {
            const deltaY = innerStartY - e.clientY;
            innerRotation = Math.max(-135, Math.min(135, innerStartRotation + deltaY));
            innerKnob.style.transform = `rotate(${innerRotation}deg)`;
        }
    });

    document.addEventListener('mouseup', () => {
        innerDragging = false;
        innerKnob.style.cursor = 'grab';
    });
}

/**
 * Initialize illuminated knobs
 */
function initIlluminatedKnobs() {
    const container = document.getElementById('illuminated-knobs');
    if (!container) return;

    const colors = ['#00ff88', '#00aaff', '#ff44aa'];

    colors.forEach(color => {
        const wrapper = document.createElement('div');
        wrapper.className = 'illuminated-knob';

        const glowRing = document.createElement('div');
        glowRing.className = 'glow-ring';
        glowRing.style.background = `radial-gradient(circle, ${color}40, transparent 70%)`;

        const knobBody = document.createElement('div');
        knobBody.className = 'knob-body';

        const indicator = document.createElement('div');
        indicator.style.cssText = `
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            width: 3px;
            height: 15px;
            background: ${color};
            border-radius: 2px;
            box-shadow: 0 0 10px ${color};
        `;
        knobBody.appendChild(indicator);

        wrapper.appendChild(glowRing);
        wrapper.appendChild(knobBody);
        container.appendChild(wrapper);

        // Interaction
        let rotation = 0;
        let isDragging = false;
        let startY = 0;
        let startRotation = 0;

        knobBody.addEventListener('mousedown', (e) => {
            isDragging = true;
            startY = e.clientY;
            startRotation = rotation;
            knobBody.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaY = startY - e.clientY;
            rotation = Math.max(-135, Math.min(135, startRotation + deltaY));
            knobBody.style.transform = `rotate(${rotation}deg)`;
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            knobBody.style.cursor = 'grab';
        });
    });
}

/**
 * Initialize step/detent knobs
 */
function initStepKnobs() {
    const container = document.getElementById('step-knobs');
    if (!container) return;

    const configs = [
        { steps: 8, label: '8 Step' },
        { steps: 12, label: '12 Step' }
    ];

    configs.forEach(config => {
        const wrapper = document.createElement('div');
        wrapper.className = 'step-knob';

        // Create step markers
        const markers = document.createElement('div');
        markers.className = 'step-markers';

        const stepMarkers = [];
        for (let i = 0; i < config.steps; i++) {
            const marker = document.createElement('div');
            marker.className = 'step-marker';
            const angle = -135 + (i / (config.steps - 1)) * 270;
            marker.style.transform = `rotate(${angle}deg)`;
            stepMarkers.push(marker);
            markers.appendChild(marker);
        }

        const knobBody = document.createElement('div');
        knobBody.className = 'step-knob-body';

        const labelEl = document.createElement('div');
        labelEl.className = 'knob-label';
        labelEl.textContent = config.label;
        labelEl.style.cssText = 'position: absolute; bottom: -25px; width: 100%; text-align: center;';

        wrapper.appendChild(markers);
        wrapper.appendChild(knobBody);
        wrapper.appendChild(labelEl);
        container.appendChild(wrapper);

        // Interaction with snapping
        let currentStep = 0;
        let isDragging = false;
        let startY = 0;
        let startStep = 0;

        const updateDisplay = () => {
            const rotation = -135 + (currentStep / (config.steps - 1)) * 270;
            knobBody.style.transform = `rotate(${rotation}deg)`;

            stepMarkers.forEach((marker, i) => {
                if (i === currentStep) {
                    marker.classList.add('active');
                } else {
                    marker.classList.remove('active');
                }
            });
        };

        knobBody.addEventListener('mousedown', (e) => {
            isDragging = true;
            startY = e.clientY;
            startStep = currentStep;
            knobBody.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaY = Math.floor((startY - e.clientY) / 20);
            currentStep = Math.max(0, Math.min(config.steps - 1, startStep + deltaY));
            updateDisplay();
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            knobBody.style.cursor = 'grab';
        });

        updateDisplay();
    });
}

// Helper functions
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

// ============================================================================
// NEW EXPERIMENTS
// ============================================================================

/**
 * Ferrofluid Knob
 * Magnetic fluid simulation reacting to rotation
 */
function initFerrofluidKnob() {
    const container = document.getElementById('ferrofluid-knob');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 200px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 50%;
        cursor: grab;
    `;

    wrapper.appendChild(canvas);
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
    uniform float u_rotation;
    uniform float u_magnetStrength;
    out vec4 fragColor;

    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    void main() {
        vec2 uv = v_uv * 2.0 - 1.0;
        float dist = length(uv);
        float angle = atan(uv.y, uv.x);
        
        vec3 color = vec3(0.02, 0.02, 0.03);
        
        // Ferrofluid base in center
        float fluidRadius = 0.3 + u_magnetStrength * 0.1;
        
        // Create magnetic spikes
        float numSpikes = 12.0 + u_magnetStrength * 8.0;
        float spikeAmplitude = 0.15 * u_magnetStrength;
        
        // Spikes follow rotation
        float spikeAngle = angle - u_rotation * 0.02;
        float spikes = sin(spikeAngle * numSpikes) * spikeAmplitude;
        spikes *= smoothstep(0.5, 0.2, dist); // Spikes extend outward
        
        // Random variation
        float variation = hash(vec2(floor(spikeAngle * numSpikes), u_time * 0.1)) * 0.05;
        
        float fluidEdge = fluidRadius + spikes + variation;
        float fluid = smoothstep(fluidEdge + 0.02, fluidEdge - 0.02, dist);
        
        // Metallic sheen on fluid
        float sheen = 0.3 + 0.7 * pow(abs(sin(angle * 3.0 + u_rotation * 0.01)), 2.0);
        
        // Fluid color - dark metallic with highlights
        vec3 fluidColor = vec3(0.05, 0.05, 0.08);
        fluidColor += vec3(0.1, 0.15, 0.2) * sheen;
        fluidColor += vec3(0.2, 0.25, 0.3) * pow(sheen, 4.0); // Specular
        
        color = mix(color, fluidColor, fluid);
        
        // Outer ring (knob body)
        float ringInner = 0.7;
        float ringOuter = 0.9;
        float ring = smoothstep(ringInner - 0.02, ringInner, dist) * 
                     smoothstep(ringOuter + 0.02, ringOuter, dist);
        
        vec3 ringColor = vec3(0.15, 0.15, 0.18);
        ringColor += vec3(0.1) * sin(angle * 20.0 + u_rotation * 0.05); // Texture
        
        color = mix(color, ringColor, ring * 0.8);
        
        // Indicator notch
        float notchAngle = u_rotation * 0.01 - 1.57; // Top position
        float notchDist = abs(angle - notchAngle);
        notchDist = min(notchDist, 6.28 - notchDist);
        float notch = smoothstep(0.15, 0.05, notchDist) * 
                      smoothstep(ringInner - 0.1, ringInner + 0.05, dist) *
                      smoothstep(ringOuter + 0.05, ringOuter - 0.05, dist);
        
        color += vec3(0.0, 0.8, 0.6) * notch;
        
        // Magnetic field lines (subtle)
        float fieldLines = sin(dist * 30.0 - u_time * 2.0) * 0.5 + 0.5;
        fieldLines *= smoothstep(0.6, 0.3, dist) * u_magnetStrength * 0.1;
        color += vec3(0.1, 0.2, 0.3) * fieldLines;
        
        fragColor = vec4(color, 1.0);
    }
    `;

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uRotation = gl.getUniformLocation(program, 'u_rotation');
    const uMagnetStrength = gl.getUniformLocation(program, 'u_magnetStrength');

    let rotation = 0;
    let magnetStrength = 0.5;
    let isDragging = false;
    let startY = 0;
    let startRotation = 0;

    canvas.addEventListener('mousedown', e => {
        isDragging = true;
        startY = e.clientY;
        startRotation = rotation;
        canvas.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const deltaY = startY - e.clientY;
        rotation = startRotation + deltaY * 2;
        magnetStrength = Math.min(1, Math.max(0, 0.5 + Math.abs(deltaY) * 0.005));
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        canvas.style.cursor = 'grab';
    });

    function render(time) {
        // Decay magnet strength back to base
        magnetStrength = magnetStrength * 0.98 + 0.5 * 0.02;

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.02, 0.02, 0.03, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform1f(uRotation, rotation);
        gl.uniform1f(uMagnetStrength, magnetStrength);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

/**
 * Time Warp Knob
 * Temporal distortion visual effects
 */
function initTimeWarpKnob() {
    const container = document.getElementById('time-warp-knob');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 200px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 50%;
        cursor: grab;
    `;

    wrapper.appendChild(canvas);
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
    uniform float u_warp;
    out vec4 fragColor;

    void main() {
        vec2 uv = v_uv * 2.0 - 1.0;
        float dist = length(uv);
        float angle = atan(uv.y, uv.x);
        
        // Time warp distortion
        float warpFactor = u_warp * 0.5;
        float distortedDist = dist + sin(angle * 6.0 + u_time * warpFactor) * 0.05 * warpFactor;
        float distortedAngle = angle + sin(dist * 10.0 - u_time * warpFactor * 0.5) * 0.2 * warpFactor;
        
        vec2 distortedUV = vec2(
            cos(distortedAngle) * distortedDist,
            sin(distortedAngle) * distortedDist
        );
        
        vec3 color = vec3(0.02, 0.03, 0.05);
        
        // Time streams - concentric rings
        float timeSpeed = 1.0 + (u_warp - 0.5) * 4.0;
        float rings = sin(distortedDist * 20.0 - u_time * timeSpeed) * 0.5 + 0.5;
        rings *= smoothstep(0.9, 0.3, dist);
        
        // Color based on time direction
        vec3 forwardColor = vec3(0.2, 0.5, 1.0);  // Blue for forward
        vec3 backwardColor = vec3(1.0, 0.3, 0.2); // Red for backward
        vec3 timeColor = mix(backwardColor, forwardColor, u_warp);
        
        color += timeColor * rings * 0.5;
        
        // Spiral arms
        float spirals = sin(distortedAngle * 3.0 + distortedDist * 5.0 - u_time * timeSpeed * 0.5);
        spirals = smoothstep(0.5, 1.0, spirals) * smoothstep(0.8, 0.2, dist);
        color += timeColor * spirals * 0.3;
        
        // Central clock face
        float clockRadius = 0.3;
        if (dist < clockRadius) {
            // Clock marks
            float marks = smoothstep(0.02, 0.0, abs(sin(angle * 6.0)));
            marks *= smoothstep(clockRadius - 0.05, clockRadius - 0.1, dist);
            color += vec3(0.5) * marks;
            
            // Hour hand
            float hourAngle = u_time * 0.1 * timeSpeed;
            float hourHand = smoothstep(0.02, 0.0, abs(angle - hourAngle));
            hourHand *= smoothstep(0.0, 0.2, dist) * smoothstep(clockRadius - 0.05, 0.1, dist);
            color += vec3(0.8) * hourHand;
            
            // Minute hand
            float minAngle = u_time * timeSpeed;
            float minHand = smoothstep(0.015, 0.0, abs(angle - minAngle));
            minHand *= smoothstep(0.0, 0.25, dist) * smoothstep(clockRadius - 0.02, 0.08, dist);
            color += timeColor * minHand;
        }
        
        // Outer ring
        float ring = smoothstep(0.03, 0.0, abs(dist - 0.85));
        color += timeColor * ring * 0.5;
        
        // Warp glow at center
        float glow = 0.02 / (dist * dist + 0.01) * abs(u_warp - 0.5);
        color += timeColor * glow * 0.1;
        
        fragColor = vec4(color, 1.0);
    }
    `;

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uWarp = gl.getUniformLocation(program, 'u_warp');

    let warp = 0.5; // 0=reverse, 0.5=normal, 1=fast forward
    let isDragging = false;
    let startY = 0;
    let startWarp = 0.5;

    canvas.addEventListener('mousedown', e => {
        isDragging = true;
        startY = e.clientY;
        startWarp = warp;
        canvas.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const deltaY = startY - e.clientY;
        warp = Math.min(1, Math.max(0, startWarp + deltaY * 0.005));
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        canvas.style.cursor = 'grab';
    });

    function render(time) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.02, 0.03, 0.05, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform1f(uWarp, warp);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

/**
 * Audio Waveform Ring Knob
 * Circular oscilloscope around the knob
 */
function initWaveformRingKnob() {
    const container = document.getElementById('waveform-ring-knob');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 200px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 50%;
        cursor: grab;
    `;

    wrapper.appendChild(canvas);
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
    uniform float u_frequency;
    out vec4 fragColor;

    void main() {
        vec2 uv = v_uv * 2.0 - 1.0;
        float dist = length(uv);
        float angle = atan(uv.y, uv.x);
        
        vec3 color = vec3(0.02, 0.03, 0.04);
        
        // Waveform ring parameters
        float ringRadius = 0.7;
        float ringWidth = 0.15;
        
        // Generate waveform based on frequency
        float freq = 1.0 + u_frequency * 10.0;
        float wave = sin(angle * freq + u_time * 3.0);
        wave += sin(angle * freq * 2.0 - u_time * 2.0) * 0.5;
        wave += sin(angle * freq * 0.5 + u_time * 4.0) * 0.3;
        wave *= 0.5;
        
        // Map wave to ring displacement
        float waveRadius = ringRadius + wave * 0.1;
        
        // Draw waveform line
        float lineDist = abs(dist - waveRadius);
        float line = smoothstep(0.02, 0.005, lineDist);
        
        // Gradient color for waveform
        vec3 waveColor = mix(
            vec3(0.0, 0.8, 0.4),
            vec3(0.0, 0.4, 1.0),
            (wave + 1.0) * 0.5
        );
        
        color += waveColor * line;
        
        // Glow around waveform
        float glow = smoothstep(0.1, 0.0, lineDist) * 0.5;
        color += waveColor * glow * 0.3;
        
        // Center knob
        float knobRadius = 0.4;
        if (dist < knobRadius) {
            // Metallic knob
            float knobShade = 0.3 + 0.2 * sin(angle * 8.0 + u_frequency * 10.0);
            vec3 knobColor = vec3(0.15, 0.15, 0.18) * (0.8 + knobShade);
            
            // Frequency display
            float freqDisplay = sin(dist * 20.0 - u_time * freq) * 0.5 + 0.5;
            freqDisplay *= smoothstep(0.35, 0.1, dist);
            knobColor += waveColor * freqDisplay * 0.2;
            
            color = mix(color, knobColor, smoothstep(knobRadius, knobRadius - 0.02, dist));
            
            // Indicator
            float indicatorAngle = u_frequency * 3.14159 * 1.5 - 2.35;
            float indicator = smoothstep(0.1, 0.0, abs(angle - indicatorAngle));
            indicator *= smoothstep(0.15, 0.25, dist) * smoothstep(0.38, 0.3, dist);
            color += waveColor * indicator * 2.0;
        }
        
        // Outer ring
        float outerRing = smoothstep(0.02, 0.0, abs(dist - 0.9));
        color += vec3(0.1, 0.15, 0.2) * outerRing;
        
        // Background grid
        float grid = smoothstep(0.02, 0.0, abs(sin(angle * 12.0)));
        grid *= smoothstep(knobRadius + 0.05, knobRadius + 0.15, dist);
        grid *= smoothstep(0.88, 0.8, dist);
        color += vec3(0.05) * grid;
        
        fragColor = vec4(color, 1.0);
    }
    `;

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uFrequency = gl.getUniformLocation(program, 'u_frequency');

    let frequency = 0.5;
    let isDragging = false;
    let startY = 0;
    let startFreq = 0.5;

    canvas.addEventListener('mousedown', e => {
        isDragging = true;
        startY = e.clientY;
        startFreq = frequency;
        canvas.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const deltaY = startY - e.clientY;
        frequency = Math.min(1, Math.max(0, startFreq + deltaY * 0.005));
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        canvas.style.cursor = 'grab';
    });

    function render(time) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.02, 0.03, 0.04, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform1f(uFrequency, frequency);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}
