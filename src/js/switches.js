document.addEventListener('DOMContentLoaded', () => {
    initBasicSwitches();
    initLEDSwitches();
    initRockerSwitches();
    initSlideSwitches();
    initControlPanel();
    initNeonSwitches();
    initFlipSwitches();
    initRetroSwitches();
    initSegmentedSwitches();
    initPlasmaSwitches();
    initIrisSwitches();

    // New Experiments
    initPortalToggle();
    initCircuitBreaker();
    initBioMechanicalSwitch();
});

function initBasicSwitches() {
    const container = document.getElementById('basic-switches');
    if (!container) return;

    const colors = ['#00ff88', '#00aaff', '#ff44aa'];

    colors.forEach((color, i) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'switch-wrapper';
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'toggle-switch';
        if (i === 0) switchEl.classList.add('active');

        const track = document.createElement('div');
        track.className = 'toggle-track';
        track.style.setProperty('--switch-color', color);

        const thumb = document.createElement('div');
        thumb.className = 'toggle-thumb';

        // Create WebGL glow canvas
        const canvas = document.createElement('canvas');
        canvas.width = 120;
        canvas.height = 60;
        canvas.style.cssText = `
            position: absolute;
            top: -15px;
            left: -30px;
            width: 120px;
            height: 60px;
            pointer-events: none;
        `;

        track.appendChild(thumb);
        switchEl.appendChild(canvas);
        switchEl.appendChild(track);

        const label = document.createElement('div');
        label.className = 'switch-label';
        label.textContent = `Switch ${i + 1}`;
        label.style.cssText = 'font-size: 0.75rem; color: var(--text-secondary);';

        wrapper.appendChild(switchEl);
        wrapper.appendChild(label);
        container.appendChild(wrapper);

        // Initialize WebGL glow
        const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
        if (gl) {
            initSwitchGlow(gl, canvas, color, switchEl);
        }

        // Add click handler
        switchEl.addEventListener('click', () => {
            switchEl.classList.toggle('active');
        });
    });
}

function initSwitchGlow(gl, canvas, color, switchEl) {
    const fragmentShader = `
        precision mediump float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec3 u_color;
        uniform float u_on;
        uniform float u_thumbPos;
        
        void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec2 thumbCenter = vec2(u_thumbPos, 0.5);
            float dist = distance(uv, thumbCenter);
            
            float glow = 0.05 / (dist * dist + 0.01);
            glow *= u_on;
            glow *= 0.8 + 0.2 * sin(u_time * 3.0);
            
            gl_FragColor = vec4(u_color * glow, glow * 0.5);
        }
    `;

    const program = UIComponents.ShaderUtils.createProgram(
        gl,
        UIComponents.ShaderUtils.vertexShader2D,
        fragmentShader
    );

    if (!program) return;

    setupQuadWebGL1(gl, program);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const uniforms = {
        time: gl.getUniformLocation(program, 'u_time'),
        resolution: gl.getUniformLocation(program, 'u_resolution'),
        color: gl.getUniformLocation(program, 'u_color'),
        on: gl.getUniformLocation(program, 'u_on'),
        thumbPos: gl.getUniformLocation(program, 'u_thumbPos')
    };

    const colorArray = hexToRgb(color);

    const animate = (timestamp) => {
        const time = timestamp * 0.001;
        const isOn = switchEl.classList.contains('active');

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        gl.uniform1f(uniforms.time, time);
        gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
        gl.uniform3fv(uniforms.color, colorArray);
        gl.uniform1f(uniforms.on, isOn ? 1.0 : 0.0);
        gl.uniform1f(uniforms.thumbPos, isOn ? 0.75 : 0.25);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
}

function initLEDSwitches() {
    const container = document.getElementById('led-switches');
    if (!container) return;

    const labels = ['Power', 'Active', 'Ready', 'Sync'];
    const colors = ['#ff4444', '#00ff88', '#ffaa00', '#00aaff'];

    labels.forEach((label, i) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'led-toggle-switch';
        if (i % 2 === 0) wrapper.classList.add('active');

        const led = document.createElement('div');
        led.className = 'led-indicator';
        led.style.setProperty('--led-color', colors[i]);

        const switchEl = document.createElement('div');
        switchEl.className = 'toggle-switch';
        if (i % 2 === 0) switchEl.classList.add('active');

        const track = document.createElement('div');
        track.className = 'toggle-track';

        const thumb = document.createElement('div');
        thumb.className = 'toggle-thumb';
        thumb.style.background = colors[i];

        track.appendChild(thumb);
        switchEl.appendChild(track);

        const labelEl = document.createElement('div');
        labelEl.textContent = label;
        labelEl.style.cssText = 'font-size: 0.8rem; color: var(--text-secondary);';

        wrapper.appendChild(led);
        wrapper.appendChild(switchEl);
        wrapper.appendChild(labelEl);
        container.appendChild(wrapper);

        wrapper.addEventListener('click', () => {
            wrapper.classList.toggle('active');
            switchEl.classList.toggle('active');
        });
    });
}

function initRockerSwitches() {
    const container = document.getElementById('rocker-switches');
    if (!container) return;

    for (let i = 0; i < 4; i++) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'rocker-switch';
        if (i === 0) switchEl.classList.add('active');

        const toggle = document.createElement('div');
        toggle.className = 'rocker-toggle';

        switchEl.appendChild(toggle);

        const label = document.createElement('div');
        label.textContent = ['I/O', 'ON', 'OFF', 'AUTO'][i];
        label.style.cssText = 'font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase;';

        wrapper.appendChild(switchEl);
        wrapper.appendChild(label);
        container.appendChild(wrapper);

        switchEl.addEventListener('click', () => {
            switchEl.classList.toggle('active');
        });
    }
}

function initSlideSwitches() {
    const container = document.getElementById('slide-switches');
    if (!container) return;

    for (let i = 0; i < 3; i++) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'slide-switch';
        switchEl.dataset.position = '0';

        const track = document.createElement('div');
        track.className = 'slide-track';

        const positions = document.createElement('div');
        positions.className = 'slide-positions';
        for (let j = 0; j < 3; j++) {
            const pos = document.createElement('div');
            pos.className = 'slide-position';
            positions.appendChild(pos);
        }

        const thumb = document.createElement('div');
        thumb.className = 'slide-thumb';

        switchEl.appendChild(track);
        switchEl.appendChild(positions);
        switchEl.appendChild(thumb);

        const label = document.createElement('div');
        label.textContent = ['L-C-R', 'OFF-LO-HI', '1-2-3'][i];
        label.style.cssText = 'font-size: 0.7rem; color: var(--text-secondary);';

        wrapper.appendChild(switchEl);
        wrapper.appendChild(label);
        container.appendChild(wrapper);

        switchEl.addEventListener('click', () => {
            let pos = parseInt(switchEl.dataset.position);
            pos = (pos + 1) % 3;
            switchEl.dataset.position = pos;
        });
    }
}

function initControlPanel() {
    const container = document.getElementById('control-panel');
    if (!container) return;

    const controls = [
        { type: 'toggle', label: 'Power' },
        { type: 'toggle', label: 'Mute' },
        { type: 'rocker', label: 'Bypass' },
        { type: 'toggle', label: 'Lock' },
        { type: 'led-toggle', label: 'Record' },
        { type: 'toggle', label: 'Monitor' },
        { type: 'rocker', label: 'Solo' },
        { type: 'toggle', label: 'FX' }
    ];

    controls.forEach(ctrl => {
        const item = document.createElement('div');
        item.className = 'panel-item';

        const label = document.createElement('div');
        label.className = 'panel-label';
        label.textContent = ctrl.label;

        let switchEl;

        if (ctrl.type === 'toggle') {
            switchEl = document.createElement('div');
            switchEl.className = 'toggle-switch';

            const track = document.createElement('div');
            track.className = 'toggle-track';

            const thumb = document.createElement('div');
            thumb.className = 'toggle-thumb';

            track.appendChild(thumb);
            switchEl.appendChild(track);

            switchEl.addEventListener('click', () => {
                switchEl.classList.toggle('active');
            });
        } else if (ctrl.type === 'rocker') {
            switchEl = document.createElement('div');
            switchEl.className = 'rocker-switch';
            switchEl.style.transform = 'scale(0.8)';

            const toggle = document.createElement('div');
            toggle.className = 'rocker-toggle';

            switchEl.appendChild(toggle);

            switchEl.addEventListener('click', () => {
                switchEl.classList.toggle('active');
            });
        } else if (ctrl.type === 'led-toggle') {
            switchEl = document.createElement('div');
            switchEl.className = 'led-toggle-switch';
            switchEl.style.flexDirection = 'column';

            const led = document.createElement('div');
            led.className = 'led-indicator';
            led.style.setProperty('--led-color', '#ff4444');

            const toggle = document.createElement('div');
            toggle.className = 'toggle-switch';

            const track = document.createElement('div');
            track.className = 'toggle-track';

            const thumb = document.createElement('div');
            thumb.className = 'toggle-thumb';
            thumb.style.background = '#ff4444';

            track.appendChild(thumb);
            toggle.appendChild(track);
            switchEl.appendChild(led);
            switchEl.appendChild(toggle);

            switchEl.addEventListener('click', () => {
                switchEl.classList.toggle('active');
                toggle.classList.toggle('active');
            });
        }

        item.appendChild(label);
        item.appendChild(switchEl);
        container.appendChild(item);
    });
}

function initNeonSwitches() {
    const container = document.getElementById('neon-switches');
    if (!container) return;

    const colors = ['#00ff88', '#ff44aa', '#00aaff'];

    colors.forEach((color, i) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'neon-switch';
        if (i === 0) switchEl.classList.add('active');

        const tube = document.createElement('div');
        tube.className = 'neon-tube';

        const gas = document.createElement('div');
        gas.className = 'neon-gas';
        gas.style.setProperty('--neon-color', color);

        tube.appendChild(gas);
        switchEl.appendChild(tube);

        wrapper.appendChild(switchEl);
        container.appendChild(wrapper);

        switchEl.addEventListener('click', () => {
            switchEl.classList.toggle('active');
            if (switchEl.classList.contains('active')) {
                gas.style.background = color;
                gas.style.boxShadow = `0 0 20px ${color}, inset 0 0 10px rgba(255, 255, 255, 0.3)`;
            } else {
                gas.style.background = '#222';
                gas.style.boxShadow = 'none';
            }
        });
    });
}

function initFlipSwitches() {
    const container = document.getElementById('flip-switches');
    if (!container) return;

    for (let i = 0; i < 3; i++) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'flip-switch';

        const guard = document.createElement('div');
        guard.className = 'flip-guard';

        const base = document.createElement('div');
        base.className = 'flip-base';

        const lever = document.createElement('div');
        lever.className = 'flip-lever';

        switchEl.appendChild(guard);
        switchEl.appendChild(base);
        switchEl.appendChild(lever);

        const label = document.createElement('div');
        label.textContent = ['ARM', 'FIRE', 'EJECT'][i];
        label.style.cssText = 'font-size: 0.7rem; color: var(--accent-danger); font-weight: bold;';

        wrapper.appendChild(switchEl);
        wrapper.appendChild(label);
        container.appendChild(wrapper);

        guard.addEventListener('click', () => {
            switchEl.classList.toggle('active');
        });
    }
}

function initRetroSwitches() {
    const container = document.getElementById('retro-switches');
    if (!container) return;

    for (let i = 0; i < 3; i++) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'retro-switch';
        if (i === 1) switchEl.classList.add('active');

        const plate = document.createElement('div');
        plate.className = 'retro-plate';

        const toggle = document.createElement('div');
        toggle.className = 'retro-toggle';

        plate.appendChild(toggle);
        switchEl.appendChild(plate);

        wrapper.appendChild(switchEl);
        container.appendChild(wrapper);

        switchEl.addEventListener('click', () => {
            switchEl.classList.toggle('active');
        });
    }
}

function initSegmentedSwitches() {
    const container = document.getElementById('segmented-switches');
    if (!container) return;

    const options = [
        ['A', 'B', 'C'],
        ['Low', 'Mid', 'High'],
        ['1x', '2x', '4x']
    ];

    options.forEach((opts, i) => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'segmented-switch';

        opts.forEach((opt, j) => {
            const segment = document.createElement('div');
            segment.className = 'segment';
            if (j === 0) segment.classList.add('active');
            segment.textContent = opt;

            segment.addEventListener('click', () => {
                switchEl.querySelectorAll('.segment').forEach(s => s.classList.remove('active'));
                segment.classList.add('active');
            });

            switchEl.appendChild(segment);
        });

        wrapper.appendChild(switchEl);
        container.appendChild(wrapper);
    });
}

function initIrisSwitches() {
    const container = document.getElementById('iris-switches');
    if (!container) return;

    const labels = ['ACCESS', 'PORT', 'LENS'];

    for (let i = 0; i < 3; i++) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const switchEl = document.createElement('div');
        switchEl.className = 'iris-switch';

        // Iris structure
        // Center glow
        const glow = document.createElement('div');
        glow.className = 'iris-center-glow';

        switchEl.appendChild(glow);

        // Blades SVG
        const svgNS = "http://www.w3.org/2000/svg";
        const s = document.createElementNS(svgNS, "svg");
        s.setAttribute("viewBox", "0 0 100 100");
        s.setAttribute("width", "100%");
        s.setAttribute("height", "100%");
        s.style.zIndex = "2";
        s.style.pointerEvents = "none";

        const g = document.createElementNS(svgNS, "g");

        // Clip path to keep blades inside circle
        const defs = document.createElementNS(svgNS, "defs");
        const clip = document.createElementNS(svgNS, "clipPath");
        clip.id = "iris-clip-" + i;
        const circle = document.createElementNS(svgNS, "circle");
        circle.setAttribute("cx", "50");
        circle.setAttribute("cy", "50");
        circle.setAttribute("r", "28"); // Aperture radius
        clip.appendChild(circle);
        defs.appendChild(clip);

        for (let k = 0; k < 6; k++) {
            const blade = document.createElementNS(svgNS, "path");
            // Blade shape: A curved element.
            blade.setAttribute("d", "M50,50 L100,50 A50,50 0 0 1 75,93 Z");
            blade.setAttribute("fill", "#222");
            blade.setAttribute("stroke", "#111");
            blade.setAttribute("stroke-width", "0.5");
            blade.style.transformOrigin = "50px 50px";
            blade.style.transform = `rotate(${k * 60}deg)`;
            blade.style.transition = "transform 0.4s ease-in-out";
            blade.classList.add('blade-path');
            g.appendChild(blade);
        }
        s.appendChild(g);
        switchEl.appendChild(s);

        const label = document.createElement('div');
        label.textContent = labels[i];
        label.style.cssText = 'font-size: 0.7rem; color: var(--text-secondary);';

        wrapper.appendChild(switchEl);
        wrapper.appendChild(label);
        container.appendChild(wrapper);

        switchEl.addEventListener('click', () => {
            switchEl.classList.toggle('active');
            const isActive = switchEl.classList.contains('active');
            const blades = switchEl.querySelectorAll('.blade-path');
            blades.forEach((b, idx) => {
                const baseRot = idx * 60;
                const rot = isActive ? baseRot + 35 : baseRot;
                b.style.transform = `rotate(${rot}deg)`;
            });
            glow.style.transform = isActive ? "translate(-50%, -50%) scale(1.5)" : "translate(-50%, -50%) scale(0)";
        });
    }
}

function initPlasmaSwitches() {
    const container = document.getElementById('plasma-switches');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'position: relative; width: 100px; height: 160px; display: flex; flex-direction: column; align-items: center; gap: 10px;';

    const canvas = document.createElement('canvas');
    canvas.width = 100;
    canvas.height = 120;
    canvas.style.cssText = 'width: 100px; height: 120px; border-radius: 4px; border: 1px solid #333; background: #111; cursor: pointer;';

    wrapper.appendChild(canvas);

    const label = document.createElement('div');
    label.textContent = "HV LINK";
    label.style.cssText = 'font-weight: bold; color: #cc66ff; font-family: monospace; letter-spacing: 2px;';
    wrapper.appendChild(label);

    container.appendChild(wrapper);

    const gl = canvas.getContext('webgl', { alpha: false });
    if (!gl) return;

    const vs = `
        attribute vec2 a_position;
        varying vec2 v_uv;
        void main() {
            v_uv = a_position * 0.5 + 0.5;
            gl_Position = vec4(a_position, 0.0, 1.0);
        }
    `;

    const fs = `
        precision mediump float;
        uniform float u_time;
        uniform float u_on;
        varying vec2 v_uv;

        float noise(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }

        void main() {
            vec2 uv = v_uv;
            float electrodeTop = step(0.95, uv.y);
            float electrodeBottom = step(uv.y, 0.05);
            vec3 color = vec3(0.1);
            if(electrodeTop > 0.0 || electrodeBottom > 0.0) {
                 color = vec3(0.4);
            }

            if (u_on > 0.0) {
                float t = u_time * 10.0;
                float x = 0.5;
                x += (noise(vec2(uv.y * 5.0, t)) - 0.5) * 0.2 * u_on;
                float width = 0.02 * u_on + 0.01 * sin(t);
                float arc = 1.0 - smoothstep(width, width + 0.05, abs(uv.x - x));
                vec3 arcColor = vec3(0.6, 0.2, 1.0);
                arcColor += vec3(0.8, 0.8, 1.0) * smoothstep(0.01, 0.0, abs(uv.x - x));
                float flick = 0.8 + 0.2 * noise(vec2(t, 0.0));
                color += arcColor * arc * flick;
                float d = distance(uv.x, x);
                color += vec3(0.4, 0.1, 0.8) * (0.1 / (d + 0.01));
            }
            gl_FragColor = vec4(color, 1.0);
        }
    `;

    const program = UIComponents.ShaderUtils.createProgram(gl, vs, fs);
    setupQuadWebGL1(gl, program);

    const u = {
        time: gl.getUniformLocation(program, 'u_time'),
        on: gl.getUniformLocation(program, 'u_on')
    };

    let isOn = false;
    let transition = 0.0;

    canvas.addEventListener('click', () => {
        isOn = !isOn;
    });

    const animate = (time) => {
        const t = time * 0.001;
        const target = isOn ? 1.0 : 0.0;
        transition += (target - transition) * 0.1;

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(program);
        gl.uniform1f(u.time, t);
        gl.uniform1f(u.on, transition);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        if (transition > 0.01 || isOn) {
            label.style.textShadow = `0 0 ${transition * 10}px #cc66ff`;
        } else {
            label.style.textShadow = 'none';
        }
        requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (result) {
        return [
            parseInt(result[1], 16) / 255,
            parseInt(result[2], 16) / 255,
            parseInt(result[3], 16) / 255
        ];
    }
    return [0, 1, 0.5];
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
 * Portal Toggle Switch
 * Inter-dimensional gateway with vortex effect
 */
function initPortalToggle() {
    const container = document.getElementById('portal-toggle');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 150px;
        height: 150px;
        position: relative;
        margin: 1rem auto;
        cursor: pointer;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 50%;
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
    uniform float u_open;
    out vec4 fragColor;

    void main() {
        vec2 uv = v_uv * 2.0 - 1.0;
        float dist = length(uv);
        float angle = atan(uv.y, uv.x);
        
        vec3 color = vec3(0.02, 0.02, 0.05);
        
        // Portal ring
        float ringRadius = 0.7 + 0.05 * sin(u_time * 2.0);
        float ring = smoothstep(0.08, 0.0, abs(dist - ringRadius));
        
        vec3 ringColor = mix(
            vec3(0.5, 0.0, 0.8),  // Purple
            vec3(0.0, 0.8, 1.0),   // Cyan
            sin(angle * 3.0 + u_time * 2.0) * 0.5 + 0.5
        );
        color += ringColor * ring * 2.0;
        
        // Vortex spiral (when open)
        if (u_open > 0.1) {
            float spiralDist = dist / u_open;
            float spiral = sin(angle * 5.0 - spiralDist * 10.0 + u_time * 5.0);
            spiral *= smoothstep(ringRadius - 0.05, 0.0, dist);
            spiral = max(0.0, spiral);
            
            vec3 spiralColor = mix(
                vec3(0.2, 0.0, 0.5),
                vec3(0.0, 0.5, 1.0),
                spiralDist
            );
            color += spiralColor * spiral * u_open;
            
            // Core glow
            float core = 0.1 / (dist + 0.1) * u_open;
            color += vec3(0.5, 0.3, 1.0) * core * 0.5;
            
            // Particles
            for (float i = 0.0; i < 10.0; i++) {
                float pAngle = i * 0.628 + u_time * (1.0 + i * 0.1);
                float pDist = mod(u_time * 0.3 + i * 0.1, 1.0) * ringRadius;
                vec2 pPos = vec2(cos(pAngle), sin(pAngle)) * pDist;
                float particle = smoothstep(0.03, 0.0, length(uv - pPos));
                color += ringColor * particle * u_open;
            }
        }
        
        // Outer energy field
        float field = sin(dist * 20.0 - u_time * 3.0) * 0.5 + 0.5;
        field *= smoothstep(0.9, 0.7, dist) * smoothstep(ringRadius - 0.1, ringRadius + 0.1, dist);
        color += ringColor * field * 0.2 * u_open;
        
        fragColor = vec4(color, 1.0);
    }
    `;

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uOpen = gl.getUniformLocation(program, 'u_open');

    let isOpen = false;
    let openAmount = 0;

    wrapper.addEventListener('click', () => {
        isOpen = !isOpen;
    });

    function render(time) {
        // Animate open/close
        const target = isOpen ? 1.0 : 0.0;
        openAmount += (target - openAmount) * 0.05;

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.02, 0.02, 0.05, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform1f(uOpen, openAmount);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

/**
 * Circuit Breaker Animation
 * Industrial breaker with electrical arc
 */
function initCircuitBreaker() {
    const container = document.getElementById('circuit-breaker');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 120px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
        cursor: pointer;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 240;
    canvas.height = 400;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 8px;
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
    uniform float u_state; // 0=off, 1=on, 0.5=tripping
    uniform float u_arc;
    out vec4 fragColor;

    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    void main() {
        vec2 uv = v_uv;
        
        vec3 color = vec3(0.08, 0.08, 0.1);
        
        // Breaker body
        float body = step(0.15, uv.x) * step(uv.x, 0.85);
        body *= step(0.1, uv.y) * step(uv.y, 0.9);
        
        vec3 bodyColor = vec3(0.15, 0.15, 0.18);
        color = mix(color, bodyColor, body);
        
        // Handle position based on state
        float handleY = 0.4 + u_state * 0.3;
        float handleWidth = 0.4;
        float handleHeight = 0.15;
        
        float handle = step(0.5 - handleWidth/2.0, uv.x) * step(uv.x, 0.5 + handleWidth/2.0);
        handle *= step(handleY - handleHeight/2.0, uv.y) * step(uv.y, handleY + handleHeight/2.0);
        
        vec3 handleColor = u_state > 0.5 ? vec3(0.1, 0.5, 0.2) : vec3(0.5, 0.1, 0.1);
        handleColor = mix(handleColor, vec3(0.8, 0.5, 0.0), step(0.3, u_arc) * step(u_arc, 0.7));
        
        color = mix(color, handleColor, handle);
        
        // ON/OFF labels
        float onLabel = step(0.7, uv.y) * step(uv.y, 0.75) * step(0.35, uv.x) * step(uv.x, 0.65);
        float offLabel = step(0.25, uv.y) * step(uv.y, 0.3) * step(0.35, uv.x) * step(uv.x, 0.65);
        
        color += vec3(0.0, 0.8, 0.3) * onLabel * u_state;
        color += vec3(0.8, 0.2, 0.1) * offLabel * (1.0 - u_state);
        
        // Electrical arc during transition
        if (u_arc > 0.0) {
            float arcY = 0.5;
            float arcX = 0.5 + (hash(vec2(u_time * 100.0, 0.0)) - 0.5) * 0.2;
            
            // Multiple arc branches
            for (float i = 0.0; i < 5.0; i++) {
                float branchOffset = hash(vec2(i, u_time * 50.0)) * 0.1;
                float branchX = arcX + branchOffset;
                
                float arcDist = abs(uv.x - branchX);
                float arcLine = smoothstep(0.02, 0.0, arcDist);
                arcLine *= smoothstep(0.75, 0.5, uv.y) * smoothstep(0.25, 0.5, uv.y);
                arcLine *= hash(vec2(uv.y * 100.0, u_time * 100.0 + i));
                
                vec3 arcColor = vec3(0.5, 0.7, 1.0) + vec3(0.5, 0.3, 0.0) * hash(vec2(i, u_time));
                color += arcColor * arcLine * u_arc * 3.0;
            }
            
            // Arc glow
            float glowDist = length(uv - vec2(0.5, 0.5));
            color += vec3(0.3, 0.5, 1.0) * (0.1 / (glowDist + 0.1)) * u_arc * 0.3;
        }
        
        // Status LED
        float ledX = 0.5;
        float ledY = 0.85;
        float led = smoothstep(0.03, 0.02, length(uv - vec2(ledX, ledY)));
        vec3 ledColor = u_state > 0.5 ? vec3(0.0, 1.0, 0.3) : vec3(1.0, 0.2, 0.1);
        color += ledColor * led;
        
        // LED glow
        float ledGlow = 0.01 / (length(uv - vec2(ledX, ledY)) + 0.01);
        color += ledColor * ledGlow * 0.1;
        
        fragColor = vec4(color, 1.0);
    }
    `;

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uState = gl.getUniformLocation(program, 'u_state');
    const uArc = gl.getUniformLocation(program, 'u_arc');

    let state = 0;
    let targetState = 0;
    let arcAmount = 0;

    wrapper.addEventListener('click', () => {
        targetState = targetState > 0.5 ? 0 : 1;
        arcAmount = 1.0; // Trigger arc
    });

    function render(time) {
        // Animate state transition
        state += (targetState - state) * 0.08;

        // Decay arc
        arcAmount *= 0.92;

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.05, 0.05, 0.08, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform1f(uState, state);
        gl.uniform1f(uArc, arcAmount);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

/**
 * Bio-Mechanical Switch
 * Organic/mechanical fusion interface
 */
function initBioMechanicalSwitch() {
    const container = document.getElementById('bio-mechanical-switch');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 180px;
        height: 120px;
        position: relative;
        margin: 1rem auto;
        cursor: pointer;
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 360;
    canvas.height = 240;
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 12px;
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
    uniform float u_active;
    out vec4 fragColor;

    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    void main() {
        vec2 uv = v_uv;
        
        vec3 color = vec3(0.03, 0.02, 0.04);
        
        // Organic base texture
        float organic = 0.0;
        for (float i = 1.0; i <= 4.0; i++) {
            float scale = pow(2.0, i);
            organic += sin(uv.x * scale * 10.0 + u_time) * sin(uv.y * scale * 8.0 + u_time * 0.7) / i;
        }
        organic = organic * 0.5 + 0.5;
        
        vec3 fleshColor = mix(
            vec3(0.15, 0.05, 0.08),
            vec3(0.25, 0.08, 0.12),
            organic
        );
        color = fleshColor;
        
        // Veins
        for (float i = 0.0; i < 5.0; i++) {
            float veinX = 0.2 + i * 0.15 + sin(uv.y * 10.0 + i) * 0.05;
            float veinWidth = 0.01 + 0.005 * sin(uv.y * 20.0 + u_time * 2.0 + i);
            float vein = smoothstep(veinWidth, 0.0, abs(uv.x - veinX));
            vein *= smoothstep(0.1, 0.3, uv.y) * smoothstep(0.9, 0.7, uv.y);
            
            // Pulse
            float pulse = 0.7 + 0.3 * sin(u_time * 3.0 - uv.y * 5.0 + i);
            
            vec3 veinColor = vec3(0.4, 0.1, 0.15) * pulse;
            veinColor = mix(veinColor, vec3(0.1, 0.5, 0.3), u_active);
            
            color = mix(color, veinColor, vein);
        }
        
        // Central node
        vec2 nodePos = vec2(0.5, 0.5);
        float nodeDist = length(uv - nodePos);
        float node = smoothstep(0.15, 0.1, nodeDist);
        
        // Node color changes with activation
        vec3 nodeColor = mix(
            vec3(0.3, 0.1, 0.15),
            vec3(0.1, 0.8, 0.4),
            u_active
        );
        
        // Bioluminescent glow
        float glow = 0.05 / (nodeDist + 0.05) * u_active;
        color += nodeColor * glow;
        
        color = mix(color, nodeColor, node);
        
        // Pulse rings emanating from node
        if (u_active > 0.1) {
            for (float i = 0.0; i < 3.0; i++) {
                float ringTime = mod(u_time + i * 0.5, 2.0);
                float ringRadius = ringTime * 0.4;
                float ring = smoothstep(0.02, 0.0, abs(nodeDist - ringRadius));
                ring *= smoothstep(2.0, 0.5, ringTime);
                color += nodeColor * ring * 0.5;
            }
        }
        
        // Mechanical elements
        float mechX = step(0.1, uv.x) * step(uv.x, 0.15);
        mechX += step(0.85, uv.x) * step(uv.x, 0.9);
        float mech = mechX * step(0.2, uv.y) * step(uv.y, 0.8);
        
        vec3 mechColor = vec3(0.2, 0.2, 0.25);
        mechColor += vec3(0.0, 0.3, 0.2) * u_active * (0.5 + 0.5 * sin(u_time * 5.0));
        
        color = mix(color, mechColor, mech);
        
        // Neural sparks when active
        if (u_active > 0.5) {
            float spark = hash(vec2(floor(uv.x * 50.0), floor(uv.y * 30.0) + u_time * 10.0));
            spark = step(0.98, spark);
            color += vec3(0.5, 1.0, 0.7) * spark;
        }
        
        fragColor = vec4(color, 1.0);
    }
    `;

    const program = createProgram(gl, vs, fs);
    if (!program) return;

    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uActive = gl.getUniformLocation(program, 'u_active');

    let isActive = false;
    let activeAmount = 0;

    wrapper.addEventListener('click', () => {
        isActive = !isActive;
    });

    function render(time) {
        // Animate activation
        const target = isActive ? 1.0 : 0.0;
        activeAmount += (target - activeAmount) * 0.05;

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.03, 0.02, 0.04, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform1f(uActive, activeAmount);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

// Helper functions for WebGL2
function createProgram(gl, vsSource, fsSource) {
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
        console.error('VS:', gl.getShaderInfoLog(vs));
        return null;
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        console.error('FS:', gl.getShaderInfoLog(fs));
        return null;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Link:', gl.getProgramInfoLog(program));
        return null;
    }

    return program;
}

function setupQuad(gl, program) {
    const vertices = new Float32Array([
        -1, -1,
        1, -1,
        -1, 1,
        1, 1
    ]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const loc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
}
