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