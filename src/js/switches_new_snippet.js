
/**
 * Initialize Plasma Switches
 */
function initPlasmaSwitches() {
    const container = document.getElementById('plasma-switches');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'position: relative; width: 100px; height: 160px; display: flex; flex-direction: column; align-items: center; gap: 10px;';

    // Canvas for plasma effect
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

        // Simple noise
        float noise(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }

        void main() {
            vec2 uv = v_uv;
            
            // Electrodes
            float electrodeTop = step(0.95, uv.y);
            float electrodeBottom = step(uv.y, 0.05);
            
            vec3 color = vec3(0.1);
            if(electrodeTop > 0.0 || electrodeBottom > 0.0) {
                 color = vec3(0.4);
            }

            // Arc
            if (u_on > 0.0) {
                float t = u_time * 10.0;
                float x = 0.5;
                // Wiggle
                x += (noise(vec2(uv.y * 5.0, t)) - 0.5) * 0.2 * u_on;
                
                float width = 0.02 * u_on + 0.01 * sin(t);
                float arc = 1.0 - smoothstep(width, width + 0.05, abs(uv.x - x));
                
                vec3 arcColor = vec3(0.6, 0.2, 1.0);
                // Core
                arcColor += vec3(0.8, 0.8, 1.0) * smoothstep(0.01, 0.0, abs(uv.x - x));
                
                // Beating/Flickering
                float flick = 0.8 + 0.2 * noise(vec2(t, 0.0));
                
                color += arcColor * arc * flick;
                
                // Glow
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

        // Smooth transition for arc intensity
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
