/**
 * Advanced Physics Experiments JavaScript
 * Initializes the advanced physics experiments on their own page.
 */

document.addEventListener('DOMContentLoaded', () => {
    initQuantumTunnelingButtons();
    initPlasmaCoreButtons();
    initCrystallineFractalButtons();
    initTemporalEchoButtons();
});

function initTemporalEchoButtons() {
    const container = document.getElementById('temporal-echo-buttons');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;
    container.appendChild(wrapper);

    const layeredCanvas = new UIComponents.LayeredCanvas(wrapper, { width: 300, height: 200 });
    const webglLayer = layeredCanvas.addLayer('webgl', 'webgl2', 0);
    if (!webglLayer.context) {
        container.innerHTML = '<p style="color: #ff6666;">WebGL2 required</p>';
        return;
    }
    const gl = webglLayer.context;

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
        uniform sampler2D u_prev_frame;
        uniform vec2 u_mouse;
        uniform float u_click;
        out vec4 fragColor;

        void main() {
            vec4 prev_color = texture(u_prev_frame, v_uv);
            prev_color.rgb *= 0.95; // Fade out previous frame

            float circle = 1.0 - smoothstep(0.0, 0.1, length(v_uv * u_resolution - u_mouse));
            vec3 current_color = vec3(0.8, 0.2, 0.5) * circle * u_click;

            fragColor = vec4(prev_color.rgb + current_color, 1.0);
        }
    `;

    const program = UIComponents.ShaderUtils.createProgram(gl, vs, fs);
    if (!program) return;

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uResolution = gl.getUniformLocation(program, 'u_resolution');
    const uPrevFrame = gl.getUniformLocation(program, 'u_prev_frame');
    const uMouse = gl.getUniformLocation(program, 'u_mouse');
    const uClick = gl.getUniformLocation(program, 'u_click');

    // Create framebuffers
    const fbs = [gl.createFramebuffer(), gl.createFramebuffer()];
    const textures = [gl.createTexture(), gl.createTexture()];

    for (let i = 0; i < 2; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbs[i]);
        gl.bindTexture(gl.TEXTURE_2D, textures[i]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 300, 200, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textures[i], 0);
    }

    let mouseX = 0, mouseY = 0;
    let click = 0;
    webglLayer.canvas.addEventListener('mousemove', (e) => {
        const rect = webglLayer.canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = rect.height - (e.clientY - rect.top);
    });
    webglLayer.canvas.addEventListener('mousedown', () => { click = 1; });
    webglLayer.canvas.addEventListener('mouseup', () => { click = 0; });
    
    let frame = 0;
    layeredCanvas.setRenderFunction('webgl', (layer, timestamp) => {
        const time = timestamp * 0.001;
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbs[frame % 2]);
        gl.clear(gl.COLOR_BUFFER_BIT);
        
        gl.useProgram(program);
        gl.uniform1f(uTime, time);
        gl.uniform2f(uResolution, layer.canvas.width, layer.canvas.height);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, textures[(frame + 1) % 2]);
        gl.uniform1i(uPrevFrame, 0);
        gl.uniform2f(uMouse, mouseX, mouseY);
        gl.uniform1f(uClick, click);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        frame++;
    });

    layeredCanvas.startAnimation();
}

function initCrystallineFractalButtons() {
    const container = document.getElementById('crystalline-fractal-buttons');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;
    container.appendChild(wrapper);

    const layeredCanvas = new UIComponents.LayeredCanvas(wrapper, { width: 300, height: 200 });
    const webglLayer = layeredCanvas.addLayer('webgl', 'webgl2', 0);
    if (!webglLayer.context) {
        container.innerHTML = '<p style="color: #ff6666;">WebGL2 required</p>';
        return;
    }
    const gl = webglLayer.context;

    const vs = `#version 300 es
        in vec4 a_position;
        void main() { gl_Position = a_position; }
    `;

    const fs = `#version 300 es
        precision highp float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_click_time;
        out vec4 fragColor;

        mat2 rotate(float a) {
            float s = sin(a);
            float c = cos(a);
            return mat2(c, -s, s, c);
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);
            float t = u_time * 0.1;
            
            vec3 color = vec3(0.0);
            
            float click_progress = u_time - u_click_time;
            if (u_click_time > 0.0 && click_progress < 5.0) {
                t += click_progress;
            }

            uv *= rotate(t * 0.5);

            for (int i = 0; i < 6; i++) {
                uv = abs(uv) - 0.5;
                uv *= 1.2;
                uv *= rotate(t * 0.3 + float(i) * 0.5);
                color += vec3(0.01, 0.02, 0.04) / (length(uv) + 0.01);
            }

            fragColor = vec4(color, 1.0);
        }
    `;

    const program = UIComponents.ShaderUtils.createProgram(gl, vs, fs);
    if (!program) return;

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uResolution = gl.getUniformLocation(program, 'u_resolution');
    const uClickTime = gl.getUniformLocation(program, 'u_click_time');

    let clickTime = -10.0;
    webglLayer.canvas.addEventListener('click', () => {
        clickTime = performance.now() * 0.001;
    });

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    layeredCanvas.setRenderFunction('webgl', (layer, timestamp) => {
        const time = timestamp * 0.001;
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.uniform1f(uTime, time);
        gl.uniform2f(uResolution, layer.canvas.width, layer.canvas.height);
        gl.uniform1f(uClickTime, clickTime);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    layeredCanvas.startAnimation();
}

function initPlasmaCoreButtons() {
    const container = document.getElementById('plasma-core-buttons');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;
    container.appendChild(wrapper);

    const layeredCanvas = new UIComponents.LayeredCanvas(wrapper, { width: 300, height: 200 });
    const webglLayer = layeredCanvas.addLayer('webgl', 'webgl2', 0);
    if (!webglLayer.context) {
        container.innerHTML = '<p style="color: #ff6666;">WebGL2 required</p>';
        return;
    }
    const gl = webglLayer.context;

    const vs = `#version 300 es
        in vec4 a_position;
        void main() { gl_Position = a_position; }
    `;

    const fs = `#version 300 es
        precision highp float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_click_time;
        out vec4 fragColor;

        float noise(vec3 p) {
            vec3 i = floor(p);
            vec3 f = fract(p);
            f = f*f*(3.0-2.0*f);
            
            vec2 uv = (i.xy+vec2(37.0,17.0)*i.z) + f.xy;
            vec2 rg = texture(TEXTURE_NOISE, (uv+ 0.5)/256.0, -100.0).yx;
            return mix(rg.x, rg.y, f.z);
        }

        float fbm(vec3 p) {
            float v = 0.0;
            v += noise(p*1.0)*0.5;
            v += noise(p*2.0)*0.25;
            v += noise(p*4.0)*0.125;
            return v;
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);
            float t = u_time * 0.3;
            
            vec3 p = vec3(uv * 2.0, t);
            float noise_val = fbm(p + fbm(p));

            float d = length(uv);
            float circle = smoothstep(0.8, 0.2, d);
            
            float flare = 1.0 + (u_time - u_click_time) < 0.5 ? 2.0 : 0.0;
            
            vec3 color = vec3(noise_val * 1.5, pow(noise_val, 2.0) * 0.5, pow(noise_val, 3.0) * 0.2);
            color = mix(color, vec3(1.0, 0.9, 0.8), flare);

            fragColor = vec4(color * circle, circle);
        }
    `;

    const program = UIComponents.ShaderUtils.createProgram(gl, vs, fs);
    if (!program) return;

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uResolution = gl.getUniformLocation(program, 'u_resolution');
    const uClickTime = gl.getUniformLocation(program, 'u_click_time');

    let clickTime = -10.0;
    webglLayer.canvas.addEventListener('click', () => {
        clickTime = performance.now() * 0.001;
    });

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    // Create a noise texture
    const noise_size = 256;
    const noise_data = new Uint8Array(noise_size * noise_size * 2);
    for (let i = 0; i < noise_size * noise_size; i++) {
        noise_data[i*2] = Math.random() * 255;
        noise_data[i*2+1] = Math.random() * 255;
    }
    const noise_texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, noise_texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, noise_size, noise_size, 0, gl.RG, gl.UNSIGNED_BYTE, noise_data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    
    gl.useProgram(program);
    gl.uniform1i(gl.getUniformLocation(program, "TEXTURE_NOISE"), 0);

    layeredCanvas.setRenderFunction('webgl', (layer, timestamp) => {
        const time = timestamp * 0.001;
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.uniform1f(uTime, time);
        gl.uniform2f(uResolution, layer.canvas.width, layer.canvas.height);
        gl.uniform1f(uClickTime, clickTime);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    layeredCanvas.startAnimation();
}


function initQuantumTunnelingButtons() {
    const container = document.getElementById('quantum-tunneling-buttons');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = `
        width: 300px;
        height: 200px;
        position: relative;
        margin: 1rem auto;
    `;
    container.appendChild(wrapper);

    const layeredCanvas = new UIComponents.LayeredCanvas(wrapper, { width: 300, height: 200 });
    const webglLayer = layeredCanvas.addLayer('webgl', 'webgl2', 0);
    if (!webglLayer.context) {
        container.innerHTML = '<p style="color: #ff6666;">WebGL2 required</p>';
        return;
    }
    const gl = webglLayer.context;

    const vs = `#version 300 es
        in vec4 a_position;
        void main() { gl_Position = a_position; }
    `;

    const fs = `#version 300 es
        precision highp float;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_click_time;
        out vec4 fragColor;

        float noise(vec2 p) {
            return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);
            
            float t = u_time * 0.5;
            float click_progress = u_time - u_click_time;

            // Button shape
            float d = length(uv);
            float button_shape = smoothstep(0.5, 0.48, d);

            // Tunneling effect
            if (click_progress < 1.5 && u_click_time > 0.0) {
                uv.x += (noise(uv + t) - 0.5) * click_progress * 2.0;
                uv.y += (noise(uv - t) - 0.5) * click_progress * 2.0;
                d = length(uv);
                button_shape = smoothstep(0.5, 0.48, d) * (1.0 - click_progress / 1.5);
            }

            vec3 color = vec3(0.1, 0.7, 1.0) * button_shape;
            
            fragColor = vec4(color, button_shape);
        }
    `;

    const program = UIComponents.ShaderUtils.createProgram(gl, vs, fs);
    if (!program) return;

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    const uTime = gl.getUniformLocation(program, 'u_time');
    const uResolution = gl.getUniformLocation(program, 'u_resolution');
    const uClickTime = gl.getUniformLocation(program, 'u_click_time');

    let clickTime = -10.0;
    webglLayer.canvas.addEventListener('click', () => {
        clickTime = performance.now() * 0.001;
    });

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    layeredCanvas.setRenderFunction('webgl', (layer, timestamp) => {
        const time = timestamp * 0.001;
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.uniform1f(uTime, time);
        gl.uniform2f(uResolution, layer.canvas.width, layer.canvas.height);
        gl.uniform1f(uClickTime, clickTime);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    layeredCanvas.startAnimation();
}
