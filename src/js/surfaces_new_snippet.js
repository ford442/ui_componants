
function initHolographicSurface() {
    const container = document.getElementById('holographic-surface');
    if (!container) return;

    const lc = new UIComponents.LayeredCanvas(container, {
        width: container.clientWidth,
        height: container.clientHeight
    });

    const layer = lc.addLayer('holo', 'webgl2');
    const gl = layer.context;

    if (!gl) return;

    // Full screen quad shader
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
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec2 u_mouse;
        out vec4 fragColor;

        // Hash function
        float hash(vec2 p) {
            p = fract(p * vec2(123.34, 456.21));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
            vec2 mouse = (u_mouse - 0.5 * u_resolution.xy) / u_resolution.y;
            
            vec3 color = vec3(0.0);
            
            // 3D Parallax layers
            for (float i = 0.0; i < 1.0; i += 0.05) {
                float depth = fract(i + u_time * 0.1);
                float scale = 3.0 * (1.0 - depth);
                float fade = depth * smoothstep(1.0, 0.9, depth);
                
                vec2 localUV = uv * scale + vec2(i * 10.0, i * 20.0);
                
                // Mouse interaction
                localUV += mouse * depth * 0.5;
                
                vec2 grid = fract(localUV) - 0.5;
                vec2 id = floor(localUV);
                
                float rnd = hash(id);
                
                if (rnd > 0.95) {
                    float d = length(grid);
                    // Glowing particle
                    float spark = 0.01 / (d * d * 10.0 + 0.01);
                    // Color variation
                    vec3 pColor = mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 0.0, 1.0), rnd);
                    color += pColor * spark * fade * 0.5;
                }
            }
            
            // Background gradient
            color += vec3(0.0, 0.02, 0.05) * (1.0 - length(uv));

            fragColor = vec4(color, 1.0);
        }
    `;

    const program = createProgram(gl, vs, fs);
    const quad = createQuad(gl);

    const u = {
        time: gl.getUniformLocation(program, 'u_time'),
        resolution: gl.getUniformLocation(program, 'u_resolution'),
        mouse: gl.getUniformLocation(program, 'u_mouse')
    };

    let mouseX = 0;
    let mouseY = 0;
    container.addEventListener('mousemove', (e) => {
        const rect = container.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = rect.height - (e.clientY - rect.top);
    });

    lc.setRenderFunction('holo', (layerInfo, time) => {
        gl.viewport(0, 0, layer.canvas.width, layer.canvas.height);
        gl.useProgram(program);
        gl.uniform1f(u.time, time * 0.001);
        gl.uniform2f(u.resolution, layer.canvas.width, layer.canvas.height);
        gl.uniform2f(u.mouse, mouseX, mouseY);

        gl.bindVertexArray(quad.vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    lc.startAnimation();

    // Resize
    const resizeObserver = new ResizeObserver(() => {
        lc.resize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);
}

function createQuad(gl) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    return { vao };
}
