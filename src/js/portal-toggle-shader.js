// Portal Toggle WebGL2 Shader
// This module creates a WebGL2 canvas that renders a swirling vortex effect for the Portal Toggle switch.

class PortalToggleShader {
    constructor(container, options = {}) {
        this.container = container;
        this.width = options.width || container.clientWidth;
        this.height = options.height || container.clientHeight;
        this.initGL();
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    initGL() {
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.pointerEvents = 'none';
        this.container.appendChild(canvas);
        const gl = canvas.getContext('webgl2');
        if (!gl) {
            console.warn('WebGL2 not supported, falling back to CSS');
            return;
        }
        this.gl = gl;
        // Vertex shader (simple passthrough)
        const vsSource = `#version 300 es
    in vec2 a_position;
    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
    }`;
        // Fragment shader: swirling vortex using polar coordinates
        const fsSource = `#version 300 es
    precision highp float;
    uniform float u_time;
    out vec4 outColor;
    void main() {
      vec2 uv = gl_FragCoord.xy / vec2(${this.width}.0, ${this.height}.0);
      vec2 p = uv - 0.5;
      float r = length(p);
      float a = atan(p.y, p.x);
      float swirl = sin(a * 8.0 + u_time * 2.0) * 0.5 + 0.5;
      float intensity = smoothstep(0.4, 0.0, r) * swirl;
      outColor = vec4(0.0, 0.5, 1.0, intensity);
    }`;
        const vertexShader = this.compileShader(gl.VERTEX_SHADER, vsSource);
        const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Unable to initialize the shader program.');
            return;
        }
        gl.useProgram(program);
        this.program = program;
        // Set up a full-screen quad
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1,
            1, -1,
            -1, 1,
            -1, 1,
            1, -1,
            1, 1,
        ]);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        const positionLocation = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
        this.timeLocation = gl.getUniformLocation(program, 'u_time');
        this.startTime = performance.now();
    }

    compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('An error occurred compiling the shaders: ', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    animate() {
        const gl = this.gl;
        if (!gl) return;
        const now = performance.now();
        const elapsed = (now - this.startTime) / 1000.0;
        gl.uniform1f(this.timeLocation, elapsed);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        requestAnimationFrame(this.animate);
    }

    destroy() {
        if (this.container && this.container.lastChild) {
            this.container.removeChild(this.container.lastChild);
        }
    }
}

// Export for global usage
window.PortalToggleShader = PortalToggleShader;
