/**
 * Aurora Background
 * WebGL2-based aurora borealis shader effect with CSS fallback
 */

class AuroraBackground {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            colors: options.colors || [
                [0.2, 0.8, 0.9],  // cyan
                [0.9, 0.2, 0.8],  // magenta
                [0.6, 0.3, 0.9]   // purple
            ],
            speed: options.speed || 0.05,
            intensity: options.intensity || 0.8,
            useWebGL: options.useWebGL !== false,
            ...options
        };

        this.canvas = null;
        this.gl = null;
        this.program = null;
        this.animationId = null;
        this.startTime = Date.now();

        this.init();
    }

    init() {
        // Try WebGL2 first
        if (this.options.useWebGL && this.initWebGL2()) {
            this.animateWebGL();
        } else {
            // Fallback to CSS gradient
            this.initCSSFallback();
        }
    }

    initWebGL2() {
        this.canvas = document.createElement('canvas');
        this.gl = this.canvas.getContext('webgl2');

        if (!this.gl) {
            return false;
        }

        // Style canvas
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -2;
            pointer-events: none;
        `;

        this.container.appendChild(this.canvas);
        this.resize();

        // Create shader program
        const vertexShader = `#version 300 es
            in vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `;

        const fragmentShader = `#version 300 es
            precision highp float;
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec3 u_colors[3];
            uniform float u_intensity;
            out vec4 fragColor;

            // Simplex noise implementation
            vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

            float snoise(vec2 v) {
                const vec4 C = vec4(0.211324865405187,
                                    0.366025403784439,
                                   -0.577350269189626,
                                    0.024390243902439);
                vec2 i  = floor(v + dot(v, C.yy) );
                vec2 x0 = v -   i + dot(i, C.xx);
                vec2 i1;
                i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                vec4 x12 = x0.xyxy + C.xxzz;
                x12.xy -= i1;
                i = mod289(i);
                vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                    + i.x + vec3(0.0, i1.x, 1.0 ));
                vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
                m = m*m ;
                m = m*m ;
                vec3 x = 2.0 * fract(p * C.www) - 1.0;
                vec3 h = abs(x) - 0.5;
                vec3 ox = floor(x + 0.5);
                vec3 a0 = x - ox;
                m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
                vec3 g;
                g.x  = a0.x  * x0.x  + h.x  * x0.y;
                g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                return 130.0 * dot(m, g);
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                
                // Multiple layers of flowing noise
                float n1 = snoise(uv * 2.0 + vec2(u_time * 0.05, 0.0));
                float n2 = snoise(uv * 3.0 - vec2(0.0, u_time * 0.08));
                float n3 = snoise(uv * 1.5 + vec2(u_time * 0.03, u_time * 0.02));
                
                // Combine noise layers
                float combinedNoise = (n1 + n2 * 0.5 + n3 * 0.3) / 2.3;
                combinedNoise = (combinedNoise + 1.0) * 0.5; // Normalize to 0-1
                
                // Create flowing waves
                float wave1 = sin(uv.y * 3.0 + u_time * 0.1 + n1 * 2.0) * 0.5 + 0.5;
                float wave2 = sin(uv.y * 5.0 - u_time * 0.15 + n2 * 1.5) * 0.5 + 0.5;
                
                // Mix colors based on noise and waves
                vec3 color1 = mix(u_colors[0], u_colors[1], combinedNoise);
                vec3 color2 = mix(u_colors[1], u_colors[2], wave1);
                vec3 finalColor = mix(color1, color2, wave2 * 0.6);
                
                // Add vertical gradient for depth
                finalColor *= (0.3 + uv.y * 0.7);
                
                // Apply intensity
                finalColor *= u_intensity;
                
                fragColor = vec4(finalColor, 1.0);
            }
        `;

        this.program = this.createProgram(vertexShader, fragmentShader);
        if (!this.program) {
            return false;
        }

        // Setup quad
        const positionLocation = this.gl.getAttribLocation(this.program, 'a_position');
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(
            this.gl.ARRAY_BUFFER,
            new Float32Array([
                -1, -1,
                1, -1,
                -1, 1,
                1, 1
            ]),
            this.gl.STATIC_DRAW
        );

        const vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(vao);
        this.gl.enableVertexAttribArray(positionLocation);
        this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);

        // Get uniform locations
        this.uniforms = {
            time: this.gl.getUniformLocation(this.program, 'u_time'),
            resolution: this.gl.getUniformLocation(this.program, 'u_resolution'),
            colors: this.gl.getUniformLocation(this.program, 'u_colors'),
            intensity: this.gl.getUniformLocation(this.program, 'u_intensity')
        };

        // Event listener
        window.addEventListener('resize', () => this.resize());

        return true;
    }

    createProgram(vertexSource, fragmentSource) {
        const vertexShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vertexShader, vertexSource);
        this.gl.compileShader(vertexShader);

        if (!this.gl.getShaderParameter(vertexShader, this.gl.COMPILE_STATUS)) {
            console.error('Vertex shader error:', this.gl.getShaderInfoLog(vertexShader));
            return null;
        }

        const fragmentShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fragmentShader, fragmentSource);
        this.gl.compileShader(fragmentShader);

        if (!this.gl.getShaderParameter(fragmentShader, this.gl.COMPILE_STATUS)) {
            console.error('Fragment shader error:', this.gl.getShaderInfoLog(fragmentShader));
            return null;
        }

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);

        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program link error:', this.gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    resize() {
        if (this.canvas) {
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
        }
    }

    animateWebGL() {
        const time = (Date.now() - this.startTime) * 0.001 * this.options.speed;

        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0, 0, 0, 1);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        this.gl.useProgram(this.program);

        // Set uniforms
        this.gl.uniform1f(this.uniforms.time, time);
        this.gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);
        this.gl.uniform1f(this.uniforms.intensity, this.options.intensity);

        // Set color array
        const flatColors = this.options.colors.flat();
        this.gl.uniform3fv(this.uniforms.colors, flatColors);

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

        this.animationId = requestAnimationFrame(() => this.animateWebGL());
    }

    initCSSFallback() {
        const div = document.createElement('div');
        div.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -2;
            pointer-events: none;
            background: linear-gradient(
                135deg,
                rgba(51, 204, 230, 0.3) 0%,
                rgba(230, 51, 204, 0.3) 50%,
                rgba(153, 77, 230, 0.3) 100%
            );
            background-size: 400% 400%;
            animation: aurora-gradient 15s ease infinite;
        `;

        // Add keyframe animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes aurora-gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
        `;
        document.head.appendChild(style);

        this.container.appendChild(div);
        this.fallbackElement = div;
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.canvas) {
            window.removeEventListener('resize', () => this.resize());
            this.canvas.remove();
        }
        if (this.fallbackElement) {
            this.fallbackElement.remove();
        }
    }
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.AuroraBackground = AuroraBackground;
}
