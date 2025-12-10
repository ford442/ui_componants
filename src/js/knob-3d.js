// 3D Rotating Knob (WebGL2)
// Renders a 3D sphere that rotates based on user drag interaction.

class Knob3D {
    constructor(container) {
        this.container = container;
        this.rotation = 0;
        this.width = container.clientWidth || 100;
        this.height = container.clientHeight || 100;

        this.initGL();
        this.animate = this.animate.bind(this);
        this.attachEvents();
        requestAnimationFrame(this.animate);
    }

    initGL() {
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        this.container.appendChild(canvas);

        const gl = canvas.getContext('webgl2');
        if (!gl) {
            console.warn('WebGL2 not supported for Knob3D');
            return;
        }
        this.gl = gl;

        // Vertex Shader: Sphere with lighting
        const vsSource = `#version 300 es
        in vec3 a_position;
        in vec3 a_normal;
        
        uniform float u_rotation;
        uniform mat4 u_projection;
        uniform mat4 u_view;
        
        out vec3 v_normal;
        out vec3 v_pos;
        
        void main() {
            float c = cos(u_rotation);
            float s = sin(u_rotation);
            mat3 rotateY = mat3(
                c, 0.0, -s,
                0.0, 1.0, 0.0,
                s, 0.0, c
            );
            
            vec3 pos = rotateY * a_position;
            v_normal = rotateY * a_normal;
            v_pos = pos;
            
            gl_Position = u_projection * u_view * vec4(pos, 1.0);
        }`;

        // Fragment Shader: Phong shading
        const fsSource = `#version 300 es
        precision highp float;
        
        in vec3 v_normal;
        in vec3 v_pos;
        
        out vec4 outColor;
        
        uniform vec3 u_lightPos;
        uniform vec3 u_viewPos;
        
        void main() {
            // Ambient
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
            
            // Diffuse
            vec3 norm = normalize(v_normal);
            vec3 lightDir = normalize(u_lightPos - v_pos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * vec3(0.0, 0.8, 1.0); // Cyan color
            
            // Specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(u_viewPos - v_pos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
            vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
            
            vec3 result = (ambient + diffuse + specular);
            outColor = vec4(result, 1.0);
        }`;

        const program = this.createProgram(gl, vsSource, fsSource);
        this.program = program;

        // Generate Sphere Data
        const sphere = this.createSphere(0.8, 30, 30);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sphere.vertices), gl.STATIC_DRAW);
        const positionLoc = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);

        const normalBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sphere.normals), gl.STATIC_DRAW);
        const normalLoc = gl.getAttribLocation(program, 'a_normal');
        gl.enableVertexAttribArray(normalLoc);
        gl.vertexAttribPointer(normalLoc, 3, gl.FLOAT, false, 0, 0);

        // Bind index buffer to VAO
        const indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(sphere.indices), gl.STATIC_DRAW);
        this.numIndices = sphere.indices.length;

        this.numVertices = sphere.vertices.length / 3;
        this.vao = vao;

        gl.enable(gl.DEPTH_TEST);
        gl.viewport(0, 0, this.width, this.height);

        // Uniform locations
        this.uRotation = gl.getUniformLocation(program, 'u_rotation');
        this.uProjection = gl.getUniformLocation(program, 'u_projection');
        this.uView = gl.getUniformLocation(program, 'u_view');
        this.uLightPos = gl.getUniformLocation(program, 'u_lightPos');
        this.uViewPos = gl.getUniformLocation(program, 'u_viewPos');

        // Constants
        const fieldOfView = 45 * Math.PI / 180;
        const aspect = this.width / this.height;
        const zNear = 0.1;
        const zFar = 100.0;
        const projectionMatrix = this.perspective(fieldOfView, aspect, zNear, zFar);

        gl.useProgram(program);
        gl.uniformMatrix4fv(this.uProjection, false, projectionMatrix);
        gl.uniformMatrix4fv(this.uView, false, new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, -3, 1
        ])); // Move camera back
        gl.uniform3fv(this.uLightPos, [2.0, 2.0, 2.0]);
        gl.uniform3fv(this.uViewPos, [0.0, 0.0, 3.0]);
    }

    createProgram(gl, vsSource, fsSource) {
        const vs = this.compileShader(gl, gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
        const program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        return program;
    }

    compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    createSphere(radius, latBands, longBands) {
        const vertices = [];
        const normals = [];
        for (let latNumber = 0; latNumber <= latBands; latNumber++) {
            const theta = latNumber * Math.PI / latBands;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let longNumber = 0; longNumber <= longBands; longNumber++) {
                const phi = longNumber * 2 * Math.PI / longBands;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);

                const x = cosPhi * sinTheta;
                const y = cosTheta;
                const z = sinPhi * sinTheta;

                normals.push(x, y, z);
                vertices.push(radius * x, radius * y, radius * z);
            }
        }

        // Generate indices for triangle rendering
        const indices = [];
        for (let latNumber = 0; latNumber < latBands; latNumber++) {
            for (let longNumber = 0; longNumber < longBands; longNumber++) {
                const first = (latNumber * (longBands + 1)) + longNumber;
                const second = first + longBands + 1;

                indices.push(first);
                indices.push(second);
                indices.push(first + 1);

                indices.push(second);
                indices.push(second + 1);
                indices.push(first + 1);
            }
        }

        return { vertices, normals, indices };
    }

    attachEvents() {
        let isDragging = false;
        let lastX = 0;

        this.container.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.clientX;
        });

        window.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaX = e.clientX - lastX;
                this.rotation += deltaX * 0.01;
                lastX = e.clientX;
            }
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }

    perspective(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ]);
    }

    animate() {
        if (!this.gl) return;

        this.gl.clearColor(0.0, 0.0, 0.0, 0.0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

        this.gl.useProgram(this.program);
        this.gl.uniform1f(this.uRotation, this.rotation);

        this.gl.bindVertexArray(this.vao);
        this.gl.drawElements(this.gl.TRIANGLES, this.numIndices, this.gl.UNSIGNED_SHORT, 0);

        requestAnimationFrame(this.animate);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const knobContainer = document.getElementById('knob-3d-container');
    if (knobContainer) {
        new Knob3D(knobContainer);
    }
});
