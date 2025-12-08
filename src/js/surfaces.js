/**
 * Surfaces Page JavaScript
 * 3D Material Simulation and Layering for Hardware Surfaces
 */

// --- 3D Math Helpers (Minimal) ---

const Math3D = {
    // Create an identity matrix (4x4)
    createIdentity() {
        return new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]);
    },

    // Perspective projection matrix
    perspective(out, fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const rangeInv = 1 / (near - far);

        out[0] = f / aspect;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;

        out[4] = 0;
        out[5] = f;
        out[6] = 0;
        out[7] = 0;

        out[8] = 0;
        out[9] = 0;
        out[10] = (far + near) * rangeInv;
        out[11] = -1;

        out[12] = 0;
        out[13] = 0;
        out[14] = (2 * far * near) * rangeInv;
        out[15] = 0;
        return out;
    },

    // LookAt view matrix
    lookAt(out, eye, center, up) {
        let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;

        const eyex = eye[0], eyey = eye[1], eyez = eye[2];
        const upx = up[0], upy = up[1], upz = up[2];
        const centerx = center[0], centery = center[1], centerz = center[2];

        // vec3.direction(eye, center, z)
        z0 = eyex - centerx;
        z1 = eyey - centery;
        z2 = eyez - centerz;

        // normalize z
        len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        z0 *= len;
        z1 *= len;
        z2 *= len;

        // vec3.cross(up, z, x)
        x0 = upy * z2 - upz * z1;
        x1 = upz * z0 - upx * z2;
        x2 = upx * z1 - upy * z0;

        // normalize x
        len = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
        if (!len) {
            x0 = 0; x1 = 0; x2 = 0;
        } else {
            len = 1 / len;
            x0 *= len; x1 *= len; x2 *= len;
        }

        // vec3.cross(z, x, y)
        y0 = z1 * x2 - z2 * x1;
        y1 = z2 * x0 - z0 * x2;
        y2 = z0 * x1 - z1 * x0;

        // normalize y
        len = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
        if (!len) {
            y0 = 0; y1 = 0; y2 = 0;
        } else {
            len = 1 / len;
            y0 *= len; y1 *= len; y2 *= len;
        }

        out[0] = x0;
        out[1] = y0;
        out[2] = z0;
        out[3] = 0;
        out[4] = x1;
        out[5] = y1;
        out[6] = z1;
        out[7] = 0;
        out[8] = x2;
        out[9] = y2;
        out[10] = z2;
        out[11] = 0;
        out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
        out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
        out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
        out[15] = 1;

        return out;
    },

    // Rotation matrix around Y axis
    rotateY(out, a) {
        const c = Math.cos(a);
        const s = Math.sin(a);

        // Assuming 'out' starts as identity or is wiped,
        // but for safety in this simple renderer we'll just return a rotation matrix
        // If 'out' is null, create new.
        if (!out) out = Math3D.createIdentity();

        out[0] = c;
        out[1] = 0;
        out[2] = -s;
        out[3] = 0;

        out[4] = 0;
        out[5] = 1;
        out[6] = 0;
        out[7] = 0;

        out[8] = s;
        out[9] = 0;
        out[10] = c;
        out[11] = 0;

        out[12] = 0;
        out[13] = 0;
        out[14] = 0;
        out[15] = 1;
        return out;
    },

    // Multiply two matrices
    multiply(out, a, b) {
        let i, j, k;
        let temp = new Float32Array(16);
        for (i = 0; i < 4; i++) {
            for (j = 0; j < 4; j++) {
                let sum = 0;
                for (k = 0; k < 4; k++) {
                    sum += a[i * 4 + k] * b[k * 4 + j]; // Column-major math is tricky, usually row-major logic is easier for humans but WebGL wants col-major.
                    // Actually, standard math is: result[row][col] = sum(a[row][k] * b[k][col])
                    // In flat array (col-major): idx = col * 4 + row
                }
                // To avoid headache, let's just stick to a hardcoded multiplication for 4x4
            }
        }

        // Hardcoded standard multiplication (a * b)
        var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        var a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        var a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        var a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

        var b00 = b[0], b01 = b[1], b02 = b[2], b03 = b[3];
        var b10 = b[4], b11 = b[5], b12 = b[6], b13 = b[7];
        var b20 = b[8], b21 = b[9], b22 = b[10], b23 = b[11];
        var b30 = b[12], b31 = b[13], b32 = b[14], b33 = b[15];

        out[0] = b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30;
        out[1] = b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31;
        out[2] = b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32;
        out[3] = b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33;
        out[4] = b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30;
        out[5] = b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31;
        out[6] = b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32;
        out[7] = b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33;
        out[8] = b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30;
        out[9] = b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31;
        out[10] = b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32;
        out[11] = b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33;
        out[12] = b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30;
        out[13] = b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31;
        out[14] = b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32;
        out[15] = b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33;
        return out;
    }
};

// --- Geometry Generators ---

const Geometry = {
    // Basic cube
    createBox(width = 1, height = 1, depth = 1) {
        const w = width / 2;
        const h = height / 2;
        const d = depth / 2;

        const positions = [
            // Front face
            -w, -h, d, w, -h, d, w, h, d, -w, h, d,
            // Back face
            -w, -h, -d, -w, h, -d, w, h, -d, w, -h, -d,
            // Top face
            -w, h, -d, -w, h, d, w, h, d, w, h, -d,
            // Bottom face
            -w, -h, -d, w, -h, -d, w, -h, d, -w, -h, d,
            // Right face
            w, -h, -d, w, h, -d, w, h, d, w, -h, d,
            // Left face
            -w, -h, -d, -w, -h, d, -w, h, d, -w, h, -d,
        ];

        const normals = [
            // Front
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
            // Back
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            // Top
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
            // Bottom
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
            // Right
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
            // Left
            -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        ];

        const uvs = [
            // Front
            0, 0, 1, 0, 1, 1, 0, 1,
            // Back
            1, 0, 1, 1, 0, 1, 0, 0,
            // Top
            0, 1, 0, 0, 1, 0, 1, 1,
            // Bottom
            1, 1, 0, 1, 0, 0, 1, 0,
            // Right
            1, 0, 1, 1, 0, 1, 0, 0,
            // Left
            0, 0, 1, 0, 1, 1, 0, 1,
        ];

        const indices = [
            0, 1, 2, 0, 2, 3,    // Front
            4, 5, 6, 4, 6, 7,    // Back
            8, 9, 10, 8, 10, 11,    // Top
            12, 13, 14, 12, 14, 15,    // Bottom
            16, 17, 18, 16, 18, 19,    // Right
            20, 21, 22, 20, 22, 23     // Left
        ];

        return { positions: new Float32Array(positions), normals: new Float32Array(normals), uvs: new Float32Array(uvs), indices: new Uint16Array(indices) };
    },

    // Rack panel (essentially a very wide, thin box with maybe some extra detail in normals if we were fancy)
    createRackPanel() {
        return this.createBox(3, 0.8, 0.1);
    },

    // Flat plate
    createPlate() {
        return this.createBox(2, 2, 0.05);
    }
};

// --- Shaders ---

const Shaders = {
    vertex: `#version 300 es
        in vec3 a_position;
        in vec3 a_normal;
        in vec2 a_uv;

        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;

        out vec3 v_normal;
        out vec3 v_worldPos;
        out vec2 v_uv;

        void main() {
            vec4 worldPos = u_model * vec4(a_position, 1.0);
            gl_Position = u_projection * u_view * worldPos;

            v_worldPos = worldPos.xyz;
            v_normal = mat3(u_model) * a_normal; // Simplified normal transform (assumes uniform scale)
            v_uv = a_uv;
        }
    `,

    fragmentPBR: `#version 300 es
        precision highp float;

        in vec3 v_normal;
        in vec3 v_worldPos;
        in vec2 v_uv;

        uniform vec3 u_baseColor;
        uniform float u_roughness;
        uniform float u_metallic;
        uniform vec3 u_lightPos;
        uniform vec3 u_viewPos;
        uniform int u_materialType; // 0: Standard, 1: Brushed, 2: Holographic

        out vec4 fragColor;

        const float PI = 3.14159265359;

        // Simple noise function
        float rand(vec2 co){
            return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
        }

        // PBR Calculations (simplified Cook-Torrance)
        float DistributionGGX(vec3 N, vec3 H, float roughness) {
            float a = roughness*roughness;
            float a2 = a*a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH*NdotH;

            float num = a2;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = PI * denom * denom;

            return num / denom;
        }

        float GeometrySchlickGGX(float NdotV, float roughness) {
            float r = (roughness + 1.0);
            float k = (r*r) / 8.0;

            float num = NdotV;
            float denom = NdotV * (1.0 - k) + k;

            return num / denom;
        }

        float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx2 = GeometrySchlickGGX(NdotV, roughness);
            float ggx1 = GeometrySchlickGGX(NdotL, roughness);

            return ggx1 * ggx2;
        }

        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        }

        void main() {
            vec3 N = normalize(v_normal);
            vec3 V = normalize(u_viewPos - v_worldPos);
            vec3 L = normalize(u_lightPos - v_worldPos);
            vec3 H = normalize(V + L);

            // Texture Modifications based on material type
            float localRoughness = u_roughness;
            vec3 localBaseColor = u_baseColor;

            // Brushed Metal Effect
            if (u_materialType == 1) {
                float noise = rand(vec2(v_uv.y * 100.0, 0.0)); // Horizontal streaks
                localRoughness = mix(localRoughness, 1.0, noise * 0.3);
                N = normalize(N + vec3(0.0, (noise - 0.5) * 0.2, 0.0));
            }

            // Holographic Effect
            if (u_materialType == 2) {
                float viewAngle = dot(N, V);
                vec3 rainbow = 0.5 + 0.5 * cos(viewAngle * 10.0 + vec3(0.0, 2.0, 4.0));
                localBaseColor = mix(localBaseColor, rainbow, 0.8);
                localRoughness *= 0.2; // Glossy
            }

            // PBR
            vec3 F0 = vec3(0.04);
            F0 = mix(F0, localBaseColor, u_metallic);

            // Cook-Torrance BRDF
            float NDF = DistributionGGX(N, H, localRoughness);
            float G   = GeometrySmith(N, V, L, localRoughness);
            vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

            vec3 numerator    = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
            vec3 specular = numerator / denominator;

            vec3 kS = F;
            vec3 kD = vec3(1.0) - kS;
            kD *= 1.0 - u_metallic;

            float NdotL = max(dot(N, L), 0.0);

            // Light color (white) and intensity
            vec3 lightColor = vec3(1.0, 1.0, 1.0);
            float lightIntensity = 3.0; // Point light intensity

            // Simple distance attenuation
            float dist = length(u_lightPos - v_worldPos);
            float attenuation = 1.0 / (dist * dist * 0.1);

            vec3 Lo = (kD * localBaseColor / PI + specular) * lightColor * lightIntensity * NdotL * attenuation;

            // Ambient
            vec3 ambient = vec3(0.03) * localBaseColor;
            vec3 color = ambient + Lo;

            // Gamma correction
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0/2.2));

            fragColor = vec4(color, 1.0);
        }
    `,

    fragmentGlass: `#version 300 es
        precision mediump float;

        in vec3 v_normal;
        in vec3 v_worldPos;
        in vec2 v_uv;

        uniform vec3 u_viewPos;
        uniform float u_opacity;
        uniform float u_roughness;

        out vec4 fragColor;

        void main() {
            vec3 N = normalize(v_normal);
            vec3 V = normalize(u_viewPos - v_worldPos);

            float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.0);

            vec3 glassColor = vec3(0.8, 0.9, 1.0); // Slightly blueish
            float alpha = u_opacity + fresnel * 0.5;

            // Make it look a bit frosted based on roughness
            float noise = fract(sin(dot(v_uv, vec2(12.9898, 78.233))) * 43758.5453);
            if (u_roughness > 0.0) {
                 glassColor += (noise - 0.5) * u_roughness * 0.5;
            }

            fragColor = vec4(glassColor, clamp(alpha, 0.0, 0.95));
        }
    const lc = new UIComponents.LayeredCanvas(container, {
        width: container.clientWidth,
        height: container.clientHeight
    });

    // We only need one WebGL2 layer for this demo
    const layer = lc.addLayer('main', 'webgl2');
    const gl = layer.context;

    const state = {
        rotation: 0,
        materialType: 0, // 0: Plastic, 1: Metal, 2: Holo
        roughness: 0.5,
        metallic: 0.0,
        color: [0.2, 0.2, 0.2],
        lightPos: [2, 2, 5],
    };

    // Controls
    // We setup controls regardless of WebGL support so the UI doesn't break
    setupControls(state, (geo) => {
        if (gl && currentGeometry) {
             uploadGeometry(geo);
        }
    });

    if (!gl) {
        console.error('WebGL2 not supported');
        const ctx = layer.canvas.getContext('2d');
        if (ctx) {
            ctx.fillStyle = '#330000';
            ctx.fillRect(0, 0, layer.canvas.width, layer.canvas.height);
            ctx.fillStyle = '#ff0000';
            ctx.font = '16px monospace';
            ctx.fillText('WebGL2 Not Supported', 20, 30);
        }
        return;
    }

    // Compile Program
    const program = createProgram(gl, Shaders.vertex, Shaders.fragmentPBR);

    // Geometry Data Helpers
    let currentGeometry = Geometry.createBox();
    let vao = gl.createVertexArray();
    let buffers = {};

    function uploadGeometry(geo) {
        gl.bindVertexArray(vao);

        // Positions
        if(!buffers.pos) buffers.pos = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.pos);
        gl.bufferData(gl.ARRAY_BUFFER, geo.positions, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        // Normals
        if(!buffers.norm) buffers.norm = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.norm);
        gl.bufferData(gl.ARRAY_BUFFER, geo.normals, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        // UVs
        if(!buffers.uv) buffers.uv = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.uv);
        gl.bufferData(gl.ARRAY_BUFFER, geo.uvs, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

        // Indices
        if(!buffers.idx) buffers.idx = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.idx);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.indices, gl.STATIC_DRAW);

        currentGeometry.count = geo.indices.length;
    }

    uploadGeometry(currentGeometry);

    // Uniform Locations
    const uniforms = {
        model: gl.getUniformLocation(program, 'u_model'),
        view: gl.getUniformLocation(program, 'u_view'),
        projection: gl.getUniformLocation(program, 'u_projection'),
        baseColor: gl.getUniformLocation(program, 'u_baseColor'),
        roughness: gl.getUniformLocation(program, 'u_roughness'),
        metallic: gl.getUniformLocation(program, 'u_metallic'),
        lightPos: gl.getUniformLocation(program, 'u_lightPos'),
        viewPos: gl.getUniformLocation(program, 'u_viewPos'),
        materialType: gl.getUniformLocation(program, 'u_materialType')
    };

    // Render Loop
    lc.setRenderFunction('main', (layerInfo, time) => {
        const t = time * 0.001;
        state.rotation += 0.005;

        gl.viewport(0, 0, layer.canvas.width, layer.canvas.height);
        gl.clearColor(0.05, 0.05, 0.05, 1.0);
        gl.enable(gl.DEPTH_TEST);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(program);

        // Matrices
        const aspect = layer.canvas.width / layer.canvas.height;
        const projection = Math3D.createIdentity();
        Math3D.perspective(projection, 45 * Math.PI / 180, aspect, 0.1, 100.0);

        const view = Math3D.createIdentity();
        const eye = [0, 0, 5];
        Math3D.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);

        const model = Math3D.createIdentity();
        Math3D.rotateY(model, state.rotation);

        // Set Uniforms
        gl.uniformMatrix4fv(uniforms.projection, false, projection);
        gl.uniformMatrix4fv(uniforms.view, false, view);
        gl.uniformMatrix4fv(uniforms.model, false, model);

        gl.uniform3fv(uniforms.baseColor, state.color);
        gl.uniform1f(uniforms.roughness, state.roughness);
        gl.uniform1f(uniforms.metallic, state.metallic);
        gl.uniform3fv(uniforms.lightPos, state.lightPos);
        gl.uniform3fv(uniforms.viewPos, eye);
        gl.uniform1i(uniforms.materialType, state.materialType);

        gl.bindVertexArray(vao);
        gl.drawElements(gl.TRIANGLES, currentGeometry.count, gl.UNSIGNED_SHORT, 0);
    });

    lc.startAnimation();

    // Resize handling
    const resizeObserver = new ResizeObserver(() => {
        lc.resize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);
}

function setupControls(state, uploadGeometryFn) {
    // Material Type
    document.getElementById('ctrl-material').addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'plastic') state.materialType = 0;
        if (val === 'metal') state.materialType = 1;
        if (val === 'holographic') state.materialType = 2;
    });

    // Shape
    document.getElementById('ctrl-shape').addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'box') uploadGeometryFn(Geometry.createBox());
        if (val === 'rack') uploadGeometryFn(Geometry.createRackPanel());
        if (val === 'plate') uploadGeometryFn(Geometry.createPlate());
    });

    // Color
    document.getElementById('ctrl-color').addEventListener('input', (e) => {
        const hex = e.target.value;
        const r = parseInt(hex.substr(1,2), 16) / 255;
        const g = parseInt(hex.substr(3,2), 16) / 255;
        const b = parseInt(hex.substr(5,2), 16) / 255;
        state.color = [r, g, b];
    });

    // Sliders
    const linkSlider = (id, prop, displayId) => {
        const el = document.getElementById(id);
        el.addEventListener('input', (e) => {
            state[prop] = parseFloat(e.target.value);
            if (displayId) document.getElementById(displayId).textContent = state[prop];
        });
    };

    linkSlider('ctrl-roughness', 'roughness', 'val-roughness');
    linkSlider('ctrl-metallic', 'metallic', 'val-metallic');

    // Lights
    document.getElementById('ctrl-light-x').addEventListener('input', (e) => {
        state.lightPos[0] = parseFloat(e.target.value);
    });
    document.getElementById('ctrl-light-y').addEventListener('input', (e) => {
        state.lightPos[1] = parseFloat(e.target.value);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initSurfaceViewer(); // Assuming this is the original init function
    initTranslucencyDemo();
    initLiquidChrome();
    initForceField();
});

function initSurfaceViewer() {
    const container = document.getElementById('surface-viewer');
    if (!container) return;

    // Create LayeredCanvas
    const lc = new UIComponents.LayeredCanvas(container, {
        width: container.clientWidth,
        height: container.clientHeight
    });

    // We only need one WebGL2 layer for this demo
    const layer = lc.addLayer('main', 'webgl2');
    const gl = layer.context;

    const state = {
        rotation: 0,
        materialType: 0, // 0: Plastic, 1: Metal, 2: Holo
        roughness: 0.5,
        metallic: 0.0,
        color: [0.2, 0.2, 0.2],
        lightPos: [2, 2, 5],
    };

    // Controls
    // We setup controls regardless of WebGL support so the UI doesn't break
    setupControls(state, (geo) => {
        if (gl && currentGeometry) {
             uploadGeometry(geo);
        }
    });

    if (!gl) {
        console.error('WebGL2 not supported');
        const ctx = layer.canvas.getContext('2d');
        if (ctx) {
            ctx.fillStyle = '#330000';
            ctx.fillRect(0, 0, layer.canvas.width, layer.canvas.height);
            ctx.fillStyle = '#ff0000';
            ctx.font = '16px monospace';
            ctx.fillText('WebGL2 Not Supported', 20, 30);
        }
        return;
    }

    // Compile Program
    const program = createProgram(gl, Shaders.vertex, Shaders.fragmentPBR);

    // Geometry Data Helpers
    let currentGeometry = Geometry.createBox();
    let vao = gl.createVertexArray();
    let buffers = {};

    function uploadGeometry(geo) {
        gl.bindVertexArray(vao);

        // Positions
        if(!buffers.pos) buffers.pos = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.pos);
        gl.bufferData(gl.ARRAY_BUFFER, geo.positions, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        // Normals
        if(!buffers.norm) buffers.norm = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.norm);
        gl.bufferData(gl.ARRAY_BUFFER, geo.normals, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        // UVs
        if(!buffers.uv) buffers.uv = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.uv);
        gl.bufferData(gl.ARRAY_BUFFER, geo.uvs, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

        // Indices
        if(!buffers.idx) buffers.idx = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.idx);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.indices, gl.STATIC_DRAW);

        currentGeometry.count = geo.indices.length;
    }

    uploadGeometry(currentGeometry);

    // Uniform Locations
    const uniforms = {
        model: gl.getUniformLocation(program, 'u_model'),
        view: gl.getUniformLocation(program, 'u_view'),
        projection: gl.getUniformLocation(program, 'u_projection'),
        baseColor: gl.getUniformLocation(program, 'u_baseColor'),
        roughness: gl.getUniformLocation(program, 'u_roughness'),
        metallic: gl.getUniformLocation(program, 'u_metallic'),
        lightPos: gl.getUniformLocation(program, 'u_lightPos'),
        viewPos: gl.getUniformLocation(program, 'u_viewPos'),
        materialType: gl.getUniformLocation(program, 'u_materialType')
    };

    // Render Loop
    lc.setRenderFunction('main', (layerInfo, time) => {
        const t = time * 0.001;
        state.rotation += 0.005;

        gl.viewport(0, 0, layer.canvas.width, layer.canvas.height);
        gl.clearColor(0.05, 0.05, 0.05, 1.0);
        gl.enable(gl.DEPTH_TEST);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(program);

        // Matrices
        const aspect = layer.canvas.width / layer.canvas.height;
        const projection = Math3D.createIdentity();
        Math3D.perspective(projection, 45 * Math.PI / 180, aspect, 0.1, 100.0);

        const view = Math3D.createIdentity();
        const eye = [0, 0, 5];
        Math3D.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);

        const model = Math3D.createIdentity();
        Math3D.rotateY(model, state.rotation);

        // Set Uniforms
        gl.uniformMatrix4fv(uniforms.projection, false, projection);
        gl.uniformMatrix4fv(uniforms.view, false, view);
        gl.uniformMatrix4fv(uniforms.model, false, model);

        gl.uniform3fv(uniforms.baseColor, state.color);
        gl.uniform1f(uniforms.roughness, state.roughness);
        gl.uniform1f(uniforms.metallic, state.metallic);
        gl.uniform3fv(uniforms.lightPos, state.lightPos);
        gl.uniform3fv(uniforms.viewPos, eye);
        gl.uniform1i(uniforms.materialType, state.materialType);

        gl.bindVertexArray(vao);
        gl.drawElements(gl.TRIANGLES, currentGeometry.count, gl.UNSIGNED_SHORT, 0);
    });

    lc.startAnimation();

    // Resize handling
    const resizeObserver = new ResizeObserver(() => {
        lc.resize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);
}

function initTranslucencyDemo() {
    const container = document.getElementById('translucency-demo');
    if (!container) return;

    // Create LayeredCanvas
        width: container.clientWidth,
        height: container.clientHeight
    });

    // Layer 1: Background SVG (The "Inside" of the case)
    const svgLayer = lc.addSVGLayer('internal');
    const svgNS = "http://www.w3.org/2000/svg";

    // Create some fake circuitry or text
    const text = document.createElementNS(svgNS, 'text');
    text.setAttribute('x', '50%');
    text.setAttribute('y', '50%');
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('fill', '#ffaa00');
    text.setAttribute('font-family', 'monospace');
    text.setAttribute('font-size', '24');
    text.textContent = "INTERNAL COMPONENT A1";
    svgLayer.element.appendChild(text);

    const rect = document.createElementNS(svgNS, 'rect');
    rect.setAttribute('x', '10%');
    rect.setAttribute('y', '40%');
    rect.setAttribute('width', '80%');
    rect.setAttribute('height', '20%');
    rect.setAttribute('fill', 'none');
    rect.setAttribute('stroke', '#ffaa00');
    rect.setAttribute('stroke-width', '2');
    rect.setAttribute('stroke-dasharray', '5,5');
    svgLayer.element.appendChild(rect);


    // Layer 2: WebGL Glass (The "Case")
    const glLayer = lc.addLayer('glass', 'webgl2', 10); // Higher Z-index
    const gl = glLayer.context;

    // Setup GL
    const program = createProgram(gl, Shaders.vertex, Shaders.fragmentGlass);

    // Using a simple plate for the glass
    const geo = Geometry.createPlate();
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, geo.positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const normBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, geo.normals, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const uvBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, geo.uvs, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

    const idxBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.indices, gl.STATIC_DRAW);

    // Uniforms
    const u = {
        model: gl.getUniformLocation(program, 'u_model'),
        view: gl.getUniformLocation(program, 'u_view'),
        projection: gl.getUniformLocation(program, 'u_projection'),
        opacity: gl.getUniformLocation(program, 'u_opacity'),
        roughness: gl.getUniformLocation(program, 'u_roughness'),
        viewPos: gl.getUniformLocation(program, 'u_viewPos'),
    };

    const state = {
        opacity: 0.3,
        roughness: 0.2,
        rotation: 0
    };

    // Render
    lc.setRenderFunction('glass', (layerInfo, time) => {
        state.rotation = Math.sin(time * 0.001) * 0.2; // Gentle wobble

        gl.viewport(0, 0, glLayer.canvas.width, glLayer.canvas.height);
        gl.clearColor(0, 0, 0, 0); // Transparent clear!
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        // gl.enable(gl.DEPTH_TEST); // Disable depth test to ensure it blends over SVG nicely without complications

        gl.useProgram(program);

        const aspect = glLayer.canvas.width / glLayer.canvas.height;
        const projection = Math3D.createIdentity();
        Math3D.perspective(projection, 45 * Math.PI / 180, aspect, 0.1, 100.0);

        const view = Math3D.createIdentity();
        const eye = [0, 0, 4];
        Math3D.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);

        const model = Math3D.createIdentity();
        Math3D.rotateY(model, state.rotation);
        // Push it slightly forward so it covers the SVG visually if we were doing 3D compositing,
        // but here the SVG is a DOM element behind the canvas.

        gl.uniformMatrix4fv(u.projection, false, projection);
        gl.uniformMatrix4fv(u.view, false, view);
        gl.uniformMatrix4fv(u.model, false, model);
        gl.uniform1f(u.opacity, state.opacity);
        gl.uniform1f(u.roughness, state.roughness);
        gl.uniform3fv(u.viewPos, eye);

        gl.bindVertexArray(vao);
        gl.drawElements(gl.TRIANGLES, geo.indices.length, gl.UNSIGNED_SHORT, 0);
    });

    lc.startAnimation();

    // Controls
    document.getElementById('ctrl-glass-opacity').addEventListener('input', (e) => state.opacity = parseFloat(e.target.value));
    document.getElementById('ctrl-glass-roughness').addEventListener('input', (e) => state.roughness = parseFloat(e.target.value));

    // Resize
    const resizeObserver = new ResizeObserver(() => {
        lc.resize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);
}

// Helpers
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

/**
 * Liquid Chrome Experiment
 * WebGL2 shader creating a distorting reflective surface
 */
function initLiquidChrome() {
    const container = document.getElementById('liquid-chrome-demo');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);

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
        uniform vec2 u_resolution;
        uniform vec2 u_mouse;
        out vec4 fragColor;

        // Psuedo-environment map reflection
        vec3 getEnv(vec3 dir) {
            float y = dir.y * 0.5 + 0.5;
            vec3 sky = mix(vec3(0.1, 0.15, 0.2), vec3(0.8, 0.9, 1.0), pow(y, 0.5));
            vec3 ground = mix(vec3(0.05), vec3(0.2, 0.1, 0.1), 1.0 - y);
            float horizon = smoothstep(-0.1, 0.1, dir.y);
    return mix(ground, sky, horizon);
}

void main() {
            vec2 uv = gl_FragCoord.xy / u_resolution.xy;
            vec2 p = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);

            // Mouse interaction
            vec2 m = (u_mouse * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);
            float d = length(p - m);
            float interaction = exp(-d * 3.0) * 0.5 * sin(u_time * 5.0);

            // Fluid distortion based on noise/sine waves
            float t = u_time * 0.5;
            float height = sin(p.x * 4.0 + t) * cos(p.y * 4.0 + t * 0.8) * 0.5;
    height += sin(p.x * 10.0 - t * 2.0) * 0.2;
    height += interaction;

            // Calculate Normal
            vec2 e = vec2(0.01, 0.0);
            float hx = sin((p.x + e.x) * 4.0 + t) * cos(p.y * 4.0 + t * 0.8) * 0.5 - height;
            float hy = sin(p.x * 4.0 + t) * cos((p.y + e.x) * 4.0 + t * 0.8) * 0.5 - height;
            
            vec3 normal = normalize(vec3(-hx, -hy, e.x));

            // Lighting
            vec3 viewDir = normalize(vec3(p, -2.0)); // Fake orthographic view
            vec3 reflectDir = reflect(viewDir, normal);

            // Fresnel
            float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 3.0);

            // Color composition
            vec3 reflection = getEnv(reflectDir);
            vec3 baseColor = vec3(0.1, 0.1, 0.15); // Dark chrome base

            // Add some "dispersion" colors
            vec3 color = mix(baseColor, reflection, 0.8);
    color += vec3(0.5, 0.2, 0.1) * fresnel;

            // Specular highlight
            vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
            float spec = pow(max(dot(reflectDir, lightDir), 0.0), 30.0);
    color += vec3(1.0) * spec;

    fragColor = vec4(color, 1.0);
}
`;

    const program = createProgram(gl, vs, fs);
    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uRes = gl.getUniformLocation(program, 'u_resolution');
    const uMouse = gl.getUniformLocation(program, 'u_mouse');

    let mouseX = 0, mouseY = 0;
    container.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = rect.height - (e.clientY - rect.top);
    });

    function render(time) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(program);
        gl.uniform1f(uTime, time * 0.001);
        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform2f(uMouse, mouseX, mouseY);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

/**
 * Hex Force Field Experiment
 */
function initForceField() {
    const container = document.getElementById('force-field-demo');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);

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
        uniform vec2 u_resolution;
        uniform vec3 u_impacts[5]; // x,y, time of impact
        out vec4 fragColor;

        // Hexagon Distance Function
        float hexDist(vec2 p) {
    p = abs(p);
    return max(dot(p, normalize(vec2(1.0, 1.73))), p.x);
}

        vec4 getHex(vec2 uv) {
            vec2 r = vec2(1.0, 1.73);
            vec2 h = r * 0.5;
            vec2 a = mod(uv, r) - h;
            vec2 b = mod(uv - h, r) - h;
            vec2 gv = dot(a, a) < dot(b, b) ? a : b;
            float x = atan(gv.x, gv.y);
            float y = 0.5 - hexDist(gv);
            vec2 id = uv - gv;
    return vec4(x, y, id.x, id.y);
}

void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
            vec2 gridUV = uv * 10.0;
            
            vec4 hex = getHex(gridUV);
            float distToEdge = smoothstep(0.01, 0.05, hex.y);

            // Base Color
            vec3 color = vec3(0.0, 0.1, 0.2) * distToEdge;

    // Impact Ripples
    for (int i = 0; i < 5; i++) {
                vec3 impact = u_impacts[i]; // x, y, startTime
                float t = u_time - impact.z;

        if (t > 0.0 && t < 2.0) {
                    vec2 impactUV = (impact.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                    float d = length(uv - impactUV);

                    // Wavefront
                    float wave = sin(d * 20.0 - t * 10.0) * exp(-d * 2.0) * exp(-t * 2.0);

            // Highlight hexes based on wave
            if (wave > 0.1) {
                        float highlight = smoothstep(0.0, 0.5, wave) * distToEdge;
                color += vec3(0.2, 0.6, 1.0) * highlight;
            }
        }
    }

            // Pulse
            float pulse = sin(u_time + hex.z * 0.5 + hex.w * 0.5) * 0.5 + 0.5;
    color += vec3(0.0, 0.05, 0.1) * pulse * distToEdge;

    fragColor = vec4(color, 1.0);
}
`;

    const program = createProgram(gl, vs, fs);
    setupQuad(gl, program);

    const uTime = gl.getUniformLocation(program, 'u_time');
    const uRes = gl.getUniformLocation(program, 'u_resolution');
    const uImpacts = gl.getUniformLocation(program, 'u_impacts');

    // Store up to 5 impacts [x, y, time]
    let impacts = new Float32Array(15).fill(-100.0); 
    let impactIdx = 0;

    canvas.addEventListener('mousedown', e => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = rect.height - (e.clientY - rect.top);
        
        impacts[impactIdx * 3] = x;
        impacts[impactIdx * 3 + 1] = y;
        impacts[impactIdx * 3 + 2] = performance.now() * 0.001;
        
        impactIdx = (impactIdx + 1) % 5;
    });

    function render(time) {
        const t = time * 0.001;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(program);
        gl.uniform1f(uTime, t);
        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform3fv(uImpacts, impacts);
        
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

// Add setupQuad helper if it doesn't exist in your file scope
function setupQuad(gl, program) {
    const vertices = new Float32Array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1,  1, 1, 1
    ]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const loc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 16, 0); // Stride 16 (4 floats), offset 0
}