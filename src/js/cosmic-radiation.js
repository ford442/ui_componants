export class CosmicRadiation {
  constructor(container) {
    this.container = container;
    this.canvasGL = document.createElement('canvas');
    this.canvasGPU = document.createElement('canvas');

    // Setup styles for layering
    this.container.style.position = 'relative';
    this.container.style.width = '100%';
    this.container.style.height = '100%';
    this.container.style.overflow = 'hidden';
    this.container.style.backgroundColor = '#050505';

    [this.canvasGL, this.canvasGPU].forEach(canvas => {
      canvas.style.position = 'absolute';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      this.container.appendChild(canvas);
    });

    this.canvasGL.style.zIndex = '1';
    this.canvasGPU.style.zIndex = '2'; // Particles on top

    this.isPlaying = true;
    this.time = 0;

    // Bind methods
    this.resize = this.resize.bind(this);
    this.render = this.render.bind(this);

    this.init();
  }

  async init() {
    this.initWebGL();
    await this.initWebGPU();

    window.addEventListener('resize', this.resize);
    this.resize();
    requestAnimationFrame(this.render);
  }

  initWebGL() {
    this.gl = this.canvasGL.getContext('webgl2');
    if (!this.gl) {
      console.error('WebGL2 not supported');
      return;
    }

    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Simple shader for wireframe sphere
    const vsSource = `#version 300 es
    in vec3 a_position;
    uniform float u_time;
    uniform vec2 u_resolution;
    void main() {
      // Simple rotation
      float c = cos(u_time * 0.5);
      float s = sin(u_time * 0.5);
      mat3 rotY = mat3(
        c, 0, s,
        0, 1, 0,
        -s, 0, c
      );
      mat3 rotX = mat3(
        1, 0, 0,
        0, c, -s,
        0, s, c
      );

      vec3 pos = rotY * rotX * a_position;

      // Perspective projection
      float aspect = u_resolution.x / u_resolution.y;
      pos.z -= 3.0;
      pos.x /= aspect;

      gl_Position = vec4(pos.x, pos.y, pos.z, -pos.z); // Simple perspective division
      gl_PointSize = 4.0;
    }`;

    const fsSource = `#version 300 es
    precision highp float;
    out vec4 outColor;
    void main() {
      outColor = vec4(0.8, 0.2, 0.2, 0.5); // Reddish wireframe
    }`;

    this.program = this.createProgram(gl, vsSource, fsSource);
    this.positionLoc = gl.getAttribLocation(this.program, 'a_position');
    this.timeLoc = gl.getUniformLocation(this.program, 'u_time');
    this.resLoc = gl.getUniformLocation(this.program, 'u_resolution');

    // Create Sphere Data
    const positions = [];
    const rings = 20;
    const segments = 30;
    for (let i = 0; i <= rings; i++) {
      const lat = (i / rings) * Math.PI - Math.PI / 2;
      for (let j = 0; j <= segments; j++) {
        const lon = (j / segments) * Math.PI * 2;
        const x = Math.cos(lat) * Math.cos(lon);
        const y = Math.sin(lat);
        const z = Math.cos(lat) * Math.sin(lon);
        positions.push(x, y, z);
      }
    }

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    gl.enableVertexAttribArray(this.positionLoc);
    gl.vertexAttribPointer(this.positionLoc, 3, gl.FLOAT, false, 0, 0);

    this.vertexCount = positions.length / 3;
  }

  createProgram(gl, vsSource, fsSource) {
    const vs = this.compileShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = this.compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  async initWebGPU() {
    if (!navigator.gpu) {
        console.warn("WebGPU not supported");
        return;
    }

    try {
        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) return; // Add check here
        this.device = await this.adapter.requestDevice();
    } catch (e) {
        console.warn("WebGPU init failed", e);
        return;
    }

    this.contextGPU = this.canvasGPU.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this.contextGPU.configure({
        device: this.device,
        format: this.format,
        alphaMode: 'premultiplied',
    });

    const particleCount = 10000;
    this.particleCount = particleCount;

    // Particles: pos(2), vel(2) -> 4 floats -> 16 bytes per particle
    const particleData = new Float32Array(particleCount * 4);
    for (let i = 0; i < particleCount; i++) {
        // Random pos around center
        const angle = Math.random() * Math.PI * 2;
        const r = 0.5 + Math.random() * 0.5;
        particleData[i * 4] = Math.cos(angle) * r;     // x
        particleData[i * 4 + 1] = Math.sin(angle) * r; // y
        // Orbit velocity perpendicular to pos
        particleData[i * 4 + 2] = -Math.sin(angle) * 0.01; // vx
        particleData[i * 4 + 3] = Math.cos(angle) * 0.01;  // vy
    }

    this.particleBuffer = this.device.createBuffer({
        size: particleData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
    this.particleBuffer.unmap();

    // Simulation Parameters Uniform
    this.simParamsBuffer = this.device.createBuffer({
        size: 16, // time(f32), padding(3*f32) to align to 16 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute Shader
    const computeShader = `
      struct Particle {
        pos : vec2f,
        vel : vec2f,
      }

      @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

      struct SimParams {
        time : f32,
      }
      @group(0) @binding(1) var<uniform> params : SimParams;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
        let index = GlobalInvocationID.x;
        if (index >= arrayLength(&particles)) { return; }

        var p = particles[index];

        // Gravity well at center
        let center = vec2f(0.0, 0.0);
        let diff = center - p.pos;
        let dist = length(diff);
        let dir = normalize(diff);

        // Swirling force
        let tangent = vec2f(-dir.y, dir.x);

        let force = dir * (0.0001 / (dist + 0.1)) + tangent * 0.0005;

        p.vel += force;
        p.pos += p.vel;

        // Damping
        p.vel *= 0.99;

        // Reset if too close or too far
        if (dist < 0.1 || dist > 1.5) {
            let angle = params.time * 0.1 + f32(index) * 0.01;
            p.pos = vec2f(cos(angle), sin(angle)) * 0.8;
            p.vel = vec2f(-sin(angle), cos(angle)) * 0.01;
        }

        particles[index] = p;
      }
    `;

    this.computePipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: this.device.createShaderModule({ code: computeShader }),
            entryPoint: 'main',
        },
    });

    this.computeBindGroup = this.device.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: this.particleBuffer } },
            { binding: 1, resource: { buffer: this.simParamsBuffer } },
        ],
    });

    // Render Shader
    const renderShader = `
      struct Particle {
        pos : vec2f,
        vel : vec2f,
      }
      @group(0) @binding(0) var<storage, read> particles : array<Particle>;

      struct VertexOutput {
        @builtin(position) position : vec4f,
        @location(0) color : vec4f,
      }

      @vertex
      fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
        let p = particles[vertexIndex];
        var out : VertexOutput;
        out.position = vec4f(p.pos, 0.0, 1.0);

        let speed = length(p.vel);
        out.color = vec4f(0.2, 0.8 + speed * 100.0, 1.0, 0.6);
        return out;
      }

      @fragment
      fn fs_main(@location(0) color : vec4f) -> @location(0) vec4f {
        return color;
      }
    `;

    this.renderPipeline = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: this.device.createShaderModule({ code: renderShader }),
            entryPoint: 'vs_main',
        },
        fragment: {
            module: this.device.createShaderModule({ code: renderShader }),
            entryPoint: 'fs_main',
            targets: [{ format: this.format }],
        },
        primitive: {
            topology: 'point-list',
        },
    });

    this.renderBindGroup = this.device.createBindGroup({
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: this.particleBuffer } },
        ],
    });
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    // WebGL
    this.canvasGL.width = width;
    this.canvasGL.height = height;
    if (this.gl) this.gl.viewport(0, 0, width, height);

    // WebGPU
    this.canvasGPU.width = width;
    this.canvasGPU.height = height;

    // Resize is handled automatically by WebGPU context configuration usually,
    // but explicit size setting on canvas is needed for correct buffer resolution
  }

  render(time) {
    if (!this.isPlaying) return;
    this.time = time * 0.001;

    // --- WebGL Render ---
    if (this.gl) {
        const gl = this.gl;
        gl.clearColor(0, 0, 0, 0); // Transparent
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program);
        gl.uniform1f(this.timeLoc, this.time);
        gl.uniform2f(this.resLoc, this.canvasGL.width, this.canvasGL.height);

        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.LINE_STRIP, 0, this.vertexCount);
    }

    // --- WebGPU Render ---
    if (this.device && this.contextGPU) {
        // Update uniforms
        const simParams = new Float32Array([this.time]); // only need 1 float, padded automatically by writeBuffer? No, need to be careful.
        // Buffer is size 16. We write 4 bytes.
        this.device.queue.writeBuffer(this.simParamsBuffer, 0, simParams);

        const commandEncoder = this.device.createCommandEncoder();

        // Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
        computePass.end();

        // Render Pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.contextGPU.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 0 }, // Transparent
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(this.particleCount);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    requestAnimationFrame(this.render);
  }
}
