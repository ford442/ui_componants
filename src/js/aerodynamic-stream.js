export class AerodynamicStreamExperiment {
  constructor(container) {
    this.container = container;
    this.canvasGL = document.createElement('canvas');
    this.canvasGPU = document.createElement('canvas');

    // Setup styles for layering
    this.container.style.position = 'relative';
    this.container.style.width = '100%';
    this.container.style.height = '100%';
    this.container.style.overflow = 'hidden';
    this.container.style.backgroundColor = '#050510';

    [this.canvasGL, this.canvasGPU].forEach(canvas => {
      canvas.style.position = 'absolute';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      this.container.appendChild(canvas);
    });

    this.canvasGL.style.zIndex = '2'; // Wireframe on top
    this.canvasGPU.style.zIndex = '1'; // Particles behind/around

    this.isPlaying = true;
    this.time = 0;
    this.mouse = { x: 0.5, y: 0.5 }; // Default speed and angle
    this.angle = 0;
    this.speed = 1.0;

    // Bind methods
    this.resize = this.resize.bind(this);
    this.render = this.render.bind(this);
    this.handleMouseMove = this.handleMouseMove.bind(this);

    this.init();
  }

  async init() {
    this.initWebGL();
    const gpuSuccess = await this.initWebGPU();

    // Fallback if WebGPU fails (though we want hybrid, the class should be safe)
    if (!gpuSuccess) {
       console.warn("AerodynamicStream: WebGPU failed to init.");
    }

    window.addEventListener('resize', this.resize);
    this.container.addEventListener('mousemove', this.handleMouseMove);
    this.container.addEventListener('touchmove', this.handleMouseMove);

    this.resize();
    requestAnimationFrame(this.render);
  }

  destroy() {
    this.isPlaying = false;
    window.removeEventListener('resize', this.resize);
    this.container.removeEventListener('mousemove', this.handleMouseMove);
    this.container.removeEventListener('touchmove', this.handleMouseMove);

    // Clean up WebGL
    if (this.gl) {
        this.gl.deleteProgram(this.program);
        this.gl.deleteBuffer(this.positionBuffer);
        this.gl.deleteVertexArray(this.vao);
    }

    // Clean up WebGPU
    if (this.device) {
        this.device.destroy();
    }
  }

  handleMouseMove(e) {
    const rect = this.container.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;

    // Normalize to 0-1
    this.mouse.x = Math.max(0, Math.min(1, x / rect.width));
    this.mouse.y = Math.max(0, Math.min(1, y / rect.height));

    // Map to physical parameters
    // Y -> Angle of Attack (-30 to +30 degrees)
    this.angle = (this.mouse.y - 0.5) * Math.PI / 3;

    // X -> Speed (0.5 to 2.0)
    this.speed = 0.5 + this.mouse.x * 1.5;
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

    // Vertex Shader
    const vsSource = `#version 300 es
    in vec3 a_position;
    uniform vec2 u_resolution;
    uniform float u_angle;

    void main() {
      // Rotate around Z axis
      float c = cos(u_angle);
      float s = sin(u_angle);
      mat3 rotZ = mat3(
        c, s, 0,
        -s, c, 0,
        0, 0, 1
      );

      vec3 pos = rotZ * a_position;

      // Scale up
      pos *= 0.5;

      // Aspect ratio correction
      float aspect = u_resolution.x / u_resolution.y;
      pos.x /= aspect;

      gl_Position = vec4(pos, 1.0);
    }`;

    // Fragment Shader
    const fsSource = `#version 300 es
    precision highp float;
    out vec4 outColor;
    void main() {
      outColor = vec4(0.0, 1.0, 0.8, 1.0); // Cyan wireframe
    }`;

    this.program = this.createProgram(gl, vsSource, fsSource);
    this.positionLoc = gl.getAttribLocation(this.program, 'a_position');
    this.resLoc = gl.getUniformLocation(this.program, 'u_resolution');
    this.angleLoc = gl.getUniformLocation(this.program, 'u_angle');

    // Generate Airfoil Geometry (NACA 0012 approximation)
    const positions = [];
    const steps = 100;
    // Top surface
    for (let i = 0; i <= steps; i++) {
        const x = i / steps; // 0 to 1
        const t = 0.12;
        const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x*x + 0.2843 * x*x*x - 0.1015 * x*x*x*x);
        // Center it: x from -0.5 to 0.5
        positions.push(x - 0.5, yt, 0);
    }
    // Bottom surface
    for (let i = steps; i >= 0; i--) {
        const x = i / steps;
        const t = 0.12;
        const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x*x + 0.2843 * x*x*x - 0.1015 * x*x*x*x);
        positions.push(x - 0.5, -yt, 0);
    }

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
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
    if (!navigator.gpu) return false;

    try {
        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) return false;
        this.device = await this.adapter.requestDevice();
    } catch (e) {
        console.warn(e);
        return false;
    }

    this.contextGPU = this.canvasGPU.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this.contextGPU.configure({
        device: this.device,
        format: this.format,
        alphaMode: 'premultiplied',
    });

    const particleCount = 20000;
    this.particleCount = particleCount;

    // Particles: pos(2), vel(2), life(1), pad(3) -> 8 floats -> 32 bytes
    const particleData = new Float32Array(particleCount * 8);
    for (let i = 0; i < particleCount; i++) {
        // Random start position
        particleData[i * 8] = (Math.random() * 2 - 1) * 1.5; // x
        particleData[i * 8 + 1] = (Math.random() * 2 - 1); // y
        particleData[i * 8 + 2] = 1.0; // vx (flow right)
        particleData[i * 8 + 3] = 0.0; // vy
        particleData[i * 8 + 4] = Math.random(); // life
    }

    this.particleBuffer = this.device.createBuffer({
        size: particleData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
    this.particleBuffer.unmap();

    // Simulation Parameters Uniform
    // time(f32), angle(f32), speed(f32), padding(f32)
    this.simParamsBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute Shader
    const computeShader = `
      struct Particle {
        pos : vec2f,
        vel : vec2f,
        life : f32,
        pad : vec3f, // Pad to 32 bytes
      }

      @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

      struct SimParams {
        time : f32,
        angle : f32,
        speed : f32,
        padding : f32,
      }
      @group(0) @binding(1) var<uniform> params : SimParams;

      // Rotate vector
      fn rotate(v: vec2f, a: f32) -> vec2f {
        let c = cos(a);
        let s = sin(a);
        return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
      }

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
        let index = GlobalInvocationID.x;
        if (index >= arrayLength(&particles)) { return; }

        var p = particles[index];

        // Base flow velocity
        var targetVel = vec2f(params.speed * 0.01, 0.0);

        // Airfoil interaction
        // Rotate particle position into airfoil local space (inverse rotation)
        let localPos = rotate(p.pos, -params.angle);

        // Approximate airfoil as an ellipse for repulsion
        // x range: -0.25 to 0.25 (since we scaled by 0.5 in shader, here we keep sim units roughly -1 to 1)
        // Airfoil length approx 0.5 sim units

        let dx = localPos.x;
        let dy = localPos.y * 4.0; // Scale Y to treat as circle
        let distSq = dx*dx + dy*dy;

        // Repulsion field
        if (distSq < 0.1) {
            // Push away
            let repelDir = normalize(rotate(vec2f(dx, dy), params.angle));
            let force = repelDir * (0.0005 / (distSq + 0.001));
            p.vel += force;

            // Add turbulence
            p.vel.y += (fract(sin(params.time * 100.0 + f32(index)) * 43758.5453) - 0.5) * 0.005;
        }

        // Apply velocity
        p.vel = mix(p.vel, targetVel, 0.02); // Return to laminar flow slowly
        p.pos += p.vel;

        // Life cycle and reset
        p.life -= 0.005 * params.speed;

        // Reset if off screen or dead
        if (p.pos.x > 1.8 || p.life <= 0.0) {
            p.pos.x = -1.8;
            p.pos.y = (fract(sin(params.time * 50.0 + f32(index)) * 43758.5453) * 2.0 - 1.0) * 0.8;
            p.vel = targetVel;
            p.life = 1.0;
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
        life : f32,
        pad : vec3f,
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

        // Color based on velocity (pressure visualization)
        let speed = length(p.vel);
        // Normalize speed approx range
        let t = smoothstep(0.0, 0.03, speed);

        // Heatmap: Blue (slow) -> Green -> Red (fast)
        out.color = vec4f(t, 0.5 + t*0.5, 1.0 - t, p.life * 0.8);
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
            targets: [{
                format: this.format,
                blend: {
                    color: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one',
                        operation: 'add',
                    },
                    alpha: {
                        srcFactor: 'zero',
                        dstFactor: 'one',
                        operation: 'add',
                    }
                }
            }],
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

    return true;
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
  }

  render(time) {
    if (!this.isPlaying) return;
    this.time = time * 0.001;

    // --- WebGL Render ---
    if (this.gl) {
        const gl = this.gl;
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program);
        gl.uniform2f(this.resLoc, this.canvasGL.width, this.canvasGL.height);
        gl.uniform1f(this.angleLoc, this.angle);

        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.LINE_STRIP, 0, this.vertexCount);
    }

    // --- WebGPU Render ---
    if (this.device && this.contextGPU) {
        const simParams = new Float32Array([
            this.time,
            this.angle,
            this.speed,
            0.0
        ]);
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
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
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
