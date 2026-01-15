/**
 * Pattern Display Tests
 * Implementation of Radial and Horizontal WGSL Pattern Renderers
 */

class PatternTests {
    constructor() {
        this.radialCanvas = document.getElementById('radial-canvas');
        this.horizontalCanvas = document.getElementById('horizontal-canvas');

        // State
        this.device = null;
        this.contextRadial = null;
        this.contextHorizontal = null;
        this.presentationFormat = null;

        this.radialPipeline = null;
        this.horizontalPipeline = null;

        this.texture = null;
        this.sampler = null;

        // Simulation State
        this.startTime = Date.now();
        this.isPlaying = true;
        this.bpm = 120;
        this.numChannels = 8;
        this.numRows = 64; // Standard pattern length

        // Buffers
        this.buffers = {
            radial: {},
            horizontal: {}
        };

        this.bindGroups = {
            radial: null,
            horizontal: null
        };

        this.init();
    }

    async init() {
        if (!navigator.gpu) {
            document.getElementById('webgpu-error').style.display = 'block';
            return;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error('No Adapter');
            this.device = await adapter.requestDevice();
        } catch (e) {
            document.getElementById('webgpu-error').style.display = 'block';
            console.error(e);
            return;
        }

        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        // Setup Contexts
        this.contextRadial = this.radialCanvas.getContext('webgpu');
        this.contextRadial.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });

        this.contextHorizontal = this.horizontalCanvas.getContext('webgpu');
        this.contextHorizontal.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });

        await this.loadAssets();
        this.initBuffers();
        this.initPipelines();

        // Start Loop
        requestAnimationFrame(() => this.render());

        // Handle Resize
        window.addEventListener('resize', () => this.resize());
        this.resize();
    }

    async loadAssets() {
        // Create Sampler
        this.sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Load Texture
        try {
            // Try to load from public root
            const response = await fetch('/buttons.png');
            if (!response.ok) throw new Error('Image not found');
            const blob = await response.blob();
            const bitmap = await createImageBitmap(blob);
            this.texture = this.createTextureFromSource(bitmap);
        } catch (e) {
            console.warn('PatternTests: Could not load buttons.png, using procedural fallback.');
            this.texture = this.createFallbackTexture();
        }
    }

    createTextureFromSource(source) {
        const texture = this.device.createTexture({
            size: [source.width, source.height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture(
            { source },
            { texture },
            [source.width, source.height]
        );
        return texture;
    }

    createFallbackTexture() {
        // Generate a 256x256 debug texture (checkerboard + noise)
        const size = 256;
        const data = new Uint8Array(size * size * 4);

        for (let i = 0; i < size * size; i++) {
            const x = i % size;
            const y = Math.floor(i / size);

            // Checkerboard pattern
            const check = ((Math.floor(x / 32) + Math.floor(y / 32)) % 2 === 0);

            // Noise
            const noise = Math.random() * 50;

            const val = check ? 100 + noise : 50 + noise;

            data[i * 4] = val;     // R
            data[i * 4 + 1] = val; // G
            data[i * 4 + 2] = val; // B
            data[i * 4 + 3] = 255; // A
        }

        const texture = this.device.createTexture({
            size: [size, size, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        this.device.queue.writeTexture(
            { texture },
            data,
            { bytesPerRow: size * 4 },
            [size, size]
        );

        return texture;
    }

    initBuffers() {
        // Common Data Sizes
        const numCells = this.numRows * this.numChannels;
        const cellsSize = numCells * 4 * 2; // u32 pairs? No, code says `cells: array<u32>`.
        // Wait, shader says `let idx = instanceIndex * 2u; let a = cells[idx]; let b = cells[idx+1];`
        // So each instance (cell) consumes 2 uint32s from the cells array.
        // Array size = numCells * 2 * 4 bytes.

        const rowFlagsSize = this.numRows * 4; // u32 per row

        // ChannelState struct: volume(f32), pan(f32), freq(f32), trigger(u32), noteAge(f32), activeEffect(u32), effectValue(f32), isMuted(u32)
        // 8 fields * 4 bytes = 32 bytes per channel
        // Total aligned size? Struct alignment rules apply.
        // In WGSL, struct members align.
        // float, float, float, u32, float, u32, float, u32 -> all 4 byte types.
        // 32 bytes is valid alignment (multiple of 16).
        const channelStateSize = this.numChannels * 32;

        // Uniforms struct size
        // Radial: 17 fields, mix of u32 and f32. All 4 bytes.
        // 17 * 4 = 68 bytes.
        // Uniform buffers must be aligned to 16 bytes. 68 -> round up to 80 (5 * 16)? Or just 16-byte aligned start.
        // Actually, size passed to createBuffer must be multiple of 16 if mapped? No, copy size must be 4-aligned.
        // But `minUniformBufferOffsetAlignment` affects binding dynamic offsets.
        // For standard binding, just ensure struct padding in WGSL matches JS.
        // JS Data: 17 floats/uints.
        // To be safe, I'll pad to 80 bytes (20 floats).
        const uniformsSize = 256; // Generous space

        // --- Init Buffers ---

        const createBuf = (size, usage) => {
            return this.device.createBuffer({
                size: Math.ceil(size / 4) * 4, // Ensure 4-byte align
                usage: usage | GPUBufferUsage.COPY_DST,
            });
        };

        // We can share the content buffers between renderers if logic is same,
        // but let's give them separate ones in case we want to diverge simulation.
        // Actually, let's share the DATA buffers (cells, channels) to show same "song" on both views.

        this.cellsBuffer = createBuf(numCells * 2 * 4, GPUBufferUsage.STORAGE);
        this.rowFlagsBuffer = createBuf(rowFlagsSize, GPUBufferUsage.STORAGE);
        this.channelsBuffer = createBuf(channelStateSize, GPUBufferUsage.STORAGE);

        // Uniform buffers need to be separate because canvas dims differ
        this.radialUniformBuffer = createBuf(uniformsSize, GPUBufferUsage.UNIFORM);
        this.horizontalUniformBuffer = createBuf(uniformsSize, GPUBufferUsage.UNIFORM);

        // Populate Initial Data
        this.generatePatternData(numCells);
    }

    generatePatternData(count) {
        // Fill cells with random notes
        const data = new Uint32Array(count * 2);
        for (let i = 0; i < count; i++) {
            // Chance of note
            if (Math.random() > 0.8) {
                // PackedA: [NoteChar][Inst][Reserved][Reserved]
                // Using bits logic from shader:
                // noteChar = (packedA >> 24) & 255
                // inst = packedA & 255

                // Notes A-G: 65-71
                const charCode = 65 + Math.floor(Math.random() * 7);
                const inst = Math.floor(Math.random() * 4);

                // PackedA = (Note << 24) | Inst
                data[i * 2] = (charCode << 24) | inst;

                // PackedB: [VolType][Res][EffCode][EffParam]
                // EffCode: 49('1'), 50('2'), 52('4'), 55('7'), 65('A')
                let effCode = 0;
                let effParam = 0;
                if (Math.random() > 0.7) {
                    const codes = [49, 50, 52, 55, 65];
                    effCode = codes[Math.floor(Math.random() * codes.length)];
                    effParam = 100 + Math.floor(Math.random() * 155);
                }

                data[i * 2 + 1] = (effCode << 8) | effParam;
            } else {
                data[i * 2] = 0;
                data[i * 2 + 1] = 0;
            }
        }
        this.device.queue.writeBuffer(this.cellsBuffer, 0, data);

        // Row Flags (markers)
        const flags = new Uint32Array(this.numRows);
        for(let i=0; i<this.numRows; i++) {
            flags[i] = (i % 4 === 0) ? 1 : 0; // Beat markers
        }
        this.device.queue.writeBuffer(this.rowFlagsBuffer, 0, flags);
    }

    initPipelines() {
        // --- Radial Pipeline ---
        const radialShader = `
            // patternv0.35_bloom.wgsl
            // - "Night Mode" (Dims housing/chrome when playing)
            // - UV Purple Ring (Outer) with Bezel Cast
            // - Channel Direction Toggle
            // - Compatible with "Donut" chassis (White center island)

            struct Uniforms {
              numRows: u32,
              numChannels: u32,
              playheadRow: u32,
              isPlaying: u32,
              cellW: f32,
              cellH: f32,
              canvasW: f32,
              canvasH: f32,
              tickOffset: f32,
              bpm: f32,
              timeSec: f32,
              beatPhase: f32,
              groove: f32,
              kickTrigger: f32,
              activeChannels: u32,
              isModuleLoaded: u32,
              bloomIntensity: f32,
              bloomThreshold: f32,
              invertChannels: u32,    // 0 = Outer Low, 1 = Inner Low
            };

            @group(0) @binding(0) var<storage, read> cells: array<u32>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;
            @group(0) @binding(2) var<storage, read> rowFlags: array<u32>;

            struct ChannelState { volume: f32, pan: f32, freq: f32, trigger: u32, noteAge: f32, activeEffect: u32, effectValue: f32, isMuted: u32 };
            @group(0) @binding(3) var<storage, read> channels: array<ChannelState>;
            @group(0) @binding(4) var buttonsSampler: sampler;
            @group(0) @binding(5) var buttonsTexture: texture_2d<f32>;

            struct VertexOut {
              @builtin(position) position: vec4<f32>,
              @location(0) @interpolate(flat) row: u32,
              @location(1) @interpolate(flat) channel: u32,
              @location(2) @interpolate(linear) uv: vec2<f32>,
              @location(3) @interpolate(flat) packedA: u32,
              @location(4) @interpolate(flat) packedB: u32,
            };

            @vertex
            fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOut {
              var quad = array<vec2<f32>, 6>(
                vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
                vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
              );

              let numChannels = uniforms.numChannels;
              let row = instanceIndex / numChannels;
              let channel = instanceIndex % numChannels;

              // --- CHANNEL DIRECTION LOGIC ---
              // Default (0): Channel 0 is OUTER ring.
              // Invert  (1): Channel 0 is INNER ring.
              var ringIndex = channel;
              if (uniforms.invertChannels == 0u) {
                  ringIndex = numChannels - 1u - channel;
              }

              let center = vec2<f32>(uniforms.canvasW * 0.5, uniforms.canvasH * 0.5);
              let minDim = min(uniforms.canvasW, uniforms.canvasH);

              let maxRadius = minDim * 0.45;
              let minRadius = minDim * 0.15;
              let ringDepth = (maxRadius - minRadius) / f32(numChannels);

              let radius = minRadius + f32(ringIndex) * ringDepth;

              let totalSteps = 64.0;
              let anglePerStep = 6.2831853 / totalSteps;
              let theta = -1.570796 + f32(row % 64u) * anglePerStep;

              let circumference = 2.0 * 3.14159265 * radius;
              let arcLength = circumference / totalSteps;

              let btnW = arcLength * 0.92;
              let btnH = ringDepth * 0.92;

              let lp = quad[vertexIndex];
              let localPos = (lp - 0.5) * vec2<f32>(btnW, btnH);

              let rotAng = theta + 1.570796;
              let cA = cos(rotAng);
              let sA = sin(rotAng);

              let rotX = localPos.x * cA - localPos.y * sA;
              let rotY = localPos.x * sA + localPos.y * cA;

              let worldX = center.x + cos(theta) * radius + rotX;
              let worldY = center.y + sin(theta) * radius + rotY;

              let clipX = (worldX / uniforms.canvasW) * 2.0 - 1.0;
              let clipY = 1.0 - (worldY / uniforms.canvasH) * 2.0;

              let idx = instanceIndex * 2u;
              let a = cells[idx];
              let b = cells[idx + 1u];

              var out: VertexOut;
              out.position = vec4<f32>(clipX, clipY, 0.0, 1.0);
              out.row = row;
              out.channel = channel;
              out.uv = lp;
              out.packedA = a;
              out.packedB = b;
              return out;
            }

            // --- HELPER FUNCTIONS ---

            fn neonPalette(t: f32) -> vec3<f32> {
              let a = vec3<f32>(0.5, 0.5, 0.5);
              let b = vec3<f32>(0.5, 0.5, 0.5);
              let c = vec3<f32>(1.0, 1.0, 1.0);
              let d = vec3<f32>(0.0, 0.33, 0.67);
              return a + b * cos(6.28318 * (c * t + d));
            }

            fn sdRoundedBox(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
              let q = abs(p) - b + r;
              return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
            }

            fn toUpperAscii(code: u32) -> u32 {
              return select(code, code - 32u, (code >= 97u) & (code <= 122u));
            }

            fn pitchClassFromPacked(packed: u32) -> f32 {
              let c0 = toUpperAscii((packed >> 24) & 255u);
              var semitone: i32 = 0;
              var valid = true;
              switch (c0) {
                case 65u: { semitone = 9; }
                case 66u: { semitone = 11; }
                case 67u: { semitone = 0; }
                case 68u: { semitone = 2; }
                case 69u: { semitone = 4; }
                case 70u: { semitone = 5; }
                case 71u: { semitone = 7; }
                default: { valid = false; }
              }
              if (!valid) { return 0.0; }
              let c1 = toUpperAscii((packed >> 16) & 255u);
              if ((c1 == 35u) || (c1 == 43u)) {
                semitone = (semitone + 1) % 12;
              } else if (c1 == 66u) {
                semitone = (semitone + 11) % 12;
              }
              return f32(semitone) / 12.0;
            }

            fn effectColorFromCode(code: u32, fallback: vec3<f32>) -> vec3<f32> {
              let c = toUpperAscii(code & 255u);
              switch c {
                case 49u: { return mix(fallback, vec3<f32>(0.2, 0.85, 0.4), 0.75); }
                case 50u: { return mix(fallback, vec3<f32>(0.85, 0.3, 0.3), 0.75); }
                case 52u: { return mix(fallback, vec3<f32>(0.4, 0.7, 1.0), 0.6); }
                case 55u: { return mix(fallback, vec3<f32>(0.9, 0.6, 0.2), 0.6); }
                case 65u: { return mix(fallback, vec3<f32>(0.95, 0.9, 0.25), 0.7); }
                default: { return fallback; }
              }
            }

            // --- CONSTANTS & DRAWING HELPERS ---

            struct FragmentConstants {
              bgColor: vec3<f32>,
              ledOnColor: vec3<f32>,
              ledOffColor: vec3<f32>,
              borderColor: vec3<f32>,
              housingSize: vec2<f32>,
            };

            fn getFragmentConstants() -> FragmentConstants {
              var c: FragmentConstants;
              c.bgColor = vec3<f32>(0.15, 0.16, 0.18);
              c.ledOnColor = vec3<f32>(0.0, 0.85, 0.95);
              c.ledOffColor = vec3<f32>(0.08, 0.08, 0.10);
              c.borderColor = vec3<f32>(0.0, 0.0, 0.0);
              c.housingSize = vec2<f32>(0.92, 0.92);
              return c;
            }

            fn drawChromeIndicator(
                uv: vec2<f32>,
                size: vec2<f32>,
                color: vec3<f32>,
                isOn: bool,
                aa: f32,
                dimFactor: f32
            ) -> vec4<f32> {
                let uv01 = (uv / size) + vec2<f32>(0.5);
                let lensR = 0.7;
                let bezelR = 0.9;
                let center = vec2<f32>(0.5, 0.5);
                let dist = length(uv01 - center) * 2.0;

                var col = vec3<f32>(0.0);
                var alpha = 0.0;

                if (dist < bezelR) {
                    if (dist > lensR) {
                        let angle = atan2(uv01.y - center.y, uv01.x - center.x);
                        let rim = 0.2 + 0.8 * abs(sin(angle * 10.0));
                        col = vec3<f32>(0.25, 0.28, 0.30) * rim * dimFactor;
                        alpha = 1.0;
                    } else {
                        let lensNormR = dist / lensR;
                        let z = sqrt(max(0.0, 1.0 - lensNormR * lensNormR));
                        let localXY = (uv01 - center) / lensR;
                        let normal = normalize(vec3<f32>(localXY.x, localXY.y, z));
                        let lightDir = normalize(vec3<f32>(-0.5, 0.5, 1.0));
                        let diffuse = max(0.0, dot(normal, lightDir));
                        let reflectDir = reflect(-lightDir, normal);
                        let specular = pow(max(0.0, dot(reflectDir, vec3<f32>(0.0, 0.0, 1.0))), 10.0);

                        let baseColor = color * (select(dimFactor, 1.0, isOn));
                        col = baseColor * (0.5 + 0.8 * diffuse);
                        col += vec3<f32>(1.0) * specular * 0.5 * dimFactor;
                        let rimGlow = exp(-pow(lensNormR, 2.0) * 6.0);
                        col += baseColor * rimGlow * 0.25;
                        alpha = 1.0;
                    }
                } else {
                    return vec4<f32>(vec3<f32>(0.0), 0.0);
                }
                let vignette = smoothstep(bezelR * 0.95, bezelR, dist);
                col = mix(col * (1.0 - 0.08 * vignette), vec3<f32>(0.02) * dimFactor, vignette);
                return vec4<f32>(col, alpha);
            }

            // --- MAIN FRAGMENT SHADER ---

            @fragment
            fn fs(in: VertexOut) -> @location(0) vec4<f32> {
              let fs = getFragmentConstants();
              let uv = in.uv;
              let p = uv - 0.5;
              let aa = fwidth(p.y) * 0.5;
              let bloom = uniforms.bloomIntensity;

              // STUDIO DARKNESS: Dim everything significantly when playing
              let isPlaying = (uniforms.isPlaying == 1u);
              let dimFactor = select(1.0, 0.35, isPlaying);

              if (in.channel == 0u) {
                let onPlayhead = (in.row == uniforms.playheadRow);
                let indSize = vec2(0.3, 0.3);

                let standardGray = vec3(0.2);
                let uvPurple = vec3(0.65, 0.0, 1.0);
                let activePurple = vec3(0.8, 0.4, 1.0);

                var indColor = standardGray;
                if (isPlaying) {
                    indColor = select(uvPurple, activePurple, onPlayhead);
                } else if (onPlayhead) {
                    indColor = fs.ledOnColor;
                }

                let isLit = isPlaying || onPlayhead;
                let indLed = drawChromeIndicator(p, indSize, indColor, isLit, aa, dimFactor);
                var col = indLed.rgb;
                var alpha = indLed.a;

                // UV CAST: When playing, boost the purple glow so it blooms onto the bezel
                if (isPlaying) {
                    col += uvPurple * 0.4 * bloom;
                }

                if (onPlayhead) {
                  let flashColor = select(fs.ledOnColor, activePurple, isPlaying);
                  let glow = flashColor * (bloom * 5.0) * exp(-length(p) * 4.0);
                  col += glow;
                  alpha = max(alpha, smoothstep(0.0, 0.2, length(glow)));
                }
                return vec4<f32>(col, clamp(alpha, 0.0, 1.0));
              }

              let dHousing = sdRoundedBox(p, fs.housingSize * 0.5, 0.06);
              let housingMask = 1.0 - smoothstep(0.0, aa * 1.5, dHousing);

              var finalColor = fs.bgColor * dimFactor;
              finalColor += vec3(0.04) * (0.5 - uv.y) * dimFactor;

              let btnScale = 1.05;
              let btnUV = (uv - 0.5) * btnScale + 0.5;
              var inButton = 0.0;
              if (btnUV.x > 0.0 && btnUV.x < 1.0 && btnUV.y > 0.0 && btnUV.y < 1.0) {
                let texColor = textureSampleLevel(buttonsTexture, buttonsSampler, btnUV, 0.0).rgb;
                finalColor = mix(finalColor, texColor * dimFactor, 0.7);
                inButton = 1.0;
              }

              if (inButton > 0.5) {
                let noteChar = (in.packedA >> 24) & 255u;
                let inst = in.packedA & 255u;
                let volType = (in.packedB >> 24) & 255u;
                let effCode = (in.packedB >> 8) & 255u;
                let effParam = in.packedB & 255u;

                let hasNote = (noteChar >= 65u && noteChar <= 71u);
                let hasExpression = (volType > 0u) || (effCode > 0u);
                let ch = channels[in.channel];
                let isMuted = (ch.isMuted == 1u);

                // COMPONENT 1: DATA LIGHT
                let topUV = btnUV - vec2(0.5, 0.16);
                let topSize = vec2(0.20, 0.20);
                let isDataPresent = hasExpression && !isMuted;
                let topColorBase = vec3(0.0, 0.9, 1.0);
                let topColor = topColorBase * select(0.0, 1.5 + bloom, isDataPresent);
                let topLed = drawChromeIndicator(topUV, topSize, topColor, isDataPresent, aa, dimFactor);
                finalColor = mix(finalColor, topLed.rgb, topLed.a);
                if (isDataPresent) { finalColor += topColor * topLed.a * 0.3; }

                // COMPONENT 2: MAIN NOTE LIGHT
                let mainUV = btnUV - vec2(0.5, 0.5);
                let mainSize = vec2(0.55, 0.45);
                var noteColor = vec3(0.2);
                var lightAmount = 0.0;

                if (hasNote) {
                  let pitchHue = pitchClassFromPacked(in.packedA);
                  let baseColor = neonPalette(pitchHue);
                  let instBand = inst & 15u;
                  let instBright = 0.8 + (select(0.0, f32(instBand) / 15.0, instBand > 0u)) * 0.2;
                  noteColor = baseColor * instBright;

                  let linger = exp(-ch.noteAge * 1.5);

                  // BLIP LOGIC: Instant strike when playhead hits
                  let onPlayhead = (in.row == uniforms.playheadRow);
                  let strike = select(0.0, 4.0 * exp(-uniforms.tickOffset * 10.0), onPlayhead);

                  let flash = f32(ch.trigger) * 1.0;
                  var d = f32(in.row) + uniforms.tickOffset - f32(uniforms.playheadRow);
                  let totalSteps = 64.0;
                  if (d > totalSteps * 0.5) { d = d - totalSteps; }
                  if (d < -totalSteps * 0.5) { d = d + totalSteps; }
                  let coreDist = abs(d);
                  let energy = 0.02 / (coreDist + 0.001);
                  let trail = exp(-10.0 * max(0.0, -d));
                  let activeVal = clamp(pow(energy, 1.5) + trail, 0.0, 1.0);

                  lightAmount = (activeVal * 0.8 + flash + strike + (linger * 2.0)) * clamp(ch.volume, 0.0, 1.2);
                  if (isMuted) { lightAmount *= 0.2; }
                }
                let displayColor = noteColor * max(lightAmount, 0.1) * (1.0 + bloom * 6.0);
                let isLit = (lightAmount > 0.05);
                let mainPad = drawChromeIndicator(mainUV, mainSize, displayColor, isLit, aa, dimFactor);
                finalColor = mix(finalColor, mainPad.rgb, mainPad.a);

                // COMPONENT 3: EFFECT LIGHT
                let botUV = btnUV - vec2(0.5, 0.85);
                let botSize = vec2(0.25, 0.12);
                var effColor = vec3(0.0);
                var isEffOn = false;
                if (effCode > 0u) {
                  effColor = effectColorFromCode(effCode, vec3(0.9, 0.8, 0.2));
                  let strength = clamp(f32(effParam) / 255.0, 0.2, 1.0);
                  if (!isMuted) { effColor *= strength * (1.0 + bloom * 2.5); isEffOn = true; }
                }
                let botLed = drawChromeIndicator(botUV, botSize, effColor, isEffOn, aa, dimFactor);
                finalColor = mix(finalColor, botLed.rgb, botLed.a);
              }

              if (housingMask < 0.5) { return vec4(fs.borderColor, 0.0); }
              return vec4(finalColor, 1.0);
            }
        `;

        // --- Horizontal Pipeline ---
        const horizontalShader = `
            // Horizontal Pattern Grid Shader

            struct Uniforms {
              numRows: u32,
              numChannels: u32,
              playheadRow: u32,
              isPlaying: u32,
              cellW: f32,
              cellH: f32,
              canvasW: f32,
              canvasH: f32,
              tickOffset: f32,
              bpm: f32,
              timeSec: f32,
              beatPhase: f32,
              groove: f32,
              kickTrigger: f32,
              activeChannels: u32,
              isModuleLoaded: u32,
            };

            @group(0) @binding(0) var<storage, read> cells: array<u32>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;
            @group(0) @binding(2) var<storage, read> rowFlags: array<u32>;

            struct ChannelState { volume: f32, pan: f32, freq: f32, trigger: u32, noteAge: f32, activeEffect: u32, effectValue: f32, isMuted: u32 };
            @group(0) @binding(3) var<storage, read> channels: array<ChannelState>;
            @group(0) @binding(4) var buttonsSampler: sampler;
            @group(0) @binding(5) var buttonsTexture: texture_2d<f32>;

            struct VertexOut {
              @builtin(position) position: vec4<f32>,
              @location(0) @interpolate(flat) row: u32,
              @location(1) @interpolate(flat) channel: u32,
              @location(2) @interpolate(linear) uv: vec2<f32>,
              @location(3) @interpolate(flat) packedA: u32,
              @location(4) @interpolate(flat) packedB: u32,
            };

            @vertex
            fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOut {
              var quad = array<vec2<f32>, 6>(
                vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
                vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
              );

              let numChannels = uniforms.numChannels;
              let row = instanceIndex / numChannels;
              let channel = instanceIndex % numChannels;

              let px = f32(row) * uniforms.cellW;
              let py = f32(channel) * uniforms.cellH;

              let lp = quad[vertexIndex];
              let worldX = px + lp.x * uniforms.cellW;
              let worldY = py + lp.y * uniforms.cellH;

              let clipX = (worldX / uniforms.canvasW) * 2.0 - 1.0;
              let clipY = 1.0 - (worldY / uniforms.canvasH) * 2.0;

              let idx = instanceIndex * 2u;
              let a = cells[idx];
              let b = cells[idx + 1u];

              var out: VertexOut;
              out.position = vec4<f32>(clipX, clipY, 0.0, 1.0);
              out.row = row;
              out.channel = channel;
              out.uv = lp;
              out.packedA = a;
              out.packedB = b;
              return out;
            }

            fn neonPalette(t: f32) -> vec3<f32> {
                let a = vec3<f32>(0.5, 0.5, 0.5);
                let b = vec3<f32>(0.5, 0.5, 0.5);
                let c = vec3<f32>(1.0, 1.0, 1.0);
                let d = vec3<f32>(0.0, 0.33, 0.67);
                return a + b * cos(6.28318 * (c * t + d));
            }

            fn sdRoundedBox(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
                let q = abs(p) - b + r;
                return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
            }

            fn toUpperAscii(code: u32) -> u32 {
                return select(code, code - 32u, (code >= 97u) & (code <= 122u));
            }

            fn pitchClassFromPacked(packed: u32) -> f32 {
                let c0 = toUpperAscii((packed >> 24) & 255u);
                var semitone: i32 = 0;
                var valid = true;
                switch (c0) {
                    case 65u: { semitone = 9; }
                    case 66u: { semitone = 11; }
                    case 67u: { semitone = 0; }
                    case 68u: { semitone = 2; }
                    case 69u: { semitone = 4; }
                    case 70u: { semitone = 5; }
                    case 71u: { semitone = 7; }
                    default: { valid = false; }
                }
                if (!valid) { return 0.0; }
                let c1 = toUpperAscii((packed >> 16) & 255u);
                if ((c1 == 35u) || (c1 == 43u)) {
                    semitone = (semitone + 1) % 12;
                } else if (c1 == 66u) {
                    semitone = (semitone + 11) % 12;
                }
                return f32(semitone) / 12.0;
            }

            fn effectColorFromCode(code: u32, fallback: vec3<f32>) -> vec3<f32> {
                let c = toUpperAscii(code & 255u);
                switch c {
                    case 49u: { return mix(fallback, vec3<f32>(0.2, 0.85, 0.4), 0.75); }
                    case 50u: { return mix(fallback, vec3<f32>(0.85, 0.3, 0.3), 0.75); }
                    case 52u: { return mix(fallback, vec3<f32>(0.4, 0.7, 1.0), 0.6); }
                    case 55u: { return mix(fallback, vec3<f32>(0.9, 0.6, 0.2), 0.6); }
                    case 65u: { return mix(fallback, vec3<f32>(0.95, 0.9, 0.25), 0.7); }
                    default: { return fallback; }
                }
            }

            struct FragmentConstants {
              bgColor: vec3<f32>,
              ledOnColor: vec3<f32>,
              ledOffColor: vec3<f32>,
              borderColor: vec3<f32>,
              housingSize: vec2<f32>,
            };

            fn getFragmentConstants() -> FragmentConstants {
                var c: FragmentConstants;
                c.bgColor = vec3<f32>(0.10, 0.11, 0.13); // Deep technical grey
                c.ledOnColor = vec3<f32>(0.0, 0.85, 0.95); // Precision Cyan
                c.ledOffColor = vec3<f32>(0.08, 0.12, 0.15); // Dark blue-grey
                c.borderColor = vec3<f32>(0.0, 0.0, 0.0); // Pure black gap
                c.housingSize = vec2<f32>(0.96, 0.96); // Maximized cell usage (Bigger Display)
                return c;
            }

            @fragment
            fn fs(in: VertexOut) -> @location(0) vec4<f32> {
              let fs = getFragmentConstants();
              let uv = in.uv;
              let p = uv - 0.5;

              let aa = fwidth(p.y) * 0.75;

              if (in.channel == 0u) {
                  var col = fs.bgColor * 0.5;
                  let onPlayhead = (in.row == uniforms.playheadRow);
                  let ledSize = vec2<f32>(0.3, 0.3);
                  let dLed = sdRoundedBox(p, ledSize, 0.05);
                  var ledCol = fs.ledOffColor;
                  let ledMask = 1.0 - smoothstep(-aa, aa, dLed);
                  col = mix(col, ledCol, ledMask);
                  if (onPlayhead) {
                     let glowIntensity = exp(-dLed * 5.0);
                     col += fs.ledOnColor * ledMask * 1.5;
                     col += fs.ledOnColor * glowIntensity * 0.8;
                  } else if (in.row % 4u == 0u) {
                     col += vec3<f32>(0.2, 0.2, 0.25) * ledMask * 0.3;
                  }
                  return vec4<f32>(col, 1.0);
              }

              let dHousing = sdRoundedBox(p, fs.housingSize * 0.5, 0.04);
              let housingMask = 1.0 - smoothstep(0.0, aa * 2.0, dHousing);

              var finalColor = fs.bgColor;
              finalColor += vec3<f32>(0.03) * (0.5 - uv.y);
              if (dHousing < 0.0 && dHousing > -0.04) {
                  finalColor += vec3<f32>(0.15) * smoothstep(0.0, -0.1, p.y);
              }

              let btnScale = 1.05;
              let btnUV = (uv - 0.5) * btnScale + 0.5;
              var btnColor = vec3<f32>(0.0);
              var inButton = 0.0;

              if (btnUV.x > 0.0 && btnUV.x < 1.0 && btnUV.y > 0.0 && btnUV.y < 1.0) {
                  btnColor = textureSampleLevel(buttonsTexture, buttonsSampler, btnUV, 0.0).rgb;
                  inButton = 1.0;
              }
              if (inButton > 0.5) {
                  finalColor = mix(finalColor, btnColor * 0.6, 0.9);
              }

              let noteChar = (in.packedA >> 24) & 255u;
              let inst = in.packedA & 255u;
              let effCode = (in.packedB >> 8) & 255u;
              let effParam = in.packedB & 255u;
              let hasNote = (noteChar >= 65u && noteChar <= 71u);
              let hasEffect = (effParam > 0u);
              let ch = channels[in.channel];

              if (inButton > 0.5) {
                  let x = btnUV.x;
                  let y = btnUV.y;
                  let indicatorXMask = smoothstep(0.4, 0.41, x) - smoothstep(0.6, 0.61, x);
                  let topLightMask   = (smoothstep(0.05, 0.06, y) - smoothstep(0.15, 0.16, y)) * indicatorXMask;
                  let mainButtonYMask  = smoothstep(0.23, 0.24, y) - smoothstep(0.80, 0.81, y);
                  let mainButtonXMask = smoothstep(0.13, 0.14, x) - smoothstep(0.86, 0.87, x);
                  let mainButtonMask = mainButtonYMask * mainButtonXMask;
                  let bottomLightMask = (smoothstep(0.90, 0.91, y) - smoothstep(0.95, 0.96, y)) * indicatorXMask;

                  if (ch.isMuted == 1u) {
                      finalColor *= 0.3;
                  }

                  if (step(0.1, exp(-ch.noteAge * 2.0)) > 0.5) {
                      let topGlow = vec3<f32>(0.0, 0.9, 1.0);
                      finalColor += topGlow * topLightMask * 1.5;
                  }

                  if (hasNote) {
                      let pitchHue = pitchClassFromPacked(in.packedA);
                      let base_note_color = neonPalette(pitchHue);
                      let instBand = inst & 15u;
                      let instBrightness = 0.8 + (select(0.0, f32(instBand) / 15.0, instBand > 0u)) * 0.2;
                      var noteColor = base_note_color * instBrightness;
                      let flash = f32(ch.trigger) * 0.8;
                      let activeLevel = exp(-ch.noteAge * 3.0);
                      let lightAmount = (activeLevel * 0.8 + flash) * clamp(ch.volume, 0.0, 1.2);
                      finalColor += noteColor * mainButtonMask * lightAmount * 2.0;
                      let subsurface = noteColor * housingMask * lightAmount * 0.15;
                      finalColor += subsurface;
                  }

                  if (hasEffect) {
                      let effectColor = effectColorFromCode(effCode, vec3<f32>(0.9, 0.8, 0.2));
                      let strength = clamp(f32(effParam) / 255.0, 0.2, 1.0);
                      finalColor += effectColor * bottomLightMask * strength * 2.5;
                      finalColor += effectColor * housingMask * strength * 0.05;
                  }

                  let rowDist = abs(i32(in.row) - i32(uniforms.playheadRow));
                  if (rowDist == 0 && !hasNote) {
                      finalColor += vec3<f32>(0.15, 0.2, 0.25) * mainButtonMask;
                  }
              }

              if (housingMask < 0.5) {
                  return vec4<f32>(fs.borderColor, 1.0);
              }
              return vec4<f32>(finalColor, 1.0);
            }
        `;

        // --- Create Bind Group Layout ---
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // cells
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },           // uniforms
                { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // rowFlags
                { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // channels
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // sampler
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // texture
            ],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        // --- Helper to Create Pipeline ---
        const createPipeline = (code) => {
            const module = this.device.createShaderModule({ code });
            return this.device.createRenderPipeline({
                layout: pipelineLayout,
                vertex: {
                    module,
                    entryPoint: 'vs',
                },
                fragment: {
                    module,
                    entryPoint: 'fs',
                    targets: [{
                        format: this.presentationFormat,
                        blend: {
                            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        }
                    }],
                },
                primitive: { topology: 'triangle-list' },
            });
        };

        this.radialPipeline = createPipeline(radialShader);
        this.horizontalPipeline = createPipeline(horizontalShader);

        // --- Create Bind Groups ---
        const createBG = (uniformBuffer) => {
            return this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.cellsBuffer } },
                    { binding: 1, resource: { buffer: uniformBuffer } },
                    { binding: 2, resource: { buffer: this.rowFlagsBuffer } },
                    { binding: 3, resource: { buffer: this.channelsBuffer } },
                    { binding: 4, resource: this.sampler },
                    { binding: 5, resource: this.texture.createView() },
                ],
            });
        };

        this.bindGroups.radial = createBG(this.radialUniformBuffer);
        this.bindGroups.horizontal = createBG(this.horizontalUniformBuffer);
    }

    resize() {
        if (!this.radialCanvas || !this.horizontalCanvas) return;

        const updateSize = (canvas, context) => {
            const dpr = window.devicePixelRatio || 1;
            const w = canvas.clientWidth * dpr;
            const h = canvas.clientHeight * dpr;
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
        };

        updateSize(this.radialCanvas, this.contextRadial);
        updateSize(this.horizontalCanvas, this.contextHorizontal);
    }

    updateUniforms(time) {
        // Calculate Playhead
        // BPM 120 -> 2 beats per sec -> 120 ticks per sec (if 64 ticks per pattern? No usually 16th notes)
        // Let's assume standard tracker logic: 1 row per 16th note?
        // 120 BPM = 2 beats/sec = 8 16th-notes/sec.
        // secPerRow = 1/8 = 0.125s.
        const secondsPerRow = 60.0 / (this.bpm * 4); // 4 rows per beat usually
        const totalRows = this.numRows;

        // Loop time
        const loopDuration = totalRows * secondsPerRow;
        const currentLoopTime = time % loopDuration;
        const playheadRow = Math.floor(currentLoopTime / secondsPerRow);
        const tickOffset = (currentLoopTime % secondsPerRow) / secondsPerRow; // 0.0 - 1.0 fractional row

        // Update Channel Data (Simulate Activity)
        const channelData = new Float32Array(this.numChannels * 8);
        for (let i = 0; i < this.numChannels; i++) {
            // Trigger randomness
            const trigger = (Math.random() < 0.05) ? 1 : 0;
            const noteAge = Math.random() * 2.0; // Simulated

            // struct: volume(f32), pan(f32), freq(f32), trigger(u32), noteAge(f32), activeEffect(u32), effectValue(f32), isMuted(u32)
            channelData[i*8 + 0] = 0.8 + Math.sin(time + i) * 0.2; // Volume
            channelData[i*8 + 1] = 0.0; // Pan
            channelData[i*8 + 2] = 440.0; // Freq

            // We interpret the float array as uint32 for specific fields?
            // JS Float32Array views bytes as floats.
            // If we need to write U32s into this buffer, we need a DataView or Uint32Array view of the same buffer.
            // Let's do a mixed write.
        }

        // Proper Mixed Buffer Update
        const bufferSize = this.numChannels * 32;
        const mixedBuffer = new ArrayBuffer(bufferSize);
        const f32View = new Float32Array(mixedBuffer);
        const u32View = new Uint32Array(mixedBuffer);

        for (let i = 0; i < this.numChannels; i++) {
            const base = i * 8;
            f32View[base + 0] = 0.8 + Math.sin(time + i) * 0.2; // Volume
            f32View[base + 1] = 0.0; // Pan
            f32View[base + 2] = 440.0; // Freq
            u32View[base + 3] = (playheadRow % (i+2) === 0 && tickOffset < 0.2) ? 1 : 0; // Trigger on beat
            f32View[base + 4] = (playheadRow % (i+2) === 0) ? tickOffset : 1.0 + tickOffset; // Note Age (fresh on beat)
            u32View[base + 5] = 0; // ActiveEffect
            f32View[base + 6] = 0; // EffectValue
            u32View[base + 7] = (i === 4 && time % 2 > 1) ? 1 : 0; // Mute toggle demo
        }
        this.device.queue.writeBuffer(this.channelsBuffer, 0, mixedBuffer);

        // Update Uniforms
        const writeUniforms = (buffer, width, height) => {
             // Struct:
             // numRows(u), numChannels(u), playheadRow(u), isPlaying(u)
             // cellW(f), cellH(f), canvasW(f), canvasH(f)
             // tickOffset(f), bpm(f), timeSec(f), beatPhase(f)
             // groove(f), kickTrigger(f), activeChannels(u), isModuleLoaded(u)
             // bloomIntensity(f), bloomThreshold(f), invertChannels(u)

             // 19 fields.
             const data = new ArrayBuffer(256);
             const u32 = new Uint32Array(data);
             const f32 = new Float32Array(data);

             // 0-3
             u32[0] = this.numRows;
             u32[1] = this.numChannels;
             u32[2] = playheadRow;
             u32[3] = 1; // isPlaying

             // 4-7
             // Calculate cell size
             // Horizontal: width / numRows? No, usually horizontal is time -> x.
             // Shader: px = row * cellW.
             // We want to fit numRows on screen? Or scroll?
             // "Horizontal Pattern Grid" usually implies scrolling or fitting.
             // Let's fit all 64 rows horizontally.
             const cellW = width / this.numRows;
             const cellH = height / this.numChannels;

             f32[4] = cellW;
             f32[5] = cellH;
             f32[6] = width;
             f32[7] = height;

             // 8-11
             f32[8] = tickOffset;
             f32[9] = this.bpm;
             f32[10] = time;
             f32[11] = (time * (this.bpm/60)) % 1.0; // beatPhase

             // 12-15
             f32[12] = 0.0; // groove
             f32[13] = 0.0; // kick
             u32[14] = (1 << this.numChannels) - 1; // activeChannels (all)
             u32[15] = 1; // loaded

             // 16-18
             f32[16] = 0.5 + Math.sin(time) * 0.2; // bloom dynamic
             f32[17] = 0.8; // bloom threshold
             u32[18] = 0; // invertChannels

             this.device.queue.writeBuffer(buffer, 0, data);
        };

        writeUniforms(this.radialUniformBuffer, this.radialCanvas.width, this.radialCanvas.height);

        // For horizontal, maybe we want bigger cells?
        // The shader uses `px = row * cellW`.
        // If we want to see the whole pattern, we fit it.
        writeUniforms(this.horizontalUniformBuffer, this.horizontalCanvas.width, this.horizontalCanvas.height);
    }

    render() {
        if (!this.device || !this.radialPipeline) return;

        const time = (Date.now() - this.startTime) * 0.001;
        this.updateUniforms(time);

        const renderPass = (context, pipeline, bindGroup, count) => {
            const encoder = this.device.createCommandEncoder();
            const view = context.getCurrentTexture().createView();

            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: view,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            });

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            // 6 vertices per quad (2 triangles), numRows * numChannels instances
            pass.draw(6, count, 0, 0);

            pass.end();
            this.device.queue.submit([encoder.finish()]);
        };

        const instanceCount = this.numRows * this.numChannels;

        renderPass(this.contextRadial, this.radialPipeline, this.bindGroups.radial, instanceCount);
        renderPass(this.contextHorizontal, this.horizontalPipeline, this.bindGroups.horizontal, instanceCount);

        requestAnimationFrame(() => this.render());
    }
}

// Start
window.addEventListener('DOMContentLoaded', () => {
    new PatternTests();
});
