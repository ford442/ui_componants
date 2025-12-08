/**
 * Global Experiments Module
 * Cross-page visual effects: Cursor Trails, Audio Visualizer, Holographic Interference
 */

// ============================================================================
// CURSOR TRAIL ECOSYSTEM
// Particles spawn from cursor, interact with UI, and decay with comet trails
// ============================================================================

class CursorTrailSystem {
    constructor(options = {}) {
        this.particles = [];
        this.maxParticles = options.maxParticles || 150;
        this.spawnRate = options.spawnRate || 3;
        this.trailLength = options.trailLength || 8;
        this.attractors = []; // UI elements that attract particles
        this.mousePos = { x: 0, y: 0 };
        this.prevMousePos = { x: 0, y: 0 };
        this.isActive = false;
        this.canvas = null;
        this.ctx = null;
        this.hue = 0;

        this.init();
    }

    init() {
        // Create overlay canvas
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'cursor-trail-canvas';
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
        `;
        document.body.appendChild(this.canvas);

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Event listeners
        window.addEventListener('resize', () => this.resize());
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));

        // Find attractors (buttons, knobs, etc.)
        this.findAttractors();

        this.isActive = true;
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    findAttractors() {
        // Find interactive elements that should attract particles
        const selectors = [
            '.led-button', '.knob-wrapper', '.switch-container',
            '.nav-card', '.component-card', 'button'
        ];
        this.attractors = [];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                const rect = el.getBoundingClientRect();
                this.attractors.push({
                    x: rect.left + rect.width / 2,
                    y: rect.top + rect.height / 2,
                    radius: Math.max(rect.width, rect.height) / 2,
                    strength: 0.02
                });
            });
        });
    }

    onMouseMove(e) {
        this.prevMousePos.x = this.mousePos.x;
        this.prevMousePos.y = this.mousePos.y;
        this.mousePos.x = e.clientX;
        this.mousePos.y = e.clientY;

        // Calculate velocity for spawn intensity
        const dx = this.mousePos.x - this.prevMousePos.x;
        const dy = this.mousePos.y - this.prevMousePos.y;
        const velocity = Math.sqrt(dx * dx + dy * dy);

        // Spawn particles based on movement
        const spawnCount = Math.min(velocity * 0.3, this.spawnRate);
        for (let i = 0; i < spawnCount; i++) {
            this.spawnParticle(velocity);
        }
    }

    spawnParticle(velocity) {
        if (this.particles.length >= this.maxParticles) {
            this.particles.shift(); // Remove oldest
        }

        this.hue = (this.hue + 0.5) % 360;

        this.particles.push({
            x: this.mousePos.x,
            y: this.mousePos.y,
            vx: (Math.random() - 0.5) * velocity * 0.3,
            vy: (Math.random() - 0.5) * velocity * 0.3,
            size: 2 + Math.random() * 4,
            life: 1.0,
            decay: 0.008 + Math.random() * 0.012,
            hue: this.hue,
            trail: []
        });
    }

    updateParticle(p) {
        // Store trail position
        p.trail.push({ x: p.x, y: p.y });
        if (p.trail.length > this.trailLength) {
            p.trail.shift();
        }

        // Apply attractor forces
        this.attractors.forEach(attr => {
            const dx = attr.x - p.x;
            const dy = attr.y - p.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < attr.radius * 3 && dist > 10) {
                const force = attr.strength * (1 - dist / (attr.radius * 3));
                p.vx += (dx / dist) * force;
                p.vy += (dy / dist) * force;
            }
        });

        // Apply velocity with damping
        p.x += p.vx;
        p.y += p.vy;
        p.vx *= 0.98;
        p.vy *= 0.98;

        // Gravity-like drift
        p.vy += 0.02;

        // Decay life
        p.life -= p.decay;
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.particles.forEach((p, index) => {
            // Draw trail
            if (p.trail.length > 1) {
                this.ctx.beginPath();
                this.ctx.moveTo(p.trail[0].x, p.trail[0].y);
                for (let i = 1; i < p.trail.length; i++) {
                    this.ctx.lineTo(p.trail[i].x, p.trail[i].y);
                }
                this.ctx.lineTo(p.x, p.y);
                this.ctx.strokeStyle = `hsla(${p.hue}, 80%, 60%, ${p.life * 0.3})`;
                this.ctx.lineWidth = p.size * p.life * 0.5;
                this.ctx.lineCap = 'round';
                this.ctx.stroke();
            }

            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
            this.ctx.fillStyle = `hsla(${p.hue}, 90%, 65%, ${p.life})`;
            this.ctx.fill();

            // Glow effect
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size * p.life * 2, 0, Math.PI * 2);
            this.ctx.fillStyle = `hsla(${p.hue}, 90%, 70%, ${p.life * 0.3})`;
            this.ctx.fill();
        });
    }

    animate() {
        if (!this.isActive) return;

        // Update particles
        this.particles = this.particles.filter(p => {
            this.updateParticle(p);
            return p.life > 0;
        });

        this.render();
        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.isActive = false;
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}

// ============================================================================
// AUDIO VISUALIZER INTEGRATION
// Microphone input affects UI elements globally
// ============================================================================

class AudioVisualizerSystem {
    constructor(options = {}) {
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.isActive = false;
        this.sensitivity = options.sensitivity || 1.5;
        this.smoothing = options.smoothing || 0.8;
        this.callbacks = [];
        this.currentVolume = 0;
        this.beatDetected = false;
        this.lastBeatTime = 0;
        this.beatThreshold = options.beatThreshold || 0.6;
        this.frequencyBands = { bass: 0, mid: 0, treble: 0 };
    }

    async init() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(stream);

            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = this.smoothing;

            source.connect(this.analyser);

            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.isActive = true;
            this.analyze();

            console.log('🎤 Audio Visualizer initialized');
            return true;
        } catch (err) {
            console.warn('Audio Visualizer: Microphone access denied or unavailable', err);
            return false;
        }
    }

    analyze() {
        if (!this.isActive) return;

        this.analyser.getByteFrequencyData(this.dataArray);

        // Calculate overall volume
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        const raw = sum / this.dataArray.length / 255;
        this.currentVolume = raw * this.sensitivity;

        // Calculate frequency bands
        const binCount = this.dataArray.length;
        const bassEnd = Math.floor(binCount * 0.1);
        const midEnd = Math.floor(binCount * 0.5);

        let bassSum = 0, midSum = 0, trebleSum = 0;
        for (let i = 0; i < bassEnd; i++) bassSum += this.dataArray[i];
        for (let i = bassEnd; i < midEnd; i++) midSum += this.dataArray[i];
        for (let i = midEnd; i < binCount; i++) trebleSum += this.dataArray[i];

        this.frequencyBands.bass = (bassSum / bassEnd / 255) * this.sensitivity;
        this.frequencyBands.mid = (midSum / (midEnd - bassEnd) / 255) * this.sensitivity;
        this.frequencyBands.treble = (trebleSum / (binCount - midEnd) / 255) * this.sensitivity;

        // Beat detection
        const now = performance.now();
        if (this.frequencyBands.bass > this.beatThreshold && now - this.lastBeatTime > 200) {
            this.beatDetected = true;
            this.lastBeatTime = now;
            this.onBeat();
        } else {
            this.beatDetected = false;
        }

        // Notify callbacks
        this.callbacks.forEach(cb => cb({
            volume: this.currentVolume,
            bands: this.frequencyBands,
            beat: this.beatDetected,
            rawData: this.dataArray
        }));

        requestAnimationFrame(() => this.analyze());
    }

    onBeat() {
        // Pulse effect on document
        document.body.style.transition = 'filter 0.1s ease-out';
        document.body.style.filter = `brightness(1.15)`;
        setTimeout(() => {
            document.body.style.filter = 'brightness(1)';
        }, 100);
    }

    subscribe(callback) {
        this.callbacks.push(callback);
        return () => {
            this.callbacks = this.callbacks.filter(cb => cb !== callback);
        };
    }

    getVolume() {
        return this.currentVolume;
    }

    getBands() {
        return this.frequencyBands;
    }

    destroy() {
        this.isActive = false;
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

// ============================================================================
// HOLOGRAPHIC INTERFERENCE PATTERN
// Parallax layers with moiré effects
// ============================================================================

class HolographicInterference {
    constructor(options = {}) {
        this.layers = [];
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
        this.mousePos = { x: 0.5, y: 0.5 };
        this.intensity = options.intensity || 0.15;
        this.lineSpacing = options.lineSpacing || 8;

        this.init();
    }

    init() {
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'holographic-interference';
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
            mix-blend-mode: overlay;
            opacity: ${this.intensity};
        `;
        document.body.insertBefore(this.canvas, document.body.firstChild);

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        window.addEventListener('resize', () => this.resize());
        document.addEventListener('mousemove', (e) => {
            this.mousePos.x = e.clientX / window.innerWidth;
            this.mousePos.y = e.clientY / window.innerHeight;
        });

        // Device orientation for mobile
        if (window.DeviceOrientationEvent) {
            window.addEventListener('deviceorientation', (e) => {
                if (e.gamma !== null && e.beta !== null) {
                    this.mousePos.x = 0.5 + (e.gamma / 90) * 0.5;
                    this.mousePos.y = 0.5 + (e.beta / 90) * 0.5;
                }
            });
        }

        this.isActive = true;
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    drawMoireLayer(offsetX, offsetY, angle, color) {
        this.ctx.save();
        this.ctx.translate(this.canvas.width / 2, this.canvas.height / 2);
        this.ctx.rotate(angle);
        this.ctx.translate(-this.canvas.width / 2 + offsetX, -this.canvas.height / 2 + offsetY);

        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1;

        const diagonal = Math.sqrt(this.canvas.width ** 2 + this.canvas.height ** 2);
        const numLines = Math.ceil(diagonal / this.lineSpacing) * 2;

        this.ctx.beginPath();
        for (let i = -numLines; i < numLines; i++) {
            const x = i * this.lineSpacing;
            this.ctx.moveTo(x, -diagonal);
            this.ctx.lineTo(x, diagonal);
        }
        this.ctx.stroke();

        this.ctx.restore();
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const time = performance.now() * 0.0001;
        const parallaxX = (this.mousePos.x - 0.5) * 30;
        const parallaxY = (this.mousePos.y - 0.5) * 30;

        // Layer 1 - Cyan lines
        this.drawMoireLayer(
            parallaxX * 1.5,
            parallaxY * 1.5,
            time + 0.1,
            'rgba(0, 255, 255, 0.5)'
        );

        // Layer 2 - Magenta lines
        this.drawMoireLayer(
            parallaxX * -1.2,
            parallaxY * -1.2,
            -time + 0.2,
            'rgba(255, 0, 255, 0.5)'
        );

        // Layer 3 - Yellow lines
        this.drawMoireLayer(
            parallaxX * 0.8,
            parallaxY * -0.8,
            time * 0.5,
            'rgba(255, 255, 0, 0.3)'
        );
    }

    animate() {
        if (!this.isActive) return;
        this.render();
        requestAnimationFrame(() => this.animate());
    }

    setIntensity(value) {
        this.intensity = value;
        this.canvas.style.opacity = value;
    }

    destroy() {
        this.isActive = false;
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}

// ============================================================================
// GLOBAL EXPERIMENTS MANAGER
// ============================================================================

class GlobalExperiments {
    constructor() {
        this.cursorTrail = null;
        this.audioVisualizer = null;
        this.holographic = null;
        this.config = {
            cursorTrail: true,
            audioVisualizer: false, // Requires user activation
            holographic: true
        };
    }

    init() {
        // Initialize based on config
        if (this.config.cursorTrail) {
            this.enableCursorTrail();
        }
        if (this.config.holographic) {
            this.enableHolographic();
        }

        // Add global controls UI
        this.createControlPanel();

        console.log('✨ Global Experiments initialized');
    }

    enableCursorTrail() {
        if (!this.cursorTrail) {
            this.cursorTrail = new CursorTrailSystem();
        }
    }

    disableCursorTrail() {
        if (this.cursorTrail) {
            this.cursorTrail.destroy();
            this.cursorTrail = null;
        }
    }

    async enableAudioVisualizer() {
        if (!this.audioVisualizer) {
            this.audioVisualizer = new AudioVisualizerSystem();
            await this.audioVisualizer.init();
        }
    }

    disableAudioVisualizer() {
        if (this.audioVisualizer) {
            this.audioVisualizer.destroy();
            this.audioVisualizer = null;
        }
    }

    enableHolographic() {
        if (!this.holographic) {
            this.holographic = new HolographicInterference({ intensity: 0.08 });
        }
    }

    disableHolographic() {
        if (this.holographic) {
            this.holographic.destroy();
            this.holographic = null;
        }
    }

    createControlPanel() {
        const panel = document.createElement('div');
        panel.className = 'global-experiments-panel';
        panel.innerHTML = `
            <button class="toggle-panel-btn" title="Toggle Experiments">✨</button>
            <div class="panel-content">
                <h4>Global Effects</h4>
                <label>
                    <input type="checkbox" id="exp-cursor-trail" ${this.config.cursorTrail ? 'checked' : ''}>
                    Cursor Trail
                </label>
                <label>
                    <input type="checkbox" id="exp-holographic" ${this.config.holographic ? 'checked' : ''}>
                    Holographic
                </label>
                <label>
                    <input type="checkbox" id="exp-audio">
                    Audio React 🎤
                </label>
            </div>
        `;

        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .global-experiments-panel {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 10000;
                font-family: system-ui, sans-serif;
            }
            .toggle-panel-btn {
                width: 48px;
                height: 48px;
                border-radius: 50%;
                border: none;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 20px;
                cursor: pointer;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .toggle-panel-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            .panel-content {
                display: none;
                position: absolute;
                bottom: 60px;
                right: 0;
                background: rgba(20, 20, 35, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 16px;
                min-width: 180px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            }
            .panel-content.open {
                display: block;
                animation: slideUp 0.2s ease-out;
            }
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .panel-content h4 {
                margin: 0 0 12px 0;
                color: #fff;
                font-size: 14px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                padding-bottom: 8px;
            }
            .panel-content label {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #ccc;
                font-size: 13px;
                padding: 6px 0;
                cursor: pointer;
                transition: color 0.2s;
            }
            .panel-content label:hover {
                color: #fff;
            }
            .panel-content input[type="checkbox"] {
                width: 16px;
                height: 16px;
                accent-color: #667eea;
            }
        `;
        document.head.appendChild(style);
        document.body.appendChild(panel);

        // Event handlers
        const toggleBtn = panel.querySelector('.toggle-panel-btn');
        const content = panel.querySelector('.panel-content');

        toggleBtn.addEventListener('click', () => {
            content.classList.toggle('open');
        });

        panel.querySelector('#exp-cursor-trail').addEventListener('change', (e) => {
            if (e.target.checked) this.enableCursorTrail();
            else this.disableCursorTrail();
        });

        panel.querySelector('#exp-holographic').addEventListener('change', (e) => {
            if (e.target.checked) this.enableHolographic();
            else this.disableHolographic();
        });

        panel.querySelector('#exp-audio').addEventListener('change', async (e) => {
            if (e.target.checked) await this.enableAudioVisualizer();
            else this.disableAudioVisualizer();
        });
    }

    destroy() {
        this.disableCursorTrail();
        this.disableAudioVisualizer();
        this.disableHolographic();
    }
}

// Auto-initialize on DOM ready
let globalExperiments = null;

document.addEventListener('DOMContentLoaded', () => {
    globalExperiments = new GlobalExperiments();
    globalExperiments.init();
});

// Export for module usage
export { GlobalExperiments, CursorTrailSystem, AudioVisualizerSystem, HolographicInterference };
export default globalExperiments;
