/**
 * Magnetic Particle Field
 * Creates an interactive particle system with magnetic cursor attraction
 */

class MagneticParticleField {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            particleCount: options.particleCount || 2000,
            particleSize: options.particleSize || 2,
            particleColor: options.particleColor || 'rgba(0, 255, 136, 0.6)',
            lineColor: options.lineColor || 'rgba(0, 170, 255, 0.2)',
            maxLineDistance: options.maxLineDistance || 100,
            magneticForce: options.magneticForce || 0.03,
            friction: options.friction || 0.98,
            constellationMode: options.constellationMode !== false,
            ...options
        };

        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.mouse = { x: 0, y: 0, active: false };
        this.animationId = null;

        this.init();
    }

    init() {
        // Style canvas
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        `;

        this.container.appendChild(this.canvas);
        this.resize();

        // Create particles
        this.createParticles();

        // Event listeners
        window.addEventListener('resize', () => this.resize());
        window.addEventListener('mousemove', (e) => this.onMouseMove(e));
        window.addEventListener('mouseleave', () => this.onMouseLeave());

        // Start animation
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    createParticles() {
        this.particles = [];
        for (let i = 0; i < this.options.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                prevX: 0,
                prevY: 0
            });
        }
    }

    onMouseMove(e) {
        this.mouse.x = e.clientX;
        this.mouse.y = e.clientY;
        this.mouse.active = true;
    }

    onMouseLeave() {
        this.mouse.active = false;
    }

    updateParticles() {
        const { magneticForce, friction } = this.options;

        for (let particle of this.particles) {
            // Store previous position (Verlet integration)
            particle.prevX = particle.x;
            particle.prevY = particle.y;

            // Apply magnetic force if mouse is active
            if (this.mouse.active) {
                const dx = this.mouse.x - particle.x;
                const dy = this.mouse.y - particle.y;
                const distSq = dx * dx + dy * dy;
                const dist = Math.sqrt(distSq);

                if (dist > 0 && dist < 300) {
                    const force = magneticForce / (distSq * 0.01 + 1);
                    particle.vx += (dx / dist) * force;
                    particle.vy += (dy / dist) * force;
                }
            }

            // Apply friction
            particle.vx *= friction;
            particle.vy *= friction;

            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Wrap around edges
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            if (particle.y > this.canvas.height) particle.y = 0;
        }
    }

    drawParticles() {
        const { particleSize, particleColor, lineColor, maxLineDistance, constellationMode } = this.options;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw constellation lines first (so they're behind particles)
        if (constellationMode) {
            this.ctx.strokeStyle = lineColor;
            this.ctx.lineWidth = 1;

            for (let i = 0; i < this.particles.length; i++) {
                for (let j = i + 1; j < this.particles.length; j++) {
                    const dx = this.particles[i].x - this.particles[j].x;
                    const dy = this.particles[i].y - this.particles[j].y;
                    const distSq = dx * dx + dy * dy;

                    if (distSq < maxLineDistance * maxLineDistance) {
                        const alpha = 1 - Math.sqrt(distSq) / maxLineDistance;
                        this.ctx.globalAlpha = alpha * 0.3;
                        this.ctx.beginPath();
                        this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
                        this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
                        this.ctx.stroke();
                    }
                }
            }
        }

        // Draw particles
        this.ctx.globalAlpha = 1;
        this.ctx.fillStyle = particleColor;

        for (let particle of this.particles) {
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particleSize, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    animate() {
        this.updateParticles();
        this.drawParticles();
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        window.removeEventListener('resize', () => this.resize());
        window.removeEventListener('mousemove', (e) => this.onMouseMove(e));
        window.removeEventListener('mouseleave', () => this.onMouseLeave());
        this.canvas.remove();
    }
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.MagneticParticleField = MagneticParticleField;
}
