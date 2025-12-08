// Liquid Ripple Button
// Adds a realistic water ripple effect to buttons on click using Canvas 2D.

class LiquidRipple {
    constructor(button) {
        this.button = button;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.ripples = [];

        this.init();
        this.animate = this.animate.bind(this);
    }

    init() {
        this.button.style.position = 'relative';
        this.button.style.overflow = 'hidden';

        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';

        this.button.appendChild(this.canvas);

        this.resize();
        window.addEventListener('resize', () => this.resize());

        this.button.addEventListener('mousedown', (e) => this.addRipple(e));

        requestAnimationFrame(this.animate);
    }

    resize() {
        const rect = this.button.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    addRipple(e) {
        const rect = this.button.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        this.ripples.push({
            x: x,
            y: y,
            radius: 0,
            alpha: 0.5,
            speed: 2 + Math.random() * 2
        });
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        for (let i = 0; i < this.ripples.length; i++) {
            const ripple = this.ripples[i];

            ripple.radius += ripple.speed;
            ripple.alpha -= 0.01;

            if (ripple.alpha <= 0) {
                this.ripples.splice(i, 1);
                i--;
                continue;
            }

            this.ctx.beginPath();
            this.ctx.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(255, 255, 255, ${ripple.alpha})`;
            this.ctx.fill();
        }

        requestAnimationFrame(this.animate);
    }
}

// Initialize on specific buttons
document.addEventListener('DOMContentLoaded', () => {
    const liquidButtons = document.querySelectorAll('.liquid-btn');
    liquidButtons.forEach(btn => new LiquidRipple(btn));
});
