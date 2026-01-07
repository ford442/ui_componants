/**
 * Tetris Falling Blocks Experiment
 * A visual simulation of falling blocks using Canvas 2D / WebGL2
 * Focuses on the "rain of blocks" visual aesthetic.
 */

export class TetrisExperiment {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        this.options = {
            columnCount: options.columnCount || 10,
            speed: options.speed || 1,
            colors: options.colors || [
                '#00ff88', '#00aaff', '#ff0088', '#ffaa00', '#8800ff', '#ff0000', '#ffff00'
            ],
            ...options
        };

        this.canvas = document.createElement('canvas');
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.display = 'block';
        this.container.appendChild(this.canvas);

        this.ctx = this.canvas.getContext('2d');

        this.grid = [];
        this.fallingPieces = [];
        this.lastTime = 0;
        this.dropTimer = 0;

        // Bind methods
        this.resize = this.resize.bind(this);
        this.animate = this.animate.bind(this);

        // Initial setup
        this.resize();
        window.addEventListener('resize', this.resize);

        requestAnimationFrame(this.animate);
    }

    initGrid() {
        // Ensure width/height are set
        if (!this.width || !this.height) return;

        this.cols = Math.floor(this.width / 30); // 30px block size approx
        this.rows = Math.floor(this.height / 30);
        this.blockSize = this.width / this.cols;

        // Reset grid
        this.grid = new Array(this.cols).fill(null).map(() => new Array(this.rows).fill(null));
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.container.getBoundingClientRect();

        this.width = rect.width;
        this.height = rect.height;

        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.ctx.scale(dpr, dpr);

        this.initGrid();
    }

    spawnPiece() {
        if (!this.cols) return;

        const col = Math.floor(Math.random() * this.cols);
        const type = Math.floor(Math.random() * 7);
        const color = this.options.colors[type];

        this.fallingPieces.push({
            x: col,
            y: -2,
            color: color,
            velocity: 0.1 + Math.random() * 0.2 * this.options.speed,
            landed: false
        });
    }

    update(dt) {
        if (!this.cols || !this.rows) return;

        // Spawn
        if (Math.random() < 0.1 * this.options.speed) {
            this.spawnPiece();
        }

        // Update falling
        for (let i = this.fallingPieces.length - 1; i >= 0; i--) {
            const p = this.fallingPieces[i];

            if (p.landed) continue;

            p.y += p.velocity;

            const gridY = Math.floor(p.y + 1);

            // Check collision
            if (gridY >= this.rows || (gridY >= 0 && this.grid[p.x] && this.grid[p.x][gridY])) {
                // Landed
                p.landed = true;
                p.y = Math.floor(p.y);

                // Add to grid
                if (gridY >= 0 && gridY < this.rows && this.grid[p.x]) {
                    this.grid[p.x][Math.floor(p.y)] = p.color;
                    // Also fill the next one if velocity pushed it far? simplified
                    if (gridY < this.rows) {
                         this.grid[p.x][gridY] = p.color;
                    }
                }

                this.fallingPieces.splice(i, 1);
            }
        }

        // Decay
        if (Math.random() < 0.05) {
            this.decayGrid();
        }
    }

    decayGrid() {
        if (!this.grid || !this.cols) return;

        for (let x = 0; x < this.cols; x++) {
            if (this.grid[x] && this.grid[x][this.rows - 1] && Math.random() < 0.1) {
                // Shift down column
                for (let y = this.rows - 1; y > 0; y--) {
                    this.grid[x][y] = this.grid[x][y-1];
                }
                this.grid[x][0] = null;
            }
        }
    }

    draw() {
        this.ctx.fillStyle = '#050510';
        this.ctx.fillRect(0, 0, this.width, this.height);

        if (!this.grid) return;

        // Draw Grid Blocks
        for (let x = 0; x < this.cols; x++) {
            for (let y = 0; y < this.rows; y++) {
                if (this.grid[x] && this.grid[x][y]) {
                    this.drawBlock(x, y, this.grid[x][y]);
                }
            }
        }

        // Draw Falling Blocks
        for (const p of this.fallingPieces) {
            this.drawBlock(p.x, p.y, p.color);
        }
    }

    drawBlock(gx, gy, color) {
        const x = gx * this.blockSize;
        const y = gy * this.blockSize;
        const s = this.blockSize - 2;

        this.ctx.fillStyle = color;
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = color;
        this.ctx.fillRect(x + 1, y + 1, s, s);

        this.ctx.fillStyle = 'rgba(255,255,255,0.3)';
        this.ctx.shadowBlur = 0;
        this.ctx.fillRect(x + 1, y + 1, s, s/3);
    }

    animate(time) {
        const dt = time - this.lastTime;
        this.lastTime = time;

        this.update(dt);
        this.draw();

        requestAnimationFrame(this.animate);
    }

    destroy() {
        window.removeEventListener('resize', this.resize);
        this.container.innerHTML = '';
    }
}
