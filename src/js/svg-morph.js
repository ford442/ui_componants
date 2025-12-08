/**
 * Elastic SVG Morphing
 * Morphs between two SVG paths with spring physics
 */

class SVGMorph {
    constructor(svgElement, options = {}) {
        this.svg = svgElement;
        this.options = {
            tension: options.tension || 170,
            friction: options.friction || 26,
            mass: options.mass || 1,
            duration: options.duration || 1000,
            ...options
        };

        this.pathElement = null;
        this.currentPath = '';
        this.targetPath = '';
        this.animationId = null;
        this.springState = { position: 0, velocity: 0 };
        this.startTime = 0;
    }

    /**
     * Morph from one path to another
     * @param {string} fromPath - Starting SVG path
     * @param {string} toPath - Target SVG path
     * @param {SVGPathElement} pathElement - The path element to animate
     */
    morph(fromPath, toPath, pathElement) {
        this.pathElement = pathElement;
        this.currentPath = fromPath;
        this.targetPath = toPath;

        // Parse paths into point arrays
        this.fromPoints = this.parsePath(fromPath);
        this.toPoints = this.parsePath(toPath);

        // Ensure both paths have the same number of points
        this.normalizePointCounts();

        // Reset spring
        this.springState = { position: 0, velocity: 0 };
        this.startTime = Date.now();

        // Cancel any existing animation
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        // Start animation
        this.animate();
    }

    /**
     * Simple path parser - supports M, L, C, Z commands
     */
    parsePath(pathString) {
        const points = [];
        const commands = pathString.match(/[MLCZmlcz][^MLCZmlcz]*/g) || [];

        for (let cmd of commands) {
            const type = cmd[0];
            const coords = cmd.slice(1).trim().split(/[\s,]+/).map(Number);

            if (type === 'M' || type === 'm') {
                points.push({ type: 'M', x: coords[0], y: coords[1] });
            } else if (type === 'L' || type === 'l') {
                points.push({ type: 'L', x: coords[0], y: coords[1] });
            } else if (type === 'C' || type === 'c') {
                points.push({
                    type: 'C',
                    x1: coords[0], y1: coords[1],
                    x2: coords[2], y2: coords[3],
                    x: coords[4], y: coords[5]
                });
            } else if (type === 'Z' || type === 'z') {
                points.push({ type: 'Z' });
            }
        }

        return points;
    }

    /**
     * Ensure both path point arrays have same length
     */
    normalizePointCounts() {
        const maxLen = Math.max(this.fromPoints.length, this.toPoints.length);

        while (this.fromPoints.length < maxLen) {
            const lastPoint = this.fromPoints[this.fromPoints.length - 1];
            this.fromPoints.push({ ...lastPoint });
        }

        while (this.toPoints.length < maxLen) {
            const lastPoint = this.toPoints[this.toPoints.length - 1];
            this.toPoints.push({ ...lastPoint });
        }
    }

    /**
     * Spring physics simulation
     */
    updateSpring(deltaTime) {
        const { tension, friction, mass } = this.options;
        const target = 1;

        // Spring force: F = -k * x
        const springForce = -tension * (this.springState.position - target);

        // Damping force: F = -c * v
        const dampingForce = -friction * this.springState.velocity;

        // Acceleration: a = F / m
        const acceleration = (springForce + dampingForce) / mass;

        // Update velocity and position
        this.springState.velocity += acceleration * deltaTime;
        this.springState.position += this.springState.velocity * deltaTime;

        // Check if spring has settled
        const settled = Math.abs(target - this.springState.position) < 0.001 &&
            Math.abs(this.springState.velocity) < 0.001;

        return settled;
    }

    /**
     * Interpolate between two points
     */
    interpolatePoint(from, to, progress) {
        const result = { type: from.type || to.type };

        if (result.type === 'M' || result.type === 'L') {
            result.x = from.x + (to.x - from.x) * progress;
            result.y = from.y + (to.y - from.y) * progress;
        } else if (result.type === 'C') {
            result.x1 = from.x1 + (to.x1 - from.x1) * progress;
            result.y1 = from.y1 + (to.y1 - from.y1) * progress;
            result.x2 = from.x2 + (to.x2 - from.x2) * progress;
            result.y2 = from.y2 + (to.y2 - from.y2) * progress;
            result.x = from.x + (to.x - from.x) * progress;
            result.y = from.y + (to.y - from.y) * progress;
        }

        return result;
    }

    /**
     * Convert points array back to path string
     */
    pointsToPath(points) {
        return points.map(point => {
            if (point.type === 'M') {
                return `M ${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
            } else if (point.type === 'L') {
                return `L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
            } else if (point.type === 'C') {
                return `C ${point.x1.toFixed(2)} ${point.y1.toFixed(2)}, ${point.x2.toFixed(2)} ${point.y2.toFixed(2)}, ${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
            } else if (point.type === 'Z') {
                return 'Z';
            }
            return '';
        }).join(' ');
    }

    animate() {
        const deltaTime = 0.016; // ~60fps

        // Update spring physics
        const settled = this.updateSpring(deltaTime);

        // Clamp progress between 0 and 1.2 for overshoot effect
        const progress = Math.max(0, Math.min(1.2, this.springState.position));

        // Interpolate all points
        const interpolatedPoints = this.fromPoints.map((fromPoint, i) => {
            const toPoint = this.toPoints[i];
            return this.interpolatePoint(fromPoint, toPoint, progress);
        });

        // Convert to path string and update element
        const newPath = this.pointsToPath(interpolatedPoints);
        this.pathElement.setAttribute('d', newPath);

        // Continue animation if not settled
        if (!settled && Date.now() - this.startTime < this.options.duration * 2) {
            this.animationId = requestAnimationFrame(() => this.animate());
        } else {
            // Ensure we end exactly at target
            this.pathElement.setAttribute('d', this.targetPath);
            this.animationId = null;
        }
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Helper function to create morphing buttons/icons
class MorphingButton {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            width: options.width || 100,
            height: options.height || 100,
            shape1: options.shape1 || 'circle',
            shape2: options.shape2 || 'square',
            fillColor: options.fillColor || '#00ff88',
            ...options
        };

        this.isShape1 = true;
        this.morpher = null;

        this.init();
    }

    init() {
        // Create SVG element
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', this.options.width);
        svg.setAttribute('height', this.options.height);
        svg.setAttribute('viewBox', `0 0 ${this.options.width} ${this.options.height}`);
        svg.style.cursor = 'pointer';

        // Create path element
        this.path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this.path.setAttribute('fill', this.options.fillColor);
        this.path.setAttribute('d', this.getShapePath(this.options.shape1));
        svg.appendChild(this.path);

        this.container.appendChild(svg);

        // Create morpher
        this.morpher = new SVGMorph(svg);

        // Add click handler
        svg.addEventListener('click', () => this.toggle());
    }

    getShapePath(shape) {
        const { width, height } = this.options;
        const cx = width / 2;
        const cy = height / 2;
        const r = Math.min(width, height) / 2 - 10;

        if (shape === 'circle') {
            // Circle path
            return `M ${cx - r} ${cy} 
                    A ${r} ${r} 0 0 1 ${cx + r} ${cy}
                    A ${r} ${r} 0 0 1 ${cx - r} ${cy} Z`;
        } else if (shape === 'square') {
            // Rounded square path
            const size = r * 1.4;
            const corner = size * 0.2;
            return `M ${cx - size + corner} ${cy - size}
                    L ${cx + size - corner} ${cy - size}
                    Q ${cx + size} ${cy - size} ${cx + size} ${cy - size + corner}
                    L ${cx + size} ${cy + size - corner}
                    Q ${cx + size} ${cy + size} ${cx + size - corner} ${cy + size}
                    L ${cx - size + corner} ${cy + size}
                    Q ${cx - size} ${cy + size} ${cx - size} ${cy + size - corner}
                    L ${cx - size} ${cy - size + corner}
                    Q ${cx - size} ${cy - size} ${cx - size + corner} ${cy - size} Z`;
        }

        return '';
    }

    toggle() {
        const fromShape = this.isShape1 ? this.options.shape1 : this.options.shape2;
        const toShape = this.isShape1 ? this.options.shape2 : this.options.shape1;

        const fromPath = this.getShapePath(fromShape);
        const toPath = this.getShapePath(toShape);

        this.morpher.morph(fromPath, toPath, this.path);
        this.isShape1 = !this.isShape1;
    }

    destroy() {
        if (this.morpher) {
            this.morpher.destroy();
        }
    }
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.SVGMorph = SVGMorph;
    window.MorphingButton = MorphingButton;
}
