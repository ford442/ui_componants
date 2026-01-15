/**
 * rendering-manager.js
 * 
 * Manages a single WebGL2 context to be shared across multiple components on a page,
 * preventing the "Too many active WebGL contexts" error.
 */

class WebGL2Manager {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
        this.renderables = [];
        this.programs = new Map(); // Cache compiled shader programs

        if (!this.gl) {
            console.error("WebGL2 not supported!");
            return;
        }

        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        this.quadBuffer = this.createQuadBuffer();

        this.lastTime = 0;
        this.animationFrameId = null;
        this.render = this.render.bind(this);
    }

    createQuadBuffer() {
        const gl = this.gl;
        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        return buffer;
    }

    createProgram(vertexSource, fragmentSource) {
        const gl = this.gl;
        const key = vertexSource + fragmentSource;
        if (this.programs.has(key)) {
            return this.programs.get(key);
        }

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

        this.programs.set(key, program);
        return program;
    }

    addRenderable(renderable) {
        this.renderables.push(renderable);
        if (!this.animationFrameId) {
            this.start();
        }
    }

    start() {
        this.lastTime = performance.now();
        this.animationFrameId = requestAnimationFrame(this.render);
    }

    stop() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    render(timestamp) {
        const gl = this.gl;
        const deltaTime = (timestamp - this.lastTime) * 0.001;
        this.lastTime = timestamp;

        const canvasRect = this.canvas.getBoundingClientRect();
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        for (const renderable of this.renderables) {
            // Check if the element is visible
            const rect = renderable.element.getBoundingClientRect();
            if (rect.bottom < canvasRect.top || rect.top > canvasRect.bottom || rect.right < canvasRect.left || rect.left > canvasRect.right) {
                continue; // Skip rendering if off-screen
            }
            
            gl.useProgram(renderable.program);
            
            // Basic quad setup
            const positionLoc = gl.getAttribLocation(renderable.program, 'a_position');
            gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
            gl.enableVertexAttribArray(positionLoc);
            gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

            if (renderable.customDraw) {
                // For components that need to do their own multi-viewport drawing, etc.
                renderable.customDraw(gl, timestamp * 0.001, deltaTime);
            } else if (renderable.uniformsCallback) {
                // The standard path for simple components
                
                // Set uniforms
                renderable.uniformsCallback(gl, timestamp * 0.001, deltaTime);
                
                // Set viewport and draw
                gl.viewport(rect.left - canvasRect.left, canvasRect.height - rect.bottom, rect.width, rect.height);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            }
        }

        this.animationFrameId = requestAnimationFrame(this.render);
    }
}

window.WebGL2Manager = WebGL2Manager;
