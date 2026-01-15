/**
 * webgpu-manager.js
 * 
 * Manages a single WebGPU device and context to be shared across multiple components.
 */

class WebGPUManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.adapter = null;
        this.device = null;
        this.context = null;
        this.format = null;

        this.renderables = [];
        this.lastTime = 0;
        this.animationFrameId = null;
        this.render = this.render.bind(this);
    }

    async init() {
        if (!navigator.gpu) {
            console.error("WebGPU not supported!");
            return false;
        }

        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            console.error("No WebGPU adapter found!");
            return false;
        }

        this.device = await this.adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied',
        });

        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        window.addEventListener('resize', () => {
            this.canvas.width = this.canvas.clientWidth;
            this.canvas.height = this.canvas.clientHeight;
        });

        return true;
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
        const deltaTime = (timestamp - this.lastTime) * 0.001;
        this.lastTime = timestamp;

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const renderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        };
        
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

        for (const renderable of this.renderables) {
            const rect = renderable.element.getBoundingClientRect();
            const canvasRect = this.canvas.getBoundingClientRect();

            if (rect.bottom < canvasRect.top || rect.top > canvasRect.bottom || rect.right < canvasRect.left || rect.left > canvasRect.right) {
                continue;
            }
            
            passEncoder.setViewport(
                rect.left - canvasRect.left,
                canvasRect.height - rect.bottom,
                rect.width,
                rect.height,
                0,
                1
            );
            
            if (renderable.render) {
                renderable.render(passEncoder, timestamp * 0.001, deltaTime);
            }
        }
        
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);

        this.animationFrameId = requestAnimationFrame(this.render);
    }
}

window.WebGPUManager = WebGPUManager;
