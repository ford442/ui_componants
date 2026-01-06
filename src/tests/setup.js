import { vi } from 'vitest';

// 1. Mock WebGPU API
// --------------------------------------------------------------------------
const mockGPU = {
    requestAdapter: vi.fn(() => Promise.resolve({
        requestDevice: vi.fn(() => Promise.resolve({
            createShaderModule: vi.fn(() => ({})),
            createRenderPipeline: vi.fn(() => ({
                getBindGroupLayout: vi.fn(() => ({}))
            })),
            createComputePipeline: vi.fn(() => ({
                getBindGroupLayout: vi.fn(() => ({}))
            })),
            createBindGroup: vi.fn(() => ({})),
            createBindGroupLayout: vi.fn(() => ({})),
            createPipelineLayout: vi.fn(() => ({})),
            createBuffer: vi.fn(() => ({
                destroy: vi.fn(),
                getMappedRange: vi.fn(() => new Float32Array(100).buffer),
                unmap: vi.fn(),
                mapAsync: vi.fn(() => Promise.resolve())
            })),
            createCommandEncoder: vi.fn(() => ({
                beginRenderPass: vi.fn(() => ({
                    setPipeline: vi.fn(),
                    setBindGroup: vi.fn(),
                    draw: vi.fn(),
                    end: vi.fn()
                })),
                beginComputePass: vi.fn(() => ({
                    setPipeline: vi.fn(),
                    setBindGroup: vi.fn(),
                    dispatchWorkgroups: vi.fn(),
                    end: vi.fn()
                })),
                finish: vi.fn()
            })),
            queue: {
                writeBuffer: vi.fn(),
                submit: vi.fn()
            }
        }))
    })),
    getPreferredCanvasFormat: vi.fn(() => 'rgba8unorm')
};

// Shim commonly-used WebGPU enums so tests don't hit ReferenceErrors
global.GPUBufferUsage = {
    MAP_READ: 1 << 0,
    MAP_WRITE: 1 << 1,
    COPY_SRC: 1 << 2,
    COPY_DST: 1 << 3,
    INDEX: 1 << 4,
    VERTEX: 1 << 5,
    UNIFORM: 1 << 6,
    STORAGE: 1 << 7,
    INDIRECT: 1 << 8,
    QUERY_RESOLVE: 1 << 9
};

global.GPUTextureUsage = {
    COPY_SRC: 1 << 0,
    COPY_DST: 1 << 1,
    TEXTURE_BINDING: 1 << 2,
    STORAGE_BINDING: 1 << 3,
    RENDER_ATTACHMENT: 1 << 4
};

global.GPUShaderStage = {
    VERTEX: 1 << 0,
    FRAGMENT: 1 << 1,
    COMPUTE: 1 << 2
};

// Assign to global navigator
Object.defineProperty(global.navigator, 'gpu', {
    value: mockGPU,
    writable: true
});

// 2. Mock Canvas getContext
// --------------------------------------------------------------------------
HTMLCanvasElement.prototype.getContext = vi.fn((type) => {
    // Common stubs for 2D/WebGL
    const commonStubs = {
        canvas: { width: 100, height: 100 },
        fillRect: vi.fn(),
        clearRect: vi.fn(),
        getImageData: vi.fn(() => ({ data: [] })),
        putImageData: vi.fn(),
        createImageData: vi.fn(() => []),
        setTransform: vi.fn(),
        drawImage: vi.fn(),
        save: vi.fn(),
        restore: vi.fn(),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        closePath: vi.fn(),
        stroke: vi.fn(),
        translate: vi.fn(),
        scale: vi.fn(),
        rotate: vi.fn(),
        arc: vi.fn(),
        fill: vi.fn(),
        measureText: vi.fn(() => ({ width: 0 })),
        fillText: vi.fn(),
        strokeText: vi.fn(),
        transform: vi.fn(),
        rect: vi.fn(),
        clip: vi.fn(),
        // WebGL specific
        createShader: vi.fn(),
        shaderSource: vi.fn(),
        compileShader: vi.fn(),
        createProgram: vi.fn(),
        attachShader: vi.fn(),
        linkProgram: vi.fn(),
        useProgram: vi.fn(),
        getShaderParameter: vi.fn(() => true),
        getProgramParameter: vi.fn(() => true),
        createBuffer: vi.fn(),
        bindBuffer: vi.fn(),
        bufferData: vi.fn(),
        enableVertexAttribArray: vi.fn(),
        vertexAttribPointer: vi.fn(),
        clearColor: vi.fn(),
        clear: vi.fn(),
        drawArrays: vi.fn(),
        viewport: vi.fn(),
        uniform1f: vi.fn(),
        uniform2f: vi.fn(),
        uniform3f: vi.fn(), // Added for your LEDButton
        uniform3fv: vi.fn(),
        getUniformLocation: vi.fn(),
        getAttribLocation: vi.fn(),
        enable: vi.fn(),
        blendFunc: vi.fn()
    };

    // Return WebGPU specific stubs if requested
    if (type === 'webgpu') {
        return {
            ...commonStubs,
            configure: vi.fn(),
            getCurrentTexture: vi.fn(() => ({
                createView: vi.fn(() => ({}))
            }))
        };
    }

    return commonStubs;
});

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
}));

// Mock requestAnimationFrame
global.requestAnimationFrame = vi.fn((cb) => setTimeout(cb, 16));
global.cancelAnimationFrame = vi.fn((id) => clearTimeout(id));
