import { describe, it, expect, beforeEach, vi } from 'vitest';

// Ensure setup file runs and environment is clean
beforeEach(() => {
    document.body.innerHTML = '';
    vi.resetModules();
});

describe('WebGPU Particle System', () => {
    it('allocates particle and uniform buffers with correct sizes', async () => {
        // Create a canvas in the DOM
        document.body.innerHTML = '<canvas id="gpu-canvas"></canvas>';
        const canvas = document.getElementById('gpu-canvas');

        // Import main to attach UIComponents to window
        await import('../js/main.js');
        const WebGPUParticleSystem = window.UIComponents.WebGPUParticleSystem;

        const particleCount = 20000; // large number to exercise sizing
        const sys = new WebGPUParticleSystem(canvas, { particleCount });

        const initResult = await sys.init();
        expect(initResult).toBe(true);

        const device = sys.device;
        // Ensure createBuffer was called at least twice (particles + uniform)
        expect(device.createBuffer).toHaveBeenCalled();
        expect(device.createBuffer.mock.calls.length).toBeGreaterThanOrEqual(2);

        // Particle buffer should be the first createBuffer call
        const particleCall = device.createBuffer.mock.calls[0][0];
        const expectedParticleBytes = particleCount * 8 * Float32Array.BYTES_PER_ELEMENT; // 8 floats per particle
        expect(particleCall.size).toBe(expectedParticleBytes);

        // Uniform buffer should be created (size 64)
        const uniformCall = device.createBuffer.mock.calls.find(call => call[0].size === 64);
        expect(uniformCall).toBeTruthy();

        // Also validate the returned particleBuffer has mapped range of expected size
        const mappedRange = sys.particleBuffer.getMappedRange();
        expect(mappedRange.byteLength).toBe(expectedParticleBytes);

        // Cleanup
        sys.destroy();
    });

    it('handles small particle counts correctly', async () => {
        document.body.innerHTML = '<canvas id="gpu-canvas"></canvas>';
        const canvas = document.getElementById('gpu-canvas');

        await import('../js/main.js');
        const WebGPUParticleSystem = window.UIComponents.WebGPUParticleSystem;

        const particleCount = 10;
        const sys = new WebGPUParticleSystem(canvas, { particleCount });

        const initResult = await sys.init();
        expect(initResult).toBe(true);

        const device = sys.device;
        const particleCall = device.createBuffer.mock.calls[0][0];
        const expectedParticleBytes = particleCount * 8 * Float32Array.BYTES_PER_ELEMENT;
        expect(particleCall.size).toBe(expectedParticleBytes);

        const mappedRange = sys.particleBuffer.getMappedRange();
        expect(mappedRange.byteLength).toBe(expectedParticleBytes);

        sys.destroy();
    });
});