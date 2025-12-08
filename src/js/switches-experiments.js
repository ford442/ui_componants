// switches-experiments.js – Handles the new visual switch experiments

// Utility to create a switch element with given class and id
function createSwitch(containerId, switchClass) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const el = document.createElement('div');
    el.className = switchClass;
    // default data attribute
    if (switchClass === 'neon-switch') el.dataset.on = 'false';
    if (switchClass === 'portal-toggle') el.dataset.open = 'false';
    if (switchClass === 'circuit-breaker') el.dataset.tripped = 'false';
    if (switchClass === 'bio-mech-switch') el.dataset.active = 'false';
    container.appendChild(el);

    // Click handler to toggle state
    el.addEventListener('click', () => {
        switch (switchClass) {
            case 'neon-switch':
                el.dataset.on = el.dataset.on === 'true' ? 'false' : 'true';
                break;
            case 'portal-toggle':
                el.dataset.open = el.dataset.open === 'true' ? 'false' : 'true';
                break;
            case 'circuit-breaker':
                el.dataset.tripped = el.dataset.tripped === 'true' ? 'false' : 'true';
                break;
            case 'bio-mech-switch':
                el.dataset.active = el.dataset.active === 'true' ? 'false' : 'true';
                break;
        }
    });
}

// Initialize all experiments when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Helper to initialize shaders safely
    const initShader = (container, ShaderClass) => {
        if (container && window[ShaderClass]) {
            // Check if shader instance already exists to avoid duplicates
            if (!container._shaderInstance) {
                container._shaderInstance = new window[ShaderClass](container);
            }
        }
    };

    // Portal Toggle integration
    const portalContainer = document.getElementById('portal-toggle');
    if (portalContainer) {
        createSwitch('portal-toggle', 'portal-toggle');
        // Initialize shader immediately for the vortex effect
        initShader(document.querySelector('#portal-toggle .portal-toggle'), 'PortalToggleShader');
    }

    // Circuit Breaker integration
    const circuitContainer = document.getElementById('circuit-breaker');
    if (circuitContainer) {
        createSwitch('circuit-breaker', 'circuit-breaker');
        // Initialize shader for the arc effect
        initShader(document.querySelector('#circuit-breaker .circuit-breaker'), 'CircuitBreakerShader');
    }

    createSwitch('neon-switches', 'neon-switch');
    createSwitch('bio-mechanical-switch', 'bio-mech-switch');
});
