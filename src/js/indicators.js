/**
 * Indicators Page JavaScript
 * Creates and manages LED indicators, meters, and displays with layered effects
 */

document.addEventListener('DOMContentLoaded', () => {
    initLEDIndicators();
    initVUMeters();
    initGaugeMeters();
    initSevenSegment();
    initStatusDashboard();
    initRingMeters();
    initOscilloscope();
    initSpectrumAnalyzer();
    initLEDMatrix();
    initButtonPanel();
    initMultiStateIndicators();
    initBioMeters();

    // WebGPU Advanced Examples
    checkWebGPUSupport().then(supported => {
        if (supported) {
            initHolographicVU();
            initFluidMeter();
            initPlasmaGlobe();
            initWaveformSynth();
            initCrystallinePanel();
            initDataVisualization();
        } else {
            document.getElementById('webgpu-warning')?.setAttribute('style', 'display: block;');
        }
    });
});

/**
 * Initialize LED indicators
 */
function initLEDIndicators() {
    const container = document.getElementById('led-indicators');
    if (!container) return;

    const leds = [
        { color: 'red', state: 'on' },
        { color: 'green', state: 'on blink' },
        { color: 'blue', state: 'on' },
        { color: 'yellow', state: '' },
        { color: 'orange', state: 'on' }
    ];

    leds.forEach(led => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem;';

        const indicator = document.createElement('div');
        indicator.className = `led ${led.color} ${led.state}`;

        const label = document.createElement('div');
        label.textContent = led.color.charAt(0).toUpperCase() + led.color.slice(1);
        label.style.cssText = 'font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase;';

        wrapper.appendChild(indicator);
        wrapper.appendChild(label);
        container.appendChild(wrapper);

        // Click to toggle
        indicator.addEventListener('click', () => {
            indicator.classList.toggle('on');
        });
    });
}

/**
 * Initialize VU Meters
 */
function initVUMeters() {
    const container = document.getElementById('vu-meters');
    if (!container) return;

    const createVUMeter = (label, numBars = 10) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'vu-meter';

        const barContainer = document.createElement('div');
        barContainer.className = 'vu-bar-container';

        const bars = [];
        for (let i = 0; i < numBars; i++) {
            const bar = document.createElement('div');
            bar.className = 'vu-bar';

            const fill = document.createElement('div');
            fill.className = 'vu-bar-fill';
            fill.style.height = '0%';

            bar.appendChild(fill);
            barContainer.appendChild(bar);
            bars.push(fill);
        }

        const labelEl = document.createElement('div');
        labelEl.className = 'vu-label';
        labelEl.textContent = label;

        wrapper.appendChild(barContainer);
        wrapper.appendChild(labelEl);
        container.appendChild(wrapper);

        // Animate bars
        const animateBars = () => {
            bars.forEach((fill, i) => {
                const delay = i * 50;
                setTimeout(() => {
                    const height = Math.random() * 100;
                    fill.style.height = `${height}%`;
                }, delay);
            });
        };

        setInterval(animateBars, 100);
    };

    createVUMeter('Left');
    createVUMeter('Right');
}

/**
 * Initialize Gauge Meters
 */
function initGaugeMeters() {
    const container = document.getElementById('gauge-meters');
    if (!container) return;

    const createGauge = (label, color) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'gauge-meter';

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 120 80');

        // Background arc
        const bgArc = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        bgArc.setAttribute('d', 'M 20 70 A 50 50 0 0 1 100 70');
        bgArc.setAttribute('fill', 'none');
        bgArc.setAttribute('stroke', '#333');
        bgArc.setAttribute('stroke-width', '8');
        bgArc.setAttribute('stroke-linecap', 'round');
        svg.appendChild(bgArc);

        // Value arc
        const valueArc = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        valueArc.setAttribute('d', 'M 20 70 A 50 50 0 0 1 100 70');
        valueArc.setAttribute('fill', 'none');
        valueArc.setAttribute('stroke', color);
        valueArc.setAttribute('stroke-width', '8');
        valueArc.setAttribute('stroke-linecap', 'round');
        valueArc.setAttribute('stroke-dasharray', '0 200');
        svg.appendChild(valueArc);

        // Needle
        const needle = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        needle.setAttribute('x1', '60');
        needle.setAttribute('y1', '70');
        needle.setAttribute('x2', '60');
        needle.setAttribute('y2', '25');
        needle.setAttribute('stroke', color);
        needle.setAttribute('stroke-width', '2');
        needle.setAttribute('stroke-linecap', 'round');
        needle.style.transformOrigin = '60px 70px';
        needle.style.transform = 'rotate(-90deg)';
        needle.style.transition = 'transform 0.3s ease';
        svg.appendChild(needle);

        // Center dot
        const centerDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        centerDot.setAttribute('cx', '60');
        centerDot.setAttribute('cy', '70');
        centerDot.setAttribute('r', '5');
        centerDot.setAttribute('fill', '#444');
        svg.appendChild(centerDot);

        const labelEl = document.createElement('div');
        labelEl.className = 'gauge-label';
        labelEl.textContent = label;

        wrapper.appendChild(svg);
        wrapper.appendChild(labelEl);
        container.appendChild(wrapper);

        // Animate gauge
        const arcLength = 157; // Approximate arc length

        const animateGauge = () => {
            const value = Math.random();
            const rotation = -90 + value * 180;
            needle.style.transform = `rotate(${rotation}deg)`;
            valueArc.setAttribute('stroke-dasharray', `${value * arcLength} 200`);
        };

        setInterval(animateGauge, 1000);
        animateGauge();
    };

    createGauge('Speed', '#00ff88');
    createGauge('Temp', '#ff8800');
}

/**
 * Initialize Seven Segment Display
 */
function initSevenSegment() {
    const container = document.getElementById('seven-segment');
    if (!container) return;

    const display = document.createElement('div');
    display.className = 'seven-segment-display';

    // Segment patterns for digits 0-9
    const patterns = {
        0: [1, 1, 1, 1, 1, 1, 0],
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1]
    };

    const digits = [];

    for (let d = 0; d < 4; d++) {
        const digit = document.createElement('div');
        digit.className = 'digit';

        const segments = [];
        const segmentNames = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];

        segmentNames.forEach((name, i) => {
            const segment = document.createElement('div');
            segment.className = `segment ${name} ${i < 4 || i === 6 ? 'h' : 'v'}`;
            segments.push(segment);
            digit.appendChild(segment);
        });

        digits.push({ element: digit, segments });
        display.appendChild(digit);
    }

    container.appendChild(display);

    // Update display function
    const updateDisplay = (value) => {
        const str = value.toString().padStart(4, '0');

        digits.forEach((digit, i) => {
            const num = parseInt(str[i]);
            const pattern = patterns[num];

            digit.segments.forEach((segment, j) => {
                if (pattern[j]) {
                    segment.classList.add('on');
                } else {
                    segment.classList.remove('on');
                }
            });
        });
    };

    // Animate counter
    let count = 0;
    setInterval(() => {
        count = (count + 1) % 10000;
        updateDisplay(count);
    }, 100);
}

/**
 * Initialize Status Dashboard
 */
function initStatusDashboard() {
    const container = document.getElementById('status-dashboard');
    if (!container) return;

    const statuses = [
        { title: 'CPU', value: '45%', trend: 'up' },
        { title: 'Memory', value: '2.4GB', trend: 'stable' },
        { title: 'Network', value: '125Mbps', trend: 'up' },
        { title: 'Temp', value: '62°C', trend: 'down' },
        { title: 'Power', value: '85W', trend: 'stable' },
        { title: 'FPS', value: '60', trend: 'stable' }
    ];

    statuses.forEach(status => {
        const item = document.createElement('div');
        item.className = 'status-item';

        const header = document.createElement('div');
        header.className = 'status-header';

        const title = document.createElement('div');
        title.className = 'status-title';
        title.textContent = status.title;

        const led = document.createElement('div');
        led.className = 'led green on';
        led.style.width = '8px';
        led.style.height = '8px';

        header.appendChild(title);
        header.appendChild(led);

        const value = document.createElement('div');
        value.className = 'status-value';
        value.textContent = status.value;

        item.appendChild(header);
        item.appendChild(value);
        container.appendChild(item);

        // Simulate value updates
        setInterval(() => {
            if (status.title === 'CPU') {
                const cpu = Math.floor(30 + Math.random() * 40);
                value.textContent = `${cpu}%`;
            } else if (status.title === 'FPS') {
                value.textContent = Math.floor(55 + Math.random() * 10).toString();
            }
        }, 1000);
    });
}

/**
 * Initialize Ring Meters
 */
function initRingMeters() {
    const container = document.getElementById('ring-meters');
    if (!container) return;

    const createRingMeter = (label, color, value) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'ring-meter';

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100');
        svg.setAttribute('height', '100');
        svg.setAttribute('viewBox', '0 0 100 100');

        const radius = 40;
        const circumference = 2 * Math.PI * radius;

        // Background circle
        const bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        bgCircle.setAttribute('cx', '50');
        bgCircle.setAttribute('cy', '50');
        bgCircle.setAttribute('r', radius);
        bgCircle.setAttribute('fill', 'none');
        bgCircle.setAttribute('stroke', '#333');
        bgCircle.setAttribute('stroke-width', '8');
        svg.appendChild(bgCircle);

        // Value circle
        const valueCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        valueCircle.setAttribute('cx', '50');
        valueCircle.setAttribute('cy', '50');
        valueCircle.setAttribute('r', radius);
        valueCircle.setAttribute('fill', 'none');
        valueCircle.setAttribute('stroke', color);
        valueCircle.setAttribute('stroke-width', '8');
        valueCircle.setAttribute('stroke-linecap', 'round');
        valueCircle.setAttribute('stroke-dasharray', `${value / 100 * circumference} ${circumference}`);
        valueCircle.style.filter = `drop-shadow(0 0 5px ${color})`;
        svg.appendChild(valueCircle);

        const valueEl = document.createElement('div');
        valueEl.className = 'ring-meter-value';
        valueEl.textContent = `${value}%`;

        const labelEl = document.createElement('div');
        labelEl.className = 'ring-meter-label';
        labelEl.textContent = label;

        wrapper.appendChild(svg);
        wrapper.appendChild(valueEl);
        wrapper.appendChild(labelEl);
        container.appendChild(wrapper);

        // Animate
        let currentValue = value;
        setInterval(() => {
            currentValue = Math.max(0, Math.min(100, currentValue + (Math.random() - 0.5) * 10));
            valueCircle.setAttribute('stroke-dasharray', `${currentValue / 100 * circumference} ${circumference}`);
            valueEl.textContent = `${Math.round(currentValue)}%`;
        }, 500);
    };

    createRingMeter('CPU', '#00ff88', 65);
    createRingMeter('RAM', '#00aaff', 42);
    createRingMeter('Disk', '#ff8800', 78);
}

/**
 * Initialize Oscilloscope
 */
function initOscilloscope() {
    const container = document.getElementById('oscilloscope');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'oscilloscope-display';
    wrapper.style.width = '100%';

    const grid = document.createElement('div');
    grid.className = 'oscilloscope-grid';

    const canvas = document.createElement('canvas');
    canvas.className = 'oscilloscope-canvas';
    canvas.width = 400;
    canvas.height = 120;

    wrapper.appendChild(grid);
    wrapper.appendChild(canvas);
    container.appendChild(wrapper);

    const ctx = canvas.getContext('2d');
    let offset = 0;

    const draw = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.shadowColor = '#00ff88';
        ctx.shadowBlur = 10;

        ctx.beginPath();

        for (let x = 0; x < canvas.width; x++) {
            const y = canvas.height / 2 +
                Math.sin((x + offset) * 0.05) * 30 +
                Math.sin((x + offset) * 0.02) * 15;

            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();
        offset += 3;

        requestAnimationFrame(draw);
    };

    draw();
}

/**
 * Initialize Spectrum Analyzer
 */
function initSpectrumAnalyzer() {
    const container = document.getElementById('spectrum-analyzer');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'spectrum-display';

    const numBars = 32;
    const bars = [];

    for (let i = 0; i < numBars; i++) {
        const bar = document.createElement('div');
        bar.className = 'spectrum-bar';
        bar.style.height = '10%';
        bars.push(bar);
        wrapper.appendChild(bar);
    }

    container.appendChild(wrapper);

    // Animate spectrum
    const animate = () => {
        bars.forEach((bar, i) => {
            const center = numBars / 2;
            const distance = Math.abs(i - center) / center;
            const baseHeight = 30 + (1 - distance) * 40;
            const randomness = Math.random() * 30;
            bar.style.height = `${baseHeight + randomness}%`;
        });
    };

    setInterval(animate, 50);
}

/**
 * Initialize LED Matrix
 */
function initLEDMatrix() {
    const container = document.getElementById('led-matrix');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'led-matrix';

    const rows = 8;
    const cols = 8;
    const dots = [];

    for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) {
            const dot = document.createElement('div');
            dot.className = 'matrix-dot';
            row.push(dot);
            wrapper.appendChild(dot);
        }
        dots.push(row);
    }

    container.appendChild(wrapper);

    // Animate matrix with scrolling pattern
    let frame = 0;

    const patterns = [
        // Heart
        [
            [0, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        // Smile
        [
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0]
        ]
    ];

    let patternIndex = 0;

    const animate = () => {
        const pattern = patterns[patternIndex];

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                if (pattern[r][c]) {
                    dots[r][c].classList.add('on');
                } else {
                    dots[r][c].classList.remove('on');
                }
            }
        }

        frame++;
        if (frame % 30 === 0) {
            patternIndex = (patternIndex + 1) % patterns.length;
        }
    };

    setInterval(animate, 100);
}

/**
 * Check WebGPU Support
 */
async function checkWebGPUSupport() {
    if (!navigator.gpu) {
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch (e) {
        return false;
    }
}

/**
 * Initialize Interactive Button Panel with Multi-layer Effects
 */
function initButtonPanel() {
    const container = document.getElementById('button-panel');
    if (!container) return;

    const buttons = [
        { label: 'Power', color: '#ff3333', icon: '⚡' },
        { label: 'Start', color: '#00ff88', icon: '▶' },
        { label: 'Stop', color: '#ff8800', icon: '■' },
        { label: 'Reset', color: '#00aaff', icon: '↻' },
        { label: 'Alert', color: '#ffff00', icon: '⚠' },
        { label: 'Lock', color: '#ff00ff', icon: '🔒' }
    ];

    buttons.forEach(btn => {
        const wrapper = document.createElement('div');
        wrapper.className = 'button-wrapper';
        wrapper.style.cssText = 'position: relative; margin: 0.5rem;';

        const button = document.createElement('button');
        button.className = 'control-button';
        button.style.cssText = `
            padding: 1rem 2rem;
            background: rgba(30, 30, 40, 0.8);
            border: 2px solid ${btn.color};
            border-radius: 8px;
            color: ${btn.color};
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 0 10px ${btn.color}40;
        `;

        const icon = document.createElement('span');
        icon.className = 'button-icon';
        icon.textContent = btn.icon;
        icon.style.fontSize = '1.2rem';

        const label = document.createElement('span');
        label.className = 'button-label';
        label.textContent = btn.label;

        button.appendChild(icon);
        button.appendChild(label);
        wrapper.appendChild(button);
        container.appendChild(wrapper);

        // Multi-layer press effect
        button.addEventListener('click', () => {
            button.style.transform = 'scale(0.95)';
            button.style.boxShadow = `0 0 20px ${btn.color}80`;

            // Create ripple effect layers
            for (let i = 0; i < 3; i++) {
                setTimeout(() => {
                    const ripple = document.createElement('div');
                    ripple.style.cssText = `
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        width: 10px;
                        height: 10px;
                        background: ${btn.color};
                        border-radius: 50%;
                        transform: translate(-50%, -50%);
                        pointer-events: none;
                        animation: ripple-expand 0.8s ease-out;
                        opacity: 0;
                        animation-delay: ${i * 0.1}s;
                    `;
                    button.appendChild(ripple);

                    setTimeout(() => ripple.remove(), 1000);
                }, i * 100);
            }

            setTimeout(() => {
                button.style.transform = 'scale(1)';
                button.style.boxShadow = `0 0 10px ${btn.color}40`;
            }, 200);
        });

        button.addEventListener('mouseenter', () => {
            button.style.background = `${btn.color}20`;
            button.style.boxShadow = `0 0 20px ${btn.color}60`;
        });

        button.addEventListener('mouseleave', () => {
            button.style.background = 'rgba(30, 30, 40, 0.8)';
            button.style.boxShadow = `0 0 10px ${btn.color}40`;
        });
    });

    // Add CSS animation for ripple
    if (!document.getElementById('ripple-animation')) {
        const style = document.createElement('style');
        style.id = 'ripple-animation';
        style.textContent = `
            @keyframes ripple-expand {
                0% {
                    width: 10px;
                    height: 10px;
                    opacity: 0.8;
                }
                100% {
                    width: 150px;
                    height: 150px;
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

/**
 * Initialize Multi-State Indicator Panel
 */
function initMultiStateIndicators() {
    const container = document.getElementById('multi-state-indicators');
    if (!container) return;

    const indicators = [
        { label: 'System', states: ['idle', 'active', 'error'], colors: ['#888', '#00ff88', '#ff3333'] },
        { label: 'Network', states: ['offline', 'connecting', 'online'], colors: ['#666', '#ffaa00', '#00aaff'] },
        { label: 'Battery', states: ['low', 'charging', 'full'], colors: ['#ff3333', '#ffff00', '#00ff88'] },
        { label: 'Signal', states: ['weak', 'medium', 'strong'], colors: ['#ff8800', '#ffff00', '#00ff88'] }
    ];

    indicators.forEach(ind => {
        const wrapper = document.createElement('div');
        wrapper.className = 'multi-state-indicator';
        wrapper.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            background: rgba(20, 20, 30, 0.6);
            border-radius: 8px;
            margin: 0.5rem;
            min-width: 120px;
        `;

        const label = document.createElement('div');
        label.className = 'indicator-label';
        label.textContent = ind.label;
        label.style.cssText = `
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-weight: bold;
        `;

        const stateDisplay = document.createElement('div');
        stateDisplay.className = 'state-display';
        stateDisplay.style.cssText = `
            display: flex;
            gap: 0.5rem;
            align-items: center;
        `;

        const dots = [];
        ind.states.forEach((state, idx) => {
            const dot = document.createElement('div');
            dot.className = `state-dot state-${state}`;
            dot.dataset.state = state;
            dot.style.cssText = `
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: ${ind.colors[idx]}40;
                border: 2px solid ${ind.colors[idx]};
                transition: all 0.3s;
                box-shadow: 0 0 5px ${ind.colors[idx]}40;
            `;
            dots.push({ element: dot, color: ind.colors[idx] });
            stateDisplay.appendChild(dot);
        });

        const statusText = document.createElement('div');
        statusText.className = 'status-text';
        statusText.textContent = ind.states[0];
        statusText.style.cssText = `
            font-size: 0.9rem;
            color: ${ind.colors[0]};
            font-weight: bold;
            min-height: 1.5rem;
            transition: color 0.3s;
        `;

        wrapper.appendChild(label);
        wrapper.appendChild(stateDisplay);
        wrapper.appendChild(statusText);
        container.appendChild(wrapper);

        // Cycle through states
        let currentState = 0;
        setInterval(() => {
            // Reset all dots
            dots.forEach(d => {
                d.element.style.background = `${d.color}40`;
                d.element.style.boxShadow = `0 0 5px ${d.color}40`;
                d.element.style.transform = 'scale(1)';
            });

            currentState = (currentState + 1) % ind.states.length;

            // Activate current dot
            dots[currentState].element.style.background = dots[currentState].color;
            dots[currentState].element.style.boxShadow = `0 0 15px ${dots[currentState].color}`;
            dots[currentState].element.style.transform = 'scale(1.2)';

            statusText.textContent = ind.states[currentState];
            statusText.style.color = ind.colors[currentState];
        }, 2000 + Math.random() * 1000);
    });
}

/**
 * Initialize WebGPU Holographic VU Meter with Multiple Layers
 */
async function initHolographicVU() {
    const container = document.getElementById('holographic-vu');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 300;
    canvas.style.cssText = 'width: 100%; max-width: 600px; border-radius: 8px;';
    container.appendChild(canvas);

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
        console.warn('WebGPU not supported for holographic VU');
        return;
    }

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();

    context.configure({ device, format });

    // Shader for multi-layer holographic effect
    const shaderCode = `
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, 1.0)
            );
            
            var output: VertexOutput;
            let pos = positions[vertexIndex];
            output.position = vec4<f32>(pos, 0.0, 1.0);
            output.uv = pos * 0.5 + 0.5;
            return output;
        }

        struct Uniforms {
            time: f32,
            levels: array<f32, 16>,
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            var color = vec3<f32>(0.0);
            
            // Multi-layer VU bars with depth
            for (var i: i32 = 0; i < 16; i++) {
                let barX = f32(i) / 15.0;
                let dist = abs(uv.x - barX);
                let level = uniforms.levels[i];
                
                // Layer 1: Main bar
                if (dist < 0.025 && uv.y > 1.0 - level) {
                    let hue = f32(i) / 15.0;
                    color += vec3<f32>(
                        sin(hue * 6.28 + uniforms.time) * 0.5 + 0.5,
                        sin(hue * 6.28 + uniforms.time + 2.09) * 0.5 + 0.5,
                        sin(hue * 6.28 + uniforms.time + 4.18) * 0.5 + 0.5
                    ) * 1.5;
                }
                
                // Layer 2: Glow effect
                let glow = exp(-dist * 50.0) * level * 0.3;
                color += vec3<f32>(0.0, 1.0, 0.5) * glow;
            }
            
            // Layer 3: Scanlines
            color *= 0.8 + 0.2 * sin(uv.y * 100.0 + uniforms.time * 5.0);
            
            // Layer 4: Holographic chromatic aberration
            let shift = sin(uniforms.time + uv.y * 10.0) * 0.01;
            color.r += shift;
            color.b -= shift;
            
            return vec4<f32>(color, 1.0);
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    // Create uniform buffer
    const uniformBufferSize = 4 + 16 * 4; // time (f32) + 16 levels (f32 each)
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    // Create pipeline
    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    // Animation loop
    let time = 0;
    const uniformData = new Float32Array(17); // time + 16 levels

    const frame = () => {
        time += 0.016;
        uniformData[0] = time;

        // Update VU levels with audio-like behavior
        for (let i = 0; i < 16; i++) {
            uniformData[i + 1] = (Math.sin(time * 2 + i * 0.5) * 0.5 + 0.5) * 0.8 + 0.2;
        }

        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Render frame
        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 }
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    };

    frame();
}

/**
 * Initialize WebGPU Fluid Simulation Meter
 */
async function initFluidMeter() {
    const container = document.getElementById('fluid-meter');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.cssText = 'width: 100%; max-width: 400px; border-radius: 8px;';
    container.appendChild(canvas);

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
        console.warn('WebGPU not supported for fluid meter');
        return;
    }

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format });

    // Fluid simulation shader with multiple render passes
    const shaderCode = `
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, 1.0)
            );
            
            var output: VertexOutput;
            let pos = positions[vertexIndex];
            output.position = vec4<f32>(pos, 0.0, 1.0);
            output.uv = pos * 0.5 + 0.5;
            return output;
        }

        struct Uniforms {
            time: f32,
            value: f32,
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let center = vec2<f32>(0.5, 0.5);
            let dist = length(uv - center);
            
            // Multi-layer fluid effect
            var col = vec3<f32>(0.0);
            
            // Layer 1: Liquid fill with waves
            let fillLevel = 0.3 + uniforms.value * 0.4;
            let wave1 = sin(uv.x * 20.0 - uniforms.time * 3.0) * 0.02;
            let wave2 = sin(uv.x * 15.0 + uniforms.time * 2.0) * 0.015;
            let totalWave = wave1 + wave2;
            
            if (uv.y > fillLevel + totalWave && dist < 0.45) {
                col = vec3<f32>(0.0, 0.8, 1.0);
                
                // Layer 2: Caustics
                let caustic1 = sin(uv.x * 30.0 + uniforms.time) * sin(uv.y * 30.0 - uniforms.time * 0.7);
                let caustic2 = sin(uv.x * 25.0 - uniforms.time * 1.3) * sin(uv.y * 25.0 + uniforms.time * 0.9);
                col += vec3<f32>((caustic1 + caustic2) * 0.15);
                
                // Depth shading
                let depth = (uv.y - fillLevel) / (1.0 - fillLevel);
                col *= 0.7 + depth * 0.3;
            }
            
            // Layer 3: Container border with glow
            if (dist > 0.43 && dist < 0.47) {
                col = vec3<f32>(0.3, 0.4, 0.5);
            } else if (dist > 0.47 && dist < 0.48) {
                col = vec3<f32>(0.2, 0.3, 0.4);
            }
            
            // Add shimmer on liquid surface
            if (abs(uv.y - fillLevel - totalWave) < 0.01 && dist < 0.45) {
                col += vec3<f32>(0.5, 0.8, 1.0) * 0.5;
            }
            
            return vec4<f32>(col, 1.0);
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    // Create uniform buffer
    const uniformBuffer = device.createBuffer({
        size: 8, // 2 floats (time, value)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    // Animation loop
    let time = 0;
    const uniformData = new Float32Array(2);

    const frame = () => {
        time += 0.016;
        const value = Math.sin(time * 0.5) * 0.5 + 0.5; // Oscillating value

        uniformData[0] = time;
        uniformData[1] = value;

        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 }
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    };

    frame();
}

/**
 * Initialize WebGPU Plasma Globe Effect
 */
async function initPlasmaGlobe() {
    const container = document.getElementById('plasma-globe');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.cssText = 'width: 100%; max-width: 400px; border-radius: 8px;';
    container.appendChild(canvas);

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
        console.warn('WebGPU not supported for plasma globe');
        return;
    }

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format });

    const shaderCode = `
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, 1.0)
            );
            
            var output: VertexOutput;
            let pos = positions[vertexIndex];
            output.position = vec4<f32>(pos, 0.0, 1.0);
            output.uv = pos * 0.5 + 0.5;
            return output;
        }

        @group(0) @binding(0) var<uniform> time: f32;

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let center = vec2<f32>(0.5, 0.5);
            let dist = length(uv - center);
            
            var col = vec3<f32>(0.0);
            
            // Plasma globe sphere
            if (dist < 0.4) {
                // Multiple plasma layers
                var plasma = 0.0;
                
                for (var i: i32 = 0; i < 5; i++) {
                    let fi = f32(i);
                    let angle = time * (0.5 + fi * 0.2);
                    let offset = vec2<f32>(
                        cos(angle) * 0.2,
                        sin(angle) * 0.2
                    );
                    
                    let layerDist = length(uv - center - offset);
                    plasma += 1.0 / (layerDist * 50.0 + 1.0);
                }
                
                // Color mapping
                col = vec3<f32>(
                    sin(plasma * 2.0 + time) * 0.5 + 0.5,
                    sin(plasma * 2.0 + time + 2.0) * 0.5 + 0.5,
                    sin(plasma * 2.0 + time + 4.0) * 0.5 + 0.5
                ) * 1.5;
                
                // Sphere shading
                let sphereShade = 1.0 - dist / 0.4;
                col *= sphereShade;
            }
            
            // Glass sphere border
            if (dist > 0.38 && dist < 0.42) {
                col += vec3<f32>(0.3, 0.5, 0.7) * 0.5;
            }
            
            return vec4<f32>(col, 1.0);
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const uniformBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    let time = 0;
    const uniformData = new Float32Array(1);

    const frame = () => {
        time += 0.016;
        uniformData[0] = time;

        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.02, g: 0.02, b: 0.08, a: 1 }
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    };

    frame();
}

/**
 * Initialize WebGPU Waveform Synthesizer
 */
async function initWaveformSynth() {
    const container = document.getElementById('waveform-synth');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 200;
    canvas.style.cssText = 'width: 100%; max-width: 600px; border-radius: 8px;';
    container.appendChild(canvas);

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
        console.warn('WebGPU not supported for waveform synth');
        return;
    }

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format });

    const shaderCode = `
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, 1.0)
            );
            
            var output: VertexOutput;
            let pos = positions[vertexIndex];
            output.position = vec4<f32>(pos, 0.0, 1.0);
            output.uv = pos * 0.5 + 0.5;
            return output;
        }

        @group(0) @binding(0) var<uniform> time: f32;

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            var col = vec3<f32>(0.0);
            
            // Multiple waveform layers
            let waveforms = 4;
            
            for (var i: i32 = 0; i < waveforms; i++) {
                let fi = f32(i);
                let freq = 5.0 + fi * 2.0;
                let phase = time * (1.0 + fi * 0.5);
                let amplitude = 0.1 + fi * 0.03;
                
                let wave = sin(uv.x * freq + phase) * amplitude;
                let waveY = 0.5 + wave;
                
                let dist = abs(uv.y - waveY);
                let intensity = 1.0 / (dist * 200.0 + 1.0);
                
                // Color gradient per layer
                let hue = fi / f32(waveforms);
                col += vec3<f32>(
                    sin(hue * 6.28) * 0.5 + 0.5,
                    sin(hue * 6.28 + 2.09) * 0.5 + 0.5,
                    sin(hue * 6.28 + 4.18) * 0.5 + 0.5
                ) * intensity;
            }
            
            // Grid overlay
            let gridX = fract(uv.x * 20.0);
            let gridY = fract(uv.y * 10.0);
            if (gridX < 0.02 || gridY < 0.02) {
                col += vec3<f32>(0.1, 0.1, 0.15);
            }
            
            return vec4<f32>(col, 1.0);
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const uniformBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    let time = 0;
    const uniformData = new Float32Array(1);

    const frame = () => {
        time += 0.016;
        uniformData[0] = time;

        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 }
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    };

    frame();
}

/**
 * Initialize WebGPU Crystalline Panel
 */
async function initCrystallinePanel() {
    const container = document.getElementById('crystalline-panel');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = 500;
    canvas.height = 300;
    canvas.style.cssText = 'width: 100%; max-width: 500px; border-radius: 8px;';
    container.appendChild(canvas);

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
        console.warn('WebGPU not supported for crystalline panel');
        return;
    }

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format });

    const shaderCode = `
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, 1.0)
            );
            
            var output: VertexOutput;
            let pos = positions[vertexIndex];
            output.position = vec4<f32>(pos, 0.0, 1.0);
            output.uv = pos * 0.5 + 0.5;
            return output;
        }

        @group(0) @binding(0) var<uniform> time: f32;

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            var col = vec3<f32>(0.0);
            
            // Voronoi-like crystal pattern
            let scale = 8.0;
            let p = uv * scale;
            let pi = floor(p);
            let pf = fract(p);
            
            var minDist = 1.0;
            var cellId = vec2<f32>(0.0);
            
            for (var y: i32 = -1; y <= 1; y++) {
                for (var x: i32 = -1; x <= 1; x++) {
                    let neighbor = vec2<f32>(f32(x), f32(y));
                    let point = 0.5 + 0.5 * sin(time * 0.5 + 6.2831 * (pi + neighbor));
                    let diff = neighbor + point - pf;
                    let dist = length(diff);
                    
                    if (dist < minDist) {
                        minDist = dist;
                        cellId = pi + neighbor;
                    }
                }
            }
            
            // Crystal coloring based on cell
            let cellColor = sin(cellId.x * 12.9898 + cellId.y * 78.233 + time * 0.2) * 0.5 + 0.5;
            
            // Multi-layer crystal effect
            col = vec3<f32>(
                0.2 + cellColor * 0.3,
                0.4 + cellColor * 0.4,
                0.6 + cellColor * 0.3
            );
            
            // Crystal edges
            if (minDist < 0.05) {
                col += vec3<f32>(0.5, 0.7, 1.0);
            }
            
            // Shimmer effect
            let shimmer = sin(uv.x * 50.0 + time * 3.0) * sin(uv.y * 50.0 - time * 2.0);
            col += vec3<f32>(shimmer * 0.1);
            
            return vec4<f32>(col, 1.0);
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const uniformBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    let time = 0;
    const uniformData = new Float32Array(1);

    const frame = () => {
        time += 0.016;
        uniformData[0] = time;

        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 }
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    };

    frame();
}

/**
 * Initialize WebGPU Data Visualization
 */
async function initDataVisualization() {
    const container = document.getElementById('data-visualization');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 400;
    canvas.style.cssText = 'width: 100%; max-width: 600px; border-radius: 8px;';
    container.appendChild(canvas);

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
        console.warn('WebGPU not supported for data visualization');
        return;
    }

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format });

    const shaderCode = `
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, 1.0)
            );
            
            var output: VertexOutput;
            let pos = positions[vertexIndex];
            output.position = vec4<f32>(pos, 0.0, 1.0);
            output.uv = pos * 0.5 + 0.5;
            return output;
        }

        struct Uniforms {
            time: f32,
            dataPoints: array<f32, 32>,
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            var col = vec3<f32>(0.0);
            
            // 3D bar chart effect with depth
            let numBars = 32;
            
            for (var i: i32 = 0; i < numBars; i++) {
                let fi = f32(i);
                let barX = fi / f32(numBars - 1);
                let barWidth = 0.8 / f32(numBars);
                
                let dataValue = uniforms.dataPoints[i];
                let barHeight = dataValue;
                
                // Bar position with depth effect
                let depth = 1.0 - fi / f32(numBars);
                let barXStart = barX - barWidth * 0.5 + depth * 0.1;
                let barXEnd = barX + barWidth * 0.5 + depth * 0.1;
                
                if (uv.x > barXStart && uv.x < barXEnd && uv.y > 1.0 - barHeight) {
                    // Color based on value and position
                    let hue = fi / f32(numBars);
                    col = vec3<f32>(
                        sin(hue * 6.28 + uniforms.time * 0.5) * 0.5 + 0.5,
                        sin(hue * 6.28 + uniforms.time * 0.5 + 2.09) * 0.5 + 0.5,
                        sin(hue * 6.28 + uniforms.time * 0.5 + 4.18) * 0.5 + 0.5
                    );
                    
                    // Add depth shading
                    col *= 0.7 + depth * 0.3;
                    
                    // Top edge highlight
                    if (uv.y < 1.0 - barHeight + 0.01) {
                        col += vec3<f32>(0.3);
                    }
                }
            }
            
            // Grid lines
            let gridY = fract(uv.y * 10.0);
            if (gridY < 0.01) {
                col += vec3<f32>(0.1, 0.1, 0.15);
            }
            
            return vec4<f32>(col, 1.0);
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const uniformBufferSize = 4 + 32 * 4; // time + 32 data points
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    let time = 0;
    const uniformData = new Float32Array(33);

    const frame = () => {
        time += 0.016;
        uniformData[0] = time;

        // Generate animated data points
        for (let i = 0; i < 32; i++) {
            uniformData[i + 1] = (Math.sin(time * 2 + i * 0.3) * 0.3 + 0.5) * 0.8 + 0.2;
        }

        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 }
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    };

    frame();
}

/**
 * Initialize Bioluminescent Meters
 */
function initBioMeters() {
    const container = document.getElementById('bio-meters');
    if (!container) return;

    const meters = [
        { label: 'HEALTH', color: [0.0, 1.0, 0.4] }, // Spring Green
        { label: 'MANA', color: [0.0, 0.6, 1.0] },   // Dodson Blue
        { label: 'TOXIN', color: [0.8, 0.0, 1.0] }   // Purple
    ];

    meters.forEach(m => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 0.5rem; position: relative;';

        const canvas = document.createElement('canvas');
        canvas.width = 100;
        canvas.height = 100;
        canvas.style.cssText = 'width: 100px; height: 100px; border-radius: 50%; background: #000; box-shadow: 0 0 10px rgba(0,0,0,0.5);';

        // SVG Overlay for Membrane/Border
        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.style.cssText = "position: absolute; top: 0; left: 0; width: 100px; height: 100px; pointer-events: none;";
        const circle = document.createElementNS(svgNS, "circle");
        circle.setAttribute("cx", "50");
        circle.setAttribute("cy", "50");
        circle.setAttribute("r", "48");
        circle.setAttribute("fill", "none");
        circle.setAttribute("stroke", `rgba(${m.color[0] * 255}, ${m.color[1] * 255}, ${m.color[2] * 255}, 0.3)`);
        circle.setAttribute("stroke-width", "2");
        circle.setAttribute("stroke-dasharray", "5 5");
        svg.appendChild(circle);

        const label = document.createElement('div');
        label.textContent = m.label;
        label.style.cssText = `font-size: 0.7rem; color: rgb(${m.color[0] * 255},${m.color[1] * 255},${m.color[2] * 255}); font-weight: bold; margin-top: 5px;`;

        wrapper.appendChild(canvas);
        wrapper.appendChild(svg);
        wrapper.appendChild(label);
        container.appendChild(wrapper);

        // WebGL Initialization
        const gl = canvas.getContext('webgl2');
        if (!gl) return;

        const vs = `#version 300 es
            in vec4 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position.xy * 0.5 + 0.5;
                gl_Position = a_position;
            }
        `;

        const fs = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            uniform float u_time;
            uniform vec3 u_color;
            out vec4 fragColor;

            void main() {
                vec2 uv =(v_uv - 0.5) * 2.0;
                float d = length(uv);

                // Organic movement
                float t = u_time * 0.5;
                vec2 p = uv;
                float noise = sin(p.x * 5.0 + t) * cos(p.y * 5.0 - t);
                noise += sin(p.y * 10.0 + t * 2.0) * 0.2;
                noise += cos(length(p) * 10.0 - t * 3.0) * 0.2;

                // Core glow
                float core = 0.05 / (abs(d - 0.5 + noise * 0.2) + 0.01);
                
                // Inner fill
                float fill = smoothstep(0.8, 0.0, d);
                fill *= 0.5 + 0.5 * sin(u_time * 2.0 + noise * 5.0); // Pulse

                vec3 col = u_color * (core + fill * 0.5);
                
                // Darken clear areas
                col *= smoothstep(1.0, 0.9, d);

                fragColor = vec4(col, 1.0);
            }
        `;

        // Create Program (Local helper logic)
        const createShader = (gl, type, source) => {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.error(gl.getShaderInfoLog(shader));
                return null;
            }
            return shader;
        };

        const program = gl.createProgram();
        const vShader = createShader(gl, gl.VERTEX_SHADER, vs);
        const fShader = createShader(gl, gl.FRAGMENT_SHADER, fs);
        if (!vShader || !fShader) return;

        gl.attachShader(program, vShader);
        gl.attachShader(program, fShader);
        gl.linkProgram(program);

        // Quad setup
        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

        const uTime = gl.getUniformLocation(program, 'u_time');
        const uColor = gl.getUniformLocation(program, 'u_color');

        function render(time) {
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(program);
            gl.uniform1f(uTime, time * 0.001);
            gl.uniform3fv(uColor, m.color);

            gl.bindVertexArray(vao);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            requestAnimationFrame(render);
        }
        requestAnimationFrame(render);
    });
}

