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
                const fps = Math.floor(55 + Math.random() * 10);
                value.textContent = fps;
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
            [0,1,1,0,0,1,1,0],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [0,1,1,1,1,1,1,0],
            [0,0,1,1,1,1,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,0,0,0,0,0]
        ],
        // Smile
        [
            [0,0,1,1,1,1,0,0],
            [0,1,0,0,0,0,1,0],
            [1,0,1,0,0,1,0,1],
            [1,0,0,0,0,0,0,1],
            [1,0,1,0,0,1,0,1],
            [1,0,0,1,1,0,0,1],
            [0,1,0,0,0,0,1,0],
            [0,0,1,1,1,1,0,0]
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
