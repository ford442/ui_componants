# UI Components Library - Agent Guide

> **Project:** UI Components Library  
> **Type:** Interactive UI Components & Visual Experiments  
> **Language:** English (all code comments and documentation)  
> **Last Updated:** 2026-02-08

---

## 1. Project Overview

This is a **technical showcase and component library** featuring high-performance interactive UI components with **layered canvas effects**. The project demonstrates advanced web rendering techniques by combining multiple technologies (WebGL2, WebGPU, SVG, CSS) into cohesive visual interfaces.

### Key Characteristics
- **Multi-Page Application (MPA):** Each experiment lives in its own HTML file in `src/pages/`. No client-side router.
- **Vanilla JavaScript:** Pure ES modules, no TypeScript, no heavy frontend frameworks.
- **Layered Rendering:** Multiple `<canvas>` elements stacked via CSS (`absolute` positioning, `z-index`) to merge different rendering contexts visually.
- **Component-as-Class Pattern:** Logic encapsulated in classes (e.g., `LEDButton`, `HybridEngine`) that manage their own DOM lifecycle and render loops.
- **Progressive Enhancement:** Graceful degradation for browsers without WebGPU support.

### Rendering Contexts Supported
| Context | Usage |
|---------|-------|
| WebGL | Hardware-accelerated 2D/3D graphics with shader effects |
| WebGL2 | Advanced shader effects, textures, and compute |
| WebGPU | Next-gen GPU compute and rendering (Chrome 113+) |
| Canvas 2D | Fallback and simple drawing |
| SVG | Vector overlays for UI elements |
| CSS | Filters, transforms, animations, and blend modes |

---

## 2. Technology Stack

### Core Technologies
- **JavaScript:** Vanilla ES6+ modules (no TypeScript)
- **HTML5:** Semantic markup
- **CSS3:** Custom properties, flexbox, grid, backdrop-filter

### Build System
- **Vite 5.x:** Dev server and production bundling
  - Root: `src/`
  - Public directory: `public/`
  - Output: `dist/`
  - Multi-page entry points defined in `vite.config.js`

### Testing
- **Vitest 1.x:** Unit testing with jsdom environment
- **@testing-library/dom:** DOM testing utilities
- **Test location:** `src/tests/`

### Deployment
- **Python script:** `deploy.py` using Paramiko for SFTP
- **Target:** `test.1ink.us/ui` (configured for remote server)

---

## 3. Project Structure

```
ui_componants/
├── src/
│   ├── index.html              # Main navigation page
│   ├── css/
│   │   ├── main.css            # Global styles, CSS variables
│   │   ├── buttons.css         # Button component styles
│   │   ├── knobs.css           # Knob component styles
│   │   ├── switches.css        # Switch component styles
│   │   ├── indicators.css      # Indicator component styles
│   │   ├── surfaces.css        # Surface/PBR styles
│   │   └── signage_lab.css     # Signage experiment styles
│   ├── js/
│   │   ├── main.js             # Core utilities, LayeredCanvas, ShaderUtils
│   │   ├── buttons.js          # Button page initialization
│   │   ├── knobs.js            # Knobs page initialization
│   │   ├── switches.js         # Switches page initialization
│   │   ├── indicators.js       # Indicators (2500+ lines, extensive WebGPU)
│   │   ├── surfaces.js         # Surface materials
│   │   ├── experiments-global.js # Global effects (cursor trails, audio)
│   │   └── [60+ experiment files...] # Individual visual experiments
│   ├── pages/
│   │   ├── buttons.html
│   │   ├── knobs.html
│   │   ├── switches.html
│   │   ├── indicators.html
│   │   ├── surfaces.html
│   │   ├── composite_blending.html
│   │   ├── hologram.html
│   │   ├── fireworks.html
│   │   ├── tetris.html
│   │   └── [50+ experiment pages...]
│   └── tests/
│       ├── setup.js            # Test environment mocks
│       └── buttons.test.js     # Example tests
├── public/                     # Static assets
├── dist/                       # Build output (generated)
├── vite.config.js              # Vite configuration
├── vitest.config.js            # Vitest configuration
├── package.json                # NPM scripts and dependencies
├── deploy.py                   # SFTP deployment script
└── [Documentation files...]
    ├── README.md
    ├── DEVELOPER_CONTEXT.md    # Architecture deep-dive
    ├── IMPLEMENTATION_SUMMARY.md
    ├── IMPLEMENTATION_VERIFICATION.md
    └── QUICK_REFERENCE.md
```

---

## 4. Build and Development Commands

```bash
# Install dependencies
npm install

# Start development server
npm run dev
# or
vite

# Build for production
npm run build
# or
vite build

# Preview production build
npm run preview
# or
vite preview

# Run tests
npm test
# or
vitest

# Deploy (requires Python and paramiko)
python deploy.py
# Note: Ensure dist/ exists by running build first
```

---

## 5. Core API Reference

### LayeredCanvas
Manages multiple canvas layers with different rendering contexts.

```javascript
import { LayeredCanvas } from './js/main.js';

const layered = new LayeredCanvas(container, {
    width: 800,
    height: 400
});

layered.addLayer('base', 'webgl', 0);
layered.addLayer('effects', 'webgl2', 1);
layered.addSVGLayer('overlay', 2);
layered.startAnimation();
```

### LEDButton
Creates an LED button with WebGL glow effect.

```javascript
const button = new UIComponents.LEDButton(container, {
    width: 100,
    height: 60,
    color: [0, 1, 0.5],  // RGB normalized
    label: 'Power',
    onToggle: (isOn) => console.log('Button:', isOn)
});
```

### RotaryKnob
Creates a rotary knob with WebGL glow and SVG indicator.

```javascript
const knob = new UIComponents.RotaryKnob(container, {
    size: 80,
    min: 0,
    max: 100,
    value: 50,
    color: '#00ff88',
    label: 'Volume',
    onChange: (value) => console.log('Value:', value)
});
```

### ShaderUtils
WebGL shader compilation helpers.

```javascript
import { ShaderUtils } from './js/main.js';

const program = ShaderUtils.createProgram(gl, vertexSource, fragmentSource);
```

### RenderingSupport
Feature detection for rendering contexts.

```javascript
await RenderingSupport.detect();
console.log(RenderingSupport.getStatus()); // { webgl: true, webgl2: true, webgpu: true }
```

---

## 6. Code Style Guidelines

### JavaScript Conventions
- **ES6+ modules:** Use `import`/`export`
- **Class-based components:** Each major component is a class
- **JSDoc comments:** Document public methods and parameters
- **Camel case:** `myVariableName`, `myFunctionName`, `MyClassName`
- **Constants:** UPPER_SNAKE_CASE for true constants
- **Private methods:** Prefix with underscore `_privateMethod()`

### CSS Conventions
- **Custom properties:** Define in `:root` in `main.css`
- **Naming:** kebab-case for classes (`.my-class-name`)
- **Organization:** Component-specific styles in separate files
- **BEM-like:** Use descriptive class names (`.button-wrapper`, `.indicator-label`)

### Key CSS Variables (from main.css)
```css
:root {
    --bg-dark: #0a0a0f;
    --bg-card: #14141f;
    --accent-primary: #00ff88;
    --accent-secondary: #00aaff;
    --accent-warning: #ffaa00;
    --accent-danger: #ff4444;
    --text-primary: #ffffff;
    --text-secondary: #888899;
    --glow-primary: 0 0 20px rgba(0, 255, 136, 0.5);
    --glow-secondary: 0 0 20px rgba(0, 170, 255, 0.5);
}
```

### HTML Conventions
- **Semantic elements:** Use proper HTML5 tags
- **Module scripts:** `<script type="module" src="./js/main.js"></script>`
- **Container IDs:** Use kebab-case (`#button-panel`, `#multi-state-indicators`)

---

## 7. Testing Instructions

### Test Setup
Tests use Vitest with jsdom environment. The setup file (`src/tests/setup.js`) mocks:
- Canvas 2D and WebGL contexts
- ResizeObserver
- requestAnimationFrame

### Running Tests
```bash
# Run tests once
npm test

# Run tests in watch mode
vitest --watch

# Run tests with coverage
vitest --coverage
```

### Writing Tests
```javascript
import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('My Component', () => {
    beforeEach(() => {
        document.body.innerHTML = '<div id="container"></div>';
        vi.resetModules();
    });

    it('should initialize correctly', async () => {
        const { MyComponent } = await import('../js/my-component.js');
        const container = document.getElementById('container');
        const instance = new MyComponent(container);
        expect(instance).toBeTruthy();
    });
});
```

---

## 8. Development Conventions

### Adding a New Page/Experiment

1. **Create HTML file** in `src/pages/my-experiment.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Experiment</title>
    <link rel="stylesheet" href="../css/main.css">
</head>
<body>
    <div class="container">
        <div id="experiment-container"></div>
    </div>
    <script type="module" src="../js/my-experiment.js"></script>
</body>
</html>
```

2. **Create JavaScript file** in `src/js/my-experiment.js`:
```javascript
import './main.js';  // Import core utilities

class MyExperiment {
    constructor(container) {
        this.container = container;
        this.init();
    }
    
    init() {
        // Setup code here
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('experiment-container');
    new MyExperiment(container);
});
```

3. **Add entry point** to `vite.config.js`:
```javascript
input: {
    // ...existing entries
    my_experiment: resolve(__dirname, 'src/pages/my-experiment.html'),
}
```

4. **Add navigation card** to `src/index.html` (optional, for main page visibility)

### Layered Rendering Pattern

When creating multi-layer effects:

```javascript
// 1. Create container with relative positioning
const container = document.createElement('div');
container.style.position = 'relative';
container.style.width = '100%';
container.style.height = '400px';

// 2. Add WebGL background layer
const bgCanvas = document.createElement('canvas');
bgCanvas.style.position = 'absolute';
bgCanvas.style.zIndex = '0';
container.appendChild(bgCanvas);

// 3. Add WebGPU overlay layer  
const overlayCanvas = document.createElement('canvas');
overlayCanvas.style.position = 'absolute';
overlayCanvas.style.zIndex = '1';
container.appendChild(overlayCanvas);

// 4. Add SVG UI layer
const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
svg.style.position = 'absolute';
svg.style.zIndex = '2';
svg.style.pointerEvents = 'none';
container.appendChild(svg);
```

---

## 9. Security Considerations

### Current State
- **No authentication:** This is a static showcase site
- **No sensitive data:** No user data or secrets in the codebase
- **Deployment credentials:** Hardcoded in `deploy.py` (noted as a known issue)

### Best Practices for Modifications
1. **Canvas contexts:** Always validate context creation returned non-null before use
2. **WebGPU:** Wrap adapter requests in try-catch blocks
3. **User input:** If adding forms, implement proper sanitization
4. **Dependencies:** Keep npm packages updated (`npm audit`)

### Deployment Security
The `deploy.py` script contains hardcoded credentials. For production use:
- Move credentials to environment variables
- Use SSH keys instead of passwords
- Consider CI/CD pipeline with secrets management

---

## 10. Known Limitations & Constraints

### Technical Constraints
1. **No TypeScript:** The project is pure JavaScript. Do not introduce `.ts` files.
2. **No External Math Libraries:** Matrix math is hardcoded in shaders. Do not add `gl-matrix` or similar.
3. **Dark Mode Only:** UI designed for dark backgrounds (`#000` or deep gray). Light mode not supported.
4. **WebGPU Browser Support:** Requires Chrome 113+ or Edge 113+. Firefox/Safari need experimental flags.

### Known Issues
1. **Hardcoded Asset Paths:** Some HTML files use relative paths (`../assets/`). Moving files requires path auditing.
2. **WebGPU Fallbacks:** Some files may fail silently on non-WebGPU browsers.
3. **Manual Matrix Math:** Complex 3D transformations are difficult and error-prone.

### Complexity Hotspots
| Location | Issue | Note |
|----------|-------|------|
| `src/js/hybrid-engine.js` | WebGL2 + WebGPU sync | Resize both contexts together, match DPR |
| `src/js/quantum-data-stream.js` | Memory alignment | WGSL structs need 16-byte alignment |
| `src/js/experiments-global.js` | Global event listeners | Potential leaks in SPA scenario |
| `src/js/indicators.js` | Large file (~2500 lines) | Multiple WebGPU visualizations |

---

## 11. Browser Support Matrix

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| Basic Components | 80+ | 80+ | 75+ | 14+ |
| WebGL2 Effects | 80+ | 80+ | 75+ | 15+ |
| WebGPU | 113+ | 113+ | Nightly | Preview |
| CSS backdrop-filter | 76+ | 79+ | 103+ | 9+ |

---

## 12. Quick Troubleshooting

### WebGPU Not Working
- Verify Chrome 113+ with `chrome://version`
- Enable flag: `chrome://flags/#enable-unsafe-webgpu`
- Check console for adapter request failures

### Build Errors
- Ensure Node.js 18+ installed
- Delete `node_modules` and `package-lock.json`, then `npm install`

### Canvas Rendering Issues
- Check browser DevTools Console for WebGL errors
- Verify canvas has valid width/height (not 0)
- Ensure context is created with correct attributes

### Test Failures
- Tests mock WebGL contexts; some features may not be fully testable
- Update mocks in `src/tests/setup.js` if adding new canvas methods

---

## 13. File Reference Summary

| File | Purpose |
|------|---------|
| `src/js/main.js` | Core utilities, LayeredCanvas, ShaderUtils, LEDButton, RotaryKnob |
| `src/js/experiments-global.js` | Global effects: cursor trails, audio visualizer |
| `src/js/indicators.js` | Extensive indicators with 6+ WebGPU visualizations |
| `src/js/hybrid-engine.js` | WebGL2 + WebGPU hybrid rendering POC |
| `src/js/quantum-data-stream.js` | Advanced WebGPU tunnel effect |
| `vite.config.js` | Build configuration, entry points |
| `vitest.config.js` | Test configuration |
| `deploy.py` | SFTP deployment script |

---

## 14. Additional Documentation

- **README.md** - User-facing overview and quick start
- **DEVELOPER_CONTEXT.md** - Architecture intent, complexity hotspots, critical flows
- **IMPLEMENTATION_SUMMARY.md** - Details on WebGPU indicator implementations
- **IMPLEMENTATION_VERIFICATION.md** - Checklist of completed features
- **QUICK_REFERENCE.md** - API reference for new indicator components

---

*This document is maintained for AI coding agents. Update when making architectural changes or adding significant features.*
