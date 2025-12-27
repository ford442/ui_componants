# DEVELOPER_CONTEXT

> **Role:** Senior Technical Lead / Project Archivist
> **Last Updated:** Current Session
> **Status:** Active / Maintenance

This document serves as the primary source of truth for the **UI Components Library** codebase. It encapsulates architectural intent, complexity hotspots, and critical operational knowledge. Read this before modifying core rendering logic or adding new experiments.

---

## 1. High-Level Architecture & Intent

### Core Purpose
This project is a **technical showcase and library** of high-performance interactive UI components and visual experiments. Its primary goal is to demonstrate **layered rendering techniques** by combining multiple web technologies (WebGL2, WebGPU, SVG, CSS) into cohesive visual interfaces without relying on heavy frontend frameworks.

### Tech Stack
*   **Core:** Vanilla JavaScript (ES Modules), HTML5, CSS3.
*   **Build System:** [Vite](https://vitejs.dev/) (Handles dev server and production bundling).
*   **Rendering Engines:**
    *   **WebGL2:** Standard 3D/2D rasterization.
    *   **WebGPU:** High-performance compute shaders and next-gen rendering.
    *   **Canvas 2D:** Fallback and simple drawing.
    *   **SVG:** Vector overlays for UI elements.
*   **Testing:** Vitest (Unit testing config present, but coverage is minimal/manual).

### Design Patterns
*   **Multi-Page Application (MPA):** Each experiment or component set lives in its own HTML file within `src/pages/`. There is no client-side router.
*   **Layered Rendering Composition:** A structural pattern where multiple `<canvas>` elements are stacked via CSS (`absolute` positioning, `z-index`) to allow different rendering contexts (e.g., WebGL background + WebGPU particles) to visually merge. See `LayeredCanvas` in `src/js/main.js`.
*   **Component-as-Class:** Logic is encapsulated in classes (e.g., `LEDButton`, `HybridEngine`) that manage their own DOM lifecycle, event listeners, and render loops.
*   **Progressive Enhancement:** Systems like `RenderingSupport` (`src/js/main.js`) detect capabilities (WebGPU) and allow components to degrade gracefully or display warnings.

---

## 2. Feature Map

| Feature | Description | Entry Point / Key File |
| :--- | :--- | :--- |
| **Core Utilities** | Canvas layering, shader compilation helpers, feature detection. | `src/js/main.js` |
| **Global Effects** | Cursor trails, audio visualization, holographic overlays that persist/apply across pages. | `src/js/experiments-global.js` |
| **Hybrid Engine** | Proof-of-concept combining WebGL2 terrain with WebGPU compute particles. | `src/js/hybrid-engine.js` |
| **Quantum Stream** | Advanced hybrid experiment with twisting tunnel (WebGL2) and data stream (WebGPU). | `src/js/quantum-data-stream.js` |
| **UI: Knobs** | Interactive rotary controls with SVG/WebGL visuals. | `src/js/knobs.js`, `src/js/knob-3d.js` |
| **UI: Buttons** | LED and physical-style buttons with glow effects. | `src/js/buttons.js` |
| **UI: Switches** | Toggle and rocker switches. | `src/js/switches.js` |
| **UI: Indicators** | Gauges, VU meters, and displays. | `src/js/indicators.js` |

---

## 3. Complexity Hotspots (The "Complex Parts")

### A. Hybrid Rendering Synchronization
*   **Location:** `src/js/hybrid-engine.js`, `src/js/quantum-data-stream.js`
*   **The Challenge:** These modules run two separate render contexts (WebGL2 and WebGPU) simultaneously.
*   **Complexity:**
    *   **Resize Sync:** Both canvases must be resized exactly together to maintain visual alignment. Mismatches result in "floating" visual artifacts.
    *   **Loop Coordination:** The `animate()` loop drives both. If one context crashes or hangs, it may stall the other or desync animations dependent on `u_time`.
*   **Agent Note:** When modifying `resize()` methods, ensure **Device Pixel Ratio (DPR)** is applied identically to both contexts.

### B. WebGPU Compute & Memory Alignment
*   **Location:** `src/js/quantum-data-stream.js` (and other WebGPU files)
*   **The Challenge:** Data exchange between CPU (JavaScript) and GPU (WGSL shaders) requires strict memory layout adherence.
*   **Complexity:**
    *   **Padding/Alignment:** WGSL structs (e.g., `Uniforms`) often require 16-byte alignment. JavaScript `Float32Array` buffers must be manually padded to match. A mismatch of even 4 bytes will cause invisible rendering or GPU device loss.
    *   **Async Init:** WebGPU initialization is asynchronous (`await navigator.gpu.requestAdapter()`). Constructors cannot be async, so initialization logic is often deferred to an `init()` method.
*   **Agent Note:** Always verify WGSL struct alignment rules (std140) when adding new uniform variables.

### C. Global Event Management
*   **Location:** `src/js/experiments-global.js`
*   **The Challenge:** This module attaches heavy listeners (`mousemove`, `deviceorientation`, `AudioContext`) to the global scope.
*   **Complexity:**
    *   **Resource Leaks:** While less critical in an MPA (page reload clears state), strictly speaking, these classes should implement `destroy()` methods to clean up listeners if the architecture ever shifts to SPA.
    *   **Audio Context State:** Browsers block AudioContext until user interaction. The `AudioVisualizerSystem` handles this, but it creates a complex state where the visualizer exists but is "dormant" until clicked.

---

## 4. Inherent Limitations & "Here be Dragons"

### Known Issues & Technical Debt
1.  **Manual Matrix Math:** The project does not currently use a math library (like `gl-matrix`). Matrix multiplications for 3D projections are often hardcoded or simplified in shaders.
    *   *Risk:* Complex 3D transformations are difficult to implement and error-prone.
    *   *Constraint:* Do not introduce a heavy external math library unless absolutely necessary to keep the "vanilla" ethos.
2.  **Hardcoded Asset Paths:** Some HTML files or JS modules may reference assets via relative paths (e.g., `../assets/`). Moving files requires careful auditing of these strings.
3.  **WebGPU Fallbacks:** While some files have fallbacks, others might simply fail silently or show a black screen if WebGPU is absent.
    *   *Dragon:* Testing in non-WebGPU environments (e.g., older mobiles, Safari without flags) is crucial before marking features "complete".

### Hard Constraints
*   **No TypeScript:** The project is pure JavaScript. Do not introduce `.ts` files or build steps requiring `tsc` without a full refactor mandate.
*   **Dark Mode Enforced:** The UI is designed for dark backgrounds (`#000` or deep gray). Light mode is not supported and would break contrast on glow effects.
*   **Directory Structure:** The build configuration (`vite.config.js`) relies on the specific `src/pages/` and `src/js/` structure. Do not reorganize root folders arbitrarily.

---

## 5. Dependency Graph & Key Flows

### Critical Flow: Experiment Initialization
1.  **User Access:** Opens `src/pages/experiment-name.html`.
2.  **Bootstrapping:**
    *   Browser loads HTML.
    *   Script module imports `../js/main.js` (initializes `RenderingSupport`).
    *   Script module imports specific experiment class (e.g., `../js/quantum-data-stream.js`).
3.  **Instantiation:**
    *   Inline script executes `new ExperimentClass(container)`.
    *   **Step 3a (Async):** WebGPU context requested.
    *   **Step 3b (Sync):** WebGL2 context created immediately.
4.  **Render Loop:**
    *   `requestAnimationFrame` calls `render()`.
    *   `render()` updates Physics/Compute -> draws WebGL background -> draws WebGPU overlay.

### Critical Flow: Global Effects
1.  `experiments-global.js` is imported in `main.js`.
2.  `DOMContentLoaded` triggers `GlobalExperiments.init()`.
3.  Checks config flags -> Instantiates `CursorTrailSystem` (Canvas 2D overlay on `body`).
