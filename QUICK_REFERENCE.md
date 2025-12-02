# Quick Reference: New Indicator Components

## Button Panel API

### HTML Container
```html
<div id="button-panel"></div>
```

### JavaScript Usage
```javascript
// Automatically initialized on DOMContentLoaded
// Creates 6 buttons with multi-layer ripple effects
```

### Button Types
1. **Power** (Red #ff3333) - ⚡
2. **Start** (Green #00ff88) - ▶
3. **Stop** (Orange #ff8800) - ■
4. **Reset** (Blue #00aaff) - ↻
5. **Alert** (Yellow #ffff00) - ⚠
6. **Lock** (Magenta #ff00ff) - 🔒

### Features
- 3-layer ripple animation on click
- Glow effects on hover
- Scale animation on press
- Customizable colors via border and shadow

---

## Multi-State Indicators API

### HTML Container
```html
<div id="multi-state-indicators"></div>
```

### Indicator Types

#### System Indicator
- States: idle (gray) → active (green) → error (red)
- Updates every ~2-3 seconds

#### Network Indicator
- States: offline (gray) → connecting (orange) → online (blue)
- Shows connection status

#### Battery Indicator
- States: low (red) → charging (yellow) → full (green)
- Displays power level

#### Signal Indicator
- States: weak (orange) → medium (yellow) → strong (green)
- Shows signal strength

### State Properties
- Each state has its own color
- Smooth transitions (0.3s)
- Active state gets glow effect
- Status text updates with state name

---

## WebGPU Components

### Holographic VU Meter
```javascript
// Container: #holographic-vu
// Size: 600x300px
// Features: 16 bars, 4 render layers, rainbow colors
```

**Layers:**
1. Main VU bars (variable height)
2. Exponential glow
3. Scanline effect
4. Chromatic aberration

### Fluid Simulation Meter
```javascript
// Container: #fluid-meter
// Size: 400x400px
// Features: Wave simulation, caustics, depth shading
```

**Effects:**
- Dual wave patterns
- Sin-based caustics
- Surface shimmer
- Container border

### Plasma Globe
```javascript
// Container: #plasma-globe
// Size: 400x400px
// Features: 5 plasma layers, rotating centers
```

**Visual Elements:**
- Volumetric sphere
- Dynamic color cycling
- Glass border
- Raymarched effect

### Waveform Synthesizer
```javascript
// Container: #waveform-synth
// Size: 600x200px
// Features: 4 waveforms, grid overlay
```

**Parameters:**
- Frequencies: 5, 7, 9, 11 Hz
- Phase offsets: 1.0, 1.5, 2.0, 2.5
- Color gradient per layer

### Crystalline Panel
```javascript
// Container: #crystalline-panel
// Size: 500x300px
// Features: Voronoi cells, procedural geometry
```

**Effects:**
- Animated cell morphing
- Per-cell color variation
- Bright edges
- Shimmer overlay

### Data Visualization
```javascript
// Container: #data-visualization
// Size: 600x400px
// Features: 32 bars, 3D depth effect
```

**Rendering:**
- Depth-based positioning
- HSV color mapping
- Top edge highlights
- Grid background

---

## CSS Classes Reference

### Button Panel
```css
.button-wrapper          /* Container for each button */
.control-button         /* Button element with effects */
.button-icon           /* Icon span */
.button-label          /* Label text */
```

### Multi-State Indicators
```css
.multi-state-indicator  /* Container for each indicator */
.indicator-label       /* Top label text */
.state-display         /* Container for state dots */
.state-dot            /* Individual state indicator */
.status-text          /* Current state text */
```

### WebGPU Components
```css
.webgpu-advanced-section  /* Main WebGPU section */
.webgpu-style            /* Canvas container styling */
.webgpu-warning          /* Warning for unsupported browsers */
.crystalline-panel       /* Special styling for crystalline */
```

---

## Customization Examples

### Change Button Colors
```javascript
// In initButtonPanel(), modify the buttons array:
{ label: 'Custom', color: '#00ffff', icon: '★' }
```

### Add More States
```javascript
// In initMultiStateIndicators(), extend indicators array:
{ 
  label: 'Custom', 
  states: ['state1', 'state2', 'state3'], 
  colors: ['#color1', '#color2', '#color3'] 
}
```

### Adjust WebGPU Animation Speed
```javascript
// In any WebGPU init function, modify:
time += 0.016; // Change to 0.008 for slower, 0.032 for faster
```

---

## Browser Support Matrix

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| Button Panel | ✅ All | ✅ All | ✅ All | ✅ All |
| Multi-State | ✅ All | ✅ All | ✅ All | ✅ All |
| WebGPU | ✅ 113+ | ✅ 113+ | 🚧 Nightly | 🚧 Preview |

Legend:
- ✅ Fully supported
- 🚧 Experimental/Preview
- ❌ Not supported

---

## Performance Tips

1. **Limit concurrent animations**: Don't run all WebGPU components simultaneously on low-end devices
2. **Check WebGPU support**: Always use `checkWebGPUSupport()` before initializing
3. **Cleanup on unmount**: Cancel animation frames when navigating away
4. **Throttle updates**: For real data, consider throttling to 30fps instead of 60fps

---

## Troubleshooting

### Buttons not appearing
- Check that `#button-panel` container exists in HTML
- Verify `initButtonPanel()` is called in DOMContentLoaded
- Check browser console for errors

### WebGPU not working
- Verify browser supports WebGPU (Chrome 113+)
- Check WebGPU flag is enabled: `chrome://flags/#enable-unsafe-webgpu`
- Look for adapter request failures in console

### State indicators not cycling
- Ensure `#multi-state-indicators` container exists
- Check setInterval is not being cleared
- Verify no JavaScript errors blocking execution

### Performance issues
- Reduce number of active WebGPU components
- Lower canvas resolution
- Increase animation interval (lower fps)

---

## Example Integration

### Add to existing page:
```html
<!-- In your HTML -->
<div id="button-panel"></div>
<div id="multi-state-indicators"></div>
<div id="holographic-vu"></div>

<!-- In your JavaScript -->
<script>
document.addEventListener('DOMContentLoaded', () => {
    initButtonPanel();
    initMultiStateIndicators();
    
    checkWebGPUSupport().then(supported => {
        if (supported) {
            initHolographicVU();
        }
    });
});
</script>
```

---

## License & Credits
Part of the UI Components Library
Built with WebGPU, Canvas, CSS3
Compatible with modern browsers

