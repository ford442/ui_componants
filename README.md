# UI Components

A comprehensive library of interactive UI components featuring **layered canvas effects** with WebGL, WebGL2, WebGPU, CSS, and SVG rendering contexts.

## Features

### Layered Rendering Contexts
Each component demonstrates different combinations of rendering technologies:
- **WebGL** - Hardware-accelerated 2D/3D graphics with shader effects
- **WebGL2** - Advanced shader effects, textures, and compute
- **WebGPU** - Next-generation GPU compute and rendering (where supported)
- **CSS** - Filters, transforms, animations, and blend modes
- **SVG** - Scalable vector overlays and filter effects

### Components

#### 🔘 [Buttons](pages/buttons.html)
LED-illuminated animated buttons with various styles:
- Basic LED buttons with WebGL glow
- RGB color-cycling buttons
- Momentary push buttons with spring animation
- Pulsing indicator buttons
- 4x4 button matrix with pattern animations
- Arcade, Industrial, Holographic, and Organic styles

#### 🎛️ [Knobs](pages/knobs.html)
Rotary knobs with multiple rendering layers:
- Basic rotary knobs with SVG and WebGL glow
- LED ring knobs with animated indicators
- Vintage amplifier-style knobs
- Digital encoders with step displays
- Large interactive layered knob demo
- Virtual mixer console with 8 channels
- Meter knobs, dual concentric, illuminated, and step knobs

#### 🔀 [Switches](pages/switches.html)
Toggle switches with glow and transition effects:
- Basic toggle switches with WebGL glow
- LED toggle switches with indicators
- Rocker switches (industrial style)
- 3-position slide switches
- Control panel with mixed switch types
- Neon, Flip, Retro, and Segmented styles

#### 📊 [Indicators](pages/indicators.html)
LED indicators, meters, and status displays:
- LED status indicators (various colors)
- VU meters with animated bars
- Analog gauge meters with needle animation
- Seven-segment digital displays
- Status dashboard with real-time updates
- Ring meters (circular progress)
- Oscilloscope waveform display
- Spectrum analyzer
- LED dot matrix display

## Getting Started

Simply open `index.html` in a modern web browser. No build process required.

```bash
# Open the project
open index.html
# Or serve locally
python -m http.server 8000
```

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 14+
- Edge 80+

WebGPU features require Chrome 113+ with appropriate flags enabled.

## Project Structure

```
ui_componants/
├── index.html          # Main navigation page
├── css/
│   ├── main.css        # Shared styles
│   ├── buttons.css     # Button-specific styles
│   ├── knobs.css       # Knob-specific styles
│   ├── switches.css    # Switch-specific styles
│   └── indicators.css  # Indicator-specific styles
├── js/
│   ├── main.js         # Core utilities & components
│   ├── buttons.js      # Button page initialization
│   ├── knobs.js        # Knobs page initialization
│   ├── switches.js     # Switches page initialization
│   └── indicators.js   # Indicators page initialization
└── pages/
    ├── buttons.html    # Buttons showcase
    ├── knobs.html      # Knobs showcase
    ├── switches.html   # Switches showcase
    └── indicators.html # Indicators showcase
```

## Core API

### UIComponents.LayeredCanvas
Manages multiple canvas layers with different rendering contexts.

```javascript
const layered = new UIComponents.LayeredCanvas(container, {
    width: 800,
    height: 400
});

layered.addLayer('base', 'webgl', 0);
layered.addLayer('effects', 'webgl2', 1);
layered.addSVGLayer('overlay', 2);
layered.startAnimation();
```

### UIComponents.LEDButton
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

### UIComponents.RotaryKnob
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

## License

MIT License - Feel free to use in your projects.
