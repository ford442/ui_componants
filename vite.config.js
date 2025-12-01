import { resolve } from 'path';
import { defineConfig } from 'vite';

export default defineConfig({
  root: 'src',
  build: {
    outDir: '../dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/index.html'),
        buttons: resolve(__dirname, 'src/pages/buttons.html'),
        composite: resolve(__dirname, 'src/pages/composite_blending.html'),
        indicators: resolve(__dirname, 'src/pages/indicators.html'),
        knobs: resolve(__dirname, 'src/pages/knobs.html'),
        switches: resolve(__dirname, 'src/pages/switches.html'),
      },
    },
  },
});
