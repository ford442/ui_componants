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
        surfaces: resolve(__dirname, 'src/pages/surfaces.html'),
        switches: resolve(__dirname, 'src/pages/switches.html'),
        experiments: resolve(__dirname, 'src/pages/experiments.html'),
        portal: resolve(__dirname, 'src/pages/portal_vortex.html'),
        neon_city: resolve(__dirname, 'src/pages/neon_city.html'),
        hologram: resolve(__dirname, 'src/pages/hologram.html'),
        pattern_tests: resolve(__dirname, 'src/pages/pattern_tests.html'),
        gravitational_nebula: resolve(__dirname, 'src/pages/gravitational-nebula.html'),
        crystal_cavern: resolve(__dirname, 'src/pages/crystal-cavern.html'),
        cyber_rain: resolve(__dirname, 'src/pages/cyber-rain.html'),
        biomechanical_growth: resolve(__dirname, 'src/pages/biomechanical-growth.html'),
        quantum_data_stream: resolve(__dirname, 'src/pages/quantum-data-stream.html'),
        cyber_crystal: resolve(__dirname, 'src/pages/cyber-crystal.html'),
        bioluminescent_abyss: resolve(__dirname, 'src/pages/bioluminescent-abyss.html'),
        plasma_confinement: resolve(__dirname, 'src/pages/plasma-confinement.html'),
        cosmic_string: resolve(__dirname, 'src/pages/cosmic-string.html'),
        hybrid_magnetic_field: resolve(__dirname, 'src/pages/hybrid-magnetic-field.html'),
        stellar_forge: resolve(__dirname, 'src/pages/stellar-forge.html'),
        singularity_reactor: resolve(__dirname, 'src/pages/singularity-reactor.html'),
      },
    },
  },
});
