
        import { GravityWellExperiment } from '../js/gravity-well.js';
        import { EnergyVortexExperiment } from '../js/energy-vortex.js';
        import { DarkEnergyPrism } from '../js/dark-energy-prism.js';
        import { CymaticPlate } from '../js/cymatic-plate.js';
        import { ChronoExcavation } from '../js/chrono-excavation.js';
        import { QuantumTensorExperiment } from '../js/quantum-tensor.js';
        import { Artifact404 } from '../js/artifact-404.js';
        import { ChronoDial } from '../js/chrono-dial.js';
        import { BioDigitalSynthesis } from '../js/bio-digital-synthesis.js';
        import { NanoPlexExperiment } from '../js/nano-plex.js';
        import { CosmicRadiation } from '../js/cosmic-radiation.js';
        import { PhotonContainmentExperiment } from '../js/photon-containment.js';
        import { SeismicWaveExperiment } from '../js/seismic-wave.js';
        import { VoidRift } from '../js/void-rift.js';
        import { QuantumDataStream } from '../js/quantum-data-stream.js';
        import { CyberCrystalExperiment } from '../js/cyber-crystal.js';
        import { BioluminescentAbyss } from '../js/bioluminescent-abyss.js';
        import { PlasmaConfinement } from '../js/plasma-confinement.js';
        import { StellarForge } from '../js/stellar-forge.js';
        import { FiberOpticExperiment } from '../js/fiber-optics.js';
        import { TemporalFissureExperiment } from '../js/temporal-fissure.js';
        import { NeuralLaceExperiment } from '../js/neural-lace.js';
        import { SpectralLoomExperiment } from '../js/spectral-loom.js';
        import { ChaosAttractor } from '../js/chaos-attractor.js';
        import { DysonSwarmExperiment } from '../js/dyson-swarm.js';
        import { NeuroMorphicCrystal } from '../js/neuro-morphic-crystal.js';
        import { SynapticFireExperiment } from '../js/synaptic-fire.js';
        import { HyperspaceTunnelExperiment } from '../js/hyperspace-tunnel.js';
        import { AtmosphericEntry } from '../js/atmospheric-entry.js';
        import { NeutrinoStormExperiment } from '../js/neutrino-storm.js';
        import { PrimordialSoup } from '../js/primordial-soup.js';
        import { PlanetaryTerraformingExperiment } from '../js/planetary-terraforming.js';
        import { QuantumStabilizer } from '../js/quantum-stabilizer.js';
        import { SupernovaRemnantExperiment } from '../js/supernova-remnant.js';
        import { NanobotConstruction } from '../js/nanobot-construction.js';

        // Initialize experiments when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            // Initialize Gravity Well
            const gravityWellContainer = document.getElementById('gravity-well-container');
            if (gravityWellContainer) {
                new GravityWellExperiment(gravityWellContainer, { numParticles: 30000 });
            }

            // Initialize Cymatic Plate
            const cymaticContainer = document.getElementById('cymatic-plate-container');
            if (cymaticContainer) {
                new CymaticPlate(cymaticContainer, { numParticles: 40000 });
            }

            // Initialize NanoPlex Assembly
            const nanoPlexContainer = document.getElementById('nano-plex-container');
            if (nanoPlexContainer) {
                new NanoPlexExperiment(nanoPlexContainer, { numParticles: 30000 });
            }

            // Initialize Photon Containment
            const photonContainer = document.getElementById('photon-containment-container');
            if (photonContainer) {
                new PhotonContainmentExperiment(photonContainer);
            }

            // Initialize Cosmic Radiation
            const cosmicContainer = document.getElementById('cosmic-radiation-container');
            if (cosmicContainer) {
                new CosmicRadiation(cosmicContainer);
            }

            // Initialize Void Rift
            const voidRiftContainer = document.getElementById('void-rift-container');
            if (voidRiftContainer) {
                new VoidRift(voidRiftContainer, { numParticles: 50000 });
            }

            // Initialize Stellar Forge
            const stellarForgeContainer = document.getElementById('stellar-forge-container');
            if (stellarForgeContainer) {
                new StellarForge(stellarForgeContainer, { numParticles: 50000 });
            }

            // Initialize Plasma Confinement
            const plasmaContainer = document.getElementById('plasma-confinement-container');
            if (plasmaContainer) {
                new PlasmaConfinement(plasmaContainer, { numParticles: 20000 });
            }

            // Initialize Bioluminescent Abyss
            const bioAbyssContainer = document.getElementById('bioluminescent-abyss-container');
            if (bioAbyssContainer) {
                new BioluminescentAbyss(bioAbyssContainer, { numParticles: 20000 });
            }

            // Initialize Cyber Crystal
            const cyberCrystalContainer = document.getElementById('cyber-crystal-container');
            if (cyberCrystalContainer) {
                new CyberCrystalExperiment(cyberCrystalContainer, { numParticles: 20000 });
            }

            // Initialize Quantum Data Stream
            const quantumStreamContainer = document.getElementById('quantum-data-stream-container');
            if (quantumStreamContainer) {
                new QuantumDataStream(quantumStreamContainer, { numParticles: 20000 });
            }

            // Initialize Cyber Rain
            const cyberRainContainer = document.getElementById('cyber-rain-container');
            if (cyberRainContainer && window.CyberRain) {
                new CyberRain(cyberRainContainer, { numParticles: 20000 });
            }

            // Initialize Crystal Cavern
            const crystalCavernContainer = document.getElementById('crystal-cavern-container');
            if (crystalCavernContainer && window.CrystalCavern) {
                new CrystalCavern(crystalCavernContainer, { numParticles: 15000 });
            }

            // Initialize Acoustic Levitation
            const acousticContainer = document.getElementById('acoustic-levitation-container');
            if (acousticContainer && window.AcousticLevitation) {
                new AcousticLevitation(acousticContainer, { numParticles: 20000 });
            }

            // Initialize Cosmic String
            const cosmicStringContainer = document.getElementById('cosmic-string-container');
            if (cosmicStringContainer && window.CosmicStringExperiment) {
                new CosmicStringExperiment(cosmicStringContainer, { numParticles: 20000 });
            }

            // Initialize Chrono Dial
            const chronoDialContainer = document.getElementById('chrono-dial-container');
            if (chronoDialContainer) {
                new ChronoDial(chronoDialContainer);
            }

            // Initialize Seismic Wave
            const seismicWaveContainer = document.getElementById('seismic-wave-container');
            if (seismicWaveContainer && window.SeismicWaveExperiment) {
                new SeismicWaveExperiment(seismicWaveContainer, { gridSize: 100 });
            }

            // Initialize Temporal Data Core
            const temporalCoreContainer = document.getElementById('temporal-core-container');
            if (temporalCoreContainer && window.TemporalCore) {
                new TemporalCore(temporalCoreContainer, { numParticles: 20000 });
            }

            // Initialize Gravitational Nebula
            const nebulaContainer = document.getElementById('gravitational-nebula-container');
            if (nebulaContainer && window.GravitationalNebula) {
                new GravitationalNebula(nebulaContainer, { numParticles: 50000 });
            }

            // Initialize Hyper-Cube
            const hyperCubeContainer = document.getElementById('hyper-cube-container');
            if (hyperCubeContainer && window.HyperCubeExperiment) {
                new HyperCubeExperiment(hyperCubeContainer, { numParticles: 50000 });
            }

            // Initialize Neon City
            const neonCityContainer = document.getElementById('neon-city-container');
            if (neonCityContainer && window.NeonCityExperiment) {
                new NeonCityExperiment(neonCityContainer, { instanceCount: 1000, numRainDrops: 5000 });
            }

            // Initialize Biomechanical Growth
            const biomechContainer = document.getElementById('biomechanical-growth-container');
            if (biomechContainer && window.BiomechanicalGrowth) {
                new BiomechanicalGrowth(biomechContainer, { numParticles: 15000 });
            }

            // Initialize Hybrid Engine
            const hybridContainer = document.getElementById('hybrid-engine-container');
            if (hybridContainer && window.HybridEngine) {
                new HybridEngine(hybridContainer, { numParticles: 20000 });
            }

            // Initialize Bio-Digital Synthesis
            const bioDigitalContainer = document.getElementById('bio-digital-synthesis-container');
            if (bioDigitalContainer && window.BioDigitalSynthesis) {
                new BioDigitalSynthesis(bioDigitalContainer, { numParticles: 30000 });
            }

            // Initialize Fiber Optics
            const fiberOpticsContainer = document.getElementById('fiber-optics-container');
            if (fiberOpticsContainer) {
                new FiberOpticExperiment(fiberOpticsContainer);
            }
            // Initialize Chrono Excavation
            const chronoExcavationContainer = document.getElementById('chrono-excavation-container');
            if (chronoExcavationContainer && window.ChronoExcavation) {
                new ChronoExcavation(chronoExcavationContainer, { numParticles: 40000 });
            }

            // Initialize Quantum Tensor
            const quantumTensorContainer = document.getElementById('quantum-tensor-container');
            if (quantumTensorContainer) {
                new QuantumTensorExperiment(quantumTensorContainer, { particleCount: 20000 });
            }

            // Initialize Temporal Fissure
            const temporalFissureContainer = document.getElementById('temporal-fissure-container');
            if (temporalFissureContainer) {
                new TemporalFissureExperiment(temporalFissureContainer, { numParticles: 30000 });
            }

            // Initialize Neural Lace
            const neuralLaceContainer = document.getElementById('neural-lace-container');
            if (neuralLaceContainer) {
                new NeuralLaceExperiment(neuralLaceContainer, { numParticles: 30000 });
            }

            // Initialize Spectral Loom
            const spectralLoomContainer = document.getElementById('spectral-loom-container');
            if (spectralLoomContainer) {
                new SpectralLoomExperiment(spectralLoomContainer, { threadCount: 150 });
            }

            // Initialize Chaos Attractor
            const chaosAttractorContainer = document.getElementById('chaos-attractor-container');
            if (chaosAttractorContainer) {
                new ChaosAttractor(chaosAttractorContainer, { particleCount: 50000 });
            }

            // Initialize Energy Vortex
            const energyVortexContainer = document.getElementById('energy-vortex-container');
            if (energyVortexContainer) {
                new EnergyVortexExperiment(energyVortexContainer, { numParticles: 30000 });
            }

            // Initialize Subatomic Collider
            const subatomicColliderContainer = document.getElementById('subatomic-collider-container');
            if (subatomicColliderContainer) {
                new SubatomicColliderExperiment(subatomicColliderContainer, { numParticles: 30000 });
            }

            // Initialize Dark Energy Prism
            const darkEnergyPrismContainer = document.getElementById('dark-energy-prism-container');
            if (darkEnergyPrismContainer) {
                new DarkEnergyPrism(darkEnergyPrismContainer, { particleCount: 30000 });
            }

            // Initialize Artifact 404
            const artifact404Container = document.getElementById('artifact-404-container');
            if (artifact404Container) {
                new Artifact404(artifact404Container, { particleCount: 20000 });
            }

            // Initialize Hybrid Magnetic Field
            const hybridMagneticContainer = document.getElementById('hybrid-magnetic-field-container');
            if (hybridMagneticContainer && window.HybridMagneticField) {
                new HybridMagneticField(hybridMagneticContainer, { numParticles: 30000 });
            }

            // Initialize Neural Data Core
            const neuralDataCoreContainer = document.getElementById('neural-data-core-container');
            if (neuralDataCoreContainer && window.NeuralDataCore) {
                new NeuralDataCore(neuralDataCoreContainer, { numParticles: 40000 });
            }

            // Initialize Cyber-Biology
            const cyberBioContainer = document.getElementById('cyber-biology-container');
            if (cyberBioContainer && window.CyberBiologyExperiment) {
                new CyberBiologyExperiment(cyberBioContainer, { numParticles: 15000 });
            }

            // Initialize Neural Network
            const neuralNetContainer = document.getElementById('neural-network-container');
            if (neuralNetContainer && window.NeuralNetworkExperiment) {
                new NeuralNetworkExperiment(neuralNetContainer, { nodeCount: 150, pulseCount: 2000 });
            }

            // Initialize Fluid Sim
            const fluidSimContainer = document.getElementById('fluid-sim-container');
            if (fluidSimContainer && window.FluidSimulationExperiment) {
                new FluidSimulationExperiment(fluidSimContainer, { particleCount: 5000 });
            }

            // Initialize Force Field
            const forceFieldContainer = document.getElementById('force-field-container');
            if (forceFieldContainer && window.ForceFieldExperiment) {
                new ForceFieldExperiment(forceFieldContainer, { numParticles: 20000 });
            }

            // Initialize Dyson Swarm
            const dysonSwarmContainer = document.getElementById('dyson-swarm-container');
            if (dysonSwarmContainer) {
                new DysonSwarmExperiment(dysonSwarmContainer, { particleCount: 50000 });
            }

            // Initialize Neuro-Morphic Crystal
            const neuroCrystalContainer = document.getElementById('neuro-morphic-crystal-container');
            if (neuroCrystalContainer) {
                new NeuroMorphicCrystal(neuroCrystalContainer, { particleCount: 30000 });
            }

            // Initialize Synaptic Fire
            const synapticFireContainer = document.getElementById('synaptic-fire-container');
            if (synapticFireContainer) {
                new SynapticFireExperiment(synapticFireContainer, { numParticles: 40000 });
            }

            // Initialize Neutrino Storm
            const neutrinoStormContainer = document.getElementById('neutrino-storm-container');
            if (neutrinoStormContainer) {
                new NeutrinoStormExperiment(neutrinoStormContainer, { particleCount: 30000 });
            }

            // Initialize Hyperspace Tunnel
            const hyperspaceTunnelContainer = document.getElementById('hyperspace-tunnel-container');
            if (hyperspaceTunnelContainer) {
                new HyperspaceTunnelExperiment(hyperspaceTunnelContainer, { numParticles: 20000 });
            }

            // Initialize Atmospheric Entry
            const atmosphericEntryContainer = document.getElementById('atmospheric-entry-container');
            if (atmosphericEntryContainer) {
                new AtmosphericEntry(atmosphericEntryContainer, { numParticles: 30000 });
            }

            // Initialize Primordial Soup
            const soupContainer = document.getElementById('primordial-soup-container');
            if (soupContainer) {
                new PrimordialSoup(soupContainer, { particleCount: 3000 });
            }

            // Initialize Planetary Terraforming
            const terraformingContainer = document.getElementById('planetary-terraforming-container');
            if (terraformingContainer) {
                new PlanetaryTerraformingExperiment(terraformingContainer, { particleCount: 100000 });
            }
            // Initialize Quantum Stabilizer
            const quantumStabilizerContainer = document.getElementById('quantum-stabilizer-container');
            if (quantumStabilizerContainer) {
                new QuantumStabilizer(quantumStabilizerContainer, { numParticles: 20000 });
            }

            // Initialize Supernova Remnant
            const supernovaRemnantContainer = document.getElementById('supernova-remnant-container');
            if (supernovaRemnantContainer) {
                new SupernovaRemnantExperiment(supernovaRemnantContainer, { numParticles: 40000 });
            }
            // Initialize Nanobot Construction
            const nanobotContainer = document.getElementById('nanobot-construction-container');
            if (nanobotContainer) {
                new NanobotConstruction(nanobotContainer, { numParticles: 30000 });
            }

            // Create SVG Morphing Buttons
            const morphingContainer = document.getElementById('morphing-container');
            if (morphingContainer && window.MorphingButton) {
                const colors = ['#00ff88', '#00aaff', '#ff0088', '#ffaa00', '#8800ff'];
                colors.forEach(color => {
                    new MorphingButton(morphingContainer, {
                        width: 100,
                        height: 100,
                        shape1: 'circle',
                        shape2: 'square',
                        fillColor: color
                    });
                });
            }

            // Background effects state
            let auroraInstance = null;
            let particlesInstance = null;

            // Aurora Background toggle
            const auroraToggle = document.getElementById('toggle-aurora');
            auroraToggle?.addEventListener('change', (e) => {
                if (e.target.checked) {
                    if (!auroraInstance && window.AuroraBackground) {
                        auroraInstance = new AuroraBackground(document.body, {
                            intensity: 0.4,
                            speed: 0.04,
                            colors: [
                                [0.2, 0.8, 0.9],  // cyan
                                [0.9, 0.2, 0.8],  // magenta
                                [0.5, 0.3, 0.9]   // purple
                            ]
                        });
                    }
                } else {
                    if (auroraInstance) {
                        auroraInstance.destroy();
                        auroraInstance = null;
                    }
                }
            });

            // Magnetic Particles toggle
            const particlesToggle = document.getElementById('toggle-particles');
            if (particlesToggle) {
                particlesToggle.addEventListener('change', (e) => {
                    if (e.target.checked) {
                    if (!particlesInstance && window.MagneticParticleField) {
                        particlesInstance = new MagneticParticleField(document.body, {
                            particleCount: 1800,
                            particleSize: 2,
                            particleColor: 'rgba(0, 255, 136, 0.6)',
                            lineColor: 'rgba(0, 170, 255, 0.2)',
                            maxLineDistance: 100,
                            magneticForce: 0.03,
                            friction: 0.98,
                            constellationMode: true
                        });
                    }
                } else {
                    if (particlesInstance) {
                        particlesInstance.destroy();
                        particlesInstance = null;
                    }
                }
            });
            }
        });
