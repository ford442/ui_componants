// Glitch Scroll Reveal
// Triggers a cyberpunk glitch animation when sections scroll into view.

class GlitchScroll {
    constructor(elements) {
        this.elements = elements;
        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.triggerGlitch(entry.target);
                    // Optional: stop observing once revealed
                    // this.observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });

        this.elements.forEach(el => this.observer.observe(el));
    }

    triggerGlitch(element) {
        element.style.animation = 'none';
        element.offsetHeight; // trigger reflow
        element.classList.add('glitch-active');

        // Remove class after animation completes to reset state or allow re-trigger
        setTimeout(() => {
            element.classList.remove('glitch-active');
        }, 800);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Inject required styles
    const style = document.createElement('style');
    style.textContent = `
        .glitch-reveal {
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.5s, transform 0.5s;
        }
        
        .glitch-active {
            opacity: 1;
            transform: translateY(0);
            animation: glitch-anim 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) both;
            position: relative;
        }
        
        @keyframes glitch-anim {
            0% {
                transform: translate(0);
                clip-path: polygon(0 2%, 100% 2%, 100% 5%, 0 5%);
            }
            20% {
                transform: translate(-5px, 5px);
                clip-path: polygon(0 15%, 100% 15%, 100% 15%, 0 15%);
                filter: hue-rotate(90deg);
            }
            40% {
                transform: translate(5px, -5px);
                clip-path: polygon(0 10%, 100% 10%, 100% 20%, 0 20%);
                filter: hue-rotate(180deg);
            }
            60% {
                transform: translate(-5px, 0);
                clip-path: polygon(0 1%, 100% 1%, 100% 2%, 0 2%);
            }
            80% {
                transform: translate(0);
                clip-path: polygon(0 30%, 100% 30%, 100% 30%, 0 30%);
            }
            100% {
                transform: translate(0);
                clip-path: none;
                filter: none;
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);

    const glitchElements = document.querySelectorAll('.glitch-reveal');
    if (glitchElements.length > 0) {
        new GlitchScroll(glitchElements);
    }
});
