// Data Definitions
const translations = {
    words: [
        { lang: 'English', text: 'ENTER', color: 'text-cyan-500' },
        { lang: 'Spanish', text: 'ENTRAR', color: 'text-magenta-500' },
        { lang: 'Mandarin', text: '进入', color: 'text-yellow-500' },
        { lang: 'Arabic', text: 'دخول', color: 'text-green-500' },
        { lang: 'Hindi', text: 'प्रवेश', color: 'text-orange-500' },
    ],
    numbers: [
        { lang: 'Western', text: '5', color: 'text-cyan-500' },
        { lang: 'Eastern Arabic', text: '٥', color: 'text-magenta-500' },
        { lang: 'Chinese', text: '五', color: 'text-yellow-500' },
        { lang: 'Devanagari', text: '५', color: 'text-green-500' },
        { lang: 'Roman', text: 'V', color: 'text-orange-500' },
    ],
    status: [
        { lang: 'English', text: 'SYSTEM OK', color: 'text-cyan-500' },
        { lang: 'Spanish', text: 'SISTEMA OK', color: 'text-magenta-500' },
        { lang: 'Japanese', text: 'システムOK', color: 'text-yellow-500' },
        { lang: 'French', text: 'SYSTÈME OK', color: 'text-green-500' },
        { lang: 'German', text: 'SYSTEM OK', color: 'text-orange-500' },
    ]
};

// State
const state = {
    activeTab: 'scroll', // 'scroll' | 'stack'
    contentType: 'words',
    isPlaying: true,
    scrollIndex: 0,
    focusLayer: null, // null = all
    chaosMode: true
};

// DOM Elements
const displayContainer = document.getElementById('display-container');
const observationText = document.getElementById('observation-text');

// Init
function init() {
    setupEventListeners();
    setInterval(tick, 2000); // The scroll ticker
    render();
}

function setupEventListeners() {
    // Tab Switching
    document.getElementById('btn-tab-scroll').addEventListener('click', () => setTab('scroll'));
    document.getElementById('btn-tab-stack').addEventListener('click', () => setTab('stack'));

    // Content Switching
    document.querySelectorAll('.btn-content').forEach(btn => {
        btn.addEventListener('click', (e) => {
            state.contentType = e.target.getAttribute('data-type');
            updateContentButtons();
            render();
        });
    });

    // Scroll Controls
    document.getElementById('btn-toggle-play').addEventListener('click', () => {
        state.isPlaying = !state.isPlaying;
        updatePlayButton();
    });

    // Stack Controls
    document.getElementById('btn-chaos-mode').addEventListener('click', () => {
        state.chaosMode = !state.chaosMode;
        document.getElementById('btn-chaos-mode').textContent = state.chaosMode ? "Disable Transparency" : "Enable Transparency";
        render();
    });
}

function setTab(tab) {
    state.activeTab = tab;

    // UI Updates
    const btnScroll = document.getElementById('btn-tab-scroll');
    const btnStack = document.getElementById('btn-tab-stack');

    if (tab === 'scroll') {
        btnScroll.classList.add('bg-blue-600', 'text-white');
        btnScroll.classList.remove('text-slate-400');
        btnStack.classList.remove('bg-blue-600', 'text-white');
        btnStack.classList.add('text-slate-400');

        document.getElementById('controls-scroll').classList.remove('hidden');
        document.getElementById('controls-stack').classList.add('hidden');
    } else {
        btnStack.classList.add('bg-blue-600', 'text-white');
        btnStack.classList.remove('text-slate-400');
        btnScroll.classList.remove('bg-blue-600', 'text-white');
        btnScroll.classList.add('text-slate-400');

        document.getElementById('controls-scroll').classList.add('hidden');
        document.getElementById('controls-stack').classList.remove('hidden');
    }
    render();
}

function updateContentButtons() {
    document.querySelectorAll('.btn-content').forEach(btn => {
        const type = btn.getAttribute('data-type');
        if (type === state.contentType) {
            btn.classList.add('bg-indigo-600', 'text-white');
            btn.classList.remove('text-slate-400');
        } else {
            btn.classList.remove('bg-indigo-600', 'text-white');
            btn.classList.add('text-slate-400');
        }
    });
}

function updatePlayButton() {
    const iconPause = document.getElementById('icon-pause');
    const iconPlay = document.getElementById('icon-play');
    if (state.isPlaying) {
        iconPause.classList.remove('hidden');
        iconPlay.classList.add('hidden');
    } else {
        iconPause.classList.add('hidden');
        iconPlay.classList.remove('hidden');
    }
}

// Logic Loop
function tick() {
    if (state.isPlaying && state.activeTab === 'scroll') {
        const data = translations[state.contentType];
        state.scrollIndex = (state.scrollIndex + 1) % data.length;
        renderScroll();
    }
}

// Render Functions
function render() {
    displayContainer.innerHTML = '';

    if (state.activeTab === 'scroll') {
        displayContainer.className = "relative h-80 rounded-lg overflow-hidden flex flex-col items-center justify-center transition-all duration-300 scroll-mode-container";
        renderScroll();
        renderObservationsScroll();
    } else {
        displayContainer.className = "relative h-80 rounded-lg overflow-hidden flex items-center justify-center transition-all duration-300 stack-mode-container";
        renderStack();
        renderObservationsStack();
    }
}

function renderScroll() {
    const data = translations[state.contentType];
    const prevIndex = (state.scrollIndex - 1 + data.length) % data.length;
    const nextIndex = (state.scrollIndex + 1) % data.length;
    const currIndex = state.scrollIndex;

    const html = `
        <div class="absolute top-2 left-2 text-xs text-slate-500 font-mono z-20">Simulated LED Matrix</div>
        <div class="scroll-mask"></div>
        <div class="flex flex-col items-center gap-8">
            <div class="scroll-item scroll-item-blur">${data[prevIndex].text}</div>
            <div class="scroll-item scroll-item-active">${data[currIndex].text}</div>
            <div class="scroll-item scroll-item-blur">${data[nextIndex].text}</div>
        </div>
        <div class="absolute right-4 top-1/2 -translate-y-1/2 flex flex-col gap-2 z-20">
            ${data.map((_, idx) => `
                <div class="w-2 h-2 rounded-full transition-colors ${idx === state.scrollIndex ? 'bg-cyan-400' : 'bg-slate-800'}"></div>
            `).join('')}
        </div>
    `;
    displayContainer.innerHTML = html;
}

function renderStack() {
    const data = translations[state.contentType].slice(0, 3); // Limit to 3 for CMYK effect
    const colors = [
        'text-cyan-600 mix-blend-multiply',
        'text-magenta-600 mix-blend-multiply',
        'text-yellow-500 mix-blend-multiply'
    ];

    let html = `<div class="absolute top-2 left-2 text-xs text-slate-400 font-mono">CMYK Overlay Simulation</div>`;

    data.forEach((item, idx) => {
        const isFocused = state.focusLayer === null || state.focusLayer === idx;
        const opacity = isFocused ? (state.chaosMode ? 0.8 : 1) : 0.05;
        const scale = isFocused ? 'scale(1)' : 'scale(0.95)';
        const blur = isFocused ? 'blur(0px)' : 'blur(4px)';
        const zIndex = isFocused ? 10 : 0;

        html += `
            <div class="stack-item ${colors[idx % 3]}"
                 style="opacity: ${opacity}; transform: ${scale}; filter: ${blur}; z-index: ${zIndex};">
                ${item.text}
            </div>
        `;
    });

    displayContainer.innerHTML = html;
    renderLayerToggles(data);
}

function renderLayerToggles(data) {
    const container = document.getElementById('layer-toggles');

    // "ALL" Button
    let buttonsHtml = `
        <button class="layer-btn flex-1 py-2 text-sm font-mono border rounded ${state.focusLayer === null ? 'border-blue-400 bg-blue-400/10 text-blue-300' : 'border-slate-600 text-slate-500 hover:border-slate-500'}"
                data-idx="null">
            ALL
        </button>
    `;

    // Layer Buttons
    data.forEach((item, idx) => {
        const isActive = state.focusLayer === idx;
        let borderClass = '';
        if (idx === 0) borderClass = 'border-cyan-500 text-cyan-600';
        if (idx === 1) borderClass = 'border-pink-500 text-pink-600';
        if (idx === 2) borderClass = 'border-yellow-500 text-yellow-600';

        const activeClass = isActive ? 'bg-slate-200 ring-2 ring-white/20' : 'opacity-60 hover:opacity-100';

        buttonsHtml += `
            <button class="layer-btn flex-1 py-2 text-sm font-bold border rounded transition-colors ${borderClass} ${activeClass}"
                    data-idx="${idx}">
                ${item.lang.substring(0,3).toUpperCase()}
            </button>
        `;
    });

    container.innerHTML = buttonsHtml;

    // Re-attach listeners to new buttons
    container.querySelectorAll('.layer-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const val = btn.getAttribute('data-idx');
            state.focusLayer = val === "null" ? null : parseInt(val);
            render();
        });
    });
}

function renderObservationsScroll() {
    observationText.innerHTML = `
        <p>
            <strong class="text-blue-400">The "Ticker" Effect:</strong> By showing the previous and next language partially obscured, we remove the "Glitch Feeling."
        </p>
        <p>The movement implies a system at work. If it were a static fade, users might think the screen is broken or stuck.</p>
        <div class="bg-slate-900 p-3 rounded text-xs border border-slate-700 mt-2">
            <strong>Verdict:</strong> Best for passive waiting areas (Elevators, queues). Bad for emergency controls.
        </div>
    `;
}

function renderObservationsStack() {
    observationText.innerHTML = `
        <p>
            <strong class="text-pink-400">The "Spectral" Risk:</strong>
            Notice how hard it is to read "All" unless the characters are vastly different shapes (like Numbers).
        </p>
        <p>
            When "Action" words overlap (e.g., ENTER / ENTRAR), they create a black blob because the shapes are too similar.
        </p>
        <p>
            However, overlapping <strong>Numbers</strong> (5 / ٥ / 五) is surprisingly parseable because the stroke density is lower and shapes are distinct.
        </p>
        <div class="bg-slate-900 p-3 rounded text-xs border border-slate-700 mt-2">
            <strong>Verdict:</strong> Fails for text. Plausible for single numerals or distinct icons, but relies heavily on colorblind accessibility.
        </div>
    `;
}

// Start
init();
