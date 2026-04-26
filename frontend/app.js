/* ============================================================
   PhishGuard AI — Frontend Engine
   Handles API communication, animated result rendering,
   and SHAP threat matrix visualization.
   ============================================================ */

const API_BASE = 'http://127.0.0.1:8000';

// --- DOM References ---
const analyzeBtn    = document.getElementById('analyze-btn');
const urlInput      = document.getElementById('url-input');
const loader        = document.getElementById('loader');
const results       = document.getElementById('results');
const errorBox      = document.getElementById('error-box');
const verdictBanner = document.getElementById('verdict-banner');
const verdictIcon   = document.getElementById('verdict-icon');
const verdictText   = document.getElementById('verdict-text');
const confidenceVal = document.getElementById('confidence-val');
const rfProbVal     = document.getElementById('rf-prob-val');
const rfBar         = document.getElementById('rf-bar');
const lstmProbVal   = document.getElementById('lstm-prob-val');
const lstmBar       = document.getElementById('lstm-bar');
const threatChart   = document.getElementById('threat-chart');

// --- Event Bindings ---
analyzeBtn.addEventListener('click', analyzeUrl);
urlInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') analyzeUrl(); });

// Glitch effect on title at startup
initGlitch();

// ============================================================
// Core Analysis Flow
// ============================================================

async function analyzeUrl() {
    const url = urlInput.value.trim();

    // Reset state
    errorBox.classList.add('hidden');
    results.classList.add('hidden');
    errorBox.textContent = '';

    if (!url) {
        showError('SYSTEM ERROR: NO TARGET URL PROVIDED. AWAITING INPUT.');
        return;
    }

    // Disable button during request
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '[ SCANNING... ]';
    loader.classList.remove('hidden');
    setRandomScanText();

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        if (!response.ok) {
            let detail = 'API engine failure.';
            try { detail = (await response.json()).detail || detail; } catch (_) {}
            throw new Error(detail);
        }

        const data = await response.json();
        renderResults(data);

    } catch (e) {
        if (e.message.toLowerCase().includes('failed to fetch') ||
            e.message.toLowerCase().includes('networkerror')) {
            showError('NETWORK ERROR: Cannot reach PhishGuard server at localhost:8000.\nEnsure the backend is running: python -m uvicorn backend.main:app --port 8000');
        } else {
            showError(`ENGINE ERROR: ${e.message}`);
        }
    } finally {
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '[ EXECUTE ]';
    }
}

// ============================================================
// Result Rendering
// ============================================================

function renderResults(data) {
    const isPhishing = data.verdict === 'Phishing';

    // Verdict panel
    verdictBanner.className = `verdict-panel ${isPhishing ? 'danger' : 'safe'}`;
    verdictIcon.textContent  = isPhishing ? '[!]' : '[✓]';
    verdictText.textContent  = isPhishing ? '🚨  PHISHING DETECTED' : '✅  URL IS SAFE';

    // Animated confidence counter
    animateCounter(confidenceVal, 0, data.confidence * 100, 900, '%');

    // RF probability bar
    const rfPct  = (data.rf_prob  * 100);
    const lstmPct = (data.lstm_prob * 100);
    renderProbBar(rfBar,   rfProbVal,   rfPct,   isPhishing);
    renderProbBar(lstmBar, lstmProbVal, lstmPct, isPhishing);

    // SHAP Threat Matrix
    renderThreatMatrix(data.threat_matrix);

    // Show results with entrance animation
    results.classList.remove('hidden');
    results.style.animation = 'none';
    requestAnimationFrame(() => {
        results.style.animation = 'fade-in 0.4s ease-out';
    });
}

function renderProbBar(barEl, labelEl, pct, isPhishing) {
    // Color the bar: high probability in a phishing verdcit = danger red
    const pctFormatted = pct.toFixed(1);
    barEl.style.width = '0%';

    // Determine dangerous threshold: RF prob > 50% in phishing context
    const isDangerous = pct > 50;
    barEl.style.background = isDangerous
        ? 'var(--red)'
        : (pct > 30 ? '#f59e0b' : 'var(--green)');
    barEl.style.boxShadow = isDangerous
        ? '0 0 8px var(--red)'
        : (pct > 30 ? '0 0 8px #f59e0b' : '0 0 8px var(--green)');

    // Animate width
    requestAnimationFrame(() => {
        setTimeout(() => { barEl.style.width = `${pct}%`; }, 50);
    });

    animateCounter(labelEl, 0, pct, 800, '%');
}

function renderThreatMatrix(threatMatrix) {
    threatChart.innerHTML = '<div class="chart-center-line"></div>';

    if (!threatMatrix || threatMatrix.length === 0) {
        threatChart.innerHTML += '<p style="color: var(--text-muted); font-size:0.85rem; padding:1rem 0;">No SHAP data available.</p>';
        return;
    }

    let maxImpact = 0;
    threatMatrix.forEach(f => {
        if (Math.abs(f.shap_score) > maxImpact) maxImpact = Math.abs(f.shap_score);
    });
    maxImpact = (maxImpact * 1.1) || 0.01;

    threatMatrix.forEach((threat, idx) => {
        const row = document.createElement('div');
        row.className = 'chart-row';
        row.style.animationDelay = `${idx * 60}ms`;

        const widthPct    = (Math.abs(threat.shap_score) / maxImpact) * 100;
        const formatImp   = Math.abs(threat.shap_score).toFixed(4);
        const isPhishPush = threat.shap_score > 0;
        const featureLabel = threat.feature.replace(/_/g, ' ').toUpperCase();

        const featureValueLabel = formatFeatureValue(threat.feature, threat.value);

        let leftBar  = '';
        let rightBar = '';

        if (!isPhishPush) {
            leftBar = `
                <span class="bar-val bar-val-left">-${formatImp}</span>
                <div class="bar-fill bar-safe" style="width:0%" data-target="${widthPct}"></div>`;
        } else {
            rightBar = `
                <div class="bar-fill bar-phish" style="width:0%" data-target="${widthPct}"></div>
                <span class="bar-val bar-val-right">+${formatImp}</span>`;
        }

        row.innerHTML = `
            <div class="chart-label" title="${threat.feature}">
                ${featureLabel}
                <span class="feature-val">${featureValueLabel}</span>
            </div>
            <div class="chart-bars-area">
                <div class="chart-left">${leftBar}</div>
                <div class="chart-right">${rightBar}</div>
            </div>
        `;
        threatChart.appendChild(row);
    });

    // Animate all bars after DOM insertion
    requestAnimationFrame(() => {
        setTimeout(() => {
            threatChart.querySelectorAll('.bar-fill[data-target]').forEach(bar => {
                bar.style.width = `${bar.dataset.target}%`;
            });
        }, 100);
    });
}

// ============================================================
// Utility Functions
// ============================================================

function formatFeatureValue(featureName, value) {
    // Map numeric values to human-readable labels
    if (value === 1)  return '[ SAFE ]';
    if (value === -1) return '[ PHISH ]';
    if (value === 0)  return '[ SUSPICIOUS ]';
    return `[ ${value} ]`;
}

function animateCounter(el, from, to, duration, suffix = '') {
    const start = performance.now();
    const range = to - from;
    function update(now) {
        const elapsed = Math.min(now - start, duration);
        const progress = elapsed / duration;
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = `${(from + range * eased).toFixed(2)}${suffix}`;
        if (elapsed < duration) requestAnimationFrame(update);
        else el.textContent = `${to.toFixed(2)}${suffix}`;
    }
    requestAnimationFrame(update);
}

function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
    errorBox.style.animation = 'none';
    requestAnimationFrame(() => {
        errorBox.style.animation = 'fade-in 0.3s ease-out';
    });
}

const SCAN_TEXTS = [
    'INJECTING URL INTO ENSEMBLE ENGINE...',
    'RUNNING RANDOM FOREST CLASSIFIER...',
    'QUERYING NEURAL SEQUENCE MODEL...',
    'COMPUTING SHAP EXPLAINABILITY SCORES...',
    'CROSS-REFERENCING THREAT SIGNATURES...',
    'ANALYZING LEXICAL PATTERNS...',
    'EVALUATING DOMAIN HEURISTICS...',
    'SYNTHESIZING THREAT MATRIX...'
];

let scanTextInterval = null;
function setRandomScanText() {
    const el = document.querySelector('.scanning-text');
    if (!el) return;
    let i = 0;
    el.textContent = SCAN_TEXTS[0];
    if (scanTextInterval) clearInterval(scanTextInterval);
    scanTextInterval = setInterval(() => {
        i = (i + 1) % SCAN_TEXTS.length;
        el.textContent = SCAN_TEXTS[i];
    }, 900);

    // Auto-clear when loader hides
    const observer = new MutationObserver(() => {
        if (loader.classList.contains('hidden')) {
            clearInterval(scanTextInterval);
            observer.disconnect();
        }
    });
    observer.observe(loader, { attributes: true, attributeFilter: ['class'] });
}

// ============================================================
// Glitch Title Animation
// ============================================================

function initGlitch() {
    const glitchEl = document.querySelector('.glitch');
    if (!glitchEl) return;

    const original = glitchEl.dataset.text;
    const GLITCH_CHARS = '!<>-_\\/[]{}—=+*^?#░▒▓@%$ABCXYZ01';

    let glitchInterval = null;

    function triggerGlitch() {
        let iterations = 0;
        clearInterval(glitchInterval);
        glitchInterval = setInterval(() => {
            glitchEl.textContent = original.split('').map((char, idx) => {
                if (char === ' ') return ' ';
                if (idx < iterations) return original[idx];
                return GLITCH_CHARS[Math.floor(Math.random() * GLITCH_CHARS.length)];
            }).join('');
            if (iterations >= original.length) {
                clearInterval(glitchInterval);
                glitchEl.textContent = original;
            }
            iterations += 1 / 2;
        }, 35);
    }

    // Trigger on load + periodically
    triggerGlitch();
    setInterval(triggerGlitch, 8000);
}
