const analyzeBtn = document.getElementById('analyze-btn');
const urlInput = document.getElementById('url-input');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const errorBox = document.getElementById('error-box');

const verdictBanner = document.getElementById('verdict-banner');
const verdictIcon = document.getElementById('verdict-icon');
const verdictText = document.getElementById('verdict-text');
const confidenceVal = document.getElementById('confidence-val');
const rfProbVal = document.getElementById('rf-prob-val');
const rfBar = document.getElementById('rf-bar');
const lstmProbVal = document.getElementById('lstm-prob-val');
const lstmBar = document.getElementById('lstm-bar');
const threatChart = document.getElementById('threat-chart');

analyzeBtn.addEventListener('click', analyzeUrl);
urlInput.addEventListener('keypress', (e) => { if(e.key === 'Enter') analyzeUrl(); });

async function analyzeUrl() {
    const url = urlInput.value.trim();
    errorBox.classList.add('hidden');
    results.classList.add('hidden');

    if(!url) {
        showError("SYSTEM ERROR: PLEASE INPUT A VALID TARGET URL.");
        return;
    }

    loader.classList.remove('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'API failure');
        }

        const data = await response.json();
        updateUI(data);

    } catch (e) {
        showError('Could not connect to PhishGuard server. Make sure the backend is running. Details: ' + e.message);
    } finally {
        loader.classList.add('hidden');
    }
}

function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
}

function updateUI(data) {
    if (data.verdict === "Phishing") {
        verdictBanner.className = 'verdict-panel danger';
        verdictText.textContent = '🚨 PHISHING';
        verdictIcon.textContent = '[!]';
    } else {
        verdictBanner.className = 'verdict-panel safe';
        verdictText.textContent = '✅ SAFE';
        verdictIcon.textContent = '[O]';
    }

    confidenceVal.textContent = `${(data.confidence * 100).toFixed(2)}%`;

    const rf = (data.rf_prob * 100).toFixed(1);
    const lstm = (data.lstm_prob * 100).toFixed(1);

    rfProbVal.textContent = `${rf}%`;
    rfBar.style.width = `${rf}%`;

    lstmProbVal.textContent = `${lstm}%`;
    lstmBar.style.width = `${lstm}%`;

    threatChart.innerHTML = '<div class="chart-center-line"></div>';

    let maxImpact = 0;
    data.threat_matrix.forEach(f => {
        if(Math.abs(f.shap_score) > maxImpact) maxImpact = Math.abs(f.shap_score);
    });
    maxImpact = maxImpact * 1.1 || 0.01;

    data.threat_matrix.forEach(threat => {
        const row = document.createElement('div');
        row.className = 'chart-row';

        let widthPct = (Math.abs(threat.shap_score) / maxImpact) * 100;
        let leftBar = '';
        let rightBar = '';
        const formatImpact = Math.abs(threat.shap_score).toFixed(4);

        if (threat.shap_score < 0) {
            leftBar = `<span class="bar-val" style="margin-right:8px;">-${formatImpact}</span><div class="bar-fill bar-safe" style="width: ${widthPct}%"></div>`;
        } else {
            rightBar = `<div class="bar-fill bar-phish" style="width: ${widthPct}%"></div><span class="bar-val" style="margin-left:8px;">+${formatImpact}</span>`;
        }

        row.innerHTML = `
            <div class="chart-label" title="${threat.feature}">${threat.feature.replace(/_/g, ' ')}</div>
            <div class="chart-bars-area">
                <div class="chart-left">${leftBar}</div>
                <div class="chart-right">${rightBar}</div>
            </div>
        `;
        threatChart.appendChild(row);
    });

    results.classList.remove('hidden');
}
