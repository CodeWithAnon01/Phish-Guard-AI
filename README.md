<div align="center">

# 🛡️ PhishGuard AI

### Real-Time AI-Powered Phishing URL Detection System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

*A full-stack cybersecurity application that identifies phishing URLs in real-time using a dual-model machine learning ensemble with SHAP explainability.*

</div>

---

## 📌 Overview

PhishGuard AI is an end-to-end cybersecurity solution built as an AI course project. It combines traditional machine learning with deep learning to detect phishing URLs with high accuracy. The system provides a glass-box (explainable) prediction pipeline — every verdict comes with a visual breakdown of which URL features drove the decision.

**Key Goals:**
- ✅ Detect phishing URLs in real-time (< 200ms per prediction)
- ✅ Exceed 85% accuracy on standard phishing datasets
- ✅ Explain *why* a URL was flagged using SHAP values
- ✅ Serve predictions via a REST API consumable by any frontend

---

## 🧠 How It Works

PhishGuard uses a **dual-model ensemble** — two completely different AI models that analyze a URL from different angles, then combine their predictions:

```
URL Input
   │
   ├──► Random Forest (Structural Analysis)
   │         • 30 heuristic features extracted from URL syntax
   │         • Domain patterns, length, symbols, TLD, subdomains
   │         • Trained on UCI Phishing Websites Dataset (11,055 URLs)
   │         • Accuracy: 96.7%
   │
   ├──► LSTM Neural Network (Lexical Analysis)
   │         • Character-level tokenization of the raw URL string
   │         • Detects suspicious character sequences and patterns
   │         • Trained on Kaggle Phishing URLs Dataset (80,000 URLs)
   │         • Accuracy: 91.9%
   │
   └──► Weighted Ensemble (80% RF + 20% LSTM)
             • Final probability score
             • Verdict: PHISHING or SAFE
             • Confidence percentage

   ↓
SHAP Explainer
   • Runs TreeExplainer on the RF prediction
   • Returns top 10 most impactful features
   • Positive SHAP = pushes toward Phishing
   • Negative SHAP = pushes toward Safe
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Dual-Model Ensemble** | Random Forest + LSTM analyze every URL from structural and lexical dimensions |
| 🔍 **SHAP Explainability** | Visual bar chart showing exactly which features influenced the prediction |
| ⚡ **Real-Time** | Sub-200ms inference with fully offline feature extraction |
| 🌐 **REST API** | FastAPI backend with CORS — works with any frontend or tool |
| 🎨 **Cyber UI** | Dark terminal-themed frontend with animated scanner and live charts |
| 🔑 **VirusTotal Ready** | Optional API key integration for deterministic threat intel override |

---

## 🗂️ Project Structure

```
phishguard-ai/
│
├── backend/                    # FastAPI application
│   ├── main.py                 # API routes (/analyze, /health)
│   ├── predictor.py            # analyze_url() orchestrator + VirusTotal hook
│   ├── feature_extractor.py    # 30-feature URL heuristic extractor
│   ├── shap_explainer.py       # SHAP TreeExplainer wrapper
│   └── __init__.py
│
├── models/                     # ML model definitions & saved weights
│   ├── ensemble.py             # EnsembleEngine: loads + combines RF & LSTM
│   ├── train_random_forest.py  # RF training script (UCI dataset)
│   ├── train_lstm.py           # LSTM training script (Kaggle dataset)
│   ├── rf_model.pkl            # Trained Random Forest (96.7% accuracy)
│   ├── rf_feature_names.pkl    # Feature column order for inference alignment
│   ├── lstm_model.pt           # Trained LSTM weights (91.9% accuracy)
│   ├── lstm_config.pkl         # Character vocab + max_len config
│   └── __init__.py
│
├── data/                       # Data preparation scripts + processed data
│   ├── prepare_uci.py          # Downloads + preprocesses UCI dataset
│   ├── prepare_kaggle.py       # Preprocesses Kaggle CSV → tokenized tensors
│   ├── uci_train.pkl           # Processed UCI training set
│   ├── uci_test.pkl            # Processed UCI test set
│   ├── kaggle_train.pkl        # Tokenized Kaggle training set
│   ├── kaggle_test.pkl         # Tokenized Kaggle test set
│   └── char_vocab.pkl          # Character vocabulary (shared with LSTM)
│
├── frontend/                   # Static web UI
│   ├── index.html              # Main page (dark terminal aesthetic)
│   ├── style.css               # Cyber theme: monospace fonts, neon accents
│   └── app.js                  # API calls + dynamic chart rendering
│
├── .gitignore
├── start.sh                    # One-command server launcher
└── README.md
```

---

## 🤖 Models

### Random Forest Classifier
- **Dataset:** [UCI Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites) (11,055 URLs, 30 features)
- **Hyperparameters:** `n_estimators=150`, `max_depth=25`, `min_samples_split=2`
- **Test Accuracy:** **96.70%**
- **Features:** IP address detection, URL length, shortening service, `@` symbol, double-slash redirect, prefix-suffix (dashes), subdomain depth, SSL state, HTTPS token, TLD heuristic, DNS record, and 19 HTML-level features

### LSTM Neural Network
- **Dataset:** [Kaggle Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) (80,000 URLs balanced)
- **Architecture:** `Embedding(245, 32) → LSTM(64) → Dropout(0.5) → Linear(1) → Sigmoid`
- **Training:** 15 epochs max, early stopping on **validation loss** (patience=3)
- **Test Accuracy:** **91.97%** | **F1: 0.9144**

### Ensemble Strategy
```
Final Score = (0.80 × RF_probability) + (0.20 × LSTM_probability)
Verdict     = "Phishing" if Final Score > 0.5 else "Safe"
Confidence  = Final Score (if Phishing) or 1 - Final Score (if Safe)
```

The RF is weighted 4× higher because:
1. It was trained on a larger, cleaner dataset relative to its task
2. The LSTM subsample (40k URLs) introduces a slight false-positive bias on short legitimate domains
3. Structural heuristics are more deterministic than character pattern recognition

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/phishguard-ai.git
cd phishguard-ai
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn scikit-learn shap pandas numpy torch python-multipart python-dotenv joblib
```

### 4. (Optional) Add VirusTotal API Key
Create a `.env` file in the project root:
```
VIRUSTOTAL_API_KEY=your_key_here
```
Get a free key at [virustotal.com](https://www.virustotal.com/gui/join-community). Free tier = 500 requests/day.

### 5. Prepare Data & Train Models
> Skip this if you already have the `.pkl` and `.pt` model files.

```bash
# Download UCI dataset (auto-fetched via ucimlrepo)
python data/prepare_uci.py

# Download phishing_site_urls.csv from Kaggle and place in datasets/
# https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
python data/prepare_kaggle.py          # 40k subsample (fast)
python data/prepare_kaggle.py --full   # Full 500k dataset (slower, more accurate)

# Train models
python models/train_random_forest.py   # ~30 seconds | Expected: >96% accuracy
python models/train_lstm.py            # ~3-5 minutes | Expected: >91% accuracy
```

### 6. Start the Backend
```bash
chmod +x start.sh
./start.sh
```
The API will be live at `http://localhost:8000`

Verify it's running:
```bash
curl http://localhost:8000/health
# → {"status":"ok","message":"PhishGuard API Engine is online"}
```

### 7. Launch the Frontend
```bash
cd frontend
python3 -m http.server 8001
```
Open `http://localhost:8001` in your browser.

---

## 🔌 API Reference

### `POST /analyze`
Analyze a URL and return a phishing verdict with SHAP explanation.

**Request:**
```json
{ "url": "http://paypa1-secure-login.xyz/verify/account" }
```

**Response:**
```json
{
  "verdict": "Phishing",
  "confidence": 0.949,
  "rf_prob": 0.94,
  "lstm_prob": 0.99,
  "threat_matrix": [
    { "feature": "prefix_suffix",   "value": -1.0, "shap_score": 0.182 },
    { "feature": "sslfinal_state",  "value": -1.0, "shap_score": 0.157 },
    { "feature": "age_of_domain",   "value": -1.0, "shap_score": 0.134 },
    ...
  ]
}
```

### `GET /health`
```json
{ "status": "ok", "message": "PhishGuard API Engine is online" }
```

---

## 🌐 Deployment

PhishGuard uses a split deployment:
- **Frontend** → [Vercel](https://vercel.com) (free static hosting)
- **Backend** → [Render](https://render.com) (free Python web service)

See the detailed deployment guide in [`DEPLOY.md`](./DEPLOY.md) for step-by-step instructions including how to host large model files via Google Drive.

---

## 📊 SHAP Explainability

Every prediction includes a **SHAP Threat Matrix** — a horizontal bar chart showing the top 10 features that influenced the decision:

- 🔴 **Red bar (right)** = Feature pushed the prediction toward **Phishing** (positive SHAP)
- 🟢 **Green bar (left)** = Feature pushed the prediction toward **Safe** (negative SHAP)
- Bar length = magnitude of impact

This makes PhishGuard a **glass-box** system — not a black-box — which is critical for security tool trustworthiness.

---

## 📁 Datasets

| Dataset | Source | Size | Used For |
|---------|--------|------|----------|
| UCI Phishing Websites | [UCI ML Repo](https://archive.ics.uci.edu/dataset/327/phishing+websites) | 11,055 URLs, 30 features | Random Forest training |
| Kaggle Phishing URLs | [Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) | 549,346 URLs | LSTM training |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn (Random Forest), PyTorch (LSTM) |
| Explainability | SHAP (TreeExplainer) |
| Backend API | FastAPI + Uvicorn |
| Threat Intel | VirusTotal API v3 (optional) |
| Frontend | Vanilla HTML/CSS/JS |
| Data Processing | pandas, numpy |
| Model Serialization | joblib (RF), torch.save (LSTM) |

---

## 👨‍💻 Author

Built as an AI course project — Semester 4, FAST NUCES.

---

<div align="center">
<sub>PhishGuard AI — Because every click matters.</sub>
</div>
