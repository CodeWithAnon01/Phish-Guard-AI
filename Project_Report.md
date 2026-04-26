---
title: Final Project Report
subtitle: PhishGuard AI - Real-time Phishing Detection Engine
author: 
  - Minhal Raza (24K-0554)
  - Ali Rooman (24K-0792)
  - Madni Moazam (24K-0868)
  - Muhammad Yaseen (24K-0877)
date: \today
university: FAST NUCES Karachi
instructor: Mr. Atif Luqman
---

# FAST NUCES Karachi
## PhishGuard AI: Real-Time Phishing Detection Engine
**Submitted To:** Mr. Atif Luqman

---

## 1. Abstract
Phishing attacks remain one of the most prevalent and effective vectors for cyber intrusion. Traditional detection mechanisms often rely on static blacklists or synchronous third-party API lookups, causing them to fail against zero-day threats or suffer from high latency. This project, **PhishGuard AI**, introduces a hybrid ensemble AI architecture capable of real-time phishing URL detection. By fusing a Random Forest structural analyzer with a Bidirectional Long Short-Term Memory (LSTM) sequence model—and underpinning the predictions with a custom deterministic heuristic engine—the system provides robust defense against modern adversarial web threats (e.g., IDN homographs, hex-encoded IPs, and open redirects). 

## 2. System Architecture
The application is structured into a multi-layered pipeline to ensure speed, accuracy, and interpretability:

1. **Frontend Presentation Layer (`frontend/`)**: A dynamic, cyberpunk-themed cybersecurity terminal built using Vanilla JavaScript. It communicates synchronously with the API, featuring real-time loading states, color-reactive probability bars, and a graphical rendering of the SHAP Threat Matrix.
2. **Deterministic Heuristics Engine (`backend/predictor.py` & `backend/feature_extractor.py`)**: Before AI classification, URLs pass through a heuristics layer designed to catch obfuscation bypass techniques. This handles credential tricks (username@), hex/integer IP parsing, and subdomain typosquatting directly.
3. **Machine Learning Ensemble (`models/`)**: 
   - **Random Forest (80% Weight)**: Evaluates the 30 structural/lexical features (like `url_length`, `having_sub_domain`, `prefix_suffix`).
   - **Bidirectional LSTM (20% Weight)**: Evaluates the raw character sequence of the URL, detecting anomalous lexical structures that bypass feature-mapping.
4. **Explainable AI (XAI)**: We implemented TreeExplainer via SHAP (SHapley Additive exPlanations) to demystify the Random Forest predictions, generating a transparent threat matrix that informs the user *why* a URL was flagged.

## 3. Implementation Details & Group Contributions
The 10 core code files were logically divided among the team members, allowing distributed domain ownership over the full-stack architecture:

### Minhal Raza (24K-0554)
**Domain: Frontend UI & Data Pipeline**
- **`frontend/app.js`, `frontend/index.html`, `frontend/style.css`**: Designed and developed the entirely dynamic user interface. Implemented the frontend threat matrix rendering and asynchronous API integration.
- **`data/prepare_uci.py`**, **`data/prepare_kaggle.py`**: Engineered the data ingestion pipelines, handling tokenization, binary serialization, and feature mapping of the massive datasets to feed the models.

### Muhammad Yaseen (24K-0877)
**Domain: API Routing, Evaluation & XAI**
- **`backend/main.py`**: Developed the FastAPI router infrastructure, including the lifecycle pre-warmer that optimizes SHAP cold-starts for sub-3-second responses.
- **`models/evaluate.py`**: Wrote the evaluation and validation scripts capable of generating ROC-AUC scores, confusion matrices, and systemic cross-model validation tracking.
- **`backend/shap_explainer.py`**: Implemented the SHAP explainer component, translating matrix impacts into normalized numeric scores mapped to explicit URL structural features.

### Ali Rooman (24K-0792)
**Domain: Core Machine Learning & Deep Learning Models**
- **`models/train_lstm.py`**: Engineered and trained the PyTorch Bidirectional LSTM model for character-level NLP sequencing.
- **`models/train_random_forest.py`**: Tuned and trained the high-dimensional Random Forest classifier using Scikit-Learn.
- **`models/ensemble.py`**: Combined both architectures into a weighted ensemble singleton pattern, ensuring seamless memory sharing during concurrent inference requests.

### Madni Moazam (24K-0868)
**Domain: Feature Extraction & Deterministic Threat Logic**
- **`backend/feature_extractor.py`**: Programmed the extraction logic spanning 30 core heuristics—computing values purely via string evaluation and regex (e.g., detecting TinyURL patterns, resolving standard pathing, and calculating URL depth) to avoid network latency.
- **`backend/predictor.py`**: Established the central prediction loop, including the implementation of the advanced edge-case deterministic overrides (such as IDN homograph detection and URL unrolling).

## 4. Results and Evaluation
The model achieved its milestone goal by exceeding an 85% combined accuracy threshold on the testing split while resolving queries completely asynchronously in under 3.0 seconds per request natively. The SHAP integration successfully proved that the application operates transparently, exposing metrics back to the client interface accurately.

## 5. Conclusion
**PhishGuard AI** successfully demonstrates the feasibility and necessity of shifting away from network-reliant blacklist systems. By layering structural heuristics (Madni Moazam), Deep Learning (Ali Rooman), XAI transparency (Muhammad Yaseen), and a seamless responsive UI (Minhal Raza), the team delivered a production-ready, locally hosted cybersecurity prototype capable of defending against complex, modernized phishing attacks.
