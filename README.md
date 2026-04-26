# PhishGuard AI — Real-time Phishing Detection Engine

## Overview

PhishGuard AI is an advanced, real-time phishing URL detection engine developed at **FAST NUCES Karachi**. It identifies malicious domains by analyzing structural characteristics, lexical patterns, and heuristic behaviors without relying on slow network loop-backs (like WHOIS or traffic rankings).

The application integrates an ensemble artificial intelligence engine wrapping a **Random Forest Classifier** and a **Bidirectional Long Short-Term Memory (LSTM)** network, demystified by **SHAP** Explainable AI.

## Project Contributors
- **Minhal Raza (24K-0554)** — Frontend UI, Dynamic Components, and Data Pipelining
- **Ali Rooman (24K-0792)** — Deep Learning (LSTM) & Machine Learning Engine
- **Madni Moazam (24K-0868)** — Feature Extraction & Deterministic Heuristics Pipeline
- **Muhammad Yaseen (24K-0877)** — API Routing, Verification, and Explainable AI (XAI / SHAP)

**Instructor:** Mr. Atif Luqman

---

## 🏗 Architecture

The system operates across three tightly integrated layers:

1. **Frontend**: A vanilla JavaScript dynamic interface containing an animated threat matrix and probability color-scales.
2. **Deterministic Heuristics Framework**: Real-time interceptions of known spoofing vectors, including Hex/Integer IPs, IDN Homographs, credential trick URLs, and Cloud script abuse.
3. **Core ML Ensemble**: The dual-architecture AI combining:
   - **80% Weight:** Random Forest evaluating 30 specialized URL structure metrics.
   - **20% Weight:** PyTorch Bidirectional LSTM analyzing URLs character-by-character for zero-day obfuscation detection.

---

## 🚀 Running the Application

### 1. Requirements

Ensure you have Python 3.9+ installed and initialized in a virtual environment (`venv`). The application dependencies are fully isolated.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt # (Ensure FastAPI, Scikit-Learn, PyTorch, Uvicorn, SHAP are installed)
```

### 2. Startup

To run the application, start the backend FastAPI server and serve the frontend statically.

```bash
# Start backend server
python -m uvicorn backend.main:app --port 8000 &

# Serve UI
cd frontend
python -m http.server 8080
```

Open a browser and navigate to `http://localhost:8080`.

### 3. Usage & Explainability

Type any URL into the search bar. The tool will parse it symmetrically through both AI models simultaneously, projecting the SHAP output locally to show exactly *which* URL feature determined the verdict. Green results denote structural safety, while red denotes recognized phishing structures.

---

*This application was fully localized, removing previously defunct dependencies on synchronous VirusTotal checks or defunct Alexa rank APIs, establishing a sub-3 second total response cycle.*
