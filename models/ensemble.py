"""
ensemble.py
===========
Loads the trained Random Forest and Bidirectional LSTM models and combines
their predictions into a weighted ensemble.

Weights: 80% RF + 20% LSTM
"""

import torch
import numpy as np
import sys
import os
import pickle
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train_lstm import URL_LSTM
from backend.feature_extractor import extract_features


class EnsembleEngine:
    """Lazy-init container for both models. Loaded once, reused globally."""

    def __init__(self):
        # ── Random Forest ──────────────────────────────────────────────────
        rf_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
        self.rf_model = joblib.load(rf_path)

        rf_features_path = os.path.join(os.path.dirname(__file__), 'rf_feature_names.pkl')
        with open(rf_features_path, 'rb') as f:
            self.rf_feature_names = pickle.load(f)

        # ── LSTM ───────────────────────────────────────────────────────────
        config_path = os.path.join(os.path.dirname(__file__), 'lstm_config.pkl')
        with open(config_path, 'rb') as f:
            lstm_config      = pickle.load(f)
            self.char_to_ix  = lstm_config['vocab']
            self.max_len     = lstm_config['max_len']

        lstm_path = os.path.join(os.path.dirname(__file__), 'lstm_model.pt')
        self.device = torch.device('cpu')

        # Load state dict to inspect dimensions → infer bidirectionality
        state = torch.load(lstm_path, map_location=self.device, weights_only=True)
        vocab_size = len(self.char_to_ix)

        # Detect whether the saved model is bidirectional from the FC weight shape
        fc_weight_shape = state['fc.weight'].shape  # (1, fc_input_dim)
        fc_input_dim    = fc_weight_shape[1]
        is_bidir        = (fc_input_dim > 64)       # 128 = bidir, 64 = single

        self.lstm_model = URL_LSTM(
            vocab_size=vocab_size,
            bidirectional=is_bidir,
        )
        self.lstm_model.load_state_dict(state)
        self.lstm_model.eval()

    def tokenize_url(self, url: str) -> torch.Tensor:
        vec = [self.char_to_ix.get(c, 0) for c in url[:self.max_len]]
        if len(vec) < self.max_len:
            vec.extend([0] * (self.max_len - len(vec)))
        return torch.tensor([vec], dtype=torch.long)  # (1, max_len)

    def ensemble_predict(self, url: str) -> dict:
        # ── RF prediction ──────────────────────────────────────────────────
        features = extract_features(url)
        x_rf     = np.array([[features.get(k, 0) for k in self.rf_feature_names]])
        rf_prob  = float(self.rf_model.predict_proba(x_rf)[0][1])

        # ── LSTM prediction ────────────────────────────────────────────────
        x_lstm = self.tokenize_url(url)
        with torch.no_grad():
            lstm_prob = float(self.lstm_model(x_lstm).item())

        # ── Weighted ensemble ──────────────────────────────────────────────
        ensemble_prob = 0.80 * rf_prob + 0.20 * lstm_prob
        verdict       = "Phishing" if ensemble_prob > 0.5 else "Safe"
        confidence    = ensemble_prob if verdict == "Phishing" else 1.0 - ensemble_prob

        return {
            "verdict":    verdict,
            "confidence": float(confidence),
            "rf_prob":    rf_prob,
            "lstm_prob":  lstm_prob,
        }


# ── Global singleton ───────────────────────────────────────────────────────────
engine = None


def ensemble_predict(url: str) -> dict:
    global engine
    if engine is None:
        engine = EnsembleEngine()
    return engine.ensemble_predict(url)
