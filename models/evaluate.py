

import sys
import os
import pickle
import numpy as np
import joblib
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from models.train_lstm import URL_LSTM
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

SEP = "=" * 60


def load_rf():
    rf_path = os.path.join(MODELS_DIR, 'rf_model.pkl')
    fn_path = os.path.join(MODELS_DIR, 'rf_feature_names.pkl')
    rf_model = joblib.load(rf_path)
    with open(fn_path, 'rb') as f:
        feature_names = pickle.load(f)
    return rf_model, feature_names


def load_lstm():
    cfg_path  = os.path.join(MODELS_DIR, 'lstm_config.pkl')
    lstm_path = os.path.join(MODELS_DIR, 'lstm_model.pt')
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    model = URL_LSTM(vocab_size=len(cfg['vocab']))
    model.load_state_dict(
        torch.load(lstm_path, map_location='cpu', weights_only=True)
    )
    model.eval()
    return model


def evaluate_rf(rf_model, feature_names):
    print(f"\n{SEP}")
    print("  RANDOM FOREST EVALUATION (UCI Phishing Websites)")
    print(SEP)

    with open(os.path.join(DATA_DIR, 'uci_test.pkl'), 'rb') as f:
        test = pickle.load(f)
    X_test, y_test = test['X_test'], test['y_test']

    x_rf = np.array([[row.get(k, 0) if hasattr(row, 'get') else row[i]
                       for i, k in enumerate(feature_names)]
                     for row in [X_test.iloc[i] for i in range(len(X_test))]])
    x_rf = X_test[feature_names].values

    y_pred      = rf_model.predict(x_rf)
    y_prob      = rf_model.predict_proba(x_rf)[:, 1]
    acc         = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Safe (0)', 'Phishing (1)']))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':12s}  Pred Safe  Pred Phish")
    print(f"  {'Actual Safe':12s}  {cm[0][0]:9d}  {cm[0][1]:10d}")
    print(f"  {'Actual Phish':12s}  {cm[1][0]:9d}  {cm[1][1]:10d}")

    if acc >= 0.85:
        print(f"\n✅  TARGET ACCURACY ≥ 85% REACHED: {acc*100:.2f}%")
    else:
        print(f"\n⚠️   TARGET NOT MET — current: {acc*100:.2f}%  (target: 85.00%)")

    return y_prob, y_test


def evaluate_lstm(lstm_model):
    print(f"\n{SEP}")
    print("  LSTM EVALUATION (Character-level Kaggle URLs)")
    print(SEP)

    with open(os.path.join(DATA_DIR, 'kaggle_test.pkl'), 'rb') as f:
        test = pickle.load(f)

    X_test = torch.tensor(test['X'], dtype=torch.long)
    y_test = test['y']

    BATCH = 512
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_test), BATCH):
            batch = X_test[start:start + BATCH]
            out   = lstm_model(batch)
            probs.extend(out.cpu().numpy().tolist())

    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Safe (0)', 'Phishing (1)']))
    return y_prob, y_test


def evaluate_ensemble(rf_prob, lstm_prob, y_test):
    print(f"\n{SEP}")
    print("  ENSEMBLE EVALUATION (80% RF + 20% LSTM)")
    print(SEP)
    n         = min(len(rf_prob), len(lstm_prob), len(y_test))
    ens_prob  = 0.80 * rf_prob[:n] + 0.20 * lstm_prob[:n]
    y_pred    = (ens_prob > 0.5).astype(int)
    y_true    = np.array(y_test)[:n]
    acc       = accuracy_score(y_true, y_pred)

    print(f"\nAccuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Safe (0)', 'Phishing (1)']))
    if acc >= 0.85:
        print(f"✅  ENSEMBLE TARGET ≥ 85% REACHED")
    else:
        print(f"⚠️   ENSEMBLE BELOW TARGET — {acc*100:.2f}%")


def main():
    print(f"\n{SEP}")
    print("  PhishGuard AI — Model Evaluation Suite")
    print(SEP)

    print("\nLoading models...")
    rf_model, feature_names = load_rf()
    lstm_model               = load_lstm()
    print("Models loaded.")

    rf_prob,   y_rf   = evaluate_rf(rf_model, feature_names)
    lstm_prob, y_lstm = evaluate_lstm(lstm_model)
    if len(rf_prob) == len(lstm_prob):
        evaluate_ensemble(rf_prob, lstm_prob, y_rf)
    else:
        print(f"\n{SEP}")
        print("  NOTE: RF and LSTM test sets have different sizes")
        print(f"  RF test: {len(rf_prob)} samples | LSTM test: {len(lstm_prob)} samples")
        print("  Ensemble evaluation skipped (train both on the same source to enable).")

    print(f"\n{SEP}\n")


if __name__ == '__main__':
    main()
