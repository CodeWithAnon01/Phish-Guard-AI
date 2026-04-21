import shap
import warnings
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ensemble import EnsembleEngine
from backend.feature_extractor import extract_features
import numpy as np

warnings.filterwarnings('ignore')

class ShapExplainer:
    def __init__(self):
        self.engine = EnsembleEngine()

        uci_train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'uci_train.pkl')
        with open(uci_train_path, 'rb') as f:
            train_data = pickle.load(f)
            X_train = train_data['X_train']

        background = X_train.sample(n=100, random_state=42)
        self.explainer = shap.TreeExplainer(self.engine.rf_model, background)
        self.feature_names = getattr(self.engine, 'rf_feature_names', [])

    def generate_threat_matrix(self, url: str) -> list:
        features = extract_features(url)
        x_array = np.array([[features.get(k, -1) for k in self.feature_names]])

        shap_values = self.explainer.shap_values(x_array)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == len(self.feature_names):
            sv = shap_values[0]
        else:
            sv = shap_values[0, :, 1]

        threat_matrix = []
        for i, sv_val in enumerate(sv):
            threat_matrix.append({
                "feature": self.feature_names[i],
                "value": float(x_array[0][i]),
                "shap_score": float(sv_val)
            })

        threat_matrix.sort(key=lambda x: abs(x['shap_score']), reverse=True)
        return threat_matrix[:10]

explainer = None
def generate_threat_matrix(url: str) -> list:
    global explainer
    if explainer is None:
        explainer = ShapExplainer()
    return explainer.generate_threat_matrix(url)
