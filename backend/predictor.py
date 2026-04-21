import sys
import os
import urllib.parse
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble import ensemble_predict
from backend.shap_explainer import generate_threat_matrix
from dotenv import load_dotenv

load_dotenv()
VT_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')

def is_valid_url(url: str) -> bool:
    try:
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def check_virustotal(url: str):
    if not VT_API_KEY:
        return None
    try:
        import base64
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        headers = {"x-apikey": VT_API_KEY}
        response = requests.get(endpoint, headers=headers, timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            stats = data['data']['attributes']['last_analysis_stats']
            if stats['malicious'] > 2:
                return {
                    "verdict": "Phishing",
                    "confidence": 0.999,
                    "rf_prob": 1.0,
                    "lstm_prob": 1.0,
                    "threat_matrix": [{"feature": "VIRUSTOTAL DETERMINISTIC OVERRIDE", "value": 1.0, "shap_score": 1.0}]
                }
    except Exception as e:
        print(f"Warning: VirusTotal engine bypass -> {e}")
        return None
    return None

def analyze_url(url: str) -> dict:
    if not is_valid_url(url):
        raise ValueError("Invalid URL format. Please include proper domain structure.")

    vt_result = check_virustotal(url)
    if vt_result:
        return vt_result

    try:
        pred_results = ensemble_predict(url)
        threat_list = generate_threat_matrix(url)
    except Exception as e:
        raise RuntimeError(f"Underlying Engine Failure: {str(e)}")

    return {
        "verdict": pred_results["verdict"],
        "confidence": pred_results["confidence"],
        "rf_prob": pred_results["rf_prob"],
        "lstm_prob": pred_results["lstm_prob"],
        "threat_matrix": threat_list
    }
