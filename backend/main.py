import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from backend.predictor import analyze_url

from backend.shap_explainer import generate_threat_matrix as _warm_shap

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("SYSTEM LOG: Warming up ensemble engine (RF + LSTM)...")
    try:
        analyze_url("https://google.com")
        print("SYSTEM LOG: Ensemble engine loaded successfully.")
    except Exception as e:
        print(f"SYSTEM LOG: Warning — ensemble warm-up failed: {e}")

    print("SYSTEM LOG: Warming up SHAP explainer (this may take ~15s on first run)...")
    try:
        _warm_shap("https://google.com")
        print("SYSTEM LOG: SHAP explainer ready.")
    except Exception as e:
        print(f"SYSTEM LOG: Warning — SHAP warm-up failed: {e}")

    print("SYSTEM LOG: PhishGuard API fully online and ready.")
    yield


app = FastAPI(title="PhishGuard AI Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

@app.post("/analyze")
def analyze(req: URLRequest):
    try:
        result = analyze_url(req.url)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
def health():
    return {"status": "ok", "message": "PhishGuard API Engine is online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
