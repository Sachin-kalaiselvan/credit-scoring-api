from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np

app = FastAPI(
    title="Credit Scoring API",
    description="Predicts loan default probability and risk band for a borrower.",
    version="1.0.0"
)

# Load model and artifacts
model = xgb.Booster()
model.load_model("models/xgboost_final.json")

with open("models/best_threshold.pkl", "rb") as f:
    THRESHOLD = pickle.load(f)

with open("models/feature_list.pkl", "rb") as f:
    FEATURE_LIST = pickle.load(f)

# Input schema
class BorrowerInput(BaseModel):
    features: dict

def get_risk_band(prob):
    if prob < 0.20:   return "Low",      "Approve"
    elif prob < 0.40: return "Medium",   "Review"
    elif prob < 0.60: return "High",     "Decline or require collateral"
    else:             return "Critical", "Decline"

@app.get("/")
def root():
    return {"status": "Credit Scoring API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "features": len(FEATURE_LIST)}

@app.post("/predict")
def predict(data: BorrowerInput):
    try:
        # Build dataframe with correct feature order
        df = pd.DataFrame([data.features])
        for col in FEATURE_LIST:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_LIST]

        dmatrix = xgb.DMatrix(df)
        prob = float(model.predict(dmatrix)[0])
        decision = int(prob >= THRESHOLD)
        band, action = get_risk_band(prob)

        return {
            "default_probability": round(prob, 4),
            "risk_band": band,
            "recommended_action": action,
            "decision": "Default Risk" if decision == 1 else "Low Risk",
            "threshold_used": THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))