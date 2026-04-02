from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Dict, Union
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
    features: Dict[str, Union[float, int]]

    @field_validator("features")
    @classmethod
    def check_features(cls, v):
        if not v:
            raise ValueError("features dict cannot be empty.")
        invalid = {k: val for k, val in v.items() if not isinstance(val, (int, float))}
        if invalid:
            raise ValueError(f"All feature values must be numeric. Invalid: {invalid}")
        non_finite = {k: val for k, val in v.items() if isinstance(val, float) and not np.isfinite(val)}
        if non_finite:
            raise ValueError(f"Feature values cannot be NaN or infinite: {non_finite}")
        return v

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

@app.get("/features")
def features():
    return {"total_features": len(FEATURE_LIST), "feature_names": FEATURE_LIST}

@app.post("/predict")
def predict(data: BorrowerInput):
    try:
        df = pd.DataFrame([data.features])

        # Warn about unknown features
        unknown = [f for f in df.columns if f not in FEATURE_LIST]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown features provided: {unknown}. Use GET /features to see valid feature names."
            )

        # Fill missing features with 0 but report them
        missing = [f for f in FEATURE_LIST if f not in df.columns]
        for col in FEATURE_LIST:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_LIST]

        dmatrix  = xgb.DMatrix(df)
        prob     = float(model.predict(dmatrix)[0])
        decision = int(prob >= THRESHOLD)
        band, action = get_risk_band(prob)

        response = {
            "default_probability": round(prob, 4),
            "risk_band": band,
            "recommended_action": action,
            "decision": "Default Risk" if decision == 1 else "Low Risk",
            "threshold_used": THRESHOLD,
        }
        if missing:
            response["warning"] = f"{len(missing)} features not provided and defaulted to 0: {missing[:5]}{'...' if len(missing) > 5 else ''}"

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
