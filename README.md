# credit-scoring-api

send it a borrower, get back default probability, a risk band and what to do about it. fastapi + docker, running on railway.

**live docs:** https://credit-scoring-api-production-6abe.up.railway.app/docs

## endpoints

| endpoint | method | what |
|---|---|---|
| `/` | GET | status |
| `/health` | GET | model info + feature count |
| `/predict` | POST | risk score for a borrower |

## example

```json
POST /predict
{
  "features": {
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 150000
  }
}
```

```json
{
  "default_probability": 0.0675,
  "risk_band": "Low",
  "recommended_action": "Approve",
  "decision": "Low Risk",
  "threshold_used": 0.66
}
```

## bands

| probability | band | action |
|---|---|---|
| < 0.20 | low | approve |
| 0.20 – 0.40 | medium | review |
| 0.40 – 0.60 | high | decline or ask for collateral |
| > 0.60 | critical | decline |

model is XGBoost, decision threshold 0.66.

## run it

```bash
git clone https://github.com/Sachin-kalaiselvan/credit-scoring-api
cd credit-scoring-api
pip install -r requirements.txt
uvicorn main:app --reload
```

or docker:

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

```
main.py  Dockerfile  requirements.txt
models/  xgboost_final.json  best_threshold.pkl  feature_list.pkl
```

related: [borrower-risk-intelligence](https://github.com/Sachin-kalaiselvan/borrower-risk-intelligence) is the model this serves, [financial-document-analyzer](https://github.com/Sachin-kalaiselvan/financial-document-analyzer) is RAG over financial PDFs.
