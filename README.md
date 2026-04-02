# Credit Scoring API

REST API that accepts borrower data and returns a default probability, risk band, and recommended action. Built with FastAPI, containerized with Docker, deployed on Railway.

**Live:** https://credit-scoring-api-production-6abe.up.railway.app/docs

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Status check |
| `/health` | GET | Model info and feature count |
| `/predict` | POST | Returns risk score for a borrower |

---

## Example

**Request:**
```json
POST /predict
{
  "features": {
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 150000
  }
}
```

**Response:**
```json
{
  "default_probability": 0.0675,
  "risk_band": "Low",
  "recommended_action": "Approve",
  "decision": "Low Risk",
  "threshold_used": 0.66
}
```

---

## Risk Bands

| Probability | Band | Action |
|---|---|---|
| < 0.20 | Low | Approve |
| 0.20 – 0.40 | Medium | Review |
| 0.40 – 0.60 | High | Decline or require collateral |
| > 0.60 | Critical | Decline |

---

## Stack

- **Model:** XGBoost with threshold 0.66
- **API:** FastAPI + Uvicorn
- **Container:** Docker
- **Deployment:** Railway

---

## Run Locally

```bash
git clone https://github.com/Sachin-kalaiselvan/credit-scoring-api
cd credit-scoring-api
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run with Docker

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

---

## Project Structure

```
credit-scoring-api/
├── main.py
├── Dockerfile
├── requirements.txt
└── models/
    ├── xgboost_final.json
    ├── best_threshold.pkl
    └── feature_list.pkl
```

---

## Related

- [Borrower Risk Intelligence](https://github.com/Sachin-kalaiselvan/borrower-risk-intelligence) — the ML model this API serves
- [Financial Document Analyzer](https://github.com/Sachin-kalaiselvan/financial-document-analyzer) — RAG pipeline for financial PDFs

---

**Sachin JK** — [GitHub](https://github.com/Sachin-kalaiselvan)
