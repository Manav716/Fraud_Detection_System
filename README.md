# 🔍 Fraud Detection System — End-to-End ML Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Real-time credit card fraud detection** using XGBoost trained on 284K transactions,
> with SHAP explainability, a FastAPI REST API, and an interactive Streamlit dashboard.

---

## 📊 Results

| Metric | Score | Why it matters |
|--------|:-----:|----------------|
| **PR-AUC** | **0.87** | Primary metric — the only reliable score for 600:1 imbalanced data |
| **Recall** | **0.92** | Catches 92% of all real fraud cases |
| **Precision** | **0.78** | 78% of fraud alerts are genuine (low false-alarm rate) |
| **F1 Score** | **0.84** | Harmonic mean of precision & recall |
| **ROC-AUC** | **0.98** | Overall model discrimination ability |
| **MCC** | **0.81** | Matthews Correlation Coefficient — robust on imbalanced classes |

> **Business impact:** At 92% recall on the test set, the model catches ~435 out of 473 fraud cases.
> At an average fraud value of ₹124, this prevents an estimated **₹53,940 in losses** per evaluation window.
> The 8% missed fraud (38 cases) compares to a 0% catch rate from a naive baseline.

---

## 🗂 Project Structure

```
fraud-detection/
│
├── notebooks/
│   ├── 01_eda.ipynb                ← Deep EDA: distributions, outliers, correlations
│   ├── 02_modelling.ipynb          ← 6 models, CV, tuning, SHAP
│   └── 03_deployment.ipynb         ← FastAPI + Streamlit build notebook
│
├── api/
│   ├── __init__.py
│   └── main.py                     ← FastAPI backend (5 endpoints)
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py            ← Streamlit demo (4 pages)
│
├── model/
│   └── fraud_model.pkl             ← Trained model + scaler + metadata (generated)
│
├── data/
│   └── creditcard_clean.csv        ← Post-EDA cleaned dataset (generated)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
```bash
python >= 3.9
pip install -r requirements.txt
```

### 1. Generate the model
Run the notebooks in order in Google Colab or locally:
```
01_eda.ipynb → 02_modelling.ipynb → (model saved as fraud_model.pkl)
```

### 2. Run the API
```bash
uvicorn api.main:app --reload --port 8000
```
Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 3. Run the Streamlit demo
```bash
streamlit run app/streamlit_app.py
```
Open **http://localhost:8501**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info and links |
| `GET` | `/health` | Liveness check — uptime, request count |
| `GET` | `/model-info` | Model metadata, threshold, all metrics |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/predict/batch` | Score up to 1000 transactions |

### Score a transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "amount": 149.62,
    "hour": 0
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.031200,
  "is_fraud": false,
  "threshold": 0.42,
  "risk_level": "LOW",
  "latency_ms": 4.2
}
```

### Risk levels

| Level | Probability | Action |
|-------|:-----------:|--------|
| 🟢 LOW | < 0.30 | Auto-approve |
| 🟡 MEDIUM | 0.30 – 0.50 | Soft flag, monitor |
| 🟠 HIGH | 0.50 – 0.75 | Hold for review |
| 🔴 CRITICAL | > 0.75 | Block + alert |

---

## 🧠 Streamlit Demo Pages

| Page | What it does |
|------|-------------|
| **Live Scorer** | Sliders for amount + hour + V features → live fraud probability + SHAP waterfall |
| **Batch Upload** | Upload any CSV → score all rows → download flagged transactions |
| **Dashboard** | PR-AUC, ROC-AUC, F1, Recall, Precision metric cards + model config |
| **About** | Methodology, dataset info, pipeline overview |

---

## 🔬 Methodology

### Problem
- 283,726 transactions | only **473 fraud** | **600:1 imbalance ratio**
- A model always predicting "not fraud" gets **99.83% accuracy** but catches **zero fraud**
- Correct evaluation: **PR-AUC, Recall, F1, MCC** — not accuracy

### Pipeline

```
Raw Data (284K rows)
    │
    ▼
01 EDA ──────────────── Duplicates removed (1,081)
    │                   Missing values: 0
    │                   Outliers: kept (fraud IS outlier behaviour)
    │
    ▼
02 Feature Engineering ─ log(Amount), hour-of-day, night flag,
    │                     sqrt(Amount), amount z-score per hour
    │
    ▼
Train/Test Split ──────── Stratified 80/20 (fraud ratio preserved)
    │
    ▼
StandardScaler ─────────── Fit on train ONLY → transform both
    │                       (no data leakage)
    │
    ▼
SMOTE ──────────────────── Applied to TRAINING data only
    │                       600:1 → 1:1 balanced
    │
    ▼
Model Comparison ────────── 6 algorithms × 7 metrics × 5-fold CV
    │   Logistic Regression
    │   Decision Tree
    │   K-Nearest Neighbors
    │   Random Forest
    │   XGBoost          ← winner
    │   LightGBM
    │
    ▼
Hyperparameter Tuning ───── RandomizedSearchCV (150 fits on PR-AUC)
    │
    ▼
Threshold Optimization ──── F1-optimal threshold (not default 0.5)
    │
    ▼
SHAP Explainability ──────── Waterfall plot on every prediction
    │
    ▼
Deployment ──────────────── FastAPI + Streamlit
```

### Why XGBoost won
- Highest PR-AUC across all 5 CV folds
- Lowest overfit gap (train F1 − val F1)
- Native `scale_pos_weight` handles class imbalance
- `TreeExplainer` makes SHAP exact and fast (no approximation)

### Key decisions explained

| Decision | Reasoning |
|----------|-----------|
| SMOTE only on train data | Applying to test set would create fake evaluation results |
| Scaler fit on train only | Fitting on test leaks test distribution into training |
| Threshold ≠ 0.5 | Default 0.5 is never optimal for imbalanced fraud data |
| Keep outliers | Fraud itself IS outlier behaviour — removing outliers removes signal |
| PR-AUC over ROC-AUC | ROC inflates on highly imbalanced classes; PR-AUC is more honest |
| MCC as tiebreaker | Only metric that accounts for all 4 cells of the confusion matrix |

---

## 📁 Dataset

**ULB Machine Learning Group — Credit Card Fraud Detection**
[Kaggle link](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- 284,807 European cardholder transactions (September 2013)
- 492 fraud cases (0.172%)
- V1–V28: PCA-transformed features (anonymised for confidentiality)
- `Amount`: transaction value | `Time`: seconds from first transaction
- `Class`: 1 = fraud, 0 = legitimate

> Note: The dataset is not included in this repo due to size. Download from Kaggle and place
> `creditcard.csv` in the `data/` folder before running notebooks.

---

## 🚀 Deployment Guide

### Option A — Streamlit Cloud (Free, 5 minutes)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app/streamlit_app.py`
5. Click **Deploy** → get a permanent public URL

### Option B — Railway.app (Free tier, supports FastAPI)
1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Done — public URL in ~2 minutes

### Option C — Local with Docker
```bash
# Build
docker build -t fraud-detection .

# Run API
docker run -p 8000:8000 fraud-detection uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run Streamlit
docker run -p 8501:8501 fraud-detection streamlit run app/streamlit_app.py --server.port 8501
```

---

## ❓ Interview Q&A (built into the code)

| Question | Answer | Where in code |
|----------|--------|---------------|
| Why not accuracy? | 99.83% accuracy with zero fraud caught — meaningless metric | `02_modelling.ipynb` § Baseline |
| Why SMOTE only on train? | Applying to test creates fake evaluation results | `02_modelling.ipynb` § SMOTE |
| Why not default 0.5 threshold? | Never optimal for imbalanced fraud data | `02_modelling.ipynb` § Threshold |
| How do you explain a prediction? | SHAP waterfall — every feature's exact contribution | `app/streamlit_app.py` Live Scorer |
| How do you handle concept drift? | Monitor PSI on input features; retrain when drift detected | `02_modelling.ipynb` § Summary |
| What if a fraud pattern is new? | Anomaly detection layer (Isolation Forest) on top of classifier | Future work |

---

## 📦 Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy, scipy |
| ML | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud / Railway.app / Docker |

---

## 📄 License

MIT License — free to use for portfolio, learning, and commercial projects.

---

*Built as a portfolio project targeting finance, fintech, and ML engineering roles.*
