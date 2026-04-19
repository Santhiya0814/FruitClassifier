# Fruit Classifier — Ensemble Learning ML Web Application

> A professional, resume-level Machine Learning web application for fruit classification using Ensemble Learning, Flask, Scikit-Learn, Supabase (PostgreSQL), and Chart.js.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.4-orange?logo=scikit-learn)
![CI](https://github.com/YOUR_USERNAME/fruit-classifier/actions/workflows/ci.yml/badge.svg)

---

## Features

- **5 Ensemble ML Algorithms** — Random Forest, AdaBoost, Gradient Boosting, Voting Classifier, Bagging Classifier.
- **Cross-Validation Scoring** — Every model evaluated with 5-fold Stratified K-Fold CV alongside test-set accuracy.
- **Analytics Dashboard** — KPI cards, performance tables with CV scores, accuracy bar charts, and prediction distribution doughnut charts.
- **Supabase Integration** — Cloud PostgreSQL via Supabase tracking model performance and prediction logs.
- **REST API endpoints** — `/predict` (ensemble inference), `/models` (performance tracking), `/logs` (recent inferences).
- **Lavender Light Theme** — Custom high-end light theme using clean CSS & Bootstrap 5.
- **Robust test suite** — 27 `pytest` test cases covering UI rendering, API contracts, validations, and database entries.
- **CI/CD Pipeline** — GitHub Actions CI (lint + train + test) and CD (auto-deploy to Render).
- **Deployment Ready** — Render/Railway compatible with `gunicorn app:app`.

---

## Project Structure

```
ML/
├── app.py                        # Flask application factory
├── db/
│   └── models.py                 # SQLAlchemy schemas (ModelPerformance + PredictionLog)
├── ml/
│   └── train.py                  # Ensemble training script — generates .pkl files
├── models/                       # Auto-generated model artifacts
│   ├── random_forest.pkl
│   ├── adaboost.pkl
│   ├── gradient_boosting.pkl
│   ├── voting_classifier.pkl
│   ├── bagging_classifier.pkl
│   └── accuracies.json           # Test accuracy + CV scores
├── routes/
│   ├── dashboard.py              # View endpoints (/, /dashboard)
│   └── predict.py                # JSON API endpoints (/predict, /models, /logs)
├── templates/
│   ├── index.html                # Prediction UI
│   └── dashboard.html            # Analytics dashboard
├── static/
│   └── style.css                 # Lavender Light theme
├── tests/
│   └── test_api.py               # 27 pytest test cases
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI: lint + train + test + coverage
│       └── cd.yml                # CD: deploy to Render on CI pass
├── data.csv                      # Training dataset (100 samples, 5 classes)
├── render.yaml                   # Render deployment config
├── requirements.txt
├── Procfile
└── .env.example
```

---

## Ensemble Models

| Model | Strategy | Key Hyperparameters |
|---|---|---|
| Random Forest | Bagging of Decision Trees | 200 estimators, unlimited depth |
| AdaBoost | Boosting with weighted resampling | 200 estimators, depth-3 base tree, lr=0.5 |
| Gradient Boosting | Sequential residual boosting | 200 estimators, depth-4, lr=0.1, subsample=0.8 |
| Voting Classifier | Soft-vote ensemble (RF + GB + Ada) | Probability averaging across 3 base models |
| Bagging Classifier | Bootstrap aggregation | 200 estimators, depth-5 base tree, 80% sample |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your Supabase DATABASE_URL
```

### 3. Train Ensemble Models
```bash
python ml/train.py
```

### 4. Run the Flask Server
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## API Reference

### `POST /predict`
Classify a fruit using an ensemble model.

**Request:**
```json
{
  "weight": 160,
  "size": 7.5,
  "sweetness": 7,
  "algorithm": "Random Forest"
}
```

**Response:**
```json
{
  "prediction": "Apple",
  "confidence": "100.00%",
  "algorithm": "Random Forest"
}
```

### `GET /models`
Returns all ensemble model performance records including CV scores.

### `GET /logs`
Returns the 100 most recent prediction logs.

---

## Running Tests
```bash
python -m pytest tests/ -v
```

27 tests covering routes, API contracts, input validation, and DB logging.

---

## CI/CD Pipeline

- **CI** (`ci.yml`) — Runs on every push/PR: flake8 lint → train models → pytest with coverage
- **CD** (`cd.yml`) — Triggers on CI success on `main`: deploys to Render via deploy hook

**Required GitHub Secrets:**
- `RENDER_DEPLOY_HOOK_URL` — from Render dashboard → your service → Settings → Deploy Hook
