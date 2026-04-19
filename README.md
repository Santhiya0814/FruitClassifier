# Fruit Classifier — ML Web Application

> A professional, resume-level Machine Learning web application for fruit classification using Flask, Scikit-Learn, SQLite, and Chart.js.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![SqlAlchemy](https://img.shields.io/badge/SQLAlchemy-3.1-blue?logo=sqlite)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.4-orange?logo=scikit-learn)
![CI](https://github.com/YOUR_USERNAME/fruit-classifier/actions/workflows/ci.yml/badge.svg)

---

## ✨ Features

- **5 ML Algorithms** — KNN, Decision Tree, Naive Bayes, Logistic Regression, Random Forest.
- **Analytics Dashboard** — KPI cards, performance tables, bar charts for accuracies, and distribution doughnut charts.
- **Database Tracking** — SQLite tracking via `Flask-SQLAlchemy` for model performance and prediction logs.
- **REST API endpoints** — Including `/predict` (machine learning inference), `/models` (performance tracking), and `/logs` (recent inferences).
- **Lavender Light Theme** — Custom high-end light theme utilizing clean CSS & Bootstrap 5.
- **Robust test suite** — Fully verified via 27 specific `pytest` test cases checking UI rendering, API contracts, validations, and database entries. 
- **Deployment Ready** — Designed specifically for quick deployments (Render/Railway with `gunicorn`).

---

## 📂 Project Structure

```
ML/
├── app.py                    # Flask app — application factory and extensions
├── db/                       # Database related
│   ├── models.py             # SQLAlchemy schemas (predictions logs & performance)
├── ml/
│   └── train.py              # Training script — generates .pkl files
├── models/                   # Auto-generated models data
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── accuracies.json
├── routes/
│   ├── dashboard.py          # View endpoints routing (/, /dashboard)
│   └── predict.py            # JSON endpoints and model handling blueprints (/predict)
├── templates/                # HTML views
│   ├── index.html            
│   └── dashboard.html        
├── static/
│   └── style.css             # Lavender Light UI
├── tests/
│   └── test_api.py           # Pytest test suite (27 tests)
├── data.csv                  # Training dataset
├── requirements.txt
└── Procfile                  
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models & Init Database
Running the training script generates all `.pkl` models and directly registers their baseline accuracy stats into your SQLite database if the server initialized it. 
```bash
python ml/train.py
```

### 3. Run the Flask Server
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📡 API Reference

### `POST /predict`
Classify a fruit based on its features and permanently log the entry to the Database.

**Request (JSON):**
```json
{
  "weight": 160,
  "size": 7.5,
  "sweetness": 7,
  "algorithm": "KNN"
}
```

**Response:**
```json
{
  "prediction": "Apple",
  "confidence": "100.00%",
  "algorithm": "KNN"
}
```

---

## 🧪 Running Tests
Your application includes a completely decoupled, in-memory SQLite test suite.
```bash
python -m pytest tests/ -v
```
