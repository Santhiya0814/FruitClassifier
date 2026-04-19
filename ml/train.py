"""
ml/train.py
-----------
Trains 5 ML models on data.csv, evaluates them, saves .pkl files via joblib,
and writes accuracy stats to models/accuracies.json.

Run once before starting the Flask app:
    python ml/train.py
"""

import os
import sys
import json
import warnings
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATS_PATH = os.path.join(MODELS_DIR, "accuracies.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load & prepare data ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

X = df[["weight", "size", "sweetness"]].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}

# ── Train, evaluate, save .pkl ────────────────────────────────────────────────
accuracies = {}

print("\n" + "=" * 58)
print("  Fruit Classifier - Model Training Report")
print("=" * 58)

for name, clf in MODELS.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    accuracies[name] = round(float(acc) * 100, 2)

    slug = name.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{slug}.pkl")
    joblib.dump(clf, path)
    print(f"  [OK]  {name:<22}  Accuracy: {acc*100:6.2f}%   -> {slug}.pkl")

# ── Persist stats to JSON ─────────────────────────────────────────────────────
best = max(accuracies, key=accuracies.get)
stats = {"accuracies": accuracies, "best_model": best}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print("=" * 58)
print(f"  [BEST]  {best} ({accuracies[best]:.2f}%)")
print(f"  [DIR]   Saved to: {MODELS_DIR}")
print("=" * 58 + "\n")

# ── Optionally update the database after training ─────────────────────────────
try:
    # Only runs when called standalone; skip inside Flask context
    sys.path.insert(0, BASE_DIR)
    from app import create_app
    from db.models import db, ModelPerformance
    from datetime import datetime

    flask_app = create_app()
    with flask_app.app_context():
        for name, acc in accuracies.items():
            row = ModelPerformance.query.filter_by(model_name=name).first()
            if row:
                row.accuracy     = acc
                row.last_trained = datetime.utcnow()
            else:
                db.session.add(ModelPerformance(model_name=name, accuracy=acc))
        db.session.commit()
        print("  [DB]  Model performance updated in database.")
except Exception as e:
    print(f"  [DB]  DB update skipped ({e})")
    print("        (Run 'python app.py' first to initialise the DB, then re-train)\n")
