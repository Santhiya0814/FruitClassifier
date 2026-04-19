"""
ml/train.py
-----------
Trains 5 Ensemble Learning models on data.csv, evaluates them with
cross-validation, saves .pkl files via joblib, and writes accuracy
stats to models/accuracies.json.

Run once before starting the Flask app:
    python ml/train.py
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATS_PATH = os.path.join(MODELS_DIR, "accuracies.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load & prepare data ────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

X = df[["weight", "size", "sweetness"]].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Base estimators for ensemble methods ───────────────────────────────────────
_rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
_gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
_ada = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME")

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
        algorithm="SAMME",
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    ),
    "Voting Classifier": VotingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("ada", AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME")),
        ],
        voting="soft",
    ),
    "Bagging Classifier": BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=200,
        max_samples=0.8,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    ),
}

# ── Train, evaluate, save ──────────────────────────────────────────────────────
accuracies  = {}
cv_scores   = {}

print("\n" + "=" * 68)
print("  Fruit Classifier - Ensemble Learning Training Report")
print("=" * 68)
print(f"  Dataset : {len(df)} samples | {df['label'].nunique()} classes | "
      f"{X_train.shape[0]} train / {X_test.shape[0]} test")
print(f"  Classes : {sorted(df['label'].unique())}")
print("-" * 68)

for name, clf in MODELS.items():
    # Cross-validation
    cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    # Final fit on full training split
    clf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    accuracies[name] = round(float(test_acc) * 100, 2)
    cv_scores[name]  = round(float(cv_acc.mean()) * 100, 2)

    slug = name.lower().replace(" ", "_")
    joblib.dump(clf, os.path.join(MODELS_DIR, f"{slug}.pkl"))

    print(f"  {'[OK]':<6} {name:<22}  "
          f"Test: {test_acc*100:6.2f}%  "
          f"CV: {cv_acc.mean()*100:6.2f}% +/- {cv_acc.std()*100:.2f}%  "
          f"-> {slug}.pkl")

# ── Persist stats ──────────────────────────────────────────────────────────────
best = max(accuracies, key=accuracies.get)
stats = {
    "accuracies": accuracies,
    "cv_scores":  cv_scores,
    "best_model": best,
}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print("-" * 68)
print(f"  [BEST]  {best}  ({accuracies[best]:.2f}% test | {cv_scores[best]:.2f}% CV)")
print(f"  [DIR]   Models saved to: {MODELS_DIR}")
print("=" * 68 + "\n")

# ── Update database ────────────────────────────────────────────────────────────
try:
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
                row.cv_score     = cv_scores.get(name)
                row.last_trained = datetime.utcnow()
            else:
                db.session.add(ModelPerformance(
                    model_name=name,
                    accuracy=acc,
                    cv_score=cv_scores.get(name),
                ))
        # Remove stale models no longer in the ensemble set
        for row in ModelPerformance.query.all():
            if row.model_name not in accuracies:
                db.session.delete(row)
        db.session.commit()
        print("  [DB]  Model performance updated in Supabase.\n")
except Exception as e:
    print(f"  [DB]  DB update skipped: {e}\n")
