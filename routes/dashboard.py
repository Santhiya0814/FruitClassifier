"""
routes/dashboard.py
-------------------
Blueprint for:
  GET /           — main prediction page
  GET /dashboard  — analytics dashboard (model performance + prediction logs)
"""

import json
import os
from flask import Blueprint, render_template, current_app

dashboard_bp = Blueprint("dashboard", __name__)


def _load_stats():
    stats_path = os.path.join(current_app.config["MODELS_DIR"], "accuracies.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return {"accuracies": {}, "best_model": "N/A"}


@dashboard_bp.route("/")
def home():
    stats = _load_stats()
    return render_template(
        "index.html",
        models=list(stats["accuracies"].keys()) or [
            "KNN", "Decision Tree", "Naive Bayes",
            "Logistic Regression", "Random Forest"
        ],
        accuracies=stats["accuracies"],
        best_model=stats["best_model"],
    )


@dashboard_bp.route("/dashboard")
def dashboard():
    from db.models import ModelPerformance, PredictionLog

    perf_rows = ModelPerformance.query.order_by(ModelPerformance.accuracy.desc()).all()
    log_rows  = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(50).all()

    # Build bar-chart data
    chart_labels  = [r.model_name for r in perf_rows]
    chart_values  = [r.accuracy   for r in perf_rows]
    best_model    = perf_rows[0].model_name if perf_rows else "N/A"

    # Prediction distribution for doughnut chart
    from sqlalchemy import func
    from db.models import db
    dist = (
        db.session.query(PredictionLog.prediction, func.count(PredictionLog.id))
        .group_by(PredictionLog.prediction)
        .all()
    )
    dist_labels = [d[0] for d in dist]
    dist_values = [d[1] for d in dist]

    return render_template(
        "dashboard.html",
        perf_rows=perf_rows,
        log_rows=log_rows,
        best_model=best_model,
        chart_labels=chart_labels,
        chart_values=chart_values,
        dist_labels=dist_labels,
        dist_values=dist_values,
        total_predictions=sum(dist_values),
    )
