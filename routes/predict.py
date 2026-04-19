"""
routes/predict.py
-----------------
Blueprint for:
  POST /predict   — classify a fruit using an ensemble model and log the result
  GET  /models    — return model performance data from DB
  GET  /logs      — return prediction history from DB
"""

import numpy as np
import joblib
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

predict_bp = Blueprint("predict", __name__)

# Ensemble model display name → pkl slug
MODEL_SLUGS = {
    "Random Forest":      "random_forest",
    "AdaBoost":           "adaboost",
    "Gradient Boosting":  "gradient_boosting",
    "Voting Classifier":  "voting_classifier",
    "Bagging Classifier": "bagging_classifier",
}

_loaded_models: dict = {}


def get_model(name: str):
    """Lazy-load a model from disk and cache it in memory."""
    if name not in _loaded_models:
        slug       = MODEL_SLUGS[name]
        models_dir = current_app.config["MODELS_DIR"]
        path       = os.path.join(models_dir, f"{slug}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}. Run python ml/train.py first."
            )
        _loaded_models[name] = joblib.load(path)
    return _loaded_models[name]


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON): { weight, size, sweetness, algorithm }
    Returns    : { prediction, confidence, algorithm }
    """
    from db.models import db, PredictionLog

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400

        for field in ("weight", "size", "sweetness"):
            if field not in data or str(data[field]).strip() == "":
                return jsonify({"error": f"Missing required field: '{field}'"}), 400

        weight    = float(data["weight"])
        size      = float(data["size"])
        sweetness = float(data["sweetness"])

        if not (0 < weight < 5000):
            return jsonify({"error": "Weight must be between 0 and 5000 grams."}), 400
        if not (0 < size < 100):
            return jsonify({"error": "Size must be between 0 and 100 cm."}), 400
        if not (0 <= sweetness <= 10):
            return jsonify({"error": "Sweetness must be between 0 and 10."}), 400

        algo = data.get("algorithm", "Random Forest")
        if algo not in MODEL_SLUGS:
            return jsonify({"error": f"Unknown algorithm '{algo}'. Choose from: {list(MODEL_SLUGS)}"}), 400

        model    = get_model(algo)
        features = np.array([[weight, size, sweetness]])
        pred     = model.predict(features)[0]

        confidence_str = "N/A"
        if hasattr(model, "predict_proba"):
            prob = float(np.max(model.predict_proba(features))) * 100
            confidence_str = f"{prob:.2f}%"

        log_entry = PredictionLog(
            weight=weight, size=size, sweetness=sweetness,
            prediction=str(pred), model_used=algo,
            confidence=confidence_str, timestamp=datetime.utcnow(),
        )
        db.session.add(log_entry)
        db.session.commit()

        return jsonify({
            "prediction": str(pred),
            "confidence": confidence_str,
            "algorithm":  algo,
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@predict_bp.route("/models", methods=["GET"])
def models_api():
    """GET /models — returns ensemble model performance data from DB."""
    from db.models import ModelPerformance
    rows = ModelPerformance.query.order_by(ModelPerformance.accuracy.desc()).all()
    return jsonify([r.to_dict() for r in rows])


@predict_bp.route("/logs", methods=["GET"])
def logs_api():
    """GET /logs — returns up to 100 most recent prediction logs."""
    from db.models import PredictionLog
    rows = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(100).all()
    return jsonify([r.to_dict() for r in rows])
