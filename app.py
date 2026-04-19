"""
app.py
------
Flask application factory.
Run locally:  python app.py
Production:   gunicorn app:app
"""

import os
import json
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from db.models import db
from routes.predict import predict_bp
from routes.dashboard import dashboard_bp


def create_app() -> Flask:
    app = Flask(__name__)

    # ── Config ────────────────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_url = os.environ.get("DATABASE_URL") or f"sqlite:///{os.path.join(BASE_DIR, 'fruit_classifier.db')}"
    # Supabase returns 'postgresql://' but SQLAlchemy requires 'postgresql+psycopg2://'
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MODELS_DIR"] = os.path.join(BASE_DIR, "models")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-42")

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)

    # ── Blueprints ────────────────────────────────────────────────────────────
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(predict_bp)

    # ── Create tables & seed model performance from JSON ─────────────────────
    with app.app_context():
        db.create_all()
        _seed_model_performance(app)

    return app


def _seed_model_performance(app: Flask):
    """
    On startup, sync model accuracy data from models/accuracies.json
    into the model_performance table (upsert).
    """
    from db.models import ModelPerformance
    from datetime import datetime

    stats_path = os.path.join(app.config["MODELS_DIR"], "accuracies.json")
    if not os.path.exists(stats_path):
        return

    with open(stats_path) as f:
        stats = json.load(f)

    for name, acc in stats.get("accuracies", {}).items():
        row = ModelPerformance.query.filter_by(model_name=name).first()
        if row:
            row.accuracy     = acc
            row.last_trained = datetime.utcnow()
        else:
            db.session.add(ModelPerformance(model_name=name, accuracy=acc))
    db.session.commit()


# ── Entry point ───────────────────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)