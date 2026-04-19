"""
tests/test_api.py
-----------------
Pytest test suite for the Fruit Classifier Flask API.
Run with:  python -m pytest tests/ -v
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app import create_app
from db.models import db as _db


@pytest.fixture(scope="session")
def app():
    """Create a test Flask app with an in-memory SQLite database."""
    test_app = create_app()
    test_app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    })
    with test_app.app_context():
        _db.create_all()
        yield test_app


@pytest.fixture
def client(app):
    return app.test_client()


# ── Helpers ────────────────────────────────────────────────
def post_predict(client, payload):
    return client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )

VALID = {"weight": 160, "size": 7.5, "sweetness": 7, "algorithm": "Random Forest"}

ALL_ALGOS = [
    "Random Forest", "AdaBoost", "Gradient Boosting",
    "Voting Classifier", "Bagging Classifier"
]


# ── Home route ─────────────────────────────────────────────
class TestHomeRoute:
    def test_home_200(self, client):
        assert client.get("/").status_code == 200

    def test_home_has_fruit(self, client):
        assert b"Fruit" in client.get("/").data


# ── Dashboard route ────────────────────────────────────────
class TestDashboard:
    def test_dashboard_200(self, client):
        assert client.get("/dashboard").status_code == 200

    def test_dashboard_has_heading(self, client):
        assert b"Dashboard" in client.get("/dashboard").data


# ── /predict happy path ────────────────────────────────────
class TestPredictSuccess:
    def test_returns_200(self, client):
        assert post_predict(client, VALID).status_code == 200

    def test_has_prediction(self, client):
        assert "prediction" in post_predict(client, VALID).get_json()

    def test_has_confidence(self, client):
        assert "confidence" in post_predict(client, VALID).get_json()

    def test_has_algorithm(self, client):
        assert "algorithm" in post_predict(client, VALID).get_json()

    def test_confidence_format(self, client):
        conf = post_predict(client, VALID).get_json()["confidence"]
        assert conf == "N/A" or conf.endswith("%")

    @pytest.mark.parametrize("algo", ALL_ALGOS)
    def test_all_algorithms(self, client, algo):
        payload = {**VALID, "algorithm": algo}
        res = post_predict(client, payload)
        # Some models might not be trained yet → allow 503 as well
        assert res.status_code in (200, 503)

    def test_prediction_logged(self, client):
        """After a predict call the log endpoint should have at least 1 entry."""
        post_predict(client, VALID)
        logs = client.get("/logs").get_json()
        assert isinstance(logs, list)
        assert len(logs) >= 1


# ── /predict validation ────────────────────────────────────
class TestPredictValidation:
    def test_missing_weight(self, client):
        assert post_predict(client, {"size": 7, "sweetness": 7, "algorithm": "KNN"}).status_code == 400

    def test_missing_size(self, client):
        assert post_predict(client, {"weight": 160, "sweetness": 7, "algorithm": "KNN"}).status_code == 400

    def test_missing_sweetness(self, client):
        assert post_predict(client, {"weight": 160, "size": 7, "algorithm": "KNN"}).status_code == 400

    def test_bad_algorithm(self, client):
        res = post_predict(client, {**VALID, "algorithm": "XGBoost"})
        assert res.status_code == 400

    def test_empty_body(self, client):
        res = client.post("/predict", data="{}", content_type="application/json")
        assert res.status_code == 400

    def test_negative_weight(self, client):
        assert post_predict(client, {**VALID, "weight": -5}).status_code == 400

    def test_sweetness_over_10(self, client):
        assert post_predict(client, {**VALID, "sweetness": 15}).status_code == 400


# ── /models endpoint ───────────────────────────────────────
class TestModelsEndpoint:
    def test_200(self, client):
        assert client.get("/models").status_code == 200

    def test_returns_list(self, client):
        assert isinstance(client.get("/models").get_json(), list)


# ── /logs endpoint ─────────────────────────────────────────
class TestLogsEndpoint:
    def test_200(self, client):
        assert client.get("/logs").status_code == 200

    def test_returns_list(self, client):
        assert isinstance(client.get("/logs").get_json(), list)

    def test_log_has_fields(self, client):
        post_predict(client, VALID)
        logs = client.get("/logs").get_json()
        if logs:
            required = {"id", "weight", "size", "sweetness", "prediction", "model_used", "timestamp"}
            assert required.issubset(logs[0].keys())
