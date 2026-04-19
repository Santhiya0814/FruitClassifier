"""
db/models.py
------------
SQLAlchemy database models for:
  - ModelPerformance  : tracks accuracy of each trained ML model
  - PredictionLog     : logs every prediction made by the /predict API
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ModelPerformance(db.Model):
    __tablename__ = "model_performance"

    id           = db.Column(db.Integer, primary_key=True)
    model_name   = db.Column(db.String(100), nullable=False, unique=True)
    accuracy     = db.Column(db.Float, nullable=False)
    last_trained = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id":           self.id,
            "model_name":   self.model_name,
            "accuracy":     round(self.accuracy, 2),
            "last_trained": self.last_trained.strftime("%Y-%m-%d %H:%M:%S"),
        }


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id          = db.Column(db.Integer, primary_key=True)
    weight      = db.Column(db.Float, nullable=False)
    size        = db.Column(db.Float, nullable=False)
    sweetness   = db.Column(db.Float, nullable=False)
    prediction  = db.Column(db.String(100), nullable=False)
    model_used  = db.Column(db.String(100), nullable=False)
    confidence  = db.Column(db.String(20), nullable=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         self.id,
            "weight":     self.weight,
            "size":       self.size,
            "sweetness":  self.sweetness,
            "prediction": self.prediction,
            "model_used": self.model_used,
            "confidence": self.confidence,
            "timestamp":  self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }
