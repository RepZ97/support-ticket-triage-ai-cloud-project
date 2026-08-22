"""Loads the trained routing model and exposes prediction."""

import json
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_pipeline = None
_metadata = None


def load():
    global _pipeline, _metadata
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_DIR / "classifier.joblib")
        _metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    return _pipeline, _metadata


def metadata():
    load()
    return _metadata


def predict(text, top_k=3):
    pipeline, _ = load()
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
    top = ranked[:top_k]
    return {
        "category": top[0][0],
        "confidence": round(float(top[0][1]), 4),
        "alternatives": [
            {"category": c, "confidence": round(float(p), 4)} for c, p in top[1:]
        ],
    }
