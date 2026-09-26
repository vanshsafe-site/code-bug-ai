"""
inference.py
-------------
Loads the trained TF-IDF + LogisticRegression baseline and exposes a
single predict_bug(code) function -- no retraining, no API calls.
"""

import joblib
import sys
import os

sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_code

_VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.joblib")
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_logreg.joblib")

_vectorizer = None
_model = None


def _load_models():
    """Lazy-load so importing this module doesn't require the model
    files to exist yet (useful for testing preprocessing alone)."""
    global _vectorizer, _model
    if _vectorizer is None:
        _vectorizer = joblib.load(_VECTORIZER_PATH)
    if _model is None:
        _model = joblib.load(_MODEL_PATH)


def predict_bug(code: str) -> dict:
    """
    Takes raw source code as a string, returns a prediction dict.

    NOT a guaranteed static analyzer -- this is a statistical model
    trained on C functions from the Devign dataset (FFmpeg/Qemu),
    predicting the LIKELIHOOD of a security-relevant defect pattern,
    not verifying correctness formally.
    """
    _load_models()

    cleaned = clean_code(code)
    X = _vectorizer.transform([cleaned])

    pred = _model.predict(X)[0]
    proba = _model.predict_proba(X)[0]
    confidence = proba[pred]

    label = "Potential Bug" if pred == 1 else "No Bug / Correct Code"

    return {
        "prediction": label,
        "confidence": round(float(confidence), 4),
        "raw_label": int(pred),
    }


if __name__ == "__main__":
    sample = """
    for i in range(10)
        print(i)
    """
    result = predict_bug(sample)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
