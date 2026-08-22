"""Train the complaint routing classifier and write the served artifact."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# CFPB redacts names, dates and account numbers as runs of X before publishing.
# Left alone these dominate the vocabulary, so treat them as stop words.
REDACTION_TOKENS = {"xx", "xxx", "xxxx", "xxxxxxxx", "xxxxxxxxxx"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | REDACTION_TOKENS)


def build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        min_df=3,
        max_features=100_000,
        sublinear_tf=True,
    )


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"train {len(train):,}   test {len(test):,}")

    candidates = {
        "linear_svc": LinearSVC(C=0.5),
        "logistic_regression": LogisticRegression(C=4.0, max_iter=1000),
    }

    scores = {}
    fitted = {}
    for name, clf in candidates.items():
        print(f"\ntraining {name}...")
        pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", clf)])
        pipe.fit(train["text"], train["label"])
        pred = pipe.predict(test["text"])
        macro_f1 = f1_score(test["label"], pred, average="macro")
        scores[name] = {
            "accuracy": accuracy_score(test["label"], pred),
            "macro_f1": macro_f1,
        }
        fitted[name] = pipe
        print(f"  accuracy {scores[name]['accuracy']:.4f}   macro-F1 {macro_f1:.4f}")

    # The API returns a confidence score per prediction, which needs
    # predict_proba. LinearSVC only exposes decision_function, so it stays a
    # reported baseline and logistic regression is what actually ships.
    served = "logistic_regression"
    pipe = fitted[served]
    pred = pipe.predict(test["text"])

    print(f"\nserving: {served}")
    print(classification_report(test["label"], pred, digits=3))

    classes = sorted(train["label"].unique())
    report = classification_report(
        test["label"], pred, output_dict=True, zero_division=0
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "classifier.joblib", compress=3)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "served_model": served,
        "candidates": scores,
        "classes": classes,
        "train_rows": len(train),
        "test_rows": len(test),
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_features": 100_000,
            "sublinear_tf": True,
        },
        "metrics": {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "per_class": {c: report[c] for c in classes},
        },
        "confusion_matrix": {
            "labels": classes,
            "counts": confusion_matrix(test["label"], pred, labels=classes).tolist(),
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    size_mb = (MODEL_DIR / "classifier.joblib").stat().st_size / 1e6
    print(f"\nsaved classifier.joblib ({size_mb:.1f} MB) and metadata.json")


if __name__ == "__main__":
    main()
