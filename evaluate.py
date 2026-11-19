"""
evaluate.py
-----------
Detailed evaluation script for the trained phishing detection model.

Loads the saved model + vectorizer, re-creates the same train/test
split used during training, and reports:
    - Accuracy, Precision, Recall, F1-score
    - Full classification report
    - Confusion matrix (printed + saved as a PNG image)

Run:
    python evaluate.py

(Run this AFTER train_model.py has been executed at least once.)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so this works headlessly
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocessing import clean_corpus
from utils import load_dataset, load_artifacts, BASE_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.2
REPORT_IMAGE_PATH = os.path.join(BASE_DIR, "models", "confusion_matrix.png")


def evaluate() -> None:
    """Load artifacts, re-run the test split, and print a full report."""
    print("=" * 60)
    print("SOCIAL MEDIA PHISHING DETECTION - MODEL EVALUATION")
    print("=" * 60)

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    df["clean_text"] = clean_corpus(df["text"].tolist(), remove_stopwords_flag=False)

    # Recreate the identical split used in train_model.py so evaluation
    # is performed on the same held-out test set.
    _, X_test_text, _, y_test = train_test_split(
        df["clean_text"],
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    X_test = vectorizer.transform(X_test_text)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Summary Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(
        y_test, y_pred, target_names=["Legitimate (0)", "Phishing (1)"]
    ))

    cm = confusion_matrix(y_test, y_pred)
    print("--- Confusion Matrix ---")
    print(cm)

    # Save a visual confusion matrix for the README / presentation.
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Phishing Detection Model")
    plt.tight_layout()
    os.makedirs(os.path.dirname(REPORT_IMAGE_PATH), exist_ok=True)
    plt.savefig(REPORT_IMAGE_PATH)
    print(f"\nConfusion matrix image saved to: {REPORT_IMAGE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
