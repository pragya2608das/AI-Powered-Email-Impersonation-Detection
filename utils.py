"""
utils.py
--------
Shared helper functions used across the training, evaluation, and
Flask serving scripts of theAI-Powered-Email-Impersonation-Detection project.

Centralizing path handling and load/save logic here keeps
train_model.py, evaluate.py, predict.py, and app.py consistent and
avoids duplicated code.
"""

import os
from typing import Tuple

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "sms.tsv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

LABEL_MAP = {"ham": 0, "spam": 1}
INVERSE_LABEL_MAP = {0: "Legitimate", 1: "Phishing"}


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the SMS Spam Collection style TSV dataset.

    The raw file has two tab-separated columns with no header:
    label (ham/spam) and text (the message).

    NOTE ON DATASET RE-PURPOSING:
    This project reuses the public SMS Spam Collection dataset as a
    stand-in for AI-Powered-Email-Impersonation-Detection. We treat:
        spam -> phishing message (label = 1)
        ham  -> legitimate social-media message (label = 0)
    This substitution is documented in the README and is purely a
    practical choice due to the lack of a large labeled, public
    dataset of real social-media phishing DMs.

    Args:
        path: Path to the sms.tsv file.

    Returns:
        A DataFrame with columns ['label', 'text', 'target'] where
        'target' is the binary numeric label (0 = legitimate,
        1 = phishing).

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please place sms.tsv inside "
            f"the 'dataset/' folder before running this script."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=3,  # csv.QUOTE_NONE - some SMS messages contain stray quotes
        encoding="latin-1",
    )

    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)
    df["target"] = df["label"].map(LABEL_MAP)

    return df


def ensure_models_dir() -> None:
    """Create the models/ directory if it does not already exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_artifacts(model, vectorizer) -> None:
    """Persist the trained model and TF-IDF vectorizer to disk."""
    ensure_models_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def load_artifacts() -> Tuple[object, object]:
    """Load the trained model and TF-IDF vectorizer from disk.

    Returns:
        A tuple of (model, vectorizer).

    Raises:
        FileNotFoundError: If either artifact is missing, prompting the
            user to run train_model.py first.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        raise FileNotFoundError(
            "Trained model/vectorizer not found in 'models/'. "
            "Please run 'python train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def label_to_text(label: int) -> str:
    """Convert a numeric prediction (0/1) into a human-readable string."""
    return INVERSE_LABEL_MAP.get(int(label), "Unknown")
