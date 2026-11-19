"""
train_model.py
---------------
End-to-end training script for the AI-Powered-Email-Impersonation-Detection.

Pipeline:
    1. Load dataset (dataset/sms.tsv)
    2. Convert labels (spam -> phishing=1, ham -> legitimate=0)
    3. Clean/preprocess text
    4. TF-IDF vectorize
    5. Train/test split
    6. Train a Logistic Regression classifier
    7. Save the trained model (models/logistic_model.pkl)
    8. Save the fitted TF-IDF vectorizer (models/tfidf_vectorizer.pkl)
    9. Print a quick evaluation summary

Run:
    python train_model.py
"""

import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import clean_corpus
from utils import load_dataset, save_artifacts, MODEL_PATH, VECTORIZER_PATH


RANDOM_STATE = 42
TEST_SIZE = 0.2


def train() -> None:
    """Run the full training pipeline and persist trained artifacts."""
    start = time.time()

    print("=" * 60)
    print("AI-Powered-Email-Impersonation-Detection - MODEL TRAINING")
    print("=" * 60)

    # 1 & 2. Load dataset and convert labels ---------------------------------
    print("\n[1/8] Loading dataset ...")
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"      Loaded {len(df)} messages "
          f"({(df['target'] == 1).sum()} phishing / spam, "
          f"{(df['target'] == 0).sum()} legitimate / ham).")

    # 3. Text preprocessing ---------------------------------------------------
    print("\n[2/8] Cleaning and preprocessing text ...")
    df["clean_text"] = clean_corpus(df["text"].tolist(), remove_stopwords_flag=False)

    # 4 & 5. Train/test split (done before vectorizer fit to avoid leakage) --
    print("\n[3/8] Splitting into train/test sets "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}) ...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"],
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )
    print(f"      Train size: {len(X_train_text)} | Test size: {len(X_test_text)}")

    # TF-IDF vectorization -----------------------------------------------------
    print("\n[4/8] Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 6. Train Logistic Regression --------------------------------------------
    print("\n[5/8] Training Logistic Regression classifier ...")
    model = LogisticRegression(
        C=10,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("      Training complete.")

    # Quick evaluation on the held-out test set --------------------------------
    print("\n[6/8] Evaluating on held-out test set ...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"      Accuracy : {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall   : {rec:.4f}")
    print(f"      F1-score : {f1:.4f}")

    # 7 & 8. Save trained model and vectorizer ---------------------------------
    print("\n[7/8] Saving trained model and vectorizer ...")
    save_artifacts(model, vectorizer)
    print(f"      Model saved to      : {MODEL_PATH}")
    print(f"      Vectorizer saved to : {VECTORIZER_PATH}")

    print("\n[8/8] Done.")
    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed:.2f} seconds")
    print("\nNext steps:")
    print("  - Run 'python evaluate.py' for a detailed evaluation report.")
    print("  - Run 'python app.py' to launch the web application.")
    print("=" * 60)


if __name__ == "__main__":
    train()
