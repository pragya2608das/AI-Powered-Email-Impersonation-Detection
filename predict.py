"""
predict.py
----------
Standalone command-line prediction utility for the phishing
detection model. Also used internally by app.py to serve predictions
through the Flask web application.

Run directly for a quick CLI demo:
    python predict.py
"""

import re
from typing import Dict, Union

from preprocessing import clean_text
from utils import load_artifacts, label_to_text


# Targeted high-risk pattern missed by the generic SMS-trained classifier.
# It requires both an account-verification action and urgency, reducing
# the chance of flagging ordinary messages that merely mention Instagram.
_ACCOUNT_VERIFICATION_PATTERN = re.compile(
    r"\b(verify|verification|confirm|secure)\b.{0,45}\b(account|profile|identity)\b"
    r"|\b(account|profile|identity)\b.{0,45}\b(verify|verification|confirm|secure)\b",
    re.IGNORECASE,
)
_URGENCY_PATTERN = re.compile(
    r"\b(immediately|urgent|urgently|now|within\s+\d+\s*(minutes?|hours?|days?)|"
    r"24\s*hours?|action\s+required|will\s+be\s+suspended|suspension)\b",
    re.IGNORECASE,
)


def predict_message(message: str, model=None, vectorizer=None) -> Dict[str, Union[str, float, int]]:
    """Predict whether a single message is phishing or legitimate.

    Args:
        message: Raw input text to classify.
        model: (Optional) a pre-loaded classifier. If not provided,
            it will be loaded from disk via `load_artifacts()`.
        vectorizer: (Optional) a pre-loaded TF-IDF vectorizer. If not
            provided, it will be loaded from disk.

    Returns:
        A dictionary with:
            - 'message': the original input text
            - 'cleaned_text': the preprocessed text
            - 'prediction': 0 or 1
            - 'label': 'Legitimate' or 'Phishing'
            - 'confidence': probability (0-100) associated with the
              predicted class
    """
    if not message or not message.strip():
        raise ValueError("Input message is empty. Please provide some text to analyze.")

    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    # The training dataset is general SMS spam, so it can miss social-media
    # account takeover wording. Override only when verification language and
    # an explicit urgency/threat signal occur together.
    targeted_phishing_signal = bool(
        _ACCOUNT_VERIFICATION_PATTERN.search(message)
        and _URGENCY_PATTERN.search(message)
    )
    if targeted_phishing_signal:
        prediction = 1
        # This is a rule-triggered risk score floor, not a calibrated model
        # probability. Keep the existing result-page contract unchanged.
        confidence = round(max(float(probabilities[1]) * 100, 85.0), 2)
    else:
        confidence = round(float(probabilities[prediction]) * 100, 2)

    return {
        "message": message,
        "cleaned_text": cleaned,
        "prediction": prediction,
        "label": label_to_text(prediction),
        "confidence": confidence,
    }


def _run_cli_demo() -> None:
    """Run a small interactive/demo loop from the command line."""
    print("=" * 60)
    print("SOCIAL MEDIA PHISHING DETECTION - CLI PREDICTION DEMO")
    print("=" * 60)

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    example_messages = [
        "Verify your Instagram account immediately",
        "Claim your free iPhone reward now",
        "Your account will be suspended, click here to verify",
        "Click this link to login and secure your account",
        "Are you coming to college?",
        "Happy birthday!",
        "Meeting is scheduled for tomorrow at 10am",
        "Send me the notes from yesterday's class",
    ]

    for msg in example_messages:
        result = predict_message(msg, model=model, vectorizer=vectorizer)
        print(f"\nMessage   : {result['message']}")
        print(f"Prediction: {result['label']} (confidence: {result['confidence']}%)")

    print("\n" + "=" * 60)
    print("Enter your own message below (or press Enter to quit).")
    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            print("Exiting CLI demo.")
            break
        result = predict_message(user_input, model=model, vectorizer=vectorizer)
        print(f"Prediction: {result['label']} (confidence: {result['confidence']}%)")


if __name__ == "__main__":
    _run_cli_demo()
