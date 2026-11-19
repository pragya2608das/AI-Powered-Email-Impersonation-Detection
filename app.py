"""
app.py
------
Flask web application for the AI-Powered-Email-Impersonation-Detection
project.Provides a simple, modern UI where a user can paste a
email message and get a real-time Phishing /
Legitimate classification with a confidence score.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request

from predict import predict_message
from utils import load_artifacts

app = Flask(__name__)

# Load the trained model and vectorizer once at startup so every
# request doesn't pay the cost of reading from disk again.
try:
    MODEL, VECTORIZER = load_artifacts()
    MODEL_LOADED = True
    MODEL_LOAD_ERROR = None
except FileNotFoundError as exc:
    MODEL, VECTORIZER = None, None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(exc)

# Example messages shown on the home page for quick demoing.
PHISHING_EXAMPLES = [
    "Verify your Instagram account immediately",
    "Claim your free iPhone reward",
    "Your account will be suspended",
    "Click this link to login",
]

LEGITIMATE_EXAMPLES = [
    "Are you coming to college?",
    "Happy birthday",
    "Meeting is scheduled for tomorrow",
    "Send me the notes",
]


@app.route("/", methods=["GET"])
def home():
    """Render the home page with the input form."""
    return render_template(
        "index.html",
        model_loaded=MODEL_LOADED,
        model_error=MODEL_LOAD_ERROR,
        phishing_examples=PHISHING_EXAMPLES,
        legitimate_examples=LEGITIMATE_EXAMPLES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission, run prediction, and render the result page."""
    message = request.form.get("message", "")

    if not MODEL_LOADED:
        return render_template(
            "index.html",
            model_loaded=MODEL_LOADED,
            model_error=MODEL_LOAD_ERROR,
            phishing_examples=PHISHING_EXAMPLES,
            legitimate_examples=LEGITIMATE_EXAMPLES,
        )

    if not message.strip():
        error = "Please enter a message before clicking Detect."
        return render_template(
            "index.html",
            model_loaded=MODEL_LOADED,
            model_error=None,
            error=error,
            phishing_examples=PHISHING_EXAMPLES,
            legitimate_examples=LEGITIMATE_EXAMPLES,
        )

    result = predict_message(message, model=MODEL, vectorizer=VECTORIZER)
    is_phishing = result["prediction"] == 1

    return render_template(
        "result.html",
        message=result["message"],
        label=result["label"],
        confidence=result["confidence"],
        is_phishing=is_phishing,
    )


if __name__ == "__main__":
    # debug=True is convenient for local PBL demos; disable it if
    # deploying anywhere beyond localhost.
    app.run(debug=True)
