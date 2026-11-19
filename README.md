# AI-Powered-Email-Impersonation-Detection Using Machine Learning

**B.Tech AI & ML — Project-Based Learning (PBL)**

A complete, runnable machine learning system that classifies email messages as **Phishing** or **Legitimate**, served through a Flask web application with a live, real-time detection UI.

---

## 1. Project Overview

Phishing are done in email mostly as Attackers impersonate platform support to
send fake "verify your account," "you've won a prize," or "your
account will be suspended" messages that trick users into clicking
malicious links or handing over credentials, OTPs, and financial
details.

This project builds a supervised machine learning pipeline —
**TF-IDF vectorization + Logistic Regression** — that learns the
linguistic patterns of phishing messages (urgency, rewards, threats,
links, requests to "click" or "verify") and classifies new messages
in real time through a web interface.

## 2. Problem Statement

> Given a short email text message, automatically
> determine whether it is a **phishing attempt** or a **legitimate
> message**, and present the result with a confidence score so a
> user can make an informed decision before clicking any link or
> replying.

## 3. Dataset Description

This project uses the well-known **SMS Spam Collection dataset**
(`dataset/sms.tsv`) — 5,574 real SMS messages labeled `ham` or
`spam`.

**Important — dataset re-purposing:** There are many large, public,labeled dataset of real emial phishing sms available for
academic use. Since phishing SMS core linguistic patterns (urgency, fake rewards,
account threats, suspicious links, requests to click/verify), this
project adapts the SMS Spam Collection dataset as follows:

| Original label | Re-purposed as              | Numeric target |
|-----------------|-----------------------------|-----------------|
| `spam`          | Phishing message            | `1`             |
| `ham`           | Legitimate social-media msg | `0`             |

This substitution is used consistently across the codebase
(`utils.py`, `train_model.py`) and is called out here for full
transparency in methodology, as required for faculty evaluation.

Class balance in the dataset:

- Legitimate (ham): 4,827 messages (~86.6%)
- Phishing (spam): 747 messages (~13.4%)

## 4. Project Structure

```
AI-Powered-Email-Impersonation-Detection/
├── dataset/
│   └── sms.tsv                  # Raw labeled dataset (label \t text)
├── notebooks/
│   └── data_exploration.ipynb   # EDA: class balance, message length, word frequency
├── models/
│   ├── logistic_model.pkl       # Trained Logistic Regression model (generated)
│   ├── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer (generated)
│   └── confusion_matrix.png     # Saved evaluation plot (generated)
├── templates/
│   ├── index.html               # Home page — message input form
│   └── result.html              # Prediction result page
├── static/
│   └── style.css                # Modern, responsive UI styling
├── preprocessing.py              # Reusable text-cleaning functions
├── utils.py                      # Shared path/load/save helpers
├── train_model.py                # Full training pipeline (entry point)
├── evaluate.py                   # Detailed evaluation report + confusion matrix
├── predict.py                    # Single-message inference (CLI + library)
├── app.py                        # Flask web application (entry point)
├── requirements.txt
├── README.md
└── presentation_summary.md       # 3–4 slide PBL review summary
```

## 5. Methodology / Workflow

```
                     ┌───────────────────┐
                     │   User Input      │
                     │     (email)       │ 
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │ Text Preprocessing │
                     │  - lowercase       │
                     │  - remove URLs     │
                     │  - remove punct.   │
                     │  - remove numbers  │
                     │  - trim whitespace │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │ TF-IDF             │
                     │ Vectorization       │
                     │ (uni + bi-grams)    │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │ Logistic           │
                     │ Regression Model    │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │ Prediction +        │
                     │ Confidence Score     │
                     └─────────┬─────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
          Phishing (⚠ warning)     Legitimate (✓ safe)
```

**Pipeline steps implemented in code:**

1. Load `dataset/sms.tsv` (`utils.load_dataset`)
2. Convert labels: `spam → 1 (phishing)`, `ham → 0 (legitimate)`
3. Clean text: lowercase, strip URLs, punctuation, numbers, extra
   whitespace (`preprocessing.clean_text`)
4. Vectorize with `TfidfVectorizer` (max 5,000 features, uni +
   bigrams, `sublinear_tf=True`)
5. Stratified train/test split (80/20)
6. Train `LogisticRegression` (`C=10`, `class_weight="balanced"`)
7. Save model → `models/logistic_model.pkl`
8. Save vectorizer → `models/tfidf_vectorizer.pkl`
9. Evaluate: accuracy, precision, recall, F1, confusion matrix
10. Predict new user input (`predict.py`)
11. Serve results through the Flask app (`app.py`)

## 6. Installation

**Requirements:** Python 3.10+

```bash
# 1. Navigate into the project folder
cd AI-Powered-Email-Impersonation-Detection

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The `dataset/sms.tsv` file is already included in this project
> folder — no manual download is required.

## 7. Execution

**Step 1 — Train the model** (creates `models/logistic_model.pkl`
and `models/tfidf_vectorizer.pkl`):

```bash
python train_model.py
```

**Step 2 (optional) — Detailed evaluation report + confusion
matrix image:**

```bash
python evaluate.py
```

**Step 3 (optional) — Quick CLI prediction demo:**

```bash
python predict.py
```

**Step 4 — Launch the web application:**

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, paste a
message, and click **Detect message**.

## 8. Screenshots

> Add screenshots of the running application here after your first
> local run, e.g.:
> - `screenshots/home_page.png` — the message input screen
> - `screenshots/phishing_result.png` — a phishing detection result
> - `screenshots/legitimate_result.png` — a legitimate result

## 9. Results

Model performance on the held-out 20% test split (1,115 messages),
from the training run included with this project:

| Metric     | Score  |
|------------|--------|
| Accuracy   | 98.5%  |
| Precision  | 94.6%  |
| Recall     | 94.0%  |
| F1-score   | 94.3%  |

Confusion matrix (rows = actual, columns = predicted):

|                    | Predicted Legitimate | Predicted Phishing |
|--------------------|:---------------------:|:--------------------:|
| **Actual Legitimate** | 958                 | 8                   |
| **Actual Phishing**   | 9                   | 140                 |

Run `python evaluate.py` to regenerate this table and the saved
plot at `models/confusion_matrix.png`.

**Example predictions:**

| Message                                             | Prediction   |
|------------------------------------------------------|--------------|
| "Claim your free iPhone reward"                       | Phishing     |
| "Your account will be suspended"                      | Phishing     |
| "Click this link to login"                            | Phishing     |
| "Are you coming to college?"                           | Legitimate   |
| "Happy birthday"                                       | Legitimate   |
| "Meeting is scheduled for tomorrow"                     | Legitimate   |
| "Send me the notes"                                     | Legitimate   |

> Note: because the underlying training data is SMS text rather than
> real Instagram/WhatsApp DMs, messages that don't closely resemble
> the phrasing style of the training set (e.g. very short or
> platform-specific wording) may get lower-confidence predictions.
> See *Future Work* below.

## 10. Future Work

- Collect or simulate a genuine email  dataset
  to reduce domain mismatch with SMS text.
- Add deep learning baselines (LSTM / BERT-based classifiers) and compare against the TF-IDF + Logistic Regression baseline.
- Add URL-based features (domain reputation, shorteners, typosquat
  detection) as a complementary signal alongside text features.
- Deploy as a browser extension that scans DMs/comments in real
  time on the actual platforms.
- Add multi-language support for regional phishing campaigns.
- Add user feedback loop ("Was this correct?") to continuously
  retrain and improve the model.

## 11. Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn (TfidfVectorizer, LogisticRegression)
- **NLP:** NLTK (optional stopword removal)
- **Web:** Flask
- **Visualization:** matplotlib, seaborn
- **Model persistence:** joblib

---

*This project was built for academic demonstration as part of a
B.Tech AI & ML Project-Based Learning (PBL) module.*
