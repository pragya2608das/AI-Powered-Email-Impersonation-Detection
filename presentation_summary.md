# Presentation Summary — Social Media Phishing Detection Using Machine Learning

*(Suitable for a 3–4 slide PBL review)*

---

## Slide 1 — Title & Objective

**Title:** AI-Powered-Email-Impersonation-Detection Using Machine Learning

**Objective:** Build a supervised machine learning system that
automatically classifies raw emails style messages as **Phishing** or **Legitimate**,
and deliver real-time predictions through a web application.

**Why it matters:** Phishing on emails tricks users into
clicking malicious links or revealing credentials, OTPs, and
financial data through fake "verify your account," "you've won a
prize," or "account suspended" messages.

---

## Slide 2 — Dataset & Preprocessing

**Dataset:** SMS Spam Collection (`sms.tsv`, 5,574 labeled
messages), re-purposed for this project:
- `spam` → **Phishing** (label = 1)
- `ham` → **Legitimate** (label = 0)

**Preprocessing pipeline:**
1. Lowercase conversion
2. URL removal
3. Punctuation removal
4. Number removal
5. Extra whitespace removal
6. Optional stopword removal

---

## Slide 3 — Machine Learning Technique (Supervised Learning)

**Approach:** Supervised binary text classification

**Feature extraction:** TF-IDF (Term Frequency–Inverse Document
Frequency) with unigrams + bigrams, 5,000-word vocabulary

**Model:** Logistic Regression (`class_weight="balanced"`, `C=10`)

**Workflow:**

```
User Input → Text Preprocessing → TF-IDF Vectorization
→ Logistic Regression Model → Prediction → Phishing / Legitimate
```

**Train/test split:** 80% / 20%, stratified by class

---

## Slide 4 — Results & Expected Outcomes

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~98.5% |
| Precision  | ~94.6% |
| Recall     | ~94.0% |
| F1-score   | ~94.3% |

**Deliverable:** A Flask web application where a user pastes a
message and instantly receives:
- Phishing / Legitimate classification
- Confidence score
- Contextual warning (⚠ for phishing, ✓ for safe)

**Future scope:** Extend to real social-media DM datasets, add deep
learning models (LSTM/BERT), incorporate URL-reputation features,
and deploy as a browser extension for live protection.
