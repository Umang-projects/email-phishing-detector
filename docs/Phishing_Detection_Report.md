# Phishing Email Detection — Classical ML Experiment Guide

Welcome to the phishing detection modeling branch! Here, you will find a complete walkthrough of building a text-based phishing email classifier using **TF-IDF + Logistic Regression** as the baseline. This guide covers data preparation, model training, evaluation, and inference.

---

## Pre-requisites

Install all required libraries before starting:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## Step-by-Step Experiment Plan

### 1. Understanding the Dataset

The dataset used is `Phishing_Email.csv`. It contains **raw email body text** as the primary feature.

| Attribute | Value |
|---|---|
| Total Samples (after cleaning) | 12,469 |
| Safe Email (Label 0) | 7,577 (60.7%) |
| Phishing Email (Label 1) | 4,906 (39.3%) |
| Missing Labels Dropped | 14 |
| Feature Used | Email Text (raw body) |

The dataset is split using a **stratified 60/20/20 strategy** to preserve class proportions across all subsets:

| Split | Size | Purpose |
|---|---|---|
| Training (60%) | ~7,481 samples | Model training only |
| Validation (20%) | ~2,494 samples | Hyperparameter tuning |
| Test (20%) | ~2,497 samples | Final unbiased evaluation (used once) |

---

### 2. Text Vectorization with TF-IDF

Unlike deep learning models, classical ML cannot consume raw text. We convert it to a sparse numerical matrix using **TF-IDF**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',     # removes filler words (the, is, and...)
    max_features=8000,        # retains 8,000 most informative terms
)
```

---

### 3. Building the Pipeline

Each model is wrapped in a scikit-learn `Pipeline` that chains vectorization + classification in a single call. We selected **Logistic Regression** over SVM because Logistic Regression natively provides probabilistic confidence scores (`predict_proba`), which are vital for our Web Dashboard's Threat Indicators panel.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=8000)),
    ('model', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)       # fit on train only
model.predict(X_test)             # vectorize + classify in one call
```

---

### 4. Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|---|---|---|---|---|---|
| LinearSVC | ~97.00% | 95.00% | 98.00% | 96.00% | Rejected: No probability scores |
| **Logistic Reg. ✅** | **~96.28%** | **94.14%** | **96.53%** | **95.32%** | **Selected**: Provides confidence metrics |

> ✅ Logistic Regression was finalized because the slight drop in accuracy is worth the massive gain in explainability and confidence scoring for the user dashboard.

---

### 5. Final Test Set Evaluation (Logistic Regression)

Evaluated **once** on the held-out test set of **2,497 emails**:

| Metric | Safe Email | Phishing Email | Overall |
|---|---|---|---|
| Precision | 0.98 | 0.94 | 0.96 |
| Recall | 0.97 | 0.97 | 0.97 |
| F1 Score | 0.97 | 0.95 | 0.96 |
| Support | 1,516 | 981 | 2,497 |

---

### 6. Saving & Inspecting the Model

```python
import joblib
import numpy as np

# Save
joblib.dump(model, 'phishing_model.pkl')
print('Model saved to phishing_model.pkl')

# Inspect learned weights
model  = joblib.load('phishing_model.pkl')
tfidf  = model.named_steps['tfidf']
logreg = model.named_steps['model']
vocab  = tfidf.get_feature_names_out()

# Assuming binary classification where 1 is Phishing
idx_1 = list(logreg.classes_).index(1)
weights = logreg.coef_[idx_1]

# Top 10 phishing keywords
top_phish = vocab[np.argsort(weights)[-10:]]
print('Phishing indicators:', top_phish)
```

---

### 7. Running Inference

#### Single Email

```python
import joblib

model = joblib.load('../../models/phishing_model.pkl')

def check_email(email_text: str) -> dict:
    pred       = model.predict([email_text])[0]
    
    # Safely extract probability score
    prob_idx   = list(model.classes_).index(pred)
    confidence = model.predict_proba([email_text])[0][prob_idx] * 100
    
    label      = 'PHISHING' if pred == 1 else 'SAFE'
    
    return {'label': label, 'confidence': f"{confidence:.2f}%"}
```

**Expected Output:**

```
[SAFE    ] Confidence: 94.21%
           Text: Hi John, attached is the Q2 report. Let me know your thoughts...

[PHISHING] Confidence: 99.87%
           Text: URGENT: Your account has been compromised! Click here NOW...
```

---

## Summary

| Item | Value |
|---|---|
| Final Model | LogisticRegression (max_iter=1000, max_features=8000) |
| Test Accuracy | ~96.28% |
| Explainability | Supports exact feature weight extraction |
| Confidence | Supports `predict_proba` for the Dashboard |
| Inference | Single `predict()` call, no GPU needed |

The model is lightweight, fast, and production-deployable. A single `predict()` call handles end-to-end inference including text preprocessing. Compare these results against the BERT experiment branch to see the trade-offs between speed/transparency (Logistic Regression) and deep contextual understanding (BERT)!
