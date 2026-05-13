# Member 3 — ML Model Development

> **Module:** Phishing Email Classification with Scikit-Learn  
> **Role:** Machine Learning Engineer  
> **Approach:** Classical ML (TF-IDF + Linear Models)  
> **Models Compared:** 4  
> **Best Model:** SVM (Linear) — Selected by Validation F1-Score

---

## 1. Environment & Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
```

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Dataset loading & manipulation |
| `scikit-learn` | Pipelines, models, metrics, splitting |
| `joblib` | Model serialization (`.pkl`) |
| `matplotlib` / `seaborn` | Confusion matrix & visualizations |

---

## 2. Data Setup & Preprocessing

### 2.1 Load & Clean
```python
CSV_PATH = "/content/Phishing_Email.csv"
df = pd.read_csv(CSV_PATH, engine='python', on_bad_lines='skip')

# Drop rows with missing Email Text or Email Type
df = df.dropna(subset=["Email Text", "Email Type"])

# Encode labels: Safe Email → 0 | Phishing Email → 1
df["label"] = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})
df = df.dropna(subset=["label"]).astype(int)
```

### 2.2 Class Distribution
```
Safe Email (0):    ~11,200  (~60.2%)
Phishing (1):      ~7,400   (~39.8%)
Total:             ~18,600  samples
```

---

## 3. Dataset Splitting Strategy

**Stratified 60 / 20 / 20 Split** — maintains exact class proportions across all subsets.

```python
RANDOM_STATE = 42

# Step 1: Separate Test set (20% of total)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Step 2: From remaining 80%, split Validation (25% of 80% = 20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
)
```

| Split | Percentage | Samples | Purpose |
|-------|-----------|---------|---------|
| **Training** | 60% | ~11,178 | Model fitting |
| **Validation** | 20% | ~3,726 | Hyperparameter tuning & model selection |
| **Test** | 20% | ~3,727 | **Final evaluation — touched ONCE only** |

> ⚠️ **Critical:** The Test set is strictly held out. It is used **only once** for final metrics after the best model is selected.

---

## 4. Pipeline Construction

### 4.1 TF-IDF Vectorizer Configuration
```python
TfidfVectorizer(
    stop_words="english",      # Remove common words (the, and, is)
    max_features=10000,         # Keep top 10,000 terms
    ngram_range=(1, 2)          # Unigrams + Bigrams
)
```

### 4.2 Models Compared
All 4 models use the **exact same TF-IDF pipeline** for fair comparison.

| Model | Classifier | Key Properties |
|-------|-----------|----------------|
| **Logistic Regression** | `LogisticRegression(max_iter=1000, random_state=42)` | Binary classification, probability scores, `solver=lbfgs` |
| **Naive Bayes** | `MultinomialNB()` | Fast probabilistic baseline, strong for sparse text |
| **SVM (Linear)** | `LinearSVC(max_iter=2000)` | Best for sparse text, excellent generalization, **Best Model** |
| **SVM (C=10)** | `LinearSVC(C=10, max_iter=2000)` | Higher regularization, near-identical to linear variant |

### 4.3 Pipeline Architecture
```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("model", classifier)
])
```

---

## 5. Training & Selection Strategy

### 5.1 Workflow
1. **Train** all 4 models on the **Training set** (60%)
2. **Evaluate** on the **Validation set** (20%) — compare Accuracy, Precision, Recall, F1
3. **Select Best Model** by highest **Validation F1-Score**
4. **Final Evaluation** on the **Test set** (20%) — touched only once

### 5.2 Why F1-Score & Recall?

| Metric | Formula | Why It Matters |
|--------|---------|--------------|
| **Recall** | TP / (TP + FN) | **Most critical** — a missed phishing email (False Negative) is the most dangerous outcome |
| **Precision** | TP / (TP + FP) | Avoid false alarms — don't block legitimate emails |
| **F1-Score** | 2 × (P × R) / (P + R) | **Primary selection metric** — harmonic mean balancing Precision & Recall |
| **Accuracy** | (TP + TN) / Total | Overall correctness — supporting metric |

---

## 6. Validation Set Results

**Validation Set:** Stratified 20% of ~18,600 samples (~3,726 emails)

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| Logistic Regression | 97.21% | 0.9521 | 0.9781 | 0.9649 | — |
| Naive Bayes | 95.68% | 0.9663 | 0.9220 | 0.9436 | Baseline |
| **SVM (Linear)** 🏆 | **97.37%** | **0.9517** | **0.9829** | **0.9670** | **Best Model** |
| SVM (C=10) | 97.34% | 0.9528 | 0.9808 | 0.9666 | Runner-up |

> 🏆 **Best Model Selected:** `SVM (Linear)`  
> - Highest Validation F1: **96.70%**  
> - Highest Validation Recall: **98.29%** (fewest missed phishing emails)

---

## 7. Final Test Set Evaluation

**Held-out Test Set:** 3,727 emails — evaluated **only once** using the selected best model.

### 7.1 Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **96.97%** |
| **Precision** | **95.12%** |
| **Recall** | **97.27%** |
| **F1-Score** | **96.18%** |

### 7.2 Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Safe Email | 0.98 | 0.97 | 0.97 | 2,264 |
| Phishing Email | 0.95 | 0.97 | 0.96 | 1,463 |
| **Overall** | **0.97** | **0.97** | **0.97** | **3,727** |

### 7.3 Confusion Matrix

| | Predicted Safe | Predicted Phishing |
|---|---|---|
| **Actual Safe** | 2,191 (TN) | 73 (FP) |
| **Actual Phishing** | 40 (FN) | 1,423 (TP) |

- **96.8%** of Safe emails correctly identified
- **97.3%** of Phishing emails correctly caught
- Only **40 phishing emails missed** out of 1,463

---

## 8. Visualizations

A **4-panel dashboard** is generated and saved as `phishing_detection_results.png`:

| Panel | Chart Type | Content |
|-------|-----------|---------|
| **1. Data Split** | Pie Chart | Training / Validation / Test proportions with sample counts |
| **2. Model Comparison** | Horizontal Bar Chart | Validation F1-Score for all 4 models — Best model highlighted |
| **3. Confusion Matrix** | Seaborn Heatmap | Test set predictions with TN, FP, FN, TP counts |
| **4. Final Test Metrics** | Vertical Bar Chart | Accuracy, Precision, Recall, F1-Score with score labels |

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Phishing Email Detection - Complete Results', fontsize=16, fontweight='bold')
# ... (4 subplots as described above)
plt.savefig('phishing_detection_results.png', dpi=150, bbox_inches='tight')
```

---

## 9. Model Export & Inference

### 9.1 Save Best Model
```python
MODEL_FILENAME = "phishing_model.pkl"
joblib.dump(best_model, MODEL_FILENAME)
```
> Saved as a single `Pipeline` object containing both the TF-IDF vectorizer and the trained SVM classifier.

### 9.2 Load & Predict (Inference)
```python
import joblib

# Load model
model = joblib.load("phishing_model.pkl")

# Predict on new emails
new_emails = ["Your email text here..."]
predictions = model.predict(new_emails)
# 0 = Safe Email | 1 = Phishing Email
```

### 9.3 Sample Predictions

| Email | Prediction |
|-------|-----------|
| "Hello, please find the attached report for your review. Thanks, John" | ✅ Safe |
| "URGENT: You won $1,000,000! Click here to claim your prize now!!!" | 🚨 Phishing |
| "Meeting rescheduled to 3pm tomorrow. Let me know if that works." | ✅ Safe |
| "CONGRATULATIONS! Your account has been selected for a special reward." | 🚨 Phishing |

---

## 10. Deliverables & Artifacts

| Artifact | Description |
|----------|-------------|
| `phishing_model.pkl` | Serialized best model (TF-IDF + SVM pipeline) ready for deployment |
| `phishing_detection_results.png` | 4-panel results dashboard |
| Validation Results Table | Comparative metrics for all 4 models |
| Test Classification Report | Final unbiased evaluation on held-out test set |

---

## 11. Key Takeaways

1. **SVM (Linear) + TF-IDF** delivers **96.97% accuracy** and **97.27% recall** on unseen test data.
2. **Stratified 60/20/20 split** ensures fair evaluation and prevents data leakage.
3. **F1-Score** was the right choice for model selection — it balances catching phishing (Recall) with avoiding false alarms (Precision).
4. The complete pipeline (vectorizer + classifier) is saved as a single object, making deployment and inference seamless.
5. Classical ML with TF-IDF remains highly competitive for text classification and is **lightweight & interpretable** compared to deep learning alternatives.

---

> **End of Member 3 — ML Model Development Documentation**
