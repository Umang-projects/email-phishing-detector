# Executive Summary: Phishing Detection Models

This document serves as the finalized evaluation report for the Email Phishing Detector project. It contrasts the performance of our dual-architecture approach: a **Deep Learning Model (BERT)** and a **Classical Machine Learning Pipeline (TF-IDF + Logistic Regression)**.

---

## 1. Model Performance Showdown

Both models were evaluated on the held-out test dataset consisting of raw email text. 

| Metric | BERT (Deep Learning) | Logistic Regression (Classical ML) |
| :--- | :--- | :--- |
| **Accuracy** | 97.90% | ~96.28% |
| **Precision (Phishing)** | 96.58% | ~94.14% |
| **Recall (Phishing)** | 98.26% | ~96.53% |
| **F1-Score** | 97.41% | ~95.32% |
| **Inference Speed** | Slower (requires tensor operations) | Extremely Fast (sparse matrix mult) |
| **Explainability** | Black Box (hard to interpret) | Transparent (exact feature weights) |

**Conclusion:** 
BERT achieved slightly higher absolute metrics due to its contextual understanding of language. However, the Logistic Regression model performs exceptionally well (96%+ Accuracy) while offering instantaneous inference speeds and full mathematical transparency (we can extract the exact words driving the prediction). Both models have been integrated into the final Streamlit Dashboard for side-by-side execution.

---

## 2. Edge Case Analysis & Model Limitations

Despite high accuracy scores, qualitative edge-case testing (detailed in `docs/bert-failure.md`) revealed that both models share certain systematic vulnerabilities based on the training data distribution.

### False Negatives: Spear Phishing
* **The Vulnerability:** Highly sophisticated, grammatically perfect spear-phishing emails that mimic polite corporate communication are often misclassified as `Safe`.
* **The Cause:** The training data contains an overwhelming majority of low-effort, aggressive phishing attempts (e.g., "URGENT WINNER"). The models incorrectly learned that polite corporate jargon equates to safety.

### False Positives: Legitimate System Alerts
* **The Vulnerability:** 100% legitimate automated IT or banking alerts (e.g., "Your Active Directory password will expire") are occasionally misclassified as `Phishing`.
* **The Cause:** **Spurious Correlation**. Words like `"expire"`, `"password"`, and `"locked out"` appear frequently in phishing attempts. The models rely heavily on these heuristic triggers rather than understanding the underlying benign context of a real IT alert.

---

## 3. Threat Indicators & Feature Extraction
To mitigate the "black box" nature of these models and provide actionable intelligence to the user, we developed a deterministic Threat Indicator extraction pipeline:

1. **Suspicious Keyword Flagging:** We use regex to categorically flag emails containing urgency, financial lures, and credential requests.
2. **Link & Attachment Enumeration:** We parse raw `.eml` files to automatically extract and count embedded hyperlinks and file attachments.
3. **ML Explainability:** We cross-reference the input text with the Logistic Regression model's TF-IDF coefficients to expose the "Top 5" words driving the phishing prediction.

*These features are fully integrated into the Web Dashboard (`app.py`).*
