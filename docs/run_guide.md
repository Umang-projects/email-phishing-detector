# How to Run the Phishing Detection Model

**Model File:** `phishing_model.pkl` | **Framework:** scikit-learn

---

## Step 1: Install Requirements

```bash
pip install scikit-learn joblib
```

---

## Step 2: Load the Model

Place `phishing_model.pkl` in the `models/` folder, and ensure you are running your script from the `src/models/` directory (or update the path accordingly):

```python
import joblib

model = joblib.load('../../models/phishing_model.pkl')
```

---

## Step 3: Run a Prediction

Because our pipeline utilizes **Logistic Regression**, it natively supports confidence scoring via probabilities.

```python
import numpy as np

email = "URGENT: Click here to claim your prize!!!"

# 1. Get the raw prediction (0 = Safe, 1 = Phishing)
prediction = model.predict([email])[0]        

# 2. Extract the confidence score (probability)
prob_idx = list(model.classes_).index(prediction)
confidence = model.predict_proba([email])[0][prob_idx] * 100

if prediction == 1:
    print(f"PHISHING  (Confidence: {confidence:.2f}%)")
else:
    print(f"SAFE      (Confidence: {confidence:.2f}%)")
```

> **Note:** Always wrap the email in a list — `model.predict([email])`, not `model.predict(email)`. The `predict_proba` array returns probabilities for all classes, so we look up the probability of the *predicted* class specifically.

---

## Step 4: Understand the Output

- **Prediction:** `0` = Safe, `1` = Phishing
- **Confidence:** Represents the model's certainty as a percentage (e.g., `98.45%`). The closer to 100%, the more certain the model is about its classification.

---

## Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| `FileNotFoundError: phishing_model.pkl` | Check your relative paths. Ensure the `.pkl` file is in the `models/` directory. |
| `ValueError: Expected 2D array` | Use `model.predict([email])` not `model.predict(email)` |
| `IndexError: only integers...` | Do not use `prediction` as a numpy array index. Use `list(model.classes_).index(prediction)` as shown in Step 3. |
