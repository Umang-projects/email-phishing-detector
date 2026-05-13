import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

import os
from .core_ml import build_model

def load_data(path):
    df = pd.read_csv(path, engine='python', on_bad_lines='skip')

    df = df.dropna(subset=["Email Text"])
    df["label"] = df["Email Type"].map({
        "Safe Email": 0,
        "Phishing Email": 1
    })
    df = df.dropna(subset=["label"])

    return df

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'external', 'Phishing_Email.csv')
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Email Text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    model_path = os.path.join(base_dir, 'models', 'phishing_model.pkl')
    joblib.dump(model, model_path)

    print("✅ Model trained and saved")

if __name__ == "__main__":
    train()
