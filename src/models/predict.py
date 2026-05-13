import joblib

def load_model():
    return joblib.load("../models/phishing_model.pkl")

def predict_email(text):
    model = load_model()
    pred = model.predict([text])[0]
    
    result = "Phishing" if pred == 1 else "Safe"
    if hasattr(model, "predict_proba"):
        try:
            prob_idx = list(model.classes_).index(pred)
            prob = model.predict_proba([text])[0][prob_idx] * 100
            return f"{result} (Confidence: {prob:.2f}%)"
        except (ValueError, IndexError):
            pass
            
    return result

if __name__ == "__main__":
    text = input("Enter email: ")
    print(predict_email(text))
