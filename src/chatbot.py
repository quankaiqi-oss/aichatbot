from pathlib import Path
import random
import re

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "data" / "chatbot_dataset_clean.csv"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"[^a-z0-9\s?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts(model_type: str = "svm"):
    model_file = "svm_model.pkl" if model_type == "svm" else "logistic_model.pkl"
    model = joblib.load(MODELS_DIR / model_file)
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    responses_df = pd.read_csv(DATASET_PATH)
    return model, vectorizer, responses_df


def format_response(response: str) -> str:
    replacements = {
        "{{Order Number}}": "your order number",
        "{{Online Company Portal Info}}": "the online shopping portal",
        "{{Online Order Interaction}}": "My Orders",
        "{{Customer Support Hours}}": "9:00 AM to 6:00 PM",
        "{{Customer Support Phone Number}}": "the customer support hotline",
        "{{Website URL}}": "the official website",
        "{{Account Category}}": "customer account",
        "{{Payment Method}}": "your payment method",
        "{{Refund Amount}}": "the refund amount",
    }
    for placeholder, replacement in replacements.items():
        response = response.replace(placeholder, replacement)
    response = re.sub(r"\{\{[^}]+\}\}", "the relevant details", response)
    return response


def predict_intent(message: str, model_type: str = "svm") -> dict:
    model, vectorizer, responses_df = load_artifacts(model_type)
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    intent = model.predict(vector)[0]

    confidence = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(vector)
        confidence = float(scores.max())
    elif hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vector).max())

    matching = responses_df[responses_df["intent"] == intent]["response"].dropna().tolist()
    response = random.choice(matching) if matching else "I can help with your customer support request."

    return {
        "intent": intent,
        "confidence": confidence,
        "response": format_response(response),
    }
