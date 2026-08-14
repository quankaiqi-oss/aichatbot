from pathlib import Path
import random
import re

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
SHARED_MODELS_DIR = BASE_DIR / "shared" / "models"
MODEL_DIRS = {
    "svm": BASE_DIR / "kaiqi_svm" / "models",
    "logistic": BASE_DIR / "kathy_logistic" / "models",
}
DATASET_PATH = BASE_DIR / "data" / "chatbot_dataset_clean.csv"
SVM_CONFIDENCE_THRESHOLD = 0.15
LOGISTIC_CONFIDENCE_THRESHOLD = 0.40

FALLBACK_RESPONSE = (
    "I am not confident I understood your request. Please type a clearer topic "
    "such as order tracking, refund, payment issue, delivery, account problem, "
    "or complaint."
)

INTENT_RESPONSES = {
    "track_order": (
        "You can track your parcel by logging in to your account and opening the "
        "My Orders section. Select the order, then check the delivery status and "
        "tracking number. If the parcel is delayed, contact customer support with "
        "your order number."
    ),
    "track_refund": (
        "You can check your refund status from the refund or return section in "
        "your account. If the refund has not been updated, please prepare your "
        "order number, refund request date, and payment method before contacting support."
    ),
    "get_refund": (
        "To request a refund, open your order details and choose the refund or return "
        "option. Submit the reason and required proof if needed. The support team will "
        "review the request based on the refund policy."
    ),
    "cancel_order": (
        "To cancel an order, go to My Orders, choose the order, and select the cancel "
        "option if it is still available. If the order has already shipped, you may need "
        "to request a return or contact customer support."
    ),
    "payment_issue": (
        "For payment issues, check whether the amount was deducted and whether the order "
        "was confirmed. If payment was deducted but the order failed, contact support with "
        "your transaction reference."
    ),
    "recover_password": (
        "Use the Forgot Password option on the login page. Enter your registered email or "
        "phone number, then follow the reset instructions sent to you."
    ),
    "complaint": (
        "I am sorry about the issue. Please describe what happened and include your order "
        "number if available. The support team can then review the case and assist you further."
    ),
}


def rule_based_intent(cleaned_text: str) -> str | None:
    has_tracking_word = any(
        word in cleaned_text
        for word in ["track", "tracking", "status", "where", "parcel", "package", "shipment"]
    )
    has_order_word = any(word in cleaned_text for word in ["order", "parcel", "package", "shipment", "delivery"])
    has_refund_word = any(word in cleaned_text for word in ["refund", "rebate", "compensation", "money back"])

    if has_tracking_word and has_order_word and not has_refund_word:
        return "track_order"
    if has_tracking_word and has_refund_word:
        return "track_refund"
    if any(word in cleaned_text for word in ["cancel", "cancelling", "cancellation"]):
        return "cancel_order"
    if any(word in cleaned_text for word in ["payment", "paid", "charged", "card declined", "transaction"]):
        return "payment_issue"
    if any(word in cleaned_text for word in ["forgot password", "reset password", "cannot log in", "cant login"]):
        return "recover_password"
    if any(word in cleaned_text for word in ["complaint", "complain", "bad service", "not helpful"]):
        return "complaint"
    return None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"[^a-z0-9\s?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts(model_type: str = "svm"):
    model_file = "svm_model.pkl" if model_type == "svm" else "logistic_model.pkl"
    model = joblib.load(MODEL_DIRS[model_type] / model_file)
    vectorizer = joblib.load(SHARED_MODELS_DIR / "tfidf_vectorizer.pkl")
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
    model_intent = model.predict(vector)[0]
    rule_intent = rule_based_intent(cleaned)

    confidence = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(vector)
        confidence = float(scores.max())
    elif hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vector).max())

    used_fallback = False
    fallback_reason = None
    if rule_intent:
        intent = rule_intent
    elif model_type == "svm" and confidence is not None and confidence < SVM_CONFIDENCE_THRESHOLD:
        intent = "fallback"
        used_fallback = True
        fallback_reason = "Low SVM decision score"
    elif model_type == "logistic" and confidence is not None and confidence < LOGISTIC_CONFIDENCE_THRESHOLD:
        intent = "fallback"
        used_fallback = True
        fallback_reason = "Low Logistic Regression probability"
    else:
        intent = model_intent

    matching = responses_df[responses_df["intent"] == intent]["response"].dropna().tolist()
    response = FALLBACK_RESPONSE if used_fallback else INTENT_RESPONSES.get(intent)
    if response is None:
        response = random.choice(matching) if matching else "I can help with your customer support request."

    return {
        "intent": intent,
        "model_intent": model_intent,
        "confidence": confidence,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "response": format_response(response),
    }
