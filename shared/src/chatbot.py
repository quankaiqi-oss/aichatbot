from pathlib import Path
from functools import lru_cache
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
    "I am not confident I understood your request. Is your issue about order tracking, "
    "refund, payment, delivery, account access, cancellation, or customer service?"
)

CLARIFICATION_RESPONSE = (
    "I can assist you, but I need one more detail before giving the correct guidance.\n\n"
    "Please choose the issue category:\n"
    "- Order tracking or delivery delay\n"
    "- Refund or return request\n"
    "- Payment issue\n"
    "- Account login problem\n"
    "- Cancellation or customer support\n\n"
    "Which issue would you like help with?"
)

CONVERSATIONAL_RESPONSES = {
    "greeting": (
        "Hi, I am ShopCare MY. I can assist with common online shopping support issues.\n\n"
        "You may ask about:\n"
        "- Order tracking and delivery delay\n"
        "- Refunds, returns, or damaged items\n"
        "- Payment, account, voucher, or complaint issues\n\n"
        "What issue would you like help with?"
    ),
    "thanks": "You are welcome. Would you like help with anything else?",
    "goodbye": "Thank you for using ShopCare MY. I hope your issue is resolved soon.",
    "capability": (
        "I can help with order tracking, refunds, payment issues, delivery delays, "
        "cancellations, account access, vouchers, invoices, complaints, and contacting support."
    ),
    "positive_confirmation": "Okay. What would you like to check next?",
    "negative_confirmation": "No problem. Tell me the correct issue, such as refund, delivery, payment, or account login.",
}

EMPATHY_PREFIXES = {
    "payment_issue": "Sorry about that, payment problems can be stressful. ",
    "delivery_period": "I understand, waiting for a parcel can be frustrating. ",
    "damaged_item": "Sorry your item arrived damaged. ",
    "wrong_item": "Sorry about the wrong item. ",
    "complaint": "I am sorry you had a bad experience. ",
    "recover_password": "No worries, account login problems are common. ",
}

FOLLOW_UP_OPTIONS = {
    "clarify": [
        "My parcel is late",
        "I want a refund",
        "Payment was deducted",
        "I need customer support",
    ],
    "fallback": [
        "Track order",
        "Request refund",
        "Payment issue",
        "Contact support",
    ],
    "track_order": [
        "My parcel is late",
        "I need tracking number help",
        "Contact support",
    ],
    "delivery_period": [
        "Contact support",
        "Track order",
        "Change shipping address",
    ],
    "get_refund": [
        "Item damaged",
        "Wrong item received",
        "Check refund status",
    ],
    "track_refund": [
        "Contact support",
        "Check refund policy",
        "Payment issue",
    ],
    "payment_issue": [
        "Payment was deducted",
        "Order not created",
        "Contact support",
    ],
    "cancel_order": [
        "Check cancellation fee",
        "Change order",
        "Contact support",
    ],
    "damaged_item": [
        "Request refund",
        "Contact support",
        "Check refund policy",
    ],
    "wrong_item": [
        "Request refund",
        "Contact support",
        "Check refund policy",
    ],
    "recover_password": [
        "Cannot login",
        "Contact support",
        "Create account",
    ],
    "contact_customer_service": [
        "Payment issue",
        "Delivery delay",
        "Complaint",
    ],
}

INTENT_RESPONSES = {
    "track_order": (
        "To track your order, please follow these steps:\n\n"
        "- Open the My Orders section in your account.\n"
        "- Select the order that you want to check.\n"
        "- Review the delivery status and tracking number.\n"
        "- If the parcel is delayed, prepare your order number before contacting support."
    ),
    "track_refund": (
        "To check your refund status, please follow these steps:\n\n"
        "- Open the refund or return section in your account.\n"
        "- Check whether the refund request is still processing or already approved.\n"
        "- Prepare your order number, refund request date, and payment method if support is needed."
    ),
    "get_refund": (
        "To request a refund, please follow these steps:\n\n"
        "- Open My Orders and select the affected item.\n"
        "- Choose the Refund or Return option.\n"
        "- Provide a clear reason for the refund request.\n"
        "- Upload supporting proof if needed, such as photos for damaged or wrong items."
    ),
    "cancel_order": (
        "To cancel an order, please check the order status first:\n\n"
        "- Open My Orders and select the order.\n"
        "- Choose Cancel if the cancellation option is still available.\n"
        "- If the order has already shipped, you may need to request a return or contact support."
    ),
    "payment_issue": (
        "For payment issues, please check the following:\n\n"
        "- Confirm whether the payment amount was deducted.\n"
        "- Check whether the order was successfully created.\n"
        "- If payment was deducted but no order appears, contact support with your transaction reference."
    ),
    "recover_password": (
        "To recover your password, please follow these steps:\n\n"
        "- Go to the login page.\n"
        "- Select Forgot Password.\n"
        "- Enter your registered email or phone number.\n"
        "- Follow the reset instructions sent to you."
    ),
    "complaint": (
        "To submit a complaint, please prepare the following details:\n\n"
        "- A short explanation of what happened.\n"
        "- Your order number, if available.\n"
        "- Screenshots or proof related to the issue.\n"
        "- Any previous communication with the seller or support team."
    ),
    "delivery_period": (
        "Delivery time depends on the courier, seller processing time, and your location. "
        "Please check the estimated delivery date in My Orders. If it has passed, contact "
        "support with your order number."
    ),
    "delivery_options": (
        "Available delivery options are usually shown during checkout. You can compare courier "
        "choices, delivery fees, and estimated arrival dates before confirming the order."
    ),
    "change_shipping_address": (
        "To change the shipping address, open your order details and check whether address "
        "editing is still available. If the order has already shipped, contact customer support "
        "as soon as possible."
    ),
    "check_refund_policy": (
        "Refund eligibility depends on the item condition, return period, seller policy, and "
        "reason for refund. Please check the refund policy in the order or help centre before "
        "submitting your request."
    ),
    "contact_customer_service": (
        "You can contact customer support through the help centre, live chat, or support form. "
        "Prepare your order number, screenshots, and a short explanation of the issue."
    ),
    "contact_human_agent": (
        "I can guide you first, but if you need a human agent, open the help centre and choose "
        "live chat or submit a support ticket with your order details."
    ),
    "check_payment_methods": (
        "You can check available payment methods during checkout. Common options include card, "
        "online banking, e-wallet, and vouchers, depending on the platform."
    ),
    "check_invoice": (
        "You can check the invoice from your order details or purchase history. If it is not "
        "available, contact support and provide the order number."
    ),
    "get_invoice": (
        "To get an invoice, open My Orders, select the completed order, and look for the invoice "
        "or receipt option. Download it from there if available."
    ),
    "create_account": (
        "To create an account, choose Sign Up, enter your email or phone number, verify it, and "
        "set a secure password."
    ),
    "registration_problems": (
        "For registration problems, check whether your email or phone number is already used, "
        "then request a new verification code. If it still fails, contact support."
    ),
    "edit_account": (
        "To update account details, open account settings or profile settings, edit the relevant "
        "information, and save the changes."
    ),
    "delete_account": (
        "To delete your account, go to account settings and look for account deletion or privacy "
        "options. Make sure you resolve active orders, refunds, and wallet balances first."
    ),
    "place_order": (
        "To place an order, add the item to cart, confirm the shipping address, choose payment "
        "method, apply vouchers if needed, and submit the order."
    ),
    "change_order": (
        "To change an order, open the order details and check whether editing is still allowed. "
        "If the order is already processed, you may need to cancel it or contact support."
    ),
    "check_cancellation_fee": (
        "Cancellation fees depend on the platform policy, seller status, and whether the order "
        "has already been processed or shipped. Check the cancellation details before confirming."
    ),
    "review": (
        "You can leave a product review from your completed order page. Share clear feedback "
        "about item quality, delivery, and seller service."
    ),
    "newsletter_subscription": (
        "You can manage newsletter or promotional messages from notification settings, email "
        "preferences, or account settings."
    ),
    "switch_account": (
        "To switch account, log out from the current account and log in using the other email, "
        "phone number, or linked login method."
    ),
    "damaged_item": (
        "If the item arrived damaged, take clear photos or videos, keep the packaging, and open "
        "a return or refund request from the order page. Include the evidence when submitting."
    ),
    "wrong_item": (
        "If you received the wrong item, take photos of the parcel and product, then request a "
        "return or refund from the order page. Include your order number and evidence."
    ),
    "voucher_issue": (
        "For voucher issues, check the voucher expiry date, minimum spend, selected products, "
        "payment method, and whether it has already been used."
    ),
}


def conversational_intent(cleaned_text: str) -> str | None:
    if cleaned_text in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        return "greeting"
    if cleaned_text in ["thanks", "thank you", "tq", "thx"]:
        return "thanks"
    if cleaned_text in ["bye", "goodbye", "see you"]:
        return "goodbye"
    if cleaned_text in ["ok", "okay", "yes", "yup", "sure"]:
        return "positive_confirmation"
    if cleaned_text in ["no", "nope", "not this"]:
        return "negative_confirmation"
    if any(phrase in cleaned_text for phrase in ["what can you do", "help me with what", "who are you"]):
        return "capability"
    return None


def infer_from_previous_intent(cleaned_text: str, previous_intent: str | None) -> str | None:
    if previous_intent not in ["get_refund", "track_refund", "damaged_item", "wrong_item", "payment_issue"]:
        return None
    if any(word in cleaned_text for word in ["damaged", "broken", "cracked", "rosak", "pecah"]):
        return "damaged_item"
    if any(word in cleaned_text for word in ["wrong", "different", "salah"]):
        return "wrong_item"
    if any(word in cleaned_text for word in ["deducted", "charged", "paid", "transaction"]):
        return "payment_issue"
    if any(word in cleaned_text for word in ["status", "pending", "not received", "where"]):
        return "track_refund" if "refund" in previous_intent else previous_intent
    return None


def rule_based_intent(cleaned_text: str, previous_intent: str | None = None) -> str | None:
    conversational = conversational_intent(cleaned_text)
    if conversational:
        return conversational

    contextual = infer_from_previous_intent(cleaned_text, previous_intent)
    if contextual:
        return contextual

    vague_problem_words = ["problem", "issue", "help", "cannot", "cant", "tak boleh", "error"]
    has_only_vague_problem = (
        any(word in cleaned_text for word in vague_problem_words)
        and not any(
            word in cleaned_text
            for word in [
                "order",
                "parcel",
                "package",
                "delivery",
                "refund",
                "payment",
                "paid",
                "cancel",
                "password",
                "login",
                "account",
                "address",
                "invoice",
                "voucher",
                "damaged",
                "wrong",
                "seller",
                "support",
            ]
        )
    )
    if has_only_vague_problem:
        return "clarify"

    has_tracking_word = any(
        word in cleaned_text
        for word in [
            "track",
            "tracking",
            "status",
            "where",
            "parcel",
            "package",
            "shipment",
            "sampai mana",
            "mana barang",
            "belum sampai",
        ]
    )
    has_order_word = any(
        word in cleaned_text
        for word in ["order", "parcel", "package", "shipment", "delivery", "barang", "item"]
    )
    has_refund_word = any(
        word in cleaned_text
        for word in ["refund", "rebate", "compensation", "money back", "return money", "duit balik"]
    )

    if has_tracking_word and has_order_word and not has_refund_word:
        return "track_order"
    if has_tracking_word and has_refund_word:
        return "track_refund"
    if any(word in cleaned_text for word in ["damaged", "broken", "cracked", "rosak", "pecah"]):
        return "damaged_item"
    if any(word in cleaned_text for word in ["wrong item", "wrong product", "different item", "salah barang"]):
        return "wrong_item"
    if any(word in cleaned_text for word in ["delay", "delayed", "late", "lambat", "not arrive", "not received"]):
        return "delivery_period"
    if any(word in cleaned_text for word in ["cancel", "cancelling", "cancellation", "batalkan"]):
        return "cancel_order"
    if any(word in cleaned_text for word in ["payment method", "pay with", "cash on delivery", "cod"]):
        return "check_payment_methods"
    if any(
        word in cleaned_text
        for word in [
            "payment",
            "paid",
            "charged",
            "card declined",
            "transaction",
            "deducted",
            "kena deduct",
            "duit kena potong",
        ]
    ):
        return "payment_issue"
    if any(
        word in cleaned_text
        for word in ["forgot password", "reset password", "cannot log in", "cant login", "cannot login", "tak boleh login"]
    ):
        return "recover_password"
    if any(word in cleaned_text for word in ["refund policy", "can refund", "eligible refund"]):
        return "check_refund_policy"
    if has_refund_word:
        return "get_refund"
    if any(word in cleaned_text for word in ["change address", "wrong address", "shipping address", "alamat"]):
        return "change_shipping_address"
    if any(word in cleaned_text for word in ["invoice", "receipt", "resit"]):
        return "get_invoice"
    if any(word in cleaned_text for word in ["voucher", "coupon", "promo code", "discount code"]):
        return "voucher_issue"
    if any(word in cleaned_text for word in ["human agent", "real person", "live agent"]):
        return "contact_human_agent"
    if any(word in cleaned_text for word in ["contact support", "customer service", "seller never reply", "support"]):
        return "contact_customer_service"
    if any(word in cleaned_text for word in ["register", "sign up", "verification code"]):
        return "registration_problems"
    if any(word in cleaned_text for word in ["complaint", "complain", "bad service", "not helpful", "angry", "scam"]):
        return "complaint"
    return None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"[^a-z0-9\s?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=2)
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


def build_response(intent: str, response: str) -> str:
    if intent in EMPATHY_PREFIXES:
        response = EMPATHY_PREFIXES[intent] + response
    if FOLLOW_UP_OPTIONS.get(intent):
        response = response + "\n\nWould you like to continue with one of the options below?"
    return response


def predict_intent(message: str, model_type: str = "svm", previous_intent: str | None = None) -> dict:
    cleaned = clean_text(message)
    rule_intent = rule_based_intent(cleaned, previous_intent)

    if rule_intent in CONVERSATIONAL_RESPONSES:
        return {
            "intent": rule_intent,
            "model_intent": None,
            "confidence": None,
            "used_fallback": False,
            "fallback_reason": None,
            "response": CONVERSATIONAL_RESPONSES[rule_intent],
            "follow_up_options": FOLLOW_UP_OPTIONS.get(rule_intent, []),
        }

    model, vectorizer, responses_df = load_artifacts(model_type)
    vector = vectorizer.transform([cleaned])
    model_intent = model.predict(vector)[0]

    confidence = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(vector)
        confidence = float(scores.max())
    elif hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vector).max())

    used_fallback = False
    fallback_reason = None
    needs_clarification = rule_intent == "clarify"
    if needs_clarification:
        intent = "clarify"
        used_fallback = True
        fallback_reason = "User message is too vague"
    elif rule_intent:
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
    if intent == "clarify":
        response = CLARIFICATION_RESPONSE
    elif used_fallback:
        response = FALLBACK_RESPONSE
    else:
        response = INTENT_RESPONSES.get(intent)
    if response is None:
        response = random.choice(matching) if matching else "I can help with your customer support request."
    response = build_response(intent, format_response(response))

    return {
        "intent": intent,
        "model_intent": model_intent,
        "confidence": confidence,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "response": response,
        "follow_up_options": FOLLOW_UP_OPTIONS.get(intent, []),
    }
