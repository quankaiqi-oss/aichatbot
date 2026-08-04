from pathlib import Path
from datetime import datetime
import csv
import json

import pandas as pd
import streamlit as st

from shared.src.chatbot import predict_intent


BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

MODEL_RESULT_DIRS = {
    "svm": BASE_DIR / "kaiqi_svm" / "results",
    "logistic": BASE_DIR / "kathy_logistic" / "results",
}

st.set_page_config(
    page_title="ShopCare MY",
    layout="wide",
)

st.title("ShopCare MY")
st.caption("Customer support chatbot for Malaysian online shoppers")

page = st.sidebar.radio(
    "Menu",
    ["Chatbot", "Model Evaluation", "Model Comparison", "About Project"],
)

if page == "Chatbot":
    st.header("Real-Time Chatbot")
    model_type = st.sidebar.radio("Chatbot model", ["svm", "logistic"])

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am ShopCare MY. Ask me about order tracking, refunds, "
                    "delivery, payment, account issues, or customer support."
                ),
            }
        ]

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "intent" in message:
                details = f"Intent: {message['intent']} | Model: {message['model'].upper()}"
                if message.get("model_intent") and message["model_intent"] != message["intent"]:
                    details += f" | Raw model: {message['model_intent']}"
                if message.get("confidence") is not None:
                    details += f" | Decision score: {message['confidence']:.3f}"
                st.caption(details)

    user_message = st.chat_input("Type your customer support question...")

    if user_message:
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_message}
        )

        try:
            result = predict_intent(user_message, model_type=model_type)
            assistant_message = {
                "role": "assistant",
                "content": result["response"],
                "model": model_type,
                "intent": result["intent"],
                "model_intent": result.get("model_intent"),
                "confidence": result["confidence"],
                "user_message": user_message,
            }
            st.session_state["chat_history"].append(assistant_message)
            st.session_state["last_prediction"] = assistant_message
            st.rerun()
        except FileNotFoundError:
            st.error("Model files are missing. Please run preprocessing and training first.")

    col1, col2 = st.columns([1, 4])
    if col1.button("Clear Chat"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("last_prediction", None)
        st.rerun()

    if "last_prediction" in st.session_state:
        with st.expander("Save feedback for latest response"):
            result = st.session_state["last_prediction"]
            with st.form("feedback_form"):
                rating = st.slider("Was this response useful?", 1, 5, 4)
                comment = st.text_input("Comment")
                submitted = st.form_submit_button("Save Feedback")
                if submitted:
                    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                datetime.now().isoformat(timespec="seconds"),
                                result["user_message"],
                                result["intent"],
                                rating,
                                comment,
                            ]
                        )
                    st.success("Feedback saved.")

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    selected_model = st.selectbox("Select model", ["svm", "logistic"])
    results_dir = MODEL_RESULT_DIRS[selected_model]
    metrics_path = results_dir / f"{selected_model}_metrics.json"
    matrix_path = results_dir / f"{selected_model}_confusion_matrix.png"
    report_path = results_dir / f"{selected_model}_classification_report.csv"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
        cols[1].metric("Precision", f"{metrics['precision']:.4f}")
        cols[2].metric("Recall", f"{metrics['recall']:.4f}")
        cols[3].metric("F1 Score", f"{metrics['f1_score']:.4f}")
        st.caption(
            f"Average prediction time: {metrics.get('average_prediction_time_ms', 0):.4f} ms"
        )

        if matrix_path.exists():
            st.image(str(matrix_path), caption=f"{selected_model.upper()} confusion matrix")
        if report_path.exists():
            st.dataframe(pd.read_csv(report_path), use_container_width=True)
    else:
        st.info("Evaluation results are not generated yet.")

elif page == "Model Comparison":
    st.header("Model Comparison")
    rows = []
    for model_name in ["svm", "logistic"]:
        metrics_path = MODEL_RESULT_DIRS[model_name] / f"{model_name}_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "Model": model_name.upper(),
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1_score"],
                    "Avg Prediction Time (ms)": metrics.get("average_prediction_time_ms", 0),
                }
            )

    if rows:
        comparison_df = pd.DataFrame(rows)
        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]])
    else:
        st.info("Comparison results are not generated yet.")

else:
    st.header("About Project")
    st.subheader("Title")
    st.write("ShopCare MY: A Machine Learning Customer Support Chatbot for Malaysian Online Shoppers")
    st.subheader("Dataset")
    st.write(
        "This prototype uses the Bitext customer support chatbot dataset. "
        "The dataset is adapted to a Malaysian online-shopping support scenario."
    )
    st.subheader("Methods")
    st.write("SVM module: TF-IDF + Linear SVM")
    st.write("Comparison module: TF-IDF + Logistic Regression")
