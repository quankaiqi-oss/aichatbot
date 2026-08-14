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

st.markdown(
    """
    <style>
        :root {
            --primary: #0f766e;
            --primary-dark: #115e59;
            --accent: #f59e0b;
            --ink: #172033;
            --muted: #64748b;
            --panel: #ffffff;
            --soft: #f5f7fb;
            --line: #dbe4ef;
        }

        .stApp {
            background:
                linear-gradient(180deg, #eef7f6 0%, #f7fafc 260px, #ffffff 620px);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        [data-testid="stSidebar"] {
            background: #0f172a;
        }

        [data-testid="stSidebar"] * {
            color: #e5eef7 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            margin: 0.28rem 0;
            padding: 0.25rem 0.45rem;
        }

        .app-hero {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
        }

        .app-hero h1 {
            margin: 0;
            color: #10233f;
            font-size: 2.15rem;
            line-height: 1.1;
            letter-spacing: 0;
        }

        .app-hero p {
            margin: 0.45rem 0 0 0;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .eyebrow {
            margin-bottom: 0.35rem !important;
            color: var(--primary-dark) !important;
            font-size: 0.78rem !important;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            justify-content: flex-end;
            min-width: 250px;
        }

        .hero-pills span {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #bbf7d0;
            border-radius: 999px;
            padding: 0.38rem 0.68rem;
            font-size: 0.82rem;
            font-weight: 600;
        }

        h2, h3 {
            color: #172033;
            letter-spacing: 0;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid #99f6e4;
            background: var(--primary);
            color: white;
            font-weight: 650;
        }

        .stButton > button:hover {
            border-color: var(--primary-dark);
            background: var(--primary-dark);
            color: white;
        }

        [data-testid="stChatMessage"] {
            border-radius: 12px;
            border: 1px solid #e5edf5;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.78);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
        }

        @media (max-width: 760px) {
            .app-hero {
                display: block;
            }

            .hero-pills {
                justify-content: flex-start;
                margin-top: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-hero">
        <div>
            <p class="eyebrow">Machine Learning Chatbot Prototype</p>
            <h1>ShopCare MY</h1>
            <p>Customer support chatbot for Malaysian online shoppers, powered by TF-IDF intent classification.</p>
        </div>
        <div class="hero-pills">
            <span>Linear SVM</span>
            <span>Logistic Regression</span>
            <span>Feedback Analysis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Menu",
    ["Chatbot", "Model Evaluation", "Model Comparison", "Feedback Records", "About Project"],
)

if page == "Chatbot":
    st.header("Real-Time Chatbot")
    st.caption("Type naturally or use the quick reply dropdown. Every message is still processed by the ML model.")
    model_type = st.sidebar.radio("Chatbot model", ["svm", "logistic"])
    quick_replies = {
        "Track Order": "I want to track my order",
        "Request Refund": "I want to request a refund",
        "Payment Issue": "I have a payment issue",
        "Delivery Help": "I need help with delivery",
        "Account Help": "I have an account problem",
        "Cancel Order": "I want to cancel my order",
        "Contact Support": "I want to contact customer support",
        "Complaint": "I want to make a complaint",
    }

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
                if message.get("used_fallback"):
                    details += f" | Fallback used: {message.get('fallback_reason', 'Low confidence')}"
                if message.get("confidence") is not None:
                    details += f" | Decision score: {message['confidence']:.3f}"
                st.caption(details)

    quick_message = None
    with st.expander("Quick reply suggestions", expanded=False):
        col1, col2 = st.columns([3, 1])
        selected_quick_reply = col1.selectbox(
            "Choose a common support topic",
            ["Select a topic"] + list(quick_replies.keys()),
            label_visibility="collapsed",
        )
        if col2.button("Send", use_container_width=True) and selected_quick_reply != "Select a topic":
            quick_message = quick_replies[selected_quick_reply]

    user_message = st.chat_input("Type your customer support question...")
    submitted_message = quick_message or user_message

    if submitted_message:
        st.session_state["chat_history"].append(
            {"role": "user", "content": submitted_message}
        )

        try:
            result = predict_intent(submitted_message, model_type=model_type)
            assistant_message = {
                "role": "assistant",
                "content": result["response"],
                "model": model_type,
                "intent": result["intent"],
                "model_intent": result.get("model_intent"),
                "confidence": result["confidence"],
                "used_fallback": result.get("used_fallback", False),
                "fallback_reason": result.get("fallback_reason"),
                "user_message": submitted_message,
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
    st.caption("View the selected model's classification performance on the shared test dataset.")
    selected_model = st.selectbox("Select model", ["svm", "logistic"])
    results_dir = MODEL_RESULT_DIRS[selected_model]
    metrics_path = results_dir / f"{selected_model}_metrics.json"
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

        if report_path.exists():
            st.dataframe(pd.read_csv(report_path), use_container_width=True)
    else:
        st.info("Evaluation results are not generated yet.")

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.caption("Both models use the same dataset, preprocessing, TF-IDF settings, and train-test split.")
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

elif page == "Feedback Records":
    st.header("Feedback Records")
    st.caption("Saved user ratings support the chatbot usability and satisfaction evaluation.")

    if not FEEDBACK_PATH.exists() or FEEDBACK_PATH.stat().st_size == 0:
        st.info("No feedback records yet.")
    else:
        feedback_df = pd.read_csv(FEEDBACK_PATH)

        if feedback_df.empty:
            st.info("No feedback records yet.")
        else:
            feedback_df["rating"] = pd.to_numeric(feedback_df["rating"], errors="coerce")
            total_feedback = len(feedback_df)
            average_rating = feedback_df["rating"].mean()
            latest_rating = feedback_df["rating"].iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Feedback", total_feedback)
            col2.metric("Average Rating", f"{average_rating:.2f} / 5")
            col3.metric("Latest Rating", f"{latest_rating:.0f} / 5")

            st.subheader("Saved Feedback")
            st.dataframe(feedback_df.sort_values("timestamp", ascending=False), use_container_width=True)

            intent_summary = (
                feedback_df.groupby("predicted_intent", as_index=False)
                .agg(total_feedback=("rating", "count"), average_rating=("rating", "mean"))
                .sort_values("total_feedback", ascending=False)
            )
            st.subheader("Feedback By Intent")
            st.dataframe(intent_summary, use_container_width=True)

else:
    st.header("About Project")
    st.caption("Project scope, dataset adaptation, and individual model responsibilities.")
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
