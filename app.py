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
            --primary-dark: #0f4f4a;
            --accent: #c08a2c;
            --ink: #111827;
            --muted: #65758b;
            --panel: #ffffff;
            --soft: #f6f8fb;
            --line: #d8e0ea;
            --navy: #101827;
            --navy-soft: #1f2a3d;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 34rem),
                linear-gradient(180deg, #f4f8f7 0%, #f7f9fc 300px, #ffffff 720px);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        }

        [data-testid="stSidebar"] * {
            color: #e5eef7 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.055);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 10px;
            margin: 0.28rem 0;
            padding: 0.32rem 0.5rem;
            transition: all 120ms ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(15, 118, 110, 0.22);
            border-color: rgba(153, 246, 228, 0.35);
        }

        .app-hero {
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.9));
            border: 1px solid rgba(216, 224, 234, 0.9);
            border-radius: 18px;
            padding: 1.55rem 1.75rem;
            margin-bottom: 0.9rem;
            box-shadow:
                0 24px 60px rgba(15, 23, 42, 0.09),
                inset 0 1px 0 rgba(255, 255, 255, 0.9);
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
            color: var(--accent) !important;
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
            background: rgba(15, 118, 110, 0.08);
            color: #0f4f4a;
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 999px;
            padding: 0.38rem 0.68rem;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .status-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(216, 224, 234, 0.9);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
        }

        .status-card small {
            display: block;
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.22rem;
        }

        .status-card strong {
            color: #10233f;
            font-size: 0.95rem;
        }

        .section-panel {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(216, 224, 234, 0.92);
            border-radius: 16px;
            padding: 1.05rem 1.15rem;
            margin: 0.9rem 0 1rem 0;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.055);
        }

        h2, h3 {
            color: #172033;
            letter-spacing: 0;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(15, 118, 110, 0.8);
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            font-weight: 650;
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.18);
        }

        .stButton > button:hover {
            border-color: var(--primary-dark);
            background: linear-gradient(135deg, var(--primary-dark), #0b3f3b);
            color: white;
        }

        [data-testid="stChatMessage"] {
            border-radius: 12px;
            border: 1px solid #e5edf5;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
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

        .help-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 5px solid var(--accent);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0 1rem 0;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
        }

        .help-panel strong {
            color: #10233f;
        }

        .help-panel p {
            margin: 0.25rem 0;
            color: var(--muted);
        }

        .sample-chip {
            display: inline-block;
            margin: 0.18rem 0.25rem 0.18rem 0;
            padding: 0.32rem 0.58rem;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            background: #f8fafc;
            color: #334155;
            font-size: 0.82rem;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 10px;
            border-color: var(--line);
        }

        @media (max-width: 760px) {
            .app-hero {
                display: block;
            }

            .hero-pills {
                justify-content: flex-start;
                margin-top: 1rem;
            }

            .status-strip {
                grid-template-columns: 1fr 1fr;
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

st.markdown(
    """
    <div class="status-strip">
        <div class="status-card"><small>Prototype Type</small><strong>ML Chatbot</strong></div>
        <div class="status-card"><small>KaiQi Module</small><strong>TF-IDF + Linear SVM</strong></div>
        <div class="status-card"><small>Kathy Module</small><strong>TF-IDF + Logistic Regression</strong></div>
        <div class="status-card"><small>Evaluation</small><strong>Accuracy, F1, Feedback</strong></div>
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
    st.caption("Ask about online-shopping support issues and get an immediate chatbot reply.")
    model_label = st.sidebar.radio(
        "Chatbot model",
        ["KaiQi SVM", "Kathy Logistic Regression"],
    )
    model_type = "svm" if model_label == "KaiQi SVM" else "logistic"
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

    st.markdown(
        """
        <div class="help-panel">
            <strong>How to use</strong>
            <p>Type a customer support question, or choose a common topic from Quick reply suggestions.</p>
            <p>The message is still processed by the trained machine learning model for intent prediction.</p>
            <span class="sample-chip">Where is my parcel?</span>
            <span class="sample-chip">My payment failed</span>
            <span class="sample-chip">I want a refund</span>
            <span class="sample-chip">I forgot my password</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)

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
                with st.expander("Technical prediction details", expanded=False):
                    st.caption(details)

    quick_message = None
    with st.expander("Quick reply suggestions", expanded=True):
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
    if col1.button("Start New Chat"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("last_prediction", None)
        st.rerun()

    if "last_prediction" in st.session_state:
        with st.expander("Rate the latest response"):
            result = st.session_state["last_prediction"]
            with st.form("feedback_form"):
                rating = st.slider("How useful was this response?", 1, 5, 4)
                comment = st.text_input("Optional comment")
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
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.caption("View how well each model predicts customer support intents on the shared test dataset.")
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    selected_label = st.selectbox("Select model", ["KaiQi SVM", "Kathy Logistic Regression"])
    selected_model = "svm" if selected_label == "KaiQi SVM" else "logistic"
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
            st.subheader("Per-Intent Classification Report")
            st.caption("This table shows how well the model performs for each support intent.")
            st.dataframe(pd.read_csv(report_path), use_container_width=True)
    else:
        st.info("Evaluation results are not generated yet.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.caption("Both models use the same dataset, preprocessing, TF-IDF settings, and train-test split for fair comparison.")
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
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
        st.subheader("Overall Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        st.subheader("Metric Chart")
        st.bar_chart(comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]])
    else:
        st.info("Comparison results are not generated yet.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Feedback Records":
    st.header("Feedback Records")
    st.caption("Saved user ratings support the chatbot usability and satisfaction evaluation.")
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)

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
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.header("About Project")
    st.caption("Project scope, dataset adaptation, and individual model responsibilities.")
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)
