from pathlib import Path
from datetime import datetime
import csv
import html
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

NON_SUPPORT_INTENTS = {
    "greeting",
    "thanks",
    "goodbye",
    "capability",
    "positive_confirmation",
    "negative_confirmation",
    "clarify",
    "fallback",
}

st.set_page_config(
    page_title="ShopCare MY",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --primary: #7f8f83;
            --primary-dark: #4f5f58;
            --accent: #9b8366;
            --ink: #24211d;
            --muted: #746f66;
            --panel: #fffdf8;
            --soft: #f6f3ed;
            --line: #ded8ca;
            --charcoal: #2b2925;
            --paper: #fbf7ef;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(43, 41, 37, 0.018) 1px, transparent 1px),
                linear-gradient(180deg, #f7f3eb 0%, #fbfaf6 340px, #ffffff 820px);
            background-size: 28px 28px, auto;
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1480px;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ebe4d7 0%, #f7f3eb 100%);
            border-right: 1px solid #d8d0c0;
        }

        [data-testid="stSidebar"] * {
            color: #2d2924 !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2.2rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            font-size: 0.9rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            background: transparent !important;
            border: 0 !important;
            border-left: 3px solid transparent !important;
            border-radius: 0 !important;
            margin: 0.15rem 0;
            padding: 0.48rem 0.35rem 0.48rem 0.72rem;
            transition: background 140ms ease, border-color 140ms ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(255, 253, 248, 0.58) !important;
            border-left-color: #4f5f58 !important;
            box-shadow: none;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
            color: #2b3e37 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
            font-weight: 700;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 253, 248, 0.42) !important;
            border-left-color: rgba(79, 95, 88, 0.34) !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            transform: scale(0.72);
            opacity: 0.58;
            margin-right: 0.12rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) > div:first-child {
            opacity: 0.86;
        }

        .app-hero {
            background:
                linear-gradient(135deg, rgba(255, 253, 248, 0.96), rgba(242, 238, 228, 0.9));
            border: 1px solid rgba(184, 174, 157, 0.55);
            border-radius: 8px;
            padding: 0.72rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow:
                0 6px 16px rgba(43, 41, 37, 0.045),
                inset 0 1px 0 rgba(255, 255, 255, 0.75);
        }

        .app-hero h1 {
            margin: 0;
            color: #10233f;
            font-size: 1.38rem;
            line-height: 1.1;
            letter-spacing: 0;
        }

        .app-hero p {
            margin: 0.22rem 0 0 0;
            color: var(--muted);
            font-size: 0.84rem;
        }

        .eyebrow {
            margin-bottom: 0.18rem !important;
            color: #87633a !important;
            font-size: 0.68rem !important;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .status-card {
            background: rgba(255, 253, 248, 0.88);
            border: 1px solid rgba(184, 174, 157, 0.48);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 8px 18px rgba(43, 41, 37, 0.045);
        }

        .status-card small {
            display: block;
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.22rem;
        }

        .status-card strong {
            color: #2d2924;
            font-size: 0.95rem;
        }

        h2, h3 {
            color: #2d2924;
            letter-spacing: 0;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 253, 248, 0.9);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 9px 20px rgba(43, 41, 37, 0.05);
        }

        .stButton > button {
            border-radius: 6px;
            border: 1px solid #cfc6b8;
            background: #f5f1e8;
            color: #2d2924;
            font-weight: 600;
            box-shadow: none;
            min-height: 2.35rem;
            padding: 0.38rem 0.72rem;
        }

        .stButton > button:hover {
            border-color: #9f9484;
            background: #ebe4d7;
            color: #2d2924;
        }

        .chat-row {
            display: flex;
            gap: 0.65rem;
            align-items: flex-start;
            margin: 0.85rem 0;
            width: 100%;
        }

        .chat-row.user {
            justify-content: flex-end;
        }

        .chat-row.assistant {
            justify-content: flex-start;
        }

        .chat-avatar {
            width: 2rem;
            height: 2rem;
            min-width: 2rem;
            border-radius: 8px;
            background: #e6dfd1;
            border: 1px solid #d2c8b8;
            color: #4f5f58;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 700;
            line-height: 1;
        }

        .chat-bubble {
            display: inline-block;
            width: fit-content;
            max-width: min(760px, 72%);
            padding: 0.72rem 0.9rem;
            border-radius: 10px;
            border: 1px solid #ded8ca;
            line-height: 1.55;
            box-shadow: 0 5px 14px rgba(43, 41, 37, 0.035);
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }

        .chat-row.assistant .chat-bubble {
            background: #fffdf8;
        }

        .chat-row.user .chat-bubble {
            background: #eee8dc;
            border-color: #d6ccbc;
            max-width: min(620px, 58%);
        }

        .quick-reply-title {
            color: var(--muted);
            font-size: 0.78rem;
            margin: 0.1rem 0 0.35rem 2.65rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 253, 248, 0.72);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 7px;
            border-color: #d6ccbc;
            background-color: #fbf8f1;
        }

        .stSelectbox div[data-baseweb="select"] * {
            color: #2d2924;
        }

        [data-testid="stChatInput"] textarea {
            background-color: #fbf8f1 !important;
            border: none !important;
            color: #2d2924 !important;
            min-height: 52px !important;
            padding: 0.9rem 1rem !important;
            font-size: 0.98rem !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] > div {
            background: #fbf8f1 !important;
            border: 1px solid #d6ccbc !important;
            border-radius: 10px !important;
            min-height: 66px !important;
            padding: 0.35rem 0.45rem !important;
            box-shadow: 0 8px 18px rgba(43, 41, 37, 0.055) !important;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: #a89d8d !important;
            box-shadow: 0 10px 22px rgba(43, 41, 37, 0.08) !important;
        }

        [data-testid="stChatInput"] button {
            background-color: #e6dfd1 !important;
            color: #4f5f58 !important;
            border: 1px solid #d2c8b8 !important;
            border-radius: 8px !important;
            min-width: 44px !important;
            min-height: 44px !important;
        }

        @media (max-width: 760px) {
            .app-hero {
                display: block;
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
        "Delivery Delay": "My parcel is late and has not arrived",
        "Damaged Item": "My item arrived damaged",
        "Wrong Item": "I received the wrong item",
        "Cancel Order": "I want to cancel my order",
        "Change Address": "I need to change my shipping address",
        "Voucher Issue": "My voucher cannot be used",
        "Account Help": "I forgot my password and cannot login",
        "Contact Support": "I want to contact customer support",
        "Complaint": "I want to make a complaint",
    }

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am ShopCare MY. Tell me what happened with your order, "
                    "payment, refund, delivery, or account."
                ),
            }
        ]

    for message in st.session_state["chat_history"]:
        role = message["role"]
        avatar = "ME" if role == "user" else "AI"
        safe_content = html.escape(message["content"])
        if role == "user":
            message_html = f"""
            <div class="chat-row user">
                <div class="chat-bubble">{safe_content}</div>
                <div class="chat-avatar">{avatar}</div>
            </div>
            """
        else:
            message_html = f"""
            <div class="chat-row assistant">
                <div class="chat-avatar">{avatar}</div>
                <div class="chat-bubble">{safe_content}</div>
            </div>
            """
        st.markdown(message_html, unsafe_allow_html=True)

    quick_message = None
    show_quick_replies = len(st.session_state["chat_history"]) == 1
    if show_quick_replies:
        st.caption("Quick topics")
        quick_reply_rows = [st.columns(4) for _ in range((len(quick_replies) + 3) // 4)]
        for index, (label, message) in enumerate(quick_replies.items()):
            row = quick_reply_rows[index // 4]
            if row[index % 4].button(label, use_container_width=True):
                quick_message = message

    suggested_message = None
    if "last_prediction" in st.session_state:
        suggestions = st.session_state["last_prediction"].get("follow_up_options", [])
        if suggestions:
            st.caption("Continue with")
            suggestion_cols = st.columns([1, 1, 1, 4][: len(suggestions)])
            for index, suggestion in enumerate(suggestions[:4]):
                if suggestion_cols[index].button(suggestion, key=f"suggestion_{index}"):
                    suggested_message = suggestion

    user_message = st.chat_input("Type your customer support question...")
    submitted_message = quick_message or suggested_message or user_message

    if submitted_message:
        st.session_state["chat_history"].append(
            {"role": "user", "content": submitted_message}
        )

        try:
            previous_intent = st.session_state.get("last_support_intent")
            result = predict_intent(
                submitted_message,
                model_type=model_type,
                previous_intent=previous_intent,
            )
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
                "follow_up_options": result.get("follow_up_options", []),
            }
            st.session_state["chat_history"].append(assistant_message)
            st.session_state["last_prediction"] = assistant_message
            if result["intent"] not in NON_SUPPORT_INTENTS:
                st.session_state["last_support_intent"] = result["intent"]
            st.rerun()
        except FileNotFoundError:
            st.error("Model files are missing. Please run preprocessing and training first.")

    col1, col2 = st.columns([1, 4])
    if col1.button("Start New Chat"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_support_intent", None)
        st.rerun()

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        with st.expander("Model details for evaluation"):
            confidence = result.get("confidence")
            confidence_text = "N/A" if confidence is None else f"{confidence:.4f}"
            detail_cols = st.columns(4)
            detail_cols[0].metric("Selected Model", result["model"].upper())
            detail_cols[1].metric("Final Intent", result["intent"])
            detail_cols[2].metric("Model Intent", result.get("model_intent") or "N/A")
            detail_cols[3].metric("Confidence", confidence_text)
            if result.get("used_fallback"):
                st.warning(result.get("fallback_reason") or "Fallback or clarification was used.")

        with st.expander("Rate the latest response"):
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

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.caption("View how well each model predicts customer support intents on the shared test dataset.")
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

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.caption("Both models use the same dataset, preprocessing, TF-IDF settings, and train-test split for fair comparison.")
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
