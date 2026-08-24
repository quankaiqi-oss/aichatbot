from pathlib import Path
from datetime import datetime
from io import BytesIO
import csv
import html
import json
import textwrap

import altair as alt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

MODEL_DISPLAY_NAMES = {
    "svm": "KaiQi SVM",
    "logistic": "Kathy Logistic Regression",
}

CHART_COLORS = ["#5C4444", "#b4cfcb", "#EDE7D5", "#8f7676"]

QUICK_SUPPORT_TOPICS = {
    "Track Order": "I want to track my order",
    "Parcel Late": "My parcel is late and has not arrived",
    "Request Refund": "I want to request a refund",
    "Payment Deducted": "My payment was deducted but no order was created",
    "Damaged Item": "My item arrived damaged",
    "Wrong Item": "I received the wrong item",
    "Cancel Order": "I want to cancel my order",
    "Change Address": "I need to change my shipping address",
    "Voucher Issue": "My voucher cannot be used",
    "Cannot Login": "I forgot my password and cannot login",
    "Seller No Reply": "The seller did not reply to my message",
    "Complaint": "I want to make a complaint",
}

QUICK_SUPPORT_EXAMPLES = [
    "barang belum sampai",
    "duit kena deduct",
    "seller tak reply",
    "nak refund",
]

CHAT_CLOSING_PHRASES = {
    "no",
    "no thanks",
    "no thank you",
    "no thankyou",
    "no tq",
    "nothing",
    "nothing else",
    "thats all",
    "that's all",
    "done",
    "thank you",
    "thanks",
    "tq",
    "bye",
    "goodbye",
}

IDLE_PROMPT_SECONDS = 10


def is_closing_reply(message: str) -> bool:
    cleaned = message.lower().strip()
    cleaned = "".join(character for character in cleaned if character.isalnum() or character.isspace() or character == "'")
    cleaned = " ".join(cleaned.split())
    return cleaned in CHAT_CLOSING_PHRASES


def get_latest_support_exchange(chat_history: list[dict]) -> dict | None:
    for message in reversed(chat_history):
        if message.get("role") == "assistant" and message.get("intent") and message.get("user_message"):
            if message["intent"] not in NON_SUPPORT_INTENTS:
                return message
    return None


def build_chat_summary(chat_history: list[dict]) -> dict:
    latest_exchange = get_latest_support_exchange(chat_history)
    if latest_exchange is None:
        return {
            "main_issue": "General customer support enquiry",
            "user_problem": "The user interacted with ShopCare MY.",
            "suggested_action": "Ask a specific question about order, refund, payment, delivery, account, voucher, or support.",
            "model": "N/A",
            "confidence": "N/A",
        }

    confidence = latest_exchange.get("confidence")
    confidence_text = "N/A" if confidence is None else f"{confidence:.4f}"
    response = latest_exchange.get("content", "")
    first_action = response.split("\n")[0].strip()

    return {
        "main_issue": latest_exchange.get("intent", "N/A"),
        "user_problem": latest_exchange.get("user_message", ""),
        "suggested_action": first_action,
        "model": latest_exchange.get("model", "N/A").upper(),
        "confidence": confidence_text,
    }


def format_chat_summary(summary: dict) -> str:
    return (
        "Here is a short summary of your support chat:\n\n"
        f"- Main issue: {summary['main_issue']}\n"
        f"- User problem: {summary['user_problem']}\n"
        f"- Suggested action: {summary['suggested_action']}\n"
        f"- Model used: {summary['model']}\n"
        f"- Confidence: {summary['confidence']}\n\n"
        "You may download this summary as a PDF for reference."
    )


def build_summary_pdf(summary: dict) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        y_position = 0.93

        fig.text(0.08, y_position, "ShopCare MY Chat Summary", fontsize=18, weight="bold", color="#5C4444")
        y_position -= 0.04
        fig.text(0.08, y_position, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        y_position -= 0.07

        fields = [
            ("Main issue", summary["main_issue"]),
            ("User problem", summary["user_problem"]),
            ("Suggested action", summary["suggested_action"]),
            ("Model used", summary["model"]),
            ("Confidence", summary["confidence"]),
        ]
        for label, value in fields:
            fig.text(0.08, y_position, f"{label}:", fontsize=12, weight="bold", color="#5C4444")
            y_position -= 0.026
            for line in textwrap.wrap(str(value), width=86):
                fig.text(0.1, y_position, line, fontsize=11, color="#222222")
                y_position -= 0.025
            y_position -= 0.018

        fig.text(
            0.08,
            0.08,
            "This summary is generated by the ShopCare MY chatbot to help users keep a simple record of the support guidance.",
            fontsize=9,
            color="#756464",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def load_metrics_table() -> pd.DataFrame:
    rows = []
    for model_name, display_name in MODEL_DISPLAY_NAMES.items():
        metrics_path = MODEL_RESULT_DIRS[model_name] / f"{model_name}_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "Model": display_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1_score"],
                    "Training Time (s)": metrics.get("training_time_seconds", 0),
                    "Avg Prediction Time (ms)": metrics.get("average_prediction_time_ms", 0),
                }
            )
    return pd.DataFrame(rows)


def interactive_metric_chart(chart_df: pd.DataFrame, title: str):
    long_df = chart_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    bars = (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(
                "Metric:N",
                title=None,
                sort=["Accuracy", "Precision", "Recall", "F1 Score"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Score:Q",
                scale=alt.Scale(domain=[0, 1.05]),
                title="Score",
                axis=alt.Axis(grid=True, gridColor="#EDE7D5"),
            ),
            xOffset=alt.XOffset("Model:N"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(range=["#5C4444", "#b4cfcb", "#8f7676"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Score:Q", format=".4f"),
            ],
        )
        .properties(height=320, title=alt.TitleParams(text=title, color="#5C4444", fontSize=15))
    )
    return bars.configure_axis(
        labelColor="#5C4444",
        titleColor="#5C4444",
        gridColor="#EDE7D5",
    ).configure_view(stroke=None)


def interactive_single_metric_chart(summary_df: pd.DataFrame, title: str):
    sorted_df = summary_df.sort_values("Score", ascending=False)
    bars = (
        alt.Chart(sorted_df)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, size=58)
        .encode(
            x=alt.X("Metric:N", title=None, sort=list(sorted_df["Metric"])),
            y=alt.Y(
                "Score:Q",
                scale=alt.Scale(domain=[0, 1.05]),
                title="Score",
                axis=alt.Axis(grid=True, gridColor="#EDE7D5", labelColor="#5C4444", titleColor="#5C4444"),
            ),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=["#5C4444", "#b4cfcb", "#EDE7D5", "#8f7676"]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Score:Q", format=".4f")],
        )
        .properties(height=320, title=alt.TitleParams(text=title, color="#5C4444", fontSize=15))
    )
    labels = (
        alt.Chart(sorted_df)
        .mark_text(align="center", baseline="bottom", dy=-6, color="#5C4444", fontSize=12)
        .encode(
            x=alt.X("Metric:N", title=None, sort=list(sorted_df["Metric"])),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1.05])),
            text=alt.Text("Score:Q", format=".4f"),
        )
    )
    return (bars + labels).configure_axis(
        labelColor="#5C4444",
        titleColor="#5C4444",
        gridColor="#EDE7D5",
    ).configure_view(stroke=None)


def interactive_speed_chart(speed_df: pd.DataFrame):
    chart_df = speed_df.sort_values("Avg Prediction Time (ms)", ascending=True)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=28)
        .encode(
            y=alt.Y("Model:N", sort=list(chart_df["Model"]), title=None, axis=alt.Axis(labelLimit=190)),
            x=alt.X(
                "Avg Prediction Time (ms):Q",
                title="Average Prediction Time (ms)",
                axis=alt.Axis(grid=True, gridColor="#EDE7D5"),
            ),
            color=alt.Color("Model:N", scale=alt.Scale(range=["#5C4444", "#b4cfcb"]), legend=None),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Avg Prediction Time (ms):Q", format=".6f"),
                alt.Tooltip("Training Time (s):Q", format=".4f"),
            ],
        )
        .properties(height=230, title=alt.TitleParams(text="Prediction Speed", color="#5C4444", fontSize=15))
        .configure_axis(labelColor="#5C4444", titleColor="#5C4444")
        .configure_view(stroke=None)
    )
    return chart

st.set_page_config(
    page_title="ShopCare MY",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --primary: #5C4444;
            --primary-dark: #463333;
            --accent: #b4cfcb;
            --ink: #5C4444;
            --muted: #756464;
            --panel: #ffffff;
            --soft: #f7f3e8;
            --line: #d8d2c3;
            --charcoal: #5C4444;
            --paper: #EDE7D5;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(92, 68, 68, 0.045) 1px, transparent 1px),
                linear-gradient(180deg, #EDE7D5 0%, #f8f4e8 420px, #ffffff 100%);
            background-size: 32px 32px, auto;
            color: var(--ink);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1380px;
            padding-left: 4rem;
            padding-right: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #5C4444;
            border-right: 1px solid #463333;
        }

        [data-testid="stSidebar"] * {
            color: #ffffff !important;
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
            background: rgba(180, 207, 203, 0.24) !important;
            border-left-color: #b4cfcb !important;
            box-shadow: none;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
            font-weight: 700;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(180, 207, 203, 0.16) !important;
            border-left-color: rgba(180, 207, 203, 0.7) !important;
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
                linear-gradient(135deg, #5C4444 0%, #5C4444 68%, #6d5353 100%);
            border: 1px solid rgba(70, 51, 51, 0.95);
            border-left: 7px solid #b4cfcb;
            border-radius: 10px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1.65rem;
            max-width: 1040px;
            box-shadow:
                0 10px 24px rgba(92, 68, 68, 0.12),
                inset 0 1px 0 rgba(255, 255, 255, 0.12);
        }

        .app-hero h1 {
            margin: 0;
            color: #ffffff;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 1.62rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: 0;
        }

        .app-hero p {
            margin: 0.42rem 0 0 0;
            color: rgba(255, 255, 255, 0.84);
            font-size: 0.9rem;
        }

        .eyebrow {
            margin-bottom: 0.34rem !important;
            color: #b4cfcb !important;
            font-size: 0.7rem !important;
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
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(216, 210, 195, 0.95);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 8px 18px rgba(92, 68, 68, 0.04);
        }

        .status-card small {
            display: block;
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.22rem;
        }

        .status-card strong {
            color: #5C4444;
            font-size: 0.95rem;
        }

        h2, h3 {
            color: #5C4444;
            letter-spacing: 0;
        }

        h1 {
            color: #5C4444 !important;
            font-size: 1.9rem !important;
            line-height: 1.18 !important;
            margin-bottom: 0.8rem !important;
        }

        .app-hero h1 {
            color: #ffffff !important;
            font-size: 1.62rem !important;
            margin-bottom: 0 !important;
        }

        h2 {
            font-size: 1.28rem !important;
            line-height: 1.25 !important;
        }

        h3 {
            font-size: 1.02rem !important;
            line-height: 1.25 !important;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.82rem 0.95rem;
            box-shadow: 0 9px 20px rgba(92, 68, 68, 0.04);
        }

        [data-testid="stMetricLabel"] {
            color: #5C4444;
            font-size: 0.82rem;
        }

        [data-testid="stMetricValue"] {
            color: #172033;
            font-size: 1.55rem;
            line-height: 1.15;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.78rem;
        }

        .stButton > button {
            border-radius: 6px;
            border: 1px solid #d8d2c3;
            background: #ffffff;
            color: #5C4444;
            font-weight: 600;
            box-shadow: none;
            min-height: 2.35rem;
            padding: 0.38rem 0.72rem;
        }

        .stButton > button:hover {
            border-color: #b4cfcb;
            background: #f5f8f4;
            color: #5C4444;
        }

        .chat-row {
            display: flex;
            gap: 0.65rem;
            align-items: flex-start;
            margin: 1.15rem 0;
            width: min(1180px, 100%);
            margin-left: auto;
            margin-right: auto;
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
            background: #b4cfcb;
            border: 1px solid #a4c3be;
            color: #000000;
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
            max-width: min(680px, 68%);
            padding: 0.95rem 1.08rem;
            border-radius: 10px;
            border: 1px solid #d8d2c3;
            line-height: 1.58;
            box-shadow: 0 6px 16px rgba(92, 68, 68, 0.032);
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }

        .chat-message-stack {
            display: flex;
            flex-direction: column;
            gap: 0.28rem;
            max-width: min(720px, 72%);
        }

        .chat-meta {
            color: #8d817b;
            font-size: 0.74rem;
            line-height: 1.25;
            padding-left: 0.18rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .chat-row.assistant .chat-bubble {
            background: #ffffff;
            border-color: #d8d2c3;
        }

        .chat-row.user .chat-bubble {
            background: #b4cfcb;
            border-color: #a4c3be;
            color: #000000;
            max-width: min(520px, 48%);
        }

        .chat-control-area {
            width: min(1180px, 100%);
            margin: 0.5rem auto 0 auto;
        }

        .chat-control-area .stButton > button {
            min-width: 10.5rem;
        }

        .chat-followups {
            width: min(1180px, 100%);
            margin: 0.2rem auto 0 auto;
            padding-left: 2.65rem;
        }

        .chat-followups .stButton > button {
            min-width: 10.5rem;
        }

        .quick-reply-title {
            color: var(--muted);
            font-size: 0.78rem;
            margin: 0.1rem 0 0.35rem 2.65rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 7px;
            background: rgba(255, 255, 255, 0.88);
            max-width: 1180px;
            margin-left: auto;
            margin-right: auto;
        }

        div[data-testid="stExpander"] details > summary {
            min-height: 2.25rem;
            padding-top: 0.35rem !important;
            padding-bottom: 0.35rem !important;
            font-size: 0.88rem;
        }

        .model-detail-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.45rem 1rem;
            max-width: 620px;
            padding: 0.15rem 0 0.35rem 0;
        }

        .model-detail-item {
            display: grid;
            grid-template-columns: 8.5rem minmax(0, 1fr);
            gap: 0.65rem;
            align-items: baseline;
            font-size: 0.86rem;
        }

        .model-detail-item span {
            color: var(--muted);
        }

        .model-detail-item strong {
            color: var(--ink);
            font-weight: 650;
            overflow-wrap: anywhere;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1.1rem 0 1.35rem 0;
        }

        .insight-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #d8d2c3;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 18px rgba(92, 68, 68, 0.04);
        }

        .insight-card small {
            display: block;
            color: var(--muted);
            font-size: 0.74rem;
            margin-bottom: 0.28rem;
        }

        .insight-card strong {
            display: block;
            color: var(--ink);
            font-size: 1.05rem;
            line-height: 1.25;
        }

        .insight-card span {
            color: var(--muted);
            font-size: 0.78rem;
        }

        .clean-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.84rem;
            overflow: hidden;
            border-radius: 9px;
        }

        .clean-table th {
            background: #5C4444;
            color: #ffffff;
            font-weight: 650;
            text-align: left;
            padding: 0.68rem 0.72rem;
            white-space: nowrap;
        }

        .clean-table td {
            background: rgba(255, 255, 255, 0.94);
            border-bottom: 1px solid #e8e0cf;
            color: var(--ink);
            padding: 0.64rem 0.72rem;
            white-space: nowrap;
        }

        .clean-table tr:last-child td {
            border-bottom: none;
        }

        .soft-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.6rem;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 7px;
            border-color: #d8d2c3;
            background-color: #ffffff;
        }

        .stSelectbox {
            max-width: 720px;
        }

        .stSelectbox div[data-baseweb="select"] * {
            color: #5C4444;
        }

        [data-testid="stChatInput"] textarea {
            background-color: #ffffff !important;
            border: none !important;
            color: #5C4444 !important;
            min-height: 52px !important;
            padding: 0.9rem 1rem !important;
            font-size: 0.98rem !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] {
            max-width: 1180px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] > div {
            background: #ffffff !important;
            border: 1px solid #d8d2c3 !important;
            border-radius: 10px !important;
            min-height: 66px !important;
            padding: 0.35rem 0.45rem !important;
            box-shadow: 0 8px 20px rgba(92, 68, 68, 0.045) !important;
            width: 100% !important;
            max-width: 1180px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: #b4cfcb !important;
            box-shadow: 0 10px 24px rgba(92, 68, 68, 0.08) !important;
        }

        [data-testid="stChatInput"] button {
            background-color: #b4cfcb !important;
            color: #5C4444 !important;
            border: 1px solid #a4c3be !important;
            border-radius: 8px !important;
            min-width: 44px !important;
            min-height: 44px !important;
        }

        @media (max-width: 760px) {
            .app-hero {
                display: block;
            }

            .chat-row,
            .chat-control-area,
            .chat-followups,
            div[data-testid="stExpander"],
            [data-testid="stChatInput"],
            [data-testid="stChatInput"] > div {
                width: 100% !important;
                max-width: 100% !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
            }

            .status-strip {
                grid-template-columns: 1fr 1fr;
            }

            .insight-grid {
                grid-template-columns: 1fr;
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
            <p class="eyebrow">Customer Support Assistant</p>
            <h1>ShopCare MY</h1>
            <p>Online shopping help for orders, refunds, payments, delivery, and account issues.</p>
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
    st.caption("Ask about online-shopping support issues and get an immediate chatbot reply.")
    model_label = st.sidebar.radio(
        "Chatbot model",
        ["KaiQi SVM", "Kathy Logistic Regression"],
    )
    model_type = "svm" if model_label == "KaiQi SVM" else "logistic"

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am ShopCare MY. I can assist with online shopping support issues.\n"
                    "You may ask about:\n"
                    "- Order tracking or delivery delay\n"
                    "- Refunds, returns, or damaged items\n"
                    "- Payment, account, voucher, or complaint issues\n"
                    "What issue would you like help with?"
                ),
            }
        ]

    if st.session_state.get("pending_idle_prompt") and not st.session_state.get("idle_prompt_sent"):
        elapsed_seconds = datetime.now().timestamp() - st.session_state.get("idle_prompt_started_at", 0)
        if elapsed_seconds >= IDLE_PROMPT_SECONDS:
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": (
                        "Do you need help with anything else? "
                        "If not, you can reply no, no thanks, done, or bye and I will prepare a short chat summary."
                    ),
                }
            )
            st.session_state["pending_idle_prompt"] = False
            st.session_state["idle_prompt_sent"] = True
            st.session_state["waiting_for_more_help"] = True
        else:
            remaining_ms = max(1000, int((IDLE_PROMPT_SECONDS - elapsed_seconds) * 1000) + 250)
            components.html(
                f"""
                <script>
                    setTimeout(function() {{
                        window.parent.location.reload();
                    }}, {remaining_ms});
                </script>
                """,
                height=0,
            )

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
            if message.get("model") and message.get("intent"):
                confidence = message.get("confidence")
                confidence_text = "N/A" if confidence is None else f"{confidence:.4f}"
                model_intent = message.get("model_intent") or "N/A"
                meta_text = (
                    f"Model: {message['model'].upper()} | "
                    f"Final intent: {message['intent']} | "
                    f"Model intent: {model_intent} | "
                    f"Confidence: {confidence_text}"
                )
                if message.get("used_fallback") and message.get("fallback_reason"):
                    meta_text += f" | {message['fallback_reason']}"
                meta_html = f'<div class="chat-meta">{html.escape(meta_text)}</div>'
                message_html = f"""
                <div class="chat-row assistant">
                    <div class="chat-avatar">{avatar}</div>
                    <div class="chat-message-stack">
                        <div class="chat-bubble">{safe_content}</div>
                        {meta_html}
                    </div>
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
        st.markdown("#### Quick Support Topics")
        st.caption(
            "Click a common online-shopping issue to start the conversation. "
            f"Local examples supported: {', '.join(QUICK_SUPPORT_EXAMPLES)}."
        )
        quick_reply_rows = [st.columns(4) for _ in range((len(QUICK_SUPPORT_TOPICS) + 3) // 4)]
        for index, (label, message) in enumerate(QUICK_SUPPORT_TOPICS.items()):
            row = quick_reply_rows[index // 4]
            if row[index % 4].button(label, use_container_width=True):
                quick_message = message

    suggested_message = None
    if "last_prediction" in st.session_state and not st.session_state.get("waiting_for_more_help"):
        suggestions = st.session_state["last_prediction"].get("follow_up_options", [])
        if suggestions:
            st.markdown('<div class="chat-followups">', unsafe_allow_html=True)
            followup_left, followup_right = st.columns([1, 5])
            with followup_left:
                st.caption("Follow-up Suggestions")
                for index, suggestion in enumerate(suggestions[:4]):
                    if st.button(suggestion, key=f"suggestion_{index}"):
                        suggested_message = suggestion
            st.markdown("</div>", unsafe_allow_html=True)

    user_message = st.chat_input("Type your customer support question...")
    submitted_message = quick_message or suggested_message or user_message

    if submitted_message:
        st.session_state["chat_history"].append(
            {"role": "user", "content": submitted_message}
        )

        if st.session_state.get("waiting_for_more_help") and is_closing_reply(submitted_message):
            summary = build_chat_summary(st.session_state["chat_history"])
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": format_chat_summary(summary),
                }
            )
            st.session_state["chat_summary_pdf"] = build_summary_pdf(summary)
            st.session_state["chat_summary_filename"] = (
                f"shopcare_chat_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            st.session_state["waiting_for_more_help"] = False
            st.session_state["pending_idle_prompt"] = False
            st.session_state["idle_prompt_sent"] = False
            st.rerun()

        try:
            st.session_state["waiting_for_more_help"] = False
            st.session_state["pending_idle_prompt"] = False
            st.session_state["idle_prompt_sent"] = False

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
                st.session_state["pending_idle_prompt"] = True
                st.session_state["idle_prompt_sent"] = False
                st.session_state["idle_prompt_started_at"] = datetime.now().timestamp()
            st.rerun()
        except FileNotFoundError:
            st.error("Model files are missing. Please run preprocessing and training first.")

    st.markdown('<div class="chat-followups">', unsafe_allow_html=True)
    start_left, start_right = st.columns([1, 5])
    with start_left:
        if st.button("Start New Chat"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("last_prediction", None)
            st.session_state.pop("last_support_intent", None)
            st.session_state.pop("pending_idle_prompt", None)
            st.session_state.pop("idle_prompt_sent", None)
            st.session_state.pop("idle_prompt_started_at", None)
            st.session_state.pop("waiting_for_more_help", None)
            st.session_state.pop("chat_summary_pdf", None)
            st.session_state.pop("chat_summary_filename", None)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("chat_summary_pdf"):
        st.download_button(
            "Download Chat Summary PDF",
            data=st.session_state["chat_summary_pdf"],
            file_name=st.session_state.get("chat_summary_filename", "shopcare_chat_summary.pdf"),
            mime="application/pdf",
        )

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
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
        summary_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Score": [
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1_score"],
                ],
            }
        )
        sorted_summary_df = summary_df.sort_values("Score", ascending=False)
        speed_df = pd.DataFrame(
            {
                "Metric": ["Training Time (s)", "Prediction Time (s)", "Avg Prediction Time (ms)"],
                "Value": [
                    metrics.get("training_time_seconds", 0),
                    metrics.get("prediction_time_seconds", 0),
                    metrics.get("average_prediction_time_ms", 0),
                ],
            }
        )

        top1, top2, top3 = st.columns(3)
        top1.metric("Accuracy", f"{metrics['accuracy']:.4f}", selected_label)
        top2.metric("F1 Score", f"{metrics['f1_score']:.4f}", "Intent balance")
        top3.metric("Avg Prediction", f"{metrics.get('average_prediction_time_ms', 0):.6f} ms", "Per message")

        st.divider()

        with st.container(border=True):
            st.subheader("Performance Overview")
            st.caption("Hover over a bar to inspect the exact score.")
            st.altair_chart(
                interactive_single_metric_chart(summary_df, f"{selected_label} Metrics"),
                use_container_width=True,
            )

        table_tab, speed_tab = st.tabs(["Intent Report", "Metric & Speed Tables"])

        if report_path.exists():
            with table_tab:
                report_df = pd.read_csv(report_path)
                report_df = report_df.rename(columns={report_df.columns[0]: "Intent"})
                if "f1-score" in report_df.columns:
                    report_df = report_df.sort_values("f1-score", ascending=False)
                st.caption("Sortable table. Default view is sorted by F1 score.")
                st.dataframe(
                    report_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "precision": st.column_config.NumberColumn(format="%.4f"),
                        "recall": st.column_config.NumberColumn(format="%.4f"),
                        "f1-score": st.column_config.NumberColumn(format="%.4f"),
                        "support": st.column_config.NumberColumn(format="%.0f"),
                    },
                )

        with speed_tab:
            metric_col, speed_col = st.columns(2)
            with metric_col:
                st.caption("Metrics sorted from highest score to lowest score.")
                st.dataframe(
                    sorted_summary_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Score": st.column_config.NumberColumn(format="%.4f")},
                )
            with speed_col:
                st.caption("Speed values sorted from fastest to slowest.")
                st.dataframe(
                    speed_df.sort_values("Value", ascending=True),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Value": st.column_config.NumberColumn(format="%.6f")},
                )
    else:
        st.info("Evaluation results are not generated yet.")

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.caption("Both models use the same dataset, preprocessing, TF-IDF settings, and train-test split for fair comparison.")

    comparison_df = load_metrics_table()
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("F1 Score", ascending=False)
        best_f1 = comparison_df.iloc[0]
        fastest = comparison_df.sort_values("Avg Prediction Time (ms)", ascending=True).iloc[0]

        top1, top2, top3 = st.columns(3)
        top1.metric("Best Overall", best_f1["Model"], f"F1 {best_f1['F1 Score']:.4f}")
        top2.metric("Highest Accuracy", best_f1["Model"], f"{best_f1['Accuracy']:.4f}")
        top3.metric("Fastest Prediction", fastest["Model"], f"{fastest['Avg Prediction Time (ms)']:.6f} ms")

        st.divider()

        chart_tab, table_tab, speed_tab = st.tabs(["Classification Chart", "Comparison Table", "Speed View"])

        with chart_tab:
            with st.container(border=True):
                st.subheader("Classification Comparison")
                st.caption("Hover over a bar to inspect the exact score.")
                metric_chart_df = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
                st.altair_chart(
                    interactive_metric_chart(metric_chart_df, "Classification Metrics"),
                    use_container_width=True,
                )

        with table_tab:
            st.caption("Sortable table. Default view is sorted by F1 Score.")
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                    "Precision": st.column_config.NumberColumn(format="%.4f"),
                    "Recall": st.column_config.NumberColumn(format="%.4f"),
                    "F1 Score": st.column_config.NumberColumn(format="%.4f"),
                    "Training Time (s)": st.column_config.NumberColumn(format="%.4f"),
                    "Avg Prediction Time (ms)": st.column_config.NumberColumn(format="%.6f"),
                },
            )
            st.caption(
                f"Best overall model: {best_f1['Model']} "
                f"(F1 Score {best_f1['F1 Score']:.4f}, Accuracy {best_f1['Accuracy']:.4f})."
            )

        with speed_tab:
            speed_table = comparison_df[["Model", "Training Time (s)", "Avg Prediction Time (ms)"]].sort_values(
                "Avg Prediction Time (ms)",
                ascending=True,
            )
            st.caption("Sortable table. Default view is sorted by prediction speed.")
            st.dataframe(
                speed_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Training Time (s)": st.column_config.NumberColumn(format="%.4f"),
                    "Avg Prediction Time (ms)": st.column_config.NumberColumn(format="%.6f"),
                },
            )
            st.caption("Hover over a bar to inspect the exact timing.")
            st.altair_chart(interactive_speed_chart(comparison_df), use_container_width=True)
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
