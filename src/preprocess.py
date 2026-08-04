from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATASET_PATH = DATA_DIR / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
CLEAN_DATASET_PATH = DATA_DIR / "chatbot_dataset_clean.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"[^a-z0-9\s?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    df = pd.read_csv(RAW_DATASET_PATH)
    df = df[["instruction", "category", "intent", "response"]].dropna()
    df["text"] = df["instruction"].apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()

    clean_df = df[["text", "instruction", "category", "intent", "response"]]
    train_parts = []
    test_parts = []
    for _, group in clean_df.groupby("intent", sort=False):
        shuffled = group.sample(frac=1, random_state=42)
        test_size = max(1, int(len(shuffled) * 0.2))
        test_parts.append(shuffled.iloc[:test_size])
        train_parts.append(shuffled.iloc[test_size:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=42)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=42)

    clean_df.to_csv(CLEAN_DATASET_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Clean dataset saved: {CLEAN_DATASET_PATH} ({len(clean_df)} rows)")
    print(f"Train dataset saved: {TRAIN_PATH} ({len(train_df)} rows)")
    print(f"Test dataset saved: {TEST_PATH} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
