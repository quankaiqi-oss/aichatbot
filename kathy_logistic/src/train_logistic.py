from pathlib import Path
import sys
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared.src.evaluate import evaluate_and_save


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "kathy_logistic" / "models"
SHARED_MODELS_DIR = BASE_DIR / "shared" / "models"
RESULTS_DIR = BASE_DIR / "kathy_logistic" / "results"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    SHARED_MODELS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=12000,
        min_df=2,
    )

    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    y_train = train_df["intent"]
    y_test = test_df["intent"]

    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start_time

    prediction_start = time.perf_counter()
    predictions = model.predict(x_test)
    prediction_seconds = time.perf_counter() - prediction_start

    labels = sorted(y_train.unique())
    metrics = evaluate_and_save("logistic", y_test, predictions, labels, RESULTS_DIR)
    metrics["training_time_seconds"] = train_seconds
    metrics["prediction_time_seconds"] = prediction_seconds
    metrics["average_prediction_time_ms"] = (prediction_seconds / len(test_df)) * 1000

    joblib.dump(model, MODELS_DIR / "logistic_model.pkl")
    joblib.dump(vectorizer, SHARED_MODELS_DIR / "tfidf_vectorizer.pkl")

    import json

    with open(RESULTS_DIR / "logistic_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
