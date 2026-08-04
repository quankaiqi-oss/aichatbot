from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]


def evaluate_and_save(model_name: str, y_true, y_pred, labels, results_dir: Path) -> dict:
    results_dir.mkdir(exist_ok=True)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    metrics_path = results_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        results_dir / f"{model_name}_classification_report.csv"
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig_width = max(10, len(labels) * 0.42)
    fig_height = max(8, len(labels) * 0.36)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar=True,
        square=False,
    )
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted Intent")
    plt.ylabel("Actual Intent")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_confusion_matrix.png", dpi=180)
    plt.close()

    return metrics
