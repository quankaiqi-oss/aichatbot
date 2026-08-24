from pathlib import Path
from collections import Counter
import json
import math
import re

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


def tokenize_response(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def modified_precision(reference_tokens: list[str], candidate_tokens: list[str], n: int) -> float:
    if len(candidate_tokens) < n:
        return 0.0
    reference_counts = Counter(tuple(reference_tokens[index : index + n]) for index in range(len(reference_tokens) - n + 1))
    candidate_counts = Counter(tuple(candidate_tokens[index : index + n]) for index in range(len(candidate_tokens) - n + 1))
    overlap = sum(min(count, reference_counts[ngram]) for ngram, count in candidate_counts.items())
    total = sum(candidate_counts.values())
    return (overlap + 1) / (total + 1)


def sentence_bleu(reference: str, candidate: str, max_n: int) -> float:
    reference_tokens = tokenize_response(reference)
    candidate_tokens = tokenize_response(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    precisions = [modified_precision(reference_tokens, candidate_tokens, n) for n in range(1, max_n + 1)]
    geo_mean = math.exp(sum(math.log(max(score, 1e-12)) for score in precisions) / max_n)
    brevity_penalty = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(1 - (len(reference_tokens) / max(len(candidate_tokens), 1)))
    return brevity_penalty * geo_mean


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens = tokenize_response(reference)
    candidate_tokens = tokenize_response(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    table = [[0] * (len(candidate_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    for row, reference_token in enumerate(reference_tokens, start=1):
        for column, candidate_token in enumerate(candidate_tokens, start=1):
            if reference_token == candidate_token:
                table[row][column] = table[row - 1][column - 1] + 1
            else:
                table[row][column] = max(table[row - 1][column], table[row][column - 1])

    lcs = table[-1][-1]
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def response_generation_metrics(actual_responses: list[str], predicted_responses: list[str]) -> dict:
    pairs = list(zip(actual_responses, predicted_responses))
    if not pairs:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "rouge_l": 0.0}
    return {
        "bleu_1": sum(sentence_bleu(actual, predicted, 1) for actual, predicted in pairs) / len(pairs),
        "bleu_2": sum(sentence_bleu(actual, predicted, 2) for actual, predicted in pairs) / len(pairs),
        "rouge_l": sum(rouge_l_f1(actual, predicted) for actual, predicted in pairs) / len(pairs),
    }


def evaluate_and_save(
    model_name: str,
    y_true,
    y_pred,
    labels,
    results_dir: Path,
    actual_responses: list[str] | None = None,
    predicted_responses: list[str] | None = None,
) -> dict:
    results_dir.mkdir(exist_ok=True)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if actual_responses is not None and predicted_responses is not None:
        metrics.update(response_generation_metrics(actual_responses, predicted_responses))

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
