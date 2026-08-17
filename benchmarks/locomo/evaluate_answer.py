"""Evaluate all saved LoCoMo predictions with token-level F1 and BLEU-1.

Tokenisation and metrics mirror the LoCoMo-Refined scorer
(benchmarks/locomo_refined/evaluate_answer.py) so scores are comparable
across the bundled benchmarks. Note this differs from the original LoCoMo
paper's SQuAD-style normalisation (which strips punctuation and articles).
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

PREDICTIONS_DIR = Path("benchmarks/locomo/predictions")
RESULTS_DIR = Path("benchmarks/locomo/results")
CONVERSATION_COUNT = 10

CATEGORY_LABELS = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
}
CATEGORY_ORDER = [4, 1, 3, 2]


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def token_f1(prediction, reference) -> float:
    """Return token overlap F1 in the range [0, 1]."""

    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    overlap = sum(
        min(count, reference_counts[token])
        for token, count in Counter(prediction_tokens).items()
    )
    if overlap <= 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction, reference) -> float:
    """Return unigram BLEU with a brevity penalty."""

    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    overlap = sum(
        min(count, reference_counts[token])
        for token, count in Counter(prediction_tokens).items()
    )
    precision = overlap / len(prediction_tokens)
    if precision <= 0:
        return 0.0

    penalty = 1.0
    if len(prediction_tokens) < len(reference_tokens):
        penalty = math.exp(1.0 - len(reference_tokens) / len(prediction_tokens))
    return penalty * precision


def load_predictions() -> list[dict]:
    """Load prediction files for all ten conversations."""

    prediction_paths = [
        PREDICTIONS_DIR / f"conv_{index}.json"
        for index in range(CONVERSATION_COUNT)
    ]
    missing_paths = [
        path for path in prediction_paths if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing prediction files: "
            + ", ".join(str(path) for path in missing_paths)
            + ". Run answer_question first."
        )

    predictions = []
    for path in prediction_paths:
        with path.open("r", encoding="utf-8") as file:
            predictions.extend(json.load(file))

    if not predictions:
        raise ValueError("Prediction files do not contain any questions.")
    return predictions


def summarize(predictions: list[dict]) -> dict:
    """Calculate overall and category-level F1 and BLEU-1."""

    scores_by_category = {}
    no_information_count = 0

    for item in predictions:
        category = item["category"]
        scores_by_category.setdefault(category, []).append(
            (
                token_f1(item["predicted_answer"], item["gold_answer"]),
                bleu1(item["predicted_answer"], item["gold_answer"]),
            )
        )

        if item["predicted_answer"].strip().lower().startswith(
            "no information"
        ):
            no_information_count += 1

    all_scores = [
        score
        for scores in scores_by_category.values()
        for score in scores
    ]
    category_results = {}

    for category in CATEGORY_ORDER:
        scores = scores_by_category.get(category, [])
        category_results[CATEGORY_LABELS[category]] = {
            "count": len(scores),
            "f1": (
                sum(f1 for f1, _ in scores) / len(scores) * 100
                if scores
                else 0.0
            ),
            "bleu1": (
                sum(b for _, b in scores) / len(scores) * 100
                if scores
                else 0.0
            ),
        }

    return {
        "question_count": len(predictions),
        "overall_f1": sum(f1 for f1, _ in all_scores) / len(all_scores) * 100,
        "overall_bleu1": sum(b for _, b in all_scores) / len(all_scores) * 100,
        "no_information_count": no_information_count,
        "categories": category_results,
    }


def print_summary(summary: dict):
    print(f"{'Category':<15}{'Questions':>12}{'F1':>10}{'BLEU-1':>10}")
    print("-" * 47)

    for category, result in summary["categories"].items():
        print(
            f"{category:<15}"
            f"{result['count']:>12}"
            f"{result['f1']:>10.2f}"
            f"{result['bleu1']:>10.2f}"
        )

    print("-" * 47)
    print(
        f"{'Overall':<15}"
        f"{summary['question_count']:>12}"
        f"{summary['overall_f1']:>10.2f}"
        f"{summary['overall_bleu1']:>10.2f}"
    )
    print(
        f"\nNo-information answers: "
        f"{summary['no_information_count']}"
    )


def main():
    predictions = load_predictions()
    summary = summarize(predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "summary.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nSaved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
