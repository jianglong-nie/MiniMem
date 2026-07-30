"""Evaluate all saved LoCoMo predictions with token-level F1."""

import json
import re
import string
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


def normalize_answer(value) -> str:
    """Lowercase text and remove punctuation, articles, and whitespace."""

    text = str(value).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, gold_answer) -> float:
    """Return token overlap F1 in the range [0, 1]."""

    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(prediction_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


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
    """Calculate overall and category-level F1."""

    scores_by_category = {}
    no_information_count = 0

    for item in predictions:
        category = item["category"]
        score = token_f1(
            item["predicted_answer"],
            item["gold_answer"],
        )
        scores_by_category.setdefault(category, []).append(score)

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
                sum(scores) / len(scores) * 100
                if scores
                else 0.0
            ),
        }

    return {
        "question_count": len(predictions),
        "overall_f1": sum(all_scores) / len(all_scores) * 100,
        "no_information_count": no_information_count,
        "categories": category_results,
    }


def print_summary(summary: dict):
    print(f"{'Category':<15}{'Questions':>12}{'F1':>10}")
    print("-" * 37)

    for category, result in summary["categories"].items():
        print(
            f"{category:<15}"
            f"{result['count']:>12}"
            f"{result['f1']:>10.2f}"
        )

    print("-" * 37)
    print(
        f"{'Overall':<15}"
        f"{summary['question_count']:>12}"
        f"{summary['overall_f1']:>10.2f}"
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
