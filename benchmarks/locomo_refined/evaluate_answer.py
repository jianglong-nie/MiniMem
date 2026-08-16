"""Score LoCoMo-Refined predictions and export the official submission file.

LoCoMo-Refined ships its own scorer, so the authoritative numbers come from the
official harness (see `print_next_step` below). This module writes the
submission file that harness expects and reports the lexical metrics locally so
a run can be inspected without paying for the LLM judge.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

PREDICTIONS_DIR = Path("benchmarks/locomo_refined/predictions")
RESULTS_DIR = Path("benchmarks/locomo_refined/results")
SUBMISSION_PATH = RESULTS_DIR / "predictions.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

CATEGORY_LABELS = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-domain",
    "4": "Single-hop",
}
CATEGORY_ORDER = ["4", "1", "3", "2"]

# Tokenisation and metrics mirror src/bleu_f1.py in the official repository so
# the locally reported values match the ones the official scorer computes.
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


def answer_candidates(gold_answer) -> list[str]:
    """Return the gold answer as a list of candidate strings."""

    if isinstance(gold_answer, list):
        candidates = [str(item).strip() for item in gold_answer]
    else:
        candidates = [str(gold_answer).strip()]
    return [candidate for candidate in candidates if candidate] or [""]


def best_score(metric, prediction, gold_answer) -> float:
    """Score against every gold candidate and keep the best, as the official
    scorer does."""

    return max(
        metric(prediction, candidate)
        for candidate in answer_candidates(gold_answer)
    )


def load_predictions() -> list[dict]:
    """Load every saved prediction file."""

    prediction_paths = sorted(PREDICTIONS_DIR.glob("*.json"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction files in {PREDICTIONS_DIR}. Run answer_question first."
        )

    predictions = []
    for path in prediction_paths:
        with path.open("r", encoding="utf-8") as file:
            predictions.extend(json.load(file))

    if not predictions:
        raise ValueError("Prediction files do not contain any questions.")
    return predictions


def summarize(predictions: list[dict]) -> dict:
    """Calculate overall and category-level lexical metrics."""

    scores_by_category = {}
    no_information_count = 0

    for item in predictions:
        scores_by_category.setdefault(item["category"], []).append(
            (
                best_score(token_f1, item["predicted_answer"], item["gold_answer"]),
                best_score(bleu1, item["predicted_answer"], item["gold_answer"]),
            )
        )

        if item["predicted_answer"].strip().lower().startswith("no information"):
            no_information_count += 1

    all_scores = [
        score for scores in scores_by_category.values() for score in scores
    ]
    category_results = {}

    for category in CATEGORY_ORDER:
        scores = scores_by_category.get(category, [])
        category_results[CATEGORY_LABELS[category]] = {
            "count": len(scores),
            "f1": sum(f1 for f1, _ in scores) / len(scores) * 100 if scores else 0.0,
            "bleu1": sum(b for _, b in scores) / len(scores) * 100 if scores else 0.0,
        }

    return {
        "question_count": len(predictions),
        "overall_f1": sum(f1 for f1, _ in all_scores) / len(all_scores) * 100,
        "overall_bleu1": sum(b for _, b in all_scores) / len(all_scores) * 100,
        "no_information_count": no_information_count,
        "categories": category_results,
    }


def write_submission(predictions: list[dict]) -> None:
    """Write the {qa_id, predicted_answer} file the official scorer reads."""

    with SUBMISSION_PATH.open("w", encoding="utf-8") as file:
        for item in predictions:
            record = {
                "qa_id": item["qa_id"],
                "predicted_answer": item["predicted_answer"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(summary: dict) -> None:
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
    print(f"\nNo-information answers: {summary['no_information_count']}")


def print_next_step() -> None:
    print(
        "\nLexical metrics above are a local sanity check only. For the official"
        "\nLLM-judge score, run the LoCoMo-Refined harness:\n"
        "\n  git clone https://github.com/mem-eval-suite/LoCoMo_refined"
        "\n  cd LoCoMo_refined"
        f"\n  export LOCOMO_PREDICTIONS_PATH={SUBMISSION_PATH.resolve()}"
        "\n  export EVALUATOR_MODEL=qwen3-14b"
        "\n  ./scripts/run_eval.sh\n"
    )


def main():
    predictions = load_predictions()
    summary = summarize(predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_submission(predictions)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nSaved submission to {SUBMISSION_PATH}")
    print(f"Saved local summary to {SUMMARY_PATH}")
    print_next_step()


if __name__ == "__main__":
    main()
