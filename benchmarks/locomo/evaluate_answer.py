"""Evaluate all saved LoCoMo predictions with token-level F1 and BLEU-1.

Tokenisation and metrics mirror the LoCoMo-Refined scorer
(benchmarks/locomo_refined/evaluate_answer.py) so scores are comparable
across the bundled benchmarks. Note this differs from the original LoCoMo
paper's SQuAD-style normalisation (which strips punctuation and articles).

An LLM judge (one call per prediction) grades semantic correctness, since
lexical overlap under-credits paraphrased answers. It is gated behind
RUN_LLM_JUDGE for free lexical-only runs.
"""

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from minimem import LLMClient

PREDICTIONS_DIR = Path("benchmarks/locomo/predictions")
RESULTS_DIR = Path("benchmarks/locomo/results")
JUDGMENTS_PATH = RESULTS_DIR / "judgments.jsonl"
CONVERSATION_COUNT = 10
MAX_WORKERS = 8

# The LLM judge costs one call per prediction. Set False for free
# lexical-only runs during development.
RUN_LLM_JUDGE = True

JUDGE_PROMPT = """Your task is to label a generated answer as CORRECT or WRONG, given a
question about a past conversation and the gold answer.

Grade generously: the generated answer may be longer or phrased differently,
and it is CORRECT as long as it conveys the same information as the gold
answer. For time questions, treat different formats of the same date or time
period as CORRECT ("May 7th" vs "7 May"). If the generated answer misses the
asked information or states something different, it is WRONG.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {predicted_answer}

Reply with exactly one word: CORRECT or WRONG."""

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


def judge_prediction(index: int, prediction: dict, llm: LLMClient) -> dict:
    """Ask the judge model whether one prediction is correct."""

    prompt = JUDGE_PROMPT.format(
        question=prediction["question"],
        gold_answer=prediction["gold_answer"],
        predicted_answer=prediction["predicted_answer"],
    )
    response = llm.invoke([{"role": "user", "content": prompt}]).strip()

    return {
        "prediction_index": index,
        "category": prediction["category"],
        "judge_response": response,
        "correct": response.upper().startswith("CORRECT"),
    }


def run_llm_judge(predictions: list[dict]) -> None:
    """Judge every prediction and save the judgments."""

    llm = LLMClient()

    judgments_by_index = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(judge_prediction, index, prediction, llm): index
            for index, prediction in enumerate(predictions)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="Judging predictions",
        ):
            judgments_by_index[future_to_index[future]] = future.result()

    with JUDGMENTS_PATH.open("w", encoding="utf-8") as file:
        for index in range(len(predictions)):
            file.write(
                json.dumps(judgments_by_index[index], ensure_ascii=False) + "\n"
            )


def load_judgments() -> list[dict]:
    """Return the judgments saved so far."""

    if not JUDGMENTS_PATH.is_file():
        return []

    with JUDGMENTS_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize_judge(judgments: list[dict]) -> dict:
    """Calculate overall and category-level judge accuracy."""

    def accuracy(items: list[dict]) -> float:
        return sum(item["correct"] for item in items) / len(items) * 100

    by_category = {}
    for judgment in judgments:
        by_category.setdefault(judgment["category"], []).append(judgment)

    return {
        "overall": {"count": len(judgments), "accuracy": accuracy(judgments)},
        "categories": {
            CATEGORY_LABELS[category]: {
                "count": len(by_category[category]),
                "accuracy": accuracy(by_category[category]),
            }
            for category in CATEGORY_ORDER
            if category in by_category
        },
    }


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


def print_judge_summary(summary: dict) -> None:
    print("\nLLM-judge accuracy:")
    print(f"{'Category':<15}{'Questions':>12}{'Accuracy':>12}")
    print("-" * 39)
    for category, result in summary["categories"].items():
        print(
            f"{category:<15}{result['count']:>12}{result['accuracy']:>11.2f}%"
        )
    print("-" * 39)
    overall = summary["overall"]
    print(
        f"{'Overall':<15}{overall['count']:>12}{overall['accuracy']:>11.2f}%"
    )


def main():
    predictions = load_predictions()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_LLM_JUDGE:
        run_llm_judge(predictions)

    summary = summarize(predictions)
    judgments = load_judgments()
    if judgments:
        summary["llm_judge"] = summarize_judge(judgments)

    output_path = RESULTS_DIR / "summary.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_summary(summary)
    if judgments:
        print_judge_summary(summary["llm_judge"])
    print(f"\nSaved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
