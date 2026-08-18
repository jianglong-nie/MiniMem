"""Score LongMemEval-Oracle predictions.

Unlike LoCoMo-Refined, LongMemEval has no separate scoring harness: the paper's
only metric is an LLM judge run locally with one prompt per question type (plus
a dedicated prompt for abstention questions, whose question_id ends with
"_abs"). The prompt templates and the yes/no parsing below are ported verbatim
from the official repository (src/evaluation/evaluate_qa.py).

Because the judge costs one LLM call per question, it is gated behind
RUN_OFFICIAL_JUDGE (default False). Every run also reports free lexical
metrics (token F1 and BLEU-1, ported from the LoCoMo-Refined scorer) as a
local sanity check; they are NOT the official metric and are meaningless for
preference questions (the gold answer is a rubric) and abstention questions
(judged on refusing to answer).
"""

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from minimem import LLMClient

from .answer_question import PREDICTIONS_PATH

RESULTS_PATH = Path("benchmarks/longmemeval/results/oracle.jsonl")
SUMMARY_PATH = Path("benchmarks/longmemeval/results/summary.json")
MAX_WORKERS = 8

# The official metric costs one judge call per prediction. Keep False for free
# lexical-only runs during development; set True to produce the real score.
RUN_OFFICIAL_JUDGE = True

QUESTION_TYPE_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
]

# Official judge prompts, verbatim.
DEFAULT_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

TEMPORAL_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

KNOWLEDGE_UPDATE_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

PREFERENCE_TEMPLATE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

ABSTENTION_TEMPLATE = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."

TEMPLATES_BY_TYPE = {
    "single-session-user": DEFAULT_TEMPLATE,
    "single-session-assistant": DEFAULT_TEMPLATE,
    "multi-session": DEFAULT_TEMPLATE,
    "temporal-reasoning": TEMPORAL_TEMPLATE,
    "knowledge-update": KNOWLEDGE_UPDATE_TEMPLATE,
    "single-session-preference": PREFERENCE_TEMPLATE,
}

# Tokenisation and lexical metrics mirror the LoCoMo-Refined scorer
# (benchmarks/locomo_refined/evaluate_answer.py).
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


def is_abstention(prediction: dict) -> bool:
    return prediction["question_id"].endswith("_abs")


def judge_prompt(prediction: dict) -> str:
    """Return the official answer-checking prompt for one prediction."""

    if is_abstention(prediction):
        template = ABSTENTION_TEMPLATE
    else:
        template = TEMPLATES_BY_TYPE[prediction["question_type"]]
    return template.format(
        prediction["question"],
        prediction["gold_answer"],
        prediction["predicted_answer"],
    )


def load_predictions() -> list[dict]:
    """Return all saved predictions."""

    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_PATH}. Run answer_question first."
        )

    with PREDICTIONS_PATH.open("r", encoding="utf-8") as file:
        predictions = [json.loads(line) for line in file if line.strip()]

    if not predictions:
        raise ValueError(f"{PREDICTIONS_PATH} does not contain any predictions.")

    return predictions


def load_results() -> list[dict]:
    """Return the judgments saved so far."""

    if not RESULTS_PATH.is_file():
        return []

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def judge_prediction(prediction: dict, llm: LLMClient) -> dict:
    """Ask the judge model whether one prediction is correct."""

    messages = [{"role": "user", "content": judge_prompt(prediction)}]
    response = llm.invoke(messages).strip()

    return {
        "question_idx": prediction["question_idx"],
        "question_id": prediction["question_id"],
        "question_type": prediction["question_type"],
        "is_abstention": is_abstention(prediction),
        "judge_response": response,
        "correct": "yes" in response.lower(),
    }


def run_official_judge(predictions: list[dict]) -> None:
    """Judge every prediction and save the judgments."""

    llm = LLMClient()

    results_by_index = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(judge_prediction, prediction, llm): index
            for index, prediction in enumerate(predictions)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="Judging predictions",
        ):
            results_by_index[future_to_index[future]] = future.result()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        for index in range(len(predictions)):
            file.write(
                json.dumps(results_by_index[index], ensure_ascii=False) + "\n"
            )


def summarize_lexical(predictions: list[dict]) -> dict:
    """Calculate overall and per-type token F1 / BLEU-1."""

    by_type = {}
    for prediction in predictions:
        by_type.setdefault(prediction["question_type"], []).append(
            (
                token_f1(prediction["predicted_answer"], prediction["gold_answer"]),
                bleu1(prediction["predicted_answer"], prediction["gold_answer"]),
            )
        )

    def averages(scores: list[tuple]) -> dict:
        return {
            "count": len(scores),
            "f1": sum(f1 for f1, _ in scores) / len(scores) * 100,
            "bleu1": sum(b for _, b in scores) / len(scores) * 100,
        }

    all_scores = [score for scores in by_type.values() for score in scores]
    return {
        "overall": averages(all_scores),
        "question_types": {
            question_type: averages(by_type[question_type])
            for question_type in QUESTION_TYPE_ORDER
            if question_type in by_type
        },
    }


def summarize_official(results: list[dict]) -> dict:
    """Calculate overall, per-type, and abstention judge accuracy."""

    def accuracy(items: list[dict]) -> float:
        return sum(item["correct"] for item in items) / len(items) * 100

    by_type = {}
    for result in results:
        by_type.setdefault(result["question_type"], []).append(result)

    summary = {
        "overall": {"count": len(results), "accuracy": accuracy(results)},
        "question_types": {
            question_type: {
                "count": len(by_type[question_type]),
                "accuracy": accuracy(by_type[question_type]),
            }
            for question_type in QUESTION_TYPE_ORDER
            if question_type in by_type
        },
    }

    abstention_items = [result for result in results if result["is_abstention"]]
    if abstention_items:
        summary["abstention"] = {
            "count": len(abstention_items),
            "accuracy": accuracy(abstention_items),
        }
    return summary


def print_lexical_summary(summary: dict) -> None:
    print("\nLexical sanity check (NOT the official metric; meaningless for")
    print("preference and abstention questions):")
    print(f"{'Question type':<30}{'Questions':>12}{'F1':>10}{'BLEU-1':>10}")
    print("-" * 62)
    for question_type, result in summary["question_types"].items():
        print(
            f"{question_type:<30}"
            f"{result['count']:>12}"
            f"{result['f1']:>10.2f}"
            f"{result['bleu1']:>10.2f}"
        )
    print("-" * 62)
    overall = summary["overall"]
    print(
        f"{'Overall':<30}"
        f"{overall['count']:>12}"
        f"{overall['f1']:>10.2f}"
        f"{overall['bleu1']:>10.2f}"
    )


def print_official_summary(summary: dict) -> None:
    print("\nOfficial LLM-judge accuracy:")
    print(f"{'Question type':<30}{'Questions':>12}{'Accuracy':>12}")
    print("-" * 54)
    for question_type, result in summary["question_types"].items():
        print(
            f"{question_type:<30}{result['count']:>12}{result['accuracy']:>11.2f}%"
        )
    print("-" * 54)
    overall = summary["overall"]
    print(
        f"{'Overall':<30}{overall['count']:>12}{overall['accuracy']:>11.2f}%"
    )
    if "abstention" in summary:
        abstention = summary["abstention"]
        print(
            f"\nAbstention subset (also counted above): "
            f"{abstention['accuracy']:.2f}% of {abstention['count']}"
        )


def main():
    predictions = load_predictions()

    if RUN_OFFICIAL_JUDGE:
        run_official_judge(predictions)

    results = load_results()
    lexical = summarize_lexical(predictions)

    token_costs = [
        prediction["token_cost"]["total_text_token"] for prediction in predictions
    ]
    summary = {
        "question_count": len(predictions),
        "avg_total_text_token": sum(token_costs) / len(token_costs),
        "lexical": lexical,
        "official_judge": summarize_official(results) if results else None,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_lexical_summary(lexical)
    if summary["official_judge"]:
        print_official_summary(summary["official_judge"])
    unjudged = len(predictions) - len(results)
    if unjudged > 0:
        print(
            f"\nReminder: {unjudged} predictions have no official judgment. The"
            "\nofficial LongMemEval metric is the LLM judge; set"
            f"\nRUN_OFFICIAL_JUDGE = True to judge them ({unjudged} LLM calls)."
        )

    print(f"\nAverage answer token cost: {summary['avg_total_text_token']:.0f}")
    print(f"Saved per-question judgments to {RESULTS_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
