"""Answer every LongMemEval-Oracle question."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from minimem import LLMClient, RetrieveMemAgent

from .construct_memory import MEMORIES_PATH, load_questions

PREDICTIONS_PATH = Path("benchmarks/longmemeval/predictions/oracle.jsonl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 15
MAX_WORKERS = 8


def load_memories() -> dict[int, list[dict]]:
    """Return memories keyed by question index."""

    if not MEMORIES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MEMORIES_PATH}. Run construct_memory first."
        )

    memories_by_question = {}
    with MEMORIES_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            memories_by_question[record["question_idx"]] = record["memories"]

    if not memories_by_question:
        raise ValueError(f"{MEMORIES_PATH} does not contain any memories.")

    return memories_by_question


def process_question(
    question_idx: int,
    question: dict,
    memories: list[dict],
    model: SentenceTransformer,
    model_lock: threading.Lock,
    llm: LLMClient,
) -> dict:
    """Retrieve memory and answer one question."""

    agent = RetrieveMemAgent(model=model, llm=llm, model_lock=model_lock)
    agent.load_memory(memories)

    # Retrieve with the bare question so the date prefix does not distort
    # embedding similarity, then answer with the question date attached:
    # temporal-reasoning questions may use relative expressions ("last month")
    # that are unanswerable without the asking date.
    retrieved_memories = agent.retrieve(question["question"], top_k=TOP_K)
    dated_question = (
        f"(asked on {question['question_date']}) {question['question']}"
    )
    answer, token_cost = agent.answer(dated_question, retrieved_memories)

    return {
        "question_idx": question_idx,
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "question": question["question"],
        "question_date": question["question_date"],
        "gold_answer": question["answer"],
        "predicted_answer": answer,
        "token_cost": token_cost,
        "retrieved_memories": retrieved_memories,
    }


def main():
    questions = load_questions()
    memories_by_question = load_memories()

    model = SentenceTransformer(EMBEDDING_MODEL)
    model_lock = threading.Lock()
    llm = LLMClient()

    predictions_by_question = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(
                process_question,
                question_idx,
                question,
                memories_by_question[question_idx],
                model,
                model_lock,
                llm,
            ): question_idx
            for question_idx, question in enumerate(questions)
        }
        for future in tqdm(
            as_completed(future_to_question),
            total=len(future_to_question),
            desc="Answering questions",
        ):
            question_idx = future_to_question[future]
            predictions_by_question[question_idx] = future.result()

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as file:
        for question_idx in range(len(questions)):
            file.write(
                json.dumps(predictions_by_question[question_idx], ensure_ascii=False)
                + "\n"
            )

    print(f"Saved predictions for {len(questions)} questions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
