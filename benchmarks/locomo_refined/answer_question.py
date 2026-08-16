"""Answer every LoCoMo-Refined question."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from minimem import LLMClient, RetrieveMemAgent

QUESTIONS_PATH = Path("benchmarks/locomo_refined/data/questions.jsonl")
MEMORIES_DIR = Path("benchmarks/locomo_refined/memories")
PREDICTIONS_DIR = Path("benchmarks/locomo_refined/predictions")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 15
MAX_WORKERS = 8


def load_questions() -> dict[str, list[dict]]:
    """Return questions grouped by conversation, in file order."""

    if not QUESTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"LoCoMo-Refined questions were not found at {QUESTIONS_PATH}"
        )

    questions_by_sample = {}
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            question = json.loads(line)
            questions_by_sample.setdefault(question["sample_id"], []).append(question)

    if not questions_by_sample:
        raise ValueError(f"{QUESTIONS_PATH} does not contain any questions.")

    return questions_by_sample


def load_memories(conversation_idx: int) -> list[dict]:
    """Return the saved memories for one conversation."""

    memory_path = MEMORIES_DIR / f"conv_{conversation_idx}.json"
    if not memory_path.is_file():
        raise FileNotFoundError(
            f"Missing {memory_path}. Run construct_memory first."
        )

    with memory_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def process_question(agent: RetrieveMemAgent, qa: dict):
    """Retrieve memory and answer one question."""

    result = agent.run(qa["question"], top_k=TOP_K)
    return {
        "qa_id": qa["qa_id"],
        "question": qa["question"],
        "category": qa["category"],
        "gold_answer": qa["answer"],
        "predicted_answer": result["answer"],
        "token_cost": result["token_cost"],
        "retrieved_memories": result["retrieved_memories"],
    }


def main():
    questions_by_sample = load_questions()

    model = SentenceTransformer(EMBEDDING_MODEL)
    model_lock = Lock()
    llm = LLMClient()

    agents = {}
    tasks = []

    for sample_id in tqdm(questions_by_sample, desc="Loading memories"):
        conversation_idx = questions_by_sample[sample_id][0]["conversation_idx"]
        agent = RetrieveMemAgent(model=model, llm=llm, model_lock=model_lock)
        agent.load_memory(load_memories(conversation_idx))
        agents[sample_id] = agent

        for question_order, qa in enumerate(questions_by_sample[sample_id]):
            tasks.append((sample_id, question_order, qa))

    predictions_by_question = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(process_question, agents[sample_id], qa): (
                sample_id,
                question_order,
            )
            for sample_id, question_order, qa in tasks
        }

        for future in tqdm(
            as_completed(future_to_question),
            total=len(future_to_question),
            desc="Answering questions",
        ):
            task_key = future_to_question[future]
            predictions_by_question[task_key] = future.result()

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for sample_id, questions in questions_by_sample.items():
        predictions = [
            predictions_by_question[(sample_id, question_order)]
            for question_order in range(len(questions))
        ]

        output_path = PREDICTIONS_DIR / f"conv_{questions[0]['conversation_idx']}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)

    print(
        f"Saved predictions for {len(questions_by_sample)} conversations to "
        f"{PREDICTIONS_DIR}/"
    )


if __name__ == "__main__":
    main()
