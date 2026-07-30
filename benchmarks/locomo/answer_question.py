"""Answer every non-adversarial LoCoMo question."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from minimem import LLMClient, RetrieveMemAgent

DATA_PATH = Path("benchmarks/locomo/data/locomo10.json")
MEMORIES_DIR = Path("benchmarks/locomo/memories")
PREDICTIONS_DIR = Path("benchmarks/locomo/predictions")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MAX_WORKERS = 8


def process_question(
    agent: RetrieveMemAgent,
    question_index: int,
    qa: dict,
):
    """Retrieve memory and answer one question."""

    result = agent.run(qa["question"], top_k=TOP_K)
    return {
        "question_index": question_index,
        "question": qa["question"],
        "category": qa["category"],
        "gold_answer": qa["answer"],
        "predicted_answer": result["answer"],
        "retrieved_memories": result["retrieved_memories"],
    }


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"LoCoMo data was not found at {DATA_PATH}"
        )

    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    model = SentenceTransformer(EMBEDDING_MODEL)
    model_lock = Lock()
    llm = LLMClient()

    agents = {}
    tasks = []
    question_counts = {}

    for conversation_index, sample in enumerate(
        tqdm(data, desc="Loading memories")
    ):
        memory_path = (
            MEMORIES_DIR / f"conv_{conversation_index}.json"
        )
        if not memory_path.is_file():
            raise FileNotFoundError(
                f"Missing {memory_path}. Run construct_memory first."
            )
        with memory_path.open("r", encoding="utf-8") as file:
            memories = json.load(file)

        agent = RetrieveMemAgent(
            model=model,
            llm=llm,
            model_lock=model_lock,
        )
        agent.load_memory(memories)
        agents[conversation_index] = agent

        question_items = [
            (question_index, qa)
            for question_index, qa in enumerate(sample["qa"])
            if qa["category"] != 5
        ]
        question_counts[conversation_index] = len(question_items)

        for question_order, (question_index, qa) in enumerate(
            question_items
        ):
            tasks.append(
                (
                    conversation_index,
                    question_order,
                    question_index,
                    qa,
                )
            )

    predictions_by_question = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(
                process_question,
                agents[conversation_index],
                question_index,
                qa,
            ): (conversation_index, question_order)
            for (
                conversation_index,
                question_order,
                question_index,
                qa,
            ) in tasks
        }

        for future in tqdm(
            as_completed(future_to_question),
            total=len(future_to_question),
            desc="Answering questions",
        ):
            task_key = future_to_question[future]
            predictions_by_question[task_key] = future.result()

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for conversation_index in range(len(data)):
        predictions = [
            predictions_by_question[
                (conversation_index, question_order)
            ]
            for question_order in range(
                question_counts[conversation_index]
            )
        ]

        output_path = (
            PREDICTIONS_DIR / f"conv_{conversation_index}.json"
        )
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)

    print(
        f"Saved predictions for {len(data)} conversations to "
        f"{PREDICTIONS_DIR}/"
    )


if __name__ == "__main__":
    main()
