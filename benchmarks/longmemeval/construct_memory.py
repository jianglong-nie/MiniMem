"""Build memories for every question in LongMemEval-Oracle."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from minimem import ConstructMemAgent, LLMClient

DATA_PATH = Path("benchmarks/longmemeval/data/longmemeval_oracle")
MEMORIES_PATH = Path("benchmarks/longmemeval/memories/oracle.jsonl")
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
MAX_WORKERS = 8


def load_questions() -> list[dict]:
    """Return all questions in file order."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"LongMemEval data was not found at {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError(f"{DATA_PATH} does not contain any questions.")

    return questions


def process_session(session: list[dict], date: str, llm: LLMClient):
    """Build memory for one haystack session."""

    date_time = datetime.strptime(date, DATE_FORMAT)
    # Keep only role/content: has_answer is evaluation metadata and must not
    # reach the model.
    turns = [
        {"role": message["role"], "content": message["content"]}
        for message in session
    ]

    agent = ConstructMemAgent(llm)
    return agent.run(turns, date_time)


def process_question(question: dict, llm: LLMClient) -> list[dict]:
    """Build the merged memory list for one question's haystack sessions."""

    memories = []
    for session, date in zip(question["haystack_sessions"], question["haystack_dates"]):
        memories.extend(process_session(session, date, llm))
    return memories


def main():
    questions = load_questions()
    llm = LLMClient()

    memories_by_question = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(process_question, question, llm): question_idx
            for question_idx, question in enumerate(questions)
        }
        for future in tqdm(
            as_completed(future_to_question),
            total=len(future_to_question),
            desc="Building questions",
        ):
            question_idx = future_to_question[future]
            memories_by_question[question_idx] = future.result()

    MEMORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEMORIES_PATH.open("w", encoding="utf-8") as file:
        for question_idx, question in enumerate(questions):
            record = {
                "question_idx": question_idx,
                "question_id": question["question_id"],
                "memories": memories_by_question[question_idx],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved memories for {len(questions)} questions to {MEMORIES_PATH}")


if __name__ == "__main__":
    main()
