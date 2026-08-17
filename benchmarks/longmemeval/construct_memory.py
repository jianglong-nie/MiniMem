"""Build memories for every question in LongMemEval-Oracle."""

import json
import threading
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


def load_completed_indices() -> set[int]:
    """Return question indices already present in the memories file."""

    if not MEMORIES_PATH.is_file():
        return set()

    with MEMORIES_PATH.open("r", encoding="utf-8") as file:
        return {json.loads(line)["question_idx"] for line in file if line.strip()}


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
    completed = load_completed_indices()
    pending = [
        (idx, question)
        for idx, question in enumerate(questions)
        if idx not in completed
    ]
    if completed:
        print(f"Resuming: {len(completed)} questions already built, {len(pending)} to go.")

    llm = LLMClient()
    MEMORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    with MEMORIES_PATH.open("a", encoding="utf-8") as file:

        def build_and_save(question_idx: int, question: dict):
            memories = process_question(question, llm)
            record = {
                "question_idx": question_idx,
                "question_id": question["question_id"],
                "memories": memories,
            }
            with write_lock:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(build_and_save, question_idx, question)
                for question_idx, question in pending
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building questions",
            ):
                future.result()

    print(
        f"Saved memories for {len(completed) + len(pending)} questions to "
        f"{MEMORIES_PATH}"
    )


if __name__ == "__main__":
    main()
