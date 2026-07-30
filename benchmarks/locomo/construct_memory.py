"""Build memories for every conversation in LoCoMo."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from minimem import ConstructMemAgent, LLMClient

DATA_PATH = Path("benchmarks/locomo/data/locomo10.json")
MEMORIES_DIR = Path("benchmarks/locomo/memories")
MAX_WORKERS = 8


def get_session_keys(conversation: dict) -> list[str]:
    """Return conversation session keys in chronological order."""

    session_keys = [
        key
        for key in conversation
        if key.startswith("session_")
        and key.removeprefix("session_").isdigit()
    ]
    return sorted(
        session_keys,
        key=lambda key: int(key.removeprefix("session_")),
    )


def process_session(conversation: dict, key: str, llm: LLMClient):
    """Build memory for one session."""

    date_time = datetime.strptime(
        conversation[f"{key}_date_time"],
        "%I:%M %p on %d %B, %Y",
    )
    turns = [
        {"role": turn["speaker"], "content": turn["text"]}
        for turn in conversation[key]
    ]

    agent = ConstructMemAgent(llm)
    return agent.run(turns, date_time)


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"LoCoMo data was not found at {DATA_PATH}"
        )

    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    llm = LLMClient()
    tasks = []
    session_counts = {}

    for conversation_index, sample in enumerate(data):
        conversation = sample["conversation"]
        session_keys = get_session_keys(conversation)
        session_counts[conversation_index] = len(session_keys)

        for session_order, key in enumerate(session_keys):
            tasks.append(
                (
                    conversation_index,
                    session_order,
                    conversation,
                    key,
                )
            )

    memories_by_session = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_session = {
            executor.submit(
                process_session,
                conversation,
                key,
                llm,
            ): (conversation_index, session_order)
            for (
                conversation_index,
                session_order,
                conversation,
                key,
            ) in tasks
        }

        for future in tqdm(
            as_completed(future_to_session),
            total=len(future_to_session),
            desc="Building sessions",
        ):
            task_key = future_to_session[future]
            memories_by_session[task_key] = future.result()

    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    for conversation_index in range(len(data)):
        memories = []
        for session_order in range(session_counts[conversation_index]):
            memories.extend(
                memories_by_session[
                    (conversation_index, session_order)
                ]
            )

        output_path = (
            MEMORIES_DIR / f"conv_{conversation_index}.json"
        )
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(memories, file, ensure_ascii=False, indent=2)

    print(
        f"Saved memories for {len(data)} conversations to "
        f"{MEMORIES_DIR}/"
    )


if __name__ == "__main__":
    main()
