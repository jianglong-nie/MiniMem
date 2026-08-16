"""Build memories for every conversation in LoCoMo-Refined."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from minimem import ConstructMemAgent, LLMClient

CONVERSATIONS_PATH = Path("benchmarks/locomo_refined/data/conversations.jsonl")
MEMORIES_DIR = Path("benchmarks/locomo_refined/memories")
DATE_FORMAT = "%I:%M %p on %d %B, %Y"
MAX_WORKERS = 8

# LoCoMo-Refined marks 521 of its 1,382 questions as multi-modal, but that flag
# only means the evidence message carries an image. Measured against the gold
# answers, just 25 questions (1.8% of the benchmark) can only be answered from
# the image description; in 382 the answer is already in the message text.
#
# Including the description follows the official multimodal flattening (see the
# dataset's conversation_history_multimodal_text field) and can recover those 25
# questions, at the cost of adding machine-generated text to 1,226 of the 5,882
# messages. BLIP captions are frequently wrong, so the more reliable `query`
# field is included alongside. Image URLs are skipped: they carry no signal for
# a text-only model. Set this to False to evaluate on message text alone.
INCLUDE_IMAGE_CONTEXT = True


def load_conversations() -> list[dict]:
    """Return all conversations ordered by conversation index."""

    if not CONVERSATIONS_PATH.is_file():
        raise FileNotFoundError(
            f"LoCoMo-Refined data was not found at {CONVERSATIONS_PATH}"
        )

    with CONVERSATIONS_PATH.open("r", encoding="utf-8") as file:
        conversations = [json.loads(line) for line in file if line.strip()]

    if not conversations:
        raise ValueError(f"{CONVERSATIONS_PATH} does not contain any conversations.")

    return sorted(conversations, key=lambda item: item["conversation_idx"])


def get_sessions(conversation: dict) -> list[dict]:
    """Return conversation sessions in chronological order."""

    return sorted(conversation["sessions"], key=lambda item: item["session_index"])


def message_content(message: dict) -> str:
    """Return the message text, including its image context when present."""

    lines = [message["text"].strip()]
    if INCLUDE_IMAGE_CONTEXT:
        for field, label in (("blip_caption", "caption"), ("query", "query")):
            value = message.get(field, "").strip()
            if value:
                lines.append(f"  [{label}] {value}")
    return "\n".join(line for line in lines if line)


def process_session(session: dict, llm: LLMClient):
    """Build memory for one session."""

    date_time = datetime.strptime(session["date_time"], DATE_FORMAT)
    turns = [
        {"role": message["speaker"], "content": message_content(message)}
        for message in session["messages"]
    ]

    agent = ConstructMemAgent(llm)
    return agent.run(turns, date_time)


def main():
    conversations = load_conversations()
    llm = LLMClient()

    tasks = []
    for conversation in conversations:
        for session_order, session in enumerate(get_sessions(conversation)):
            tasks.append((conversation["sample_id"], session_order, session))

    memories_by_session = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_session = {
            executor.submit(process_session, session, llm): (
                sample_id,
                session_order,
            )
            for sample_id, session_order, session in tasks
        }

        for future in tqdm(
            as_completed(future_to_session),
            total=len(future_to_session),
            desc="Building sessions",
        ):
            task_key = future_to_session[future]
            memories_by_session[task_key] = future.result()

    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    for conversation in conversations:
        sample_id = conversation["sample_id"]
        memories = []
        for session_order in range(len(conversation["sessions"])):
            memories.extend(memories_by_session[(sample_id, session_order)])

        output_path = MEMORIES_DIR / f"conv_{conversation['conversation_idx']}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(memories, file, ensure_ascii=False, indent=2)

    print(
        f"Saved memories for {len(conversations)} conversations to "
        f"{MEMORIES_DIR}/"
    )


if __name__ == "__main__":
    main()
