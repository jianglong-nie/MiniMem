"""Run one synthetic question with two LLM calls."""

import datetime as dt
import json
from pathlib import Path

from benchmarks.locomo.evaluate_answer import token_f1
from minimem import ConstructMemAgent, LLMClient, RetrieveMemAgent


def main() -> None:
    """Build memory and answer one question from the bundled tiny example."""

    data_path = Path(__file__).with_name("tiny_conversation.json")
    with data_path.open("r", encoding="utf-8") as file:
        example = json.load(file)

    llm = LLMClient()

    print(f"Model: {llm.model}")
    print("Step 1/2: Calling the LLM to construct memory...")
    construct_agent = ConstructMemAgent(llm)
    memories = construct_agent.run(
        example["turns"],
        dt.datetime.fromisoformat(example["conversation_date"]),
    )
    print(f"Constructed {len(memories)} memories.")

    print("Step 2/2: Calling the LLM to answer the question...")
    retrieve_agent = RetrieveMemAgent(llm=llm)
    retrieve_agent.load_memory(memories)
    result = retrieve_agent.run(example["question"], top_k=2)
    print("Answer received.\n")

    result["memories"] = memories
    result["gold_answer"] = example["answer"]
    result["token_f1"] = token_f1(result["answer"], example["answer"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
