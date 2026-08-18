import datetime
import json

from .base import MemoryItem
from .llm import LLMClient

SYSTEM_PROMPT = """
Extract concise, self-contained facts from a conversation for later retrieval.
Preserve names, dates, relationships, plans, preferences, and outcomes when
they are stated. Do not invent information. Return only a valid JSON array.
Each array item must contain exactly one non-empty string field named "fact".
""".strip()

USER_PROMPT = """
Conversation date: {date}

Conversation:
{conversation}

Return the memorable facts in this format:
[{{"fact": "A self-contained fact."}}]
""".strip()


class ConstructMemAgent:
    def __init__(self, llm=None, max_retries=3):
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero.")
        self.llm = llm or LLMClient()
        self.max_retries = max_retries

    def run(
        self,
        turns: list[dict[str, str]],
        date_time: datetime.datetime,
    ):
        conversation = self._parse_input(turns)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    date=date_time.strftime("%d %B %Y"),
                    conversation=conversation,
                ),
            },
        ]

        response = self._get_llm_response(messages)
        memories = self._parse_response(response, date_time)
        return [memory.to_dict() for memory in memories]

    def _parse_input(self, turns: list[dict[str, str]]) -> str:
        if not turns:
            raise ValueError("Conversation turns cannot be empty.")

        lines = []
        for turn in turns:
            role = turn.get("role", "").strip()
            content = turn.get("content", "").strip()
            if not role or not content:
                raise ValueError(
                    "Every turn must have non-empty role and content fields."
                )
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _get_llm_response(self, messages: list[dict]) -> list[dict]:
        last_error = None

        for _ in range(self.max_retries):
            try:
                response = self.llm.invoke(messages)
                response = self._strip_json_code_fence(response)
                data = json.loads(response)

                if not isinstance(data, list):
                    raise ValueError("The response must be a JSON array.")

                for item in data:
                    if not isinstance(item, dict) or set(item) != {"fact"}:
                        raise ValueError(
                            "Every item must contain exactly one fact field."
                        )
                    fact = item["fact"]
                    if not isinstance(fact, str) or not fact.strip():
                        raise ValueError("Every fact must be a non-empty string.")
                return data
            except Exception as error:
                last_error = error

        raise RuntimeError(
            "The model did not return valid memory JSON after "
            f"{self.max_retries} attempts: {last_error}"
        ) from last_error

    def _strip_json_code_fence(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```") and response.endswith("```"):
            lines = response.splitlines()
            if len(lines) >= 3:
                response = "\n".join(lines[1:-1]).strip()
        return response

    def _parse_response(
        self,
        response: list[dict],
        date_time: datetime.datetime,
    ):
        return [
            MemoryItem(
                observed_at=date_time,
                fact=item["fact"].strip(),
            )
            for item in response
        ]
