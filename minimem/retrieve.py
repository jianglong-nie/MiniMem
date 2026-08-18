from contextlib import nullcontext

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from .llm import LLMClient

ANSWER_SYSTEM_PROMPT = """
Answer the question using the supplied memory facts as evidence. You may use
ordinary common knowledge for a simple inference, but do not invent
conversation-specific details. Give a short direct answer without explanation.
If no memory fact is relevant, reply exactly: No information available.
""".strip()

ANSWER_PROMPT_TEMPLATE = """
Answer the question below based on the retrieved memories.

Memories:
{memories}

Question: {query}

Return only the short answer.
""".strip()


class RetrieveMemAgent:
    def __init__(
        self,
        model="all-MiniLM-L6-v2",
        llm=None,
        model_lock=None,
    ):
        self.llm = llm or LLMClient()
        self.model = (
            SentenceTransformer(model)
            if isinstance(model, str)
            else model
        )
        self.model_lock = model_lock
        self.memories = []
        self.memory_vectors = None
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def run(self, query: str, top_k=5):
        retrieved_memories = self.retrieve(query, top_k)
        answer, token_cost = self.answer(query, retrieved_memories)
        return {
            "question": query,
            "retrieved_memories": retrieved_memories,
            "answer": answer,
            "token_cost": token_cost,
        }

    def load_memory(self, memories: list[dict]):
        self.memories = []
        for memory in memories:
            fact = memory.get("fact", "")
            if not isinstance(fact, str) or not fact.strip():
                raise ValueError("Every memory must contain a non-empty fact.")

            observed_at = memory.get("observed_at")
            if observed_at is not None and not isinstance(observed_at, str):
                raise ValueError("Memory observed_at must be an ISO string.")

            self.memories.append(dict(memory))

        if not self.memories:
            self.memory_vectors = None
            return

        memory_texts = [
            self._format_memory(memory)
            for memory in self.memories
        ]
        self.memory_vectors = self._encode(memory_texts)

    def retrieve(self, query: str, top_k=5):
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not self.memories or self.memory_vectors is None:
            return []

        query_vector = self._encode(query)
        memory_scores = self.memory_vectors @ query_vector
        top_k = min(top_k, len(self.memories))
        top_k_indices = memory_scores.argsort()[::-1][:top_k]

        return [
            dict(self.memories[index])
            for index in top_k_indices
        ]

    def answer(self, query: str, memories: list[dict]):
        memory_text = "\n".join(
            f"- {self._format_memory(memory)}"
            for memory in memories
        )
        if not memory_text:
            memory_text = "- No memory facts were retrieved."

        prompt = ANSWER_PROMPT_TEMPLATE.format(
            memories=memory_text,
            query=query,
        )
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        answer = self.llm.invoke(messages).strip()

        # tiktoken 口径统计:只数 content 文本,数不到 role 等消息结构开销,
        # 会略小于 API 返回的 prompt_tokens,但离线可复算、跨模型可比
        token_cost = {
            "retrieve_text_token": self.count_tokens(memory_text),
            "input_text_token": self.count_tokens(ANSWER_SYSTEM_PROMPT + prompt),
            "output_text_token": self.count_tokens(answer),
        }
        token_cost["total_text_token"] = (
            token_cost["input_text_token"] + token_cost["output_text_token"]
        )
        return answer, token_cost

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _format_memory(self, memory: dict) -> str:
        fact = memory["fact"]
        observed_at = memory.get("observed_at")
        if observed_at:
            return f"[{observed_at}] {fact}"
        return fact

    def _encode(self, text):
        with self.model_lock or nullcontext():
            vectors = self.model.encode(
                text,
                normalize_embeddings=True,
            )
        return np.asarray(vectors, dtype=float)
