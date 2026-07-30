import numpy as np
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
        max_retries=3,
    ):
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero.")

        self.llm = llm or LLMClient()
        self.model = (
            SentenceTransformer(model)
            if isinstance(model, str)
            else model
        )
        self.model_lock = model_lock
        self.max_retries = max_retries
        self.memories = []
        self.memory_vectors = None

    def run(self, query: str, top_k=5):
        retrieved_memories = self.retrieve(query, top_k)
        answer = self.answer(query, retrieved_memories)
        return {
            "question": query,
            "retrieved_memories": retrieved_memories,
            "answer": answer,
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
        return self._get_llm_response(messages)

    def _format_memory(self, memory: dict) -> str:
        fact = memory["fact"]
        observed_at = memory.get("observed_at")
        if observed_at:
            return f"[{observed_at}] {fact}"
        return fact

    def _encode(self, text):
        if self.model_lock is None:
            vectors = self.model.encode(
                text,
                normalize_embeddings=True,
            )
        else:
            with self.model_lock:
                vectors = self.model.encode(
                    text,
                    normalize_embeddings=True,
                )
        return np.asarray(vectors, dtype=float)

    def _get_llm_response(self, messages: list[dict]) -> str:
        last_error = None

        for _ in range(self.max_retries):
            try:
                response = self.llm.invoke(messages).strip()
                if not response:
                    raise ValueError("The model returned an empty answer.")
                return response
            except Exception as error:
                last_error = error

        raise RuntimeError(
            "The model did not return an answer after "
            f"{self.max_retries} attempts: {last_error}"
        ) from last_error
