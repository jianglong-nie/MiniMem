import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        timeout=None,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL_ID")
        if not self.model:
            raise ValueError("Set LLM_MODEL_ID or pass a model name.")

        api_key = api_key or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("Set LLM_API_KEY or pass an API key.")

        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        # DeepSeek thinking switch: "disabled" / "enabled" (unset = provider default)
        self.thinking_type = os.getenv("LLM_THINKING_TYPE") or None
        timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
        )
        if timeout <= 0:
            raise ValueError("LLM timeout must be greater than zero.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("LLM_BASE_URL") or None,
            timeout=timeout,
            # 网络/限流等临时错误由 SDK 带退避地重试;
            # JSON 格式错误的重试在 ConstructMemAgent 里单独处理。
            max_retries=3,
        )

    def invoke(self, messages: list[dict[str, str]]) -> str:
        extra_body = (
            {"thinking": {"type": self.thinking_type}} if self.thinking_type else None
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The model returned an empty response.")
        return content
