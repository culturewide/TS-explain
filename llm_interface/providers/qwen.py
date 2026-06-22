from __future__ import annotations

import os
from typing import List

from core.config import load_standard_env_files
from llm_interface.base import BaseLLMProvider, LLMMessage, LLMResponse


class QwenProvider(BaseLLMProvider):
    provider_name = "qwen"

    def __init__(self, config: dict):
        load_standard_env_files(config.get("_project_root"))
        cfg = config.get("llm", {}).get("qwen", {})
        self.model_name = cfg.get("model", "qwen-plus")
        self.api_key_env = cfg.get("api_key_env", "DASHSCOPE_API_KEY")
        if not os.getenv(self.api_key_env):
            raise RuntimeError(f"Missing {self.api_key_env}; set it or use provider=offline.")
        try:
            import dashscope  # type: ignore
        except Exception as exc:
            raise RuntimeError("dashscope package is not installed.") from exc
        self.dashscope = dashscope
        self.dashscope.api_key = os.getenv(self.api_key_env)

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        response = self.dashscope.Generation.call(
            model=self.model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1800),
            result_format="message",
        )
        if getattr(response, "status_code", 200) != 200:
            raise RuntimeError(f"Qwen API failed: {response}")
        content = response.output.choices[0].message.content
        return LLMResponse(content=content, provider=self.provider_name, model=self.model_name, raw={"response": str(response)})
