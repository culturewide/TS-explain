from __future__ import annotations

import json
import os
import urllib.request
from typing import List

from core.config import load_standard_env_files
from llm_interface.base import BaseLLMProvider, LLMMessage, LLMResponse


class DeepSeekProvider(BaseLLMProvider):
    provider_name = "deepseek"

    def __init__(self, config: dict):
        load_standard_env_files(config.get("_project_root"))
        cfg = config.get("llm", {}).get("deepseek", {})
        self.model_name = cfg.get("model", "deepseek-chat")
        self.base_url = cfg.get("base_url", "https://api.deepseek.com/v1/chat/completions")
        self.api_key_env = cfg.get("api_key_env", "DEEPSEEK_API_KEY")
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing {self.api_key_env}; set it or use provider=offline.")

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 1800),
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, provider=self.provider_name, model=self.model_name, raw=data)
