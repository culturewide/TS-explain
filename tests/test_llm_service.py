from __future__ import annotations

import unittest
from typing import List

from core.schema import ExplanationRequest, KnowledgeChunk, RetrievedChunk
from llm_interface.base import BaseLLMProvider, ExplanationService
from llm_interface.types import LLMMessage, LLMResponse


def hit(citation_id: str, text: str, source_type: str) -> RetrievedChunk:
    chunk = KnowledgeChunk(
        id=citation_id.lower(),
        text=text,
        source_type=source_type,
        source_path=f"{source_type}.txt",
        source_id=citation_id.lower(),
        metadata={"citation_id": citation_id},
    )
    return RetrievedChunk(chunk=chunk, score=1.0)


class CapturingProvider(BaseLLMProvider):
    provider_name = "capture"
    model_name = "capture"

    def __init__(self) -> None:
        self.messages: List[LLMMessage] = []

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        self.messages = messages
        return LLMResponse(content="Window evidence is cited [S9001].", provider=self.provider_name, model=self.model_name)


class ExplanationServiceTest(unittest.TestCase):
    def test_rag_ts_prompt_does_not_repeat_s9001_context(self) -> None:
        provider = CapturingProvider()
        service = ExplanationService(provider_name="offline")
        service.provider = provider

        service.answer(
            ExplanationRequest(question="Explain the window.", mode="rag-ts", dataset_name="SMD", model_name="DADA"),
            retrieved_override=[
                hit("S9001", "TS-Fact Card window evidence.", "ts_fact_card"),
                hit("S0001", "DADA uses adaptive bottlenecks.", "paper_summary"),
            ],
        )

        user_prompt = "\n\n".join(message.content for message in provider.messages if message.role == "user")
        self.assertEqual(user_prompt.count("S9001"), 1)
        self.assertIn("TS-Fact Card window evidence.", user_prompt)
        self.assertIn("S0001", user_prompt)

    def test_rag_only_prompt_keeps_s9001_as_regular_retrieval_if_supplied(self) -> None:
        provider = CapturingProvider()
        service = ExplanationService(provider_name="offline")
        service.provider = provider

        service.answer(
            ExplanationRequest(question="Explain evidence.", mode="rag-only", dataset_name="SMD", model_name="DADA"),
            retrieved_override=[hit("S9001", "TS-Fact Card window evidence.", "ts_fact_card")],
        )

        user_prompt = "\n\n".join(message.content for message in provider.messages if message.role == "user")
        self.assertEqual(user_prompt.count("S9001"), 1)
        self.assertIn("TS-Fact Card window evidence.", user_prompt)


if __name__ == "__main__":
    unittest.main()
