from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from core.config import load_project_config
from core.schema import ExplanationAnswer, ExplanationRequest, KnowledgeChunk, RetrievedChunk
from feature_extraction.prompt_builder import build_fact_card_text
from knowledge_base.retrieval.retriever import Retriever
from llm_interface.prompts.explanation import build_explanation_messages, retrieved_context
from llm_interface.response_parser import extract_citations
from llm_interface.types import LLMMessage, LLMResponse


class BaseLLMProvider:
    provider_name = "base"
    model_name = "unknown"

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        raise NotImplementedError


def build_provider(config: dict, provider_name: Optional[str] = None) -> BaseLLMProvider:
    provider = provider_name or config.get("llm", {}).get("default_provider", "offline")
    if provider == "qwen":
        from llm_interface.providers.qwen import QwenProvider

        return QwenProvider(config)
    if provider == "deepseek":
        from llm_interface.providers.deepseek import DeepSeekProvider

        return DeepSeekProvider(config)
    from llm_interface.providers.offline import OfflineProvider

    return OfflineProvider(config)


class ExplanationService:
    """Run No-RAG, RAG-Only, and RAG + TS-Feature explanation modes."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        provider_name: Optional[str] = None,
        retriever: Optional[Retriever] = None,
    ) -> None:
        self.config = load_project_config(config_path)
        self.provider = build_provider(self.config, provider_name)
        self.retriever = retriever or Retriever(config_path)

    def answer(
        self,
        request: ExplanationRequest,
        *,
        retrieved_override: Optional[Sequence[RetrievedChunk]] = None,
    ) -> ExplanationAnswer:
        mode = request.mode.lower().replace("_", "-")
        retrieved: List[RetrievedChunk] = []
        if mode in {"rag-only", "rag-ts", "rag+ts-feature", "rag-ts-feature"}:
            if retrieved_override is not None:
                retrieved = list(retrieved_override)
            else:
                retrieved = self.retriever.retrieve(
                    request.question,
                    model_name=request.model_name,
                    dataset_name=request.dataset_name,
                )
        fact_card_text = ""
        if mode in {"rag-ts", "rag+ts-feature", "rag-ts-feature"} and request.fact_card is not None:
            fact_card_text = request.fact_card.narrative or build_fact_card_text(request.fact_card)
            if not any(hit.citation_id == "S9001" for hit in retrieved):
                fact_chunk = KnowledgeChunk(
                    id=f"tsfact-{request.fact_card.window_id}",
                    text=fact_card_text,
                    source_type="ts_fact_card",
                    source_path=f"window:{request.fact_card.window_id}",
                    source_id=request.fact_card.window_id,
                    metadata={
                        "citation_id": "S9001",
                        "dataset_name": request.dataset_name,
                        "model_name": request.model_name,
                        "window_id": request.fact_card.window_id,
                    },
                )
                retrieved.append(RetrievedChunk(chunk=fact_chunk, score=1.0, vector_score=0.0, bm25_score=0.0))
            fact_card_text = f"[S9001] {fact_card_text}"
        elif mode in {"rag-ts", "rag+ts-feature", "rag-ts-feature"}:
            fact_hit = next((hit for hit in retrieved if hit.citation_id == "S9001"), None)
            if fact_hit is not None:
                fact_card_text = f"[S9001] {fact_hit.chunk.text}"

        prompt_retrieved = retrieved
        if fact_card_text:
            prompt_retrieved = [hit for hit in retrieved if hit.chunk.source_type != "ts_fact_card"]
        context = retrieved_context(prompt_retrieved) if mode in {"rag-only", "rag-ts", "rag+ts-feature", "rag-ts-feature"} else ""

        messages = build_explanation_messages(
            question=request.question,
            mode=request.mode,
            retrieval_context=context,
            fact_card_text=fact_card_text,
        )
        response = self.provider.generate(
            messages,
            temperature=float(self.config.get("llm", {}).get("temperature", 0.2)),
            max_tokens=int(self.config.get("llm", {}).get("max_tokens", 1800)),
        )
        return ExplanationAnswer(
            answer=response.content,
            citations=extract_citations(response.content),
            retrieved=retrieved,
            mode=request.mode,
            provider=response.provider,
            raw=response.raw,
        )
