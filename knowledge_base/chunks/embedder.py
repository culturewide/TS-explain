from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from core.text import hashing_embedding


class BaseEmbedder:
    name = "base"
    dimension = 384

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


@dataclass
class LocalHashingEmbedder(BaseEmbedder):
    dimension: int = 384
    name: str = "local_hashing"

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [hashing_embedding(text, dim=self.dimension) for text in texts]


@dataclass
class OpenAIEmbedder(BaseEmbedder):
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    dimension: int = 1536
    name: str = "openai"

    def __post_init__(self) -> None:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env}; use local_hashing or set the API key.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package is not installed.") from exc
        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [list(item.embedding) for item in response.data]


@dataclass
class SentenceTransformerEmbedder(BaseEmbedder):
    model: str = "BAAI/bge-large-zh-v1.5"
    dimension: int = 1024
    name: str = "sentence_transformer"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError("sentence-transformers package is not installed.") from exc
        self._model = SentenceTransformer(self.model)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors]


def batched(items: Sequence[str], size: int = 64) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_embedder(config: dict | None = None) -> BaseEmbedder:
    config = config or {}
    emb_cfg = config.get("embedding", config)
    provider = emb_cfg.get("provider", "local_hashing")
    if provider == "openai":
        return OpenAIEmbedder(
            model=emb_cfg.get("model", "text-embedding-3-small"),
            api_key_env=emb_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
            dimension=int(emb_cfg.get("dimension", 1536)),
        )
    if provider in {"bge", "sentence_transformer", "sentence-transformer"}:
        return SentenceTransformerEmbedder(
            model=emb_cfg.get("bge_model", "BAAI/bge-large-zh-v1.5"),
            dimension=int(emb_cfg.get("dimension", 1024)),
        )
    return LocalHashingEmbedder(dimension=int(emb_cfg.get("dimension", 384)))

