from __future__ import annotations

from typing import Any, Dict

from core.schema import KnowledgeChunk, RetrievedChunk


EvidenceItem = Dict[str, Any]


def retrieved_chunk_to_item(hit: RetrievedChunk, *, max_text_chars: int | None = None) -> EvidenceItem:
    text = hit.chunk.text
    if max_text_chars is not None:
        text = text[:max_text_chars]
    return {
        "id": hit.chunk.id,
        "citation_id": hit.citation_id,
        "score": hit.score,
        "vector_score": hit.vector_score,
        "bm25_score": hit.bm25_score,
        "rerank_score": hit.rerank_score,
        "source_type": hit.chunk.source_type,
        "source_path": hit.chunk.source_path,
        "source_id": hit.chunk.source_id,
        "text": text,
        "metadata": hit.chunk.metadata,
    }


def item_to_retrieved_chunk(item: EvidenceItem) -> RetrievedChunk:
    metadata = dict(item.get("metadata") or {})
    if item.get("citation_id"):
        metadata.setdefault("citation_id", item["citation_id"])
    chunk = KnowledgeChunk(
        id=item.get("id") or item.get("citation_id") or "unknown",
        text=item.get("text", ""),
        source_type=item.get("source_type", "unknown"),
        source_path=item.get("source_path", ""),
        source_id=item.get("source_id", item.get("id", "")),
        metadata=metadata,
    )
    return RetrievedChunk(
        chunk=chunk,
        score=float(item.get("score", 0.0)),
        vector_score=float(item.get("vector_score", 0.0)),
        bm25_score=float(item.get("bm25_score", 0.0)),
        rerank_score=item.get("rerank_score"),
    )


def fact_card_to_item(
    *,
    text: str,
    dataset_name: str | None,
    model_name: str | None,
    window_id: str,
) -> EvidenceItem:
    return {
        "id": f"tsfact-{window_id}",
        "citation_id": "S9001",
        "score": 1.0,
        "vector_score": 0.0,
        "bm25_score": 0.0,
        "rerank_score": 0.0,
        "source_type": "ts_fact_card",
        "source_path": f"window:{window_id}",
        "source_id": window_id,
        "text": text,
        "metadata": {
            "citation_id": "S9001",
            "dataset_name": dataset_name,
            "model_name": model_name,
            "window_id": window_id,
        },
    }

