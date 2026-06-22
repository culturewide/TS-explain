from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping

from core.schema import KnowledgeChunk, Metadata
from core.text import chunk_by_paragraphs, normalize_text, stable_hash


def make_chunk_id(source_path: str, source_id: str, text: str) -> str:
    return stable_hash(f"{source_path}|{source_id}|{text}", length=20)


def record_chunk(
    *,
    text: str,
    source_type: str,
    source_path: str | Path,
    source_id: str,
    metadata: Mapping[str, object] | None = None,
) -> KnowledgeChunk:
    clean = normalize_text(text)
    rel_path = str(source_path).replace("\\", "/")
    chunk_id = make_chunk_id(rel_path, source_id, clean)
    return KnowledgeChunk(
        id=chunk_id,
        text=clean,
        source_type=source_type,
        source_path=rel_path,
        source_id=source_id,
        metadata=dict(metadata or {}),
    )


def paragraph_chunks(
    *,
    text: str,
    source_type: str,
    source_path: str | Path,
    source_id_prefix: str,
    metadata: Mapping[str, object] | None = None,
    max_chars: int = 1200,
    overlap_chars: int = 120,
    min_chars: int = 80,
) -> List[KnowledgeChunk]:
    pieces = chunk_by_paragraphs(text, max_chars=max_chars, overlap_chars=overlap_chars, min_chars=min_chars)
    chunks: List[KnowledgeChunk] = []
    for idx, piece in enumerate(pieces, start=1):
        chunks.append(
            record_chunk(
                text=piece,
                source_type=source_type,
                source_path=source_path,
                source_id=f"{source_id_prefix}:{idx}",
                metadata=metadata,
            )
        )
    return chunks


def assign_citation_ids(chunks: Iterable[KnowledgeChunk], prefix: str = "S") -> List[KnowledgeChunk]:
    assigned: List[KnowledgeChunk] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata.setdefault("citation_id", f"{prefix}{idx:04d}")
        assigned.append(chunk)
    return assigned

