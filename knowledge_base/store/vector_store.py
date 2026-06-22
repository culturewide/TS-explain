from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.config import project_root, resolve_path
from core.schema import KnowledgeChunk
from core.text import cosine_similarity


FilterDict = Dict[str, Any]


def _matches_filters(chunk: KnowledgeChunk, filters: Optional[FilterDict]) -> bool:
    if not filters:
        return True
    meta = chunk.metadata
    model = filters.get("model_name")
    if model and str(meta.get("model_name", "")).lower() != str(model).lower():
        return False
    dataset = filters.get("dataset_name")
    if dataset and str(meta.get("dataset_name", "")).lower() != str(dataset).lower():
        return False
    source_types = filters.get("source_types")
    if source_types and chunk.source_type not in set(source_types):
        return False
    return True


class LocalVectorStore:
    """Small deterministic persistent vector index.

    ChromaDB is preferred for production, but this local backend lets the
    project build and test before optional dependencies are installed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.path / "chunks.jsonl"
        self.vectors_path = self.path / "vectors.npy"
        self.manifest_path = self.path / "manifest.json"
        self.chunks: List[KnowledgeChunk] = []
        self.vectors: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if self.chunks_path.exists():
            chunks = []
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                chunks.append(
                    KnowledgeChunk(
                        id=raw["id"],
                        text=raw["text"],
                        source_type=raw["source_type"],
                        source_path=raw["source_path"],
                        source_id=raw["source_id"],
                        metadata=raw.get("metadata", {}),
                    )
                )
            self.chunks = chunks
        if self.vectors_path.exists():
            self.vectors = np.load(self.vectors_path)

    def reset(self) -> None:
        for path in (self.chunks_path, self.vectors_path, self.manifest_path):
            if path.exists():
                path.unlink()
        self.chunks = []
        self.vectors = np.zeros((0, 384), dtype=np.float32)

    def add_chunks(self, chunks: Sequence[KnowledgeChunk], embeddings: Sequence[Sequence[float]]) -> None:
        existing = {chunk.id: idx for idx, chunk in enumerate(self.chunks)}
        vectors = self.vectors.tolist() if self.vectors.size else []
        for chunk, embedding in zip(chunks, embeddings):
            vector = list(map(float, embedding))
            if chunk.id in existing:
                idx = existing[chunk.id]
                self.chunks[idx] = chunk
                vectors[idx] = vector
            else:
                existing[chunk.id] = len(self.chunks)
                self.chunks.append(chunk)
                vectors.append(vector)
        self.vectors = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 384), dtype=np.float32)
        self.persist()

    def persist(self) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(
                    json.dumps(
                        {
                            "id": chunk.id,
                            "text": chunk.text,
                            "source_type": chunk.source_type,
                            "source_path": chunk.source_path,
                            "source_id": chunk.source_id,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        np.save(self.vectors_path, self.vectors)
        self.manifest_path.write_text(
            json.dumps({"backend": "local", "chunk_count": len(self.chunks)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 8,
        filters: Optional[FilterDict] = None,
    ) -> List[Tuple[KnowledgeChunk, float]]:
        if not self.chunks:
            return []
        scored: List[Tuple[KnowledgeChunk, float]] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if _matches_filters(chunk, filters):
                scored.append((chunk, cosine_similarity(embedding, vector.tolist())))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def get_all(self, filters: Optional[FilterDict] = None) -> List[KnowledgeChunk]:
        return [chunk for chunk in self.chunks if _matches_filters(chunk, filters)]


class ChromaVectorStore:
    def __init__(self, persist_dir: str | Path, collection: str):
        try:
            import chromadb  # type: ignore
        except Exception as exc:
            raise RuntimeError("chromadb package is not installed.") from exc
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(collection)

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)

    def add_chunks(self, chunks: Sequence[KnowledgeChunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    **{k: str(v) for k, v in chunk.metadata.items() if v is not None},
                    "source_type": chunk.source_type,
                    "source_path": chunk.source_path,
                    "source_id": chunk.source_id,
                }
                for chunk in chunks
            ],
            embeddings=[list(map(float, emb)) for emb in embeddings],
        )

    def _where(self, filters: Optional[FilterDict]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        clauses: List[Dict[str, Any]] = []
        if filters.get("model_name"):
            clauses.append({"model_name": str(filters["model_name"])})
        if filters.get("dataset_name"):
            clauses.append({"dataset_name": str(filters["dataset_name"])})
        if filters.get("source_types"):
            clauses.append({"source_type": {"$in": list(filters["source_types"])}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 8,
        filters: Optional[FilterDict] = None,
    ) -> List[Tuple[KnowledgeChunk, float]]:
        result = self.collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=top_k,
            where=self._where(filters),
            include=["documents", "metadatas", "distances"],
        )
        chunks: List[Tuple[KnowledgeChunk, float]] = []
        for idx, chunk_id in enumerate(result.get("ids", [[]])[0]):
            meta = result.get("metadatas", [[]])[0][idx] or {}
            document = result.get("documents", [[]])[0][idx]
            distance = result.get("distances", [[]])[0][idx]
            source_type = meta.pop("source_type", "unknown")
            source_path = meta.pop("source_path", "")
            source_id = meta.pop("source_id", chunk_id)
            score = 1.0 / (1.0 + float(distance))
            chunks.append(
                (
                    KnowledgeChunk(
                        id=chunk_id,
                        text=document,
                        source_type=source_type,
                        source_path=source_path,
                        source_id=source_id,
                        metadata=meta,
                    ),
                    score,
                )
            )
        return chunks

    def get_all(self, filters: Optional[FilterDict] = None) -> List[KnowledgeChunk]:
        result = self.collection.get(where=self._where(filters), include=["documents", "metadatas"])
        chunks: List[KnowledgeChunk] = []
        for idx, chunk_id in enumerate(result.get("ids", [])):
            meta = result.get("metadatas", [])[idx] or {}
            document = result.get("documents", [])[idx]
            source_type = meta.pop("source_type", "unknown")
            source_path = meta.pop("source_path", "")
            source_id = meta.pop("source_id", chunk_id)
            chunks.append(KnowledgeChunk(chunk_id, document, source_type, source_path, source_id, meta))
        return chunks


def build_vector_store(config: Dict[str, Any], backend: str | None = None):
    cfg = config.get("vector_store", {})
    backend = backend or cfg.get("backend", "auto")
    root = Path(config.get("_project_root", project_root()))
    if backend in {"chroma", "auto"}:
        try:
            persist_dir = resolve_path(cfg.get("persist_dir", "data/kb/chroma"), root)
            return ChromaVectorStore(persist_dir, cfg.get("collection", "ts_explain_kb"))
        except Exception:
            if backend == "chroma":
                raise
    local_dir = resolve_path(cfg.get("local_dir", "data/kb/local_index"), root)
    return LocalVectorStore(local_dir)

