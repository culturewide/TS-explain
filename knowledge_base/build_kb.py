from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import load_project_config, load_yaml, project_root, resolve_path
from core.schema import KnowledgeChunk
from knowledge_base.chunks.chunker import assign_citation_ids
from knowledge_base.chunks.embedder import batched, build_embedder
from knowledge_base.ingest import code_ingestor, config_ingestor, meta_ingestor, paper_ingestor, result_ingestor
from knowledge_base.store.metadata_store import MetadataStore
from knowledge_base.store.vector_store import build_vector_store


def build_chunks(asset_dir: Path, chunk_config: dict) -> List[KnowledgeChunk]:
    chunks: List[KnowledgeChunk] = []
    for ingestor in (paper_ingestor, result_ingestor, meta_ingestor, config_ingestor, code_ingestor):
        produced = ingestor.ingest(asset_dir, chunk_config)
        chunks.extend(produced)
    chunks = assign_citation_ids(chunks)
    return chunks


def build_knowledge_base(
    *,
    config_path: str | Path | None = None,
    assets_dir: str | Path | None = None,
    reset: bool = False,
    backend: str | None = None,
) -> dict:
    config = load_project_config(config_path)
    root = Path(config["_project_root"])
    chunk_config = load_yaml(root / "config" / "rag_chunk_config.yaml")
    assets = resolve_path(assets_dir or config["project"]["assets_dir"], root)
    chunks = build_chunks(assets, chunk_config)
    embedder = build_embedder(config)
    store = build_vector_store(config, backend=backend)
    if reset:
        store.reset()

    embeddings: List[List[float]] = []
    texts = [chunk.text for chunk in chunks]
    for batch in batched(texts, size=64):
        embeddings.extend(embedder.embed_texts(batch))
    store.add_chunks(chunks, embeddings)

    kb_dir = resolve_path(config["project"]["kb_dir"], root)
    meta_store = MetadataStore(kb_dir / "metadata.sqlite")
    if reset:
        meta_store.reset()
    meta_store.upsert_chunks(chunks)
    meta_store.close()

    counts = Counter(chunk.source_type for chunk in chunks)
    manifest = {
        "asset_dir": str(assets),
        "chunk_count": len(chunks),
        "source_type_counts": dict(counts),
        "embedding_provider": embedder.name,
        "backend": store.__class__.__name__,
    }
    (kb_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or incrementally update the TS-Explain knowledge base.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--assets", default=None, help="Path to ts-explain-assets")
    parser.add_argument("--reset", action="store_true", help="Clear existing index before adding chunks")
    parser.add_argument("--backend", default=None, choices=["auto", "chroma", "local"], help="Vector backend")
    args = parser.parse_args()
    manifest = build_knowledge_base(
        config_path=args.config,
        assets_dir=args.assets,
        reset=args.reset,
        backend=args.backend,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
