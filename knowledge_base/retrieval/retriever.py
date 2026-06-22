from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from core.config import load_project_config
from core.schema import RetrievedChunk, RetrievalQuery
from knowledge_base.chunks.embedder import build_embedder
from knowledge_base.retrieval.hybrid_search import HybridSearcher
from knowledge_base.retrieval.query_rewriter import infer_filters, rewrite_query
from knowledge_base.store.vector_store import build_vector_store


class Retriever:
    def __init__(self, config_path: str | Path | None = None):
        self.config = load_project_config(config_path)
        self.embedder = build_embedder(self.config)
        self.vector_store = build_vector_store(self.config)
        retrieval_cfg = self.config.get("retrieval", {})
        self.searcher = HybridSearcher(
            self.vector_store,
            self.embedder,
            vector_weight=float(retrieval_cfg.get("vector_weight", 0.60)),
            bm25_weight=float(retrieval_cfg.get("bm25_weight", 0.35)),
            rerank_weight=float(retrieval_cfg.get("rerank_weight", 0.05)),
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        source_types: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        inferred = infer_filters(question)
        resolved_top_k = top_k or int(self.config.get("retrieval", {}).get("top_k", 8))
        resolved_model = model_name or inferred.get("model_name")
        resolved_dataset = dataset_name or inferred.get("dataset_name")
        query = RetrievalQuery(
            query=rewrite_query(question),
            top_k=resolved_top_k,
            model_name=resolved_model,
            dataset_name=resolved_dataset,
            source_types=source_types,
        )
        hits = self.searcher.search(query)

        # Explanation questions usually need more than a single strict slice.
        # If the user filters by dataset, model-level mechanism chunks do not
        # have dataset metadata; if the user filters by model, dataset metadata
        # does not have model metadata. Add small supplemental searches so that
        # "why" answers can cite both performance evidence and mechanism/data
        # context.
        supplements: List[RetrievedChunk] = []
        if source_types is None and resolved_model:
            supplements.extend(
                self.searcher.search(
                    RetrievalQuery(
                        query=rewrite_query(question),
                        top_k=3,
                        model_name=resolved_model,
                        source_types=["paper_summary", "architecture"],
                    )
                )
            )
        if source_types is None and resolved_dataset:
            supplements.extend(
                self.searcher.search(
                    RetrievalQuery(
                        query=rewrite_query(question),
                        top_k=2,
                        dataset_name=resolved_dataset,
                        source_types=["dataset_meta"],
                    )
                )
            )

        if not supplements:
            return hits[:resolved_top_k]

        supplement_by_id = {}
        for hit in supplements:
            if hit.chunk.id not in supplement_by_id or hit.score > supplement_by_id[hit.chunk.id].score:
                supplement_by_id[hit.chunk.id] = hit
        unique_supplements = sorted(supplement_by_id.values(), key=lambda item: item.score, reverse=True)

        results: List[RetrievedChunk] = []
        reserved = min(len(unique_supplements), max(1, resolved_top_k // 2))
        for hit in hits:
            if hit.chunk.id not in supplement_by_id and len(results) < resolved_top_k - reserved:
                results.append(hit)
        for hit in unique_supplements:
            if len(results) < resolved_top_k and all(existing.chunk.id != hit.chunk.id for existing in results):
                results.append(hit)
        for hit in hits:
            if len(results) < resolved_top_k and all(existing.chunk.id != hit.chunk.id for existing in results):
                results.append(hit)
        return results[:resolved_top_k]
