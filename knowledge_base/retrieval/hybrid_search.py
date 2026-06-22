from __future__ import annotations

from typing import Dict, List, Optional

from core.schema import KnowledgeChunk, RetrievedChunk, RetrievalQuery
from core.text import bm25_scores, keyword_overlap
from knowledge_base.chunks.embedder import BaseEmbedder


class HybridSearcher:
    def __init__(
        self,
        vector_store,
        embedder: BaseEmbedder,
        vector_weight: float = 0.60,
        bm25_weight: float = 0.35,
        rerank_weight: float = 0.05,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rerank_weight = rerank_weight

    def _filters(self, query: RetrievalQuery) -> Dict[str, object]:
        filters: Dict[str, object] = {}
        if query.model_name:
            filters["model_name"] = query.model_name
        if query.dataset_name:
            filters["dataset_name"] = query.dataset_name
        if query.source_types:
            filters["source_types"] = query.source_types
        return filters

    def search(self, query: RetrievalQuery) -> List[RetrievedChunk]:
        filters = self._filters(query)
        q_emb = self.embedder.embed_query(query.query)
        vector_hits = self.vector_store.query(q_emb, top_k=max(query.top_k * 4, 20), filters=filters)
        candidates = {chunk.id: chunk for chunk, _ in vector_hits}
        vector_scores = {chunk.id: score for chunk, score in vector_hits}

        all_filtered: List[KnowledgeChunk] = self.vector_store.get_all(filters=filters)
        bm25 = bm25_scores(query.query, [chunk.text for chunk in all_filtered])
        for idx, score in sorted(bm25.items(), key=lambda item: item[1], reverse=True)[: max(query.top_k * 4, 20)]:
            chunk = all_filtered[idx]
            candidates[chunk.id] = chunk

        bm25_by_id = {all_filtered[idx].id: score for idx, score in bm25.items() if idx < len(all_filtered)}
        results: List[RetrievedChunk] = []
        for chunk_id, chunk in candidates.items():
            vector_score = float(vector_scores.get(chunk_id, 0.0))
            lexical = float(bm25_by_id.get(chunk_id, 0.0))
            rerank = keyword_overlap(query.query, chunk.text)
            score = self.vector_weight * vector_score + self.bm25_weight * lexical + self.rerank_weight * rerank
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_score,
                    bm25_score=lexical,
                    rerank_score=rerank,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: query.top_k]

