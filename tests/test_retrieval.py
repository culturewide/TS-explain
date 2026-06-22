from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.schema import KnowledgeChunk, RetrievalQuery
from knowledge_base.chunks.embedder import LocalHashingEmbedder
from knowledge_base.retrieval.hybrid_search import HybridSearcher
from knowledge_base.store.vector_store import LocalVectorStore


class RetrievalTest(unittest.TestCase):
    def test_hybrid_search_with_metadata_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks = [
                KnowledgeChunk(
                    id="a",
                    text="Anomaly Transformer 在 SMD 上 F-score 为 0.8751，Recall 较高。",
                    source_type="experiment_result",
                    source_path="results.csv",
                    source_id="row:1",
                    metadata={"model_name": "Anomaly Transformer", "dataset_name": "SMD", "citation_id": "S0001"},
                ),
                KnowledgeChunk(
                    id="b",
                    text="CATCH 使用 channel fusion module 和 frequency patching 处理多变量异常。",
                    source_type="paper_summary",
                    source_path="README.md",
                    source_id="p:1",
                    metadata={"model_name": "CATCH", "citation_id": "S0002"},
                ),
            ]
            embedder = LocalHashingEmbedder()
            store = LocalVectorStore(Path(tmp))
            store.add_chunks(chunks, embedder.embed_texts([chunk.text for chunk in chunks]))
            searcher = HybridSearcher(store, embedder)
            hits = searcher.search(RetrievalQuery("SMD F-score Recall", dataset_name="SMD", top_k=3))
            self.assertEqual(hits[0].chunk.id, "a")
            self.assertEqual(hits[0].citation_id, "S0001")


if __name__ == "__main__":
    unittest.main()

