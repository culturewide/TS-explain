from __future__ import annotations

import unittest

from core.schema import ExplanationAnswer, KnowledgeChunk, RetrievedChunk
from evaluation.evaluator import Evaluator
from evaluation.metrics.hallucination import hallucination_rate


def retrieved(citation_id: str, text: str) -> RetrievedChunk:
    chunk = KnowledgeChunk(
        id=citation_id.lower(),
        text=text,
        source_type="paper_summary",
        source_path="README.md",
        source_id=citation_id.lower(),
        metadata={"citation_id": citation_id},
    )
    return RetrievedChunk(chunk=chunk, score=1.0)


class EvaluationTest(unittest.TestCase):
    def test_citation_and_hallucination_metrics(self) -> None:
        chunk = retrieved(
            "S0001",
            "CATCH uses frequency patching and a channel fusion module for multivariate anomaly detection.",
        )
        answer = ExplanationAnswer(
            answer="CATCH uses frequency patching and a channel fusion module for anomaly detection [S0001].",
            citations=["S0001"],
            retrieved=[chunk],
        )
        metrics = Evaluator().evaluate(answer)
        self.assertLessEqual(metrics["hallucination"]["hallucination_rate"], 0.5)
        self.assertEqual(metrics["citation_fidelity"][0]["citation_id"], "S0001")

    def test_existing_citation_without_text_support_is_not_supported(self) -> None:
        report = hallucination_rate(
            "DADA uses a lunar calendar voting module [S0001].",
            [retrieved("S0001", "DADA uses adaptive bottlenecks and dual adversarial decoders.")],
            support_threshold=0.5,
        )
        self.assertEqual(report.citation_exists_claims, 1)
        self.assertEqual(report.supported_claims, 0)
        self.assertEqual(len(report.citation_weak_claims), 1)
        self.assertEqual(report.hallucination_rate, 1.0)
        self.assertEqual(report.citation_support_rate, 0.0)

    def test_existing_citation_with_text_support_is_supported(self) -> None:
        report = hallucination_rate(
            "DADA uses adaptive bottlenecks and dual adversarial decoders [S0001].",
            [retrieved("S0001", "DADA uses adaptive bottlenecks and dual adversarial decoders.")],
            support_threshold=0.5,
        )
        self.assertEqual(report.citation_exists_claims, 1)
        self.assertEqual(report.supported_claims, 1)
        self.assertEqual(report.citation_weak_claims, [])
        self.assertEqual(report.hallucination_rate, 0.0)
        self.assertEqual(report.citation_support_rate, 1.0)

    def test_uncited_claim_can_be_supported_by_high_overlap_evidence(self) -> None:
        report = hallucination_rate(
            "DADA uses adaptive bottlenecks and dual adversarial decoders.",
            [retrieved("S0001", "DADA uses adaptive bottlenecks and dual adversarial decoders.")],
            support_threshold=0.5,
        )
        self.assertEqual(report.citation_exists_claims, 0)
        self.assertEqual(report.supported_claims, 1)
        self.assertEqual(report.hallucination_rate, 0.0)
        self.assertEqual(report.citation_support_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
