from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

from core.config import load_project_config
from core.schema import ExplanationAnswer
from evaluation.metrics.answer_quality import heuristic_quality
from evaluation.metrics.citation_fidelity import citation_fidelity
from evaluation.metrics.hallucination import hallucination_rate


class Evaluator:
    def __init__(self, config_path: str | Path | None = None):
        self.config = load_project_config(config_path)
        self.support_threshold = float(self.config.get("evaluation", {}).get("support_threshold", 0.18))

    def evaluate(self, answer: ExplanationAnswer) -> Dict[str, object]:
        hallucination = hallucination_rate(
            answer.answer,
            answer.retrieved,
            support_threshold=self.support_threshold,
        )
        fidelity = citation_fidelity(answer.answer, answer.retrieved)
        quality = heuristic_quality(answer.answer)
        valid_citations = [item for item in fidelity if item.exists and item.overlap >= 0.12]
        citation_consistency_rate = (len(valid_citations) / len(fidelity)) if fidelity else 0.0
        return {
            "mode": answer.mode,
            "provider": answer.provider,
            "hallucination": asdict(hallucination),
            "citation_fidelity": [asdict(item) for item in fidelity],
            "citation_consistency_rate": citation_consistency_rate,
            "valid_citation_count": len(valid_citations),
            "citation_check_count": len(fidelity),
            "answer_quality": asdict(quality),
        }
