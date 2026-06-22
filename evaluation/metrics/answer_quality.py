from __future__ import annotations

from dataclasses import dataclass

from llm_interface.response_parser import extract_citations


@dataclass
class AnswerQuality:
    citation_count: int
    has_uncertainty: bool
    has_conclusion: bool
    score: float


def heuristic_quality(answer: str) -> AnswerQuality:
    citation_count = len(extract_citations(answer))
    has_uncertainty = any(word in answer for word in ["不确定", "证据不足", "限制", "核对"])
    has_conclusion = any(word in answer for word in ["结论", "原因", "依据"])
    score = 0.0
    score += min(citation_count, 5) / 5 * 0.45
    score += 0.30 if has_conclusion else 0.0
    score += 0.25 if has_uncertainty else 0.0
    return AnswerQuality(citation_count, has_uncertainty, has_conclusion, round(score, 3))

